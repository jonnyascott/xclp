import os
import pickle
import random
import time
import math
import numpy as np
import torch

from helpers import average_models, create_client_dataloader, create_val_loader, create_model, train, evaluate,\
    save_checkpoint

from pseudo_labelling import run_psuedo_labelling, generate_network_plabels
from config import parse_args, set_args


def main():
    args = parse_args()
    args.fully_supervised = False
    set_args(args)

    if torch.cuda.is_available():
        print(f"Using cuda device {args.gpu_id}. \n")
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        print("Using CPU. \n")
        device = 'cpu'

    base_dir = os.path.join('all_results', args.dataset, f"{args.num_labels}_labels",
                            f'{args.iid}',
                            f'{args.num_clients}_clients_{args.num_labelled_clients}_labelledclients')

    MT_str = 'MT_' if args.isMT else ''
    load_model_name = f'stage1_{MT_str}{args.architecture}_rounds_{args.start_round}_split_{args.label_split}'

    if args.plabel_network == 0:
        if args.lp_computation == 'local':
            pseudo_labelling_setting = '_localLP'
        else:
            pseudo_labelling_setting = '_CrossClientLP'
    else:
        pseudo_labelling_setting = f'_NetworkPsuedoLabels_{args.plabel_network}'

    model_name = f'stage2_{MT_str}{args.architecture}_rounds_{args.start_round}' \
                 f'_epochs_{args.local_epochs}_split_{args.label_split}' + pseudo_labelling_setting

    checkpoint_path = os.path.join(base_dir, model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    print(f"Saving model to {model_name}")

    # create client data loaders
    client_loaders = []
    for i in range(args.num_clients):
        train_loader, train_loader_noshuff, labelled_idx = create_client_dataloader(i, args)
        client_loaders.append((train_loader, train_loader_noshuff, labelled_idx))

    val_loader, test_loader = create_val_loader(args)

    # create models, one model for each client that is sampled per round, and one global model
    print(f'Using {args.architecture} architecture.')
    models = []
    clients_per_round = max(1, int(args.num_clients * args.client_frac))
    for _ in range(clients_per_round + 2):
        model = create_model(args)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0,
                                    nesterov=False,
                                    weight_decay=args.weight_decay
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_rampdown, verbose=False)

        if args.isMT:
            ema_model = create_model(args, ema=True)
            ema_model.to(device)
            models.append((model, optimizer, scheduler, ema_model))
        else:
            models.append((model, optimizer, scheduler))

    if args.isMT:
        global_model, global_optimizer, _, global_ema_model = models.pop(-1)
        feature_model, _, _, _ = models.pop(-1)
    else:
        global_model, global_optimizer, _ = models.pop(-1)
        feature_model, _, _ = models.pop(-1)

    # Load the model from Stage 1
    resume_path = os.path.join(base_dir, load_model_name, f'epoch_{args.start_round}.ckpt')
    print(f'loading model from {resume_path}')
    assert os.path.isfile(resume_path), f"=> no checkpoint found at '{resume_path}'"
    checkpoint = torch.load(resume_path, map_location=device) # f'cuda:{args.gpu_id}'
    global_model.load_state_dict(checkpoint['model_state_dict'])
    feature_model.load_state_dict(checkpoint['model_state_dict'])
    global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for param_group in global_optimizer.param_groups:
        param_group['lr'] = args.learning_rate

    if args.isMT:
        ema_resume_path = os.path.join(base_dir, load_model_name, f'EMA_epoch_{args.start_round}.ckpt')
        ema_checkpoint = torch.load(ema_resume_path, map_location=device) # f'cuda:{args.gpu_id}'
        global_ema_model.load_state_dict(ema_checkpoint['model_state_dict'])

    print('')
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'pseudo_label_accuracy': [],
               'p_weight_correct': [], 'p_weight_incorrect': [], 'ema_loss': [], 'ema_accuracy': []}
    global_step = 0
    evaluate(val_loader, global_model, device)

    if args.plabel_network == 2:
        generate_network_plabels(client_loaders, global_model, device, args)

    for round in range(1, args.rounds + 1):
        start_time = time.time()
        print(f"Starting epoch {round} / {args.rounds}")

        # update all client model weights
        sd = global_model.state_dict()
        for model, *_ in models:
            model.load_state_dict(sd)

        if args.isMT:
            # update all client ema_model weights
            ema_sd = global_ema_model.state_dict()
            for _, _, _, ema_model in models:
                ema_model.load_state_dict(ema_sd)

        feature_model.load_state_dict(sd)

        # sample clients
        labelled_clients_per_round = max(1, int(math.ceil(clients_per_round * args.labelled_client_frac)))
        if args.lp_computation == 'local':
            labelled_clients_per_round = clients_per_round
        labelled_clients = random.sample(client_loaders[:args.num_labelled_clients], labelled_clients_per_round)

        unlabelled_clients_per_round = clients_per_round - labelled_clients_per_round
        unlabelled_clients = random.sample(client_loaders[args.num_labelled_clients:], unlabelled_clients_per_round)
        sampled_clients = labelled_clients + unlabelled_clients

        # Compute and update pseudo-labels
        if args.plabel_network == 0:
            statistics = run_psuedo_labelling(sampled_clients, feature_model, device, args)
        elif args.plabel_network == 1:
            statistics = generate_network_plabels(sampled_clients, global_model, device, args)
        else:
            statistics = {'p_label_accuracy': None, 'p_weight_correct': None, 'p_weight_incorrect': None}

        # Train at each client
        client_train_losses, client_train_accuracies = [], []
        for j, ((client_data, _, _), model_tup) in enumerate(zip(sampled_clients, models)):
            if args.isMT:
                model, optimizer, _, ema_model = model_tup
            else:
                model, optimizer, _ = model_tup

            print(f'Training client {j + 1} / {clients_per_round}')
            for epoch in range(args.local_epochs):
                if args.isMT:
                    train_loss, train_accuracy, global_step = train(client_data, model, optimizer, device, args,
                                                                    round, global_step, ema_model=ema_model)
                else:
                    train_loss, train_accuracy, global_step = train(client_data, model, optimizer, device, args,
                                                                    round, global_step)
            client_train_losses.append(train_loss)
            client_train_accuracies.append(train_accuracy)

        # Average client models, set weights of global model to this average
        client_models = [x[0] for x in models]
        average_models(client_models, global_model)

        if args.isMT:
            ema_models = [x[3] for x in models]
            average_models(ema_models, global_ema_model)

        # update learning rates
        for scheduler in [x[2] for x in models]:
            scheduler.step()

        if round % args.save_checkpoint == 0:
            if args.isMT:
                save_checkpoint(round, checkpoint_path, global_model, global_optimizer, ema_model=global_ema_model)
            else:
                save_checkpoint(round, checkpoint_path, global_model, global_optimizer)

        mean_train_loss = np.mean(client_train_losses)
        mean_train_acc = np.mean(client_train_accuracies)

        print(f'Train Accuracy: {mean_train_acc:.2f}, Train Loss: {mean_train_loss:.4f}')

        history['train_loss'].append(mean_train_loss)
        history['train_accuracy'].append(mean_train_acc)
        history['pseudo_label_accuracy'].append(statistics['p_label_accuracy'])
        history['p_weight_correct'].append(statistics['p_weight_correct'])
        history['p_weight_incorrect'].append(statistics['p_weight_incorrect'])

        if round % 10 == 0:
            val_loss, val_accuracy = evaluate(val_loader, global_model, device)
            print('')
            print(f"Validation Accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.4f}")
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            if val_accuracy >= max(history['val_accuracy']):
                best_round = round
            if args.isMT:
                print('Evaluating ema model...')
                ema_loss, ema_accuracy = evaluate(val_loader, global_ema_model, device)
                print(f"Validation Accuracy: {ema_accuracy:.2f}, Validation loss: {ema_loss:.4f}")
                history['ema_loss'].append(ema_loss)
                history['ema_accuracy'].append(ema_accuracy)

        print(f"Finished round in {time.time() - start_time:.2f} seconds. \n")

    test_loss, test_accuracy = evaluate(test_loader, global_model, device)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    history['test'] = (test_loss, test_accuracy)

    best_checkpoint = torch.load(os.path.join(checkpoint_path, f'epoch_{best_round}.ckpt'))
    global_model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_accuracy = evaluate(test_loader, global_model, device)
    print('')
    print(f'Best Test Loss: {test_loss:.4f}')
    print(f'Best Test Accuracy: {test_accuracy:.2f}')
    history['best_test'] = (test_loss, test_accuracy)

    with open(os.path.join(checkpoint_path, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    main()
