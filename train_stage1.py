import numpy as np
import os
import pickle
import random
import time
import torch

from helpers import average_models, create_client_dataloader, create_val_loader, create_model, train, evaluate, \
    save_checkpoint
from config import parse_args, set_args


def main():
    args = parse_args()
    args.fully_supervised = True if not args.isMT else False
    set_args(args)

    if torch.cuda.is_available():
        print(f"Using CUDA device {args.gpu_id}. \n")
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        print("Using CPU.")
        device = 'cpu'

    # make save directory
    MT_str = 'MT_' if args.isMT else ''
    model_name = f'stage1_{MT_str}{args.architecture}_rounds_{args.rounds}_split_{args.label_split}'

    base_dir = os.path.join('all_results', args.dataset, f"{args.num_labels}_labels",
                            f'{args.iid}',
                            f'{args.num_clients}_clients_{args.num_labelled_clients}_labelledclients')
    checkpoint_path = os.path.join(base_dir, model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Saving model to {model_name}")

    client_loaders = []
    # create client data loaders for clients with labelled data or both labelled and unlabelled if isMT
    for i in range(args.num_labelled_clients):
        train_loader, _, _ = create_client_dataloader(i, args)
        client_loaders.append(train_loader)
    clients_per_round = max(1, int(args.num_clients * args.client_frac))

    val_loader, test_loader = create_val_loader(args)

    # create models, one model for each client that is sampled per round, and one global model
    print(f'Using {args.architecture} architecture.')
    models = []
    for _ in range(clients_per_round + 1):
        model = create_model(args)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
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
    else:
        global_model, global_optimizer, _ = models.pop(-1)

    print('')
    history = {'train_loss': [], 'train_accuracy': [],
               'val_loss': [], 'val_accuracy': [],
               'ema_loss': [], 'ema_accuracy': []}
    global_step = 0

    for round in range(1, args.rounds + 1):
        start_time = time.time()
        print(f"Starting round {round} / {args.rounds}")

        # update all client model weights
        sd = global_model.state_dict()
        for model, *_ in models:
            model.load_state_dict(sd)

        if args.isMT:
            # update all client ema_model weights
            sd = global_ema_model.state_dict()
            for _, _, _, ema_model in models:
                ema_model.load_state_dict(sd)

        # sample clients
        sampled_clients = random.sample(client_loaders, clients_per_round)
        client_train_losses, client_train_accuracies = [], []

        # Train at each client
        for j, (client_data, model_tup) in enumerate(zip(sampled_clients, models)):
            if args.isMT:
                model, optimizer, _, ema_model = model_tup
            else:
                model, optimizer, _ = model_tup

            print(f'Training client {j+1} / {clients_per_round}')
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

        if round % 10 == 0:
            val_loss, val_accuracy = evaluate(val_loader, global_model, device)
            print('')
            print(f"Validation Accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.4f}")
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            if args.isMT:
                print('Evaluating ema model...')
                ema_loss, ema_accuracy = evaluate(val_loader, global_ema_model, device)
                print(f"Validation Accuracy: {ema_accuracy:.2f}, Validation loss: {ema_loss:.4f}")
                history['ema_loss'].append(ema_loss)
                history['ema_accuracy'].append(ema_accuracy)

        print(f"Finished round in {time.time() - start_time:.2f} seconds. \n")

    test_loss, test_accuracy = evaluate(test_loader, global_model, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')
    history['test'] = (test_loss, test_accuracy)

    with open(os.path.join(checkpoint_path, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
