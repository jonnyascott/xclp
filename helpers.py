import numpy as np
import os
import pandas as pd
from collections import OrderedDict

import torch
from torch import nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from architectures import CNN, ResNet18
from datasets import RandomTranslateWithReflect, LocalImageDataset, DoubleDataset, TwoStreamBatchSampler
from losses import softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


# Create data loader for client
def create_client_dataloader(client_idx, args):
    path_to_labels = os.path.join('data-local', 'labels', args.dataset, 'train')
    client_label_file = f'{args.num_labels}_labels_{args.num_clients}_clients_{args.num_labelled_clients}' \
                        f'_labelledclients_{args.iid}_{args.dist_num}_split_{args.label_split}.txt'
    client_labels = pd.read_csv(os.path.join(path_to_labels, client_label_file), sep=' ', header=None, dtype=str)
    labelled_data = client_labels.loc[(client_labels.iloc[:, 2] == str(client_idx)) * (client_labels.iloc[:, 3] == '1')].iloc[:, :2].values
    unlabelled_data = client_labels.loc[(client_labels.iloc[:, 2] == str(client_idx)) * (client_labels.iloc[:, 3] == '0')].iloc[:, :2].values

    if args.dataset.startswith('cifar'):
        train_img_dir = os.path.join('data-local', 'images', 'cifar', args.dataset, 'by-image', 'train')
        channel_stats = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}
        train_transform = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    else:
        train_img_dir = os.path.join('data-local', 'images', 'miniimagenet', 'train')
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        channel_stats = {"mean": mean_pix, "std": std_pix}
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

    if args.isMT:
        train_transform = TransformTwice(train_transform)

    classes = sorted(os.listdir(train_img_dir))
    label_dict = dict(zip(classes, range(len(classes))))

    labelled_train_dataset = LocalImageDataset(labelled_data, train_img_dir, train_transform, label_dict,
                                               p_labels=[label_dict[x[1]] for x in labelled_data])

    if args.fully_supervised:
        assert labelled_data.shape[0] > 0
        train_loader = DataLoader(labelled_train_dataset, batch_size=args.labelled_batch_size, num_workers=args.workers, shuffle=True)
        train_loader_noshuff = DataLoader(labelled_train_dataset, batch_size=args.labelled_batch_size, num_workers=args.workers,
                                          shuffle=False)
        labelled_idx = np.arange(len(labelled_train_dataset))

    else:
        unlabelled_train_dataset = LocalImageDataset(unlabelled_data, train_img_dir, train_transform, label_dict)
        if client_idx < args.num_labelled_clients:
            joint_train_dataset = DoubleDataset(unlabelled_train_dataset, labelled_train_dataset)
            unlabelled_idx = np.arange(joint_train_dataset.n1).astype(int)
            labelled_idx = np.arange(joint_train_dataset.n1, len(joint_train_dataset)).astype(int)
            batch_sampler = TwoStreamBatchSampler(unlabelled_idx, labelled_idx,
                                                  args.batch_size, args.labelled_batch_size)
            train_loader = DataLoader(joint_train_dataset, batch_sampler=batch_sampler, num_workers=args.workers)

            train_loader_noshuff = DataLoader(joint_train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False)  # Applies augmentation to this
        else:
            train_loader = DataLoader(unlabelled_train_dataset, batch_size=args.batch_size, num_workers=args.workers)
            train_loader_noshuff = DataLoader(unlabelled_train_dataset, batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False)  # Applies augmentation to this
            labelled_idx = np.array([])

    return train_loader, train_loader_noshuff, labelled_idx


def create_val_loader(args):
    if args.dataset.startswith('cifar'):
        channel_stats = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}
        val_img_dir = os.path.join('data-local', 'images', 'cifar', args.dataset, 'by-image', 'val')
        test_img_dir = os.path.join('data-local', 'images', 'cifar', args.dataset, 'by-image', 'test')

    else:
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        channel_stats = {"mean": mean_pix, "std": std_pix}
        val_img_dir = os.path.join('data-local', 'images', 'miniimagenet', 'test')
        test_img_dir = os.path.join('data-local', 'images', 'miniimagenet', 'test')

    val_labels_file = os.path.join('data-local', 'labels', args.dataset, 'val', 'val_labels.txt')
    test_labels_file = os.path.join('data-local', 'labels', args.dataset, 'test', 'test_labels.txt')

    classes = sorted(os.listdir(val_img_dir))
    label_dict = dict(zip(classes, range(len(classes))))
    val_labels = pd.read_csv(val_labels_file, sep=' ', header=None).values
    test_labels = pd.read_csv(test_labels_file, sep=' ', header=None).values

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    val_dataset = LocalImageDataset(val_labels, val_img_dir, val_transform, label_dict)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_dataset = LocalImageDataset(test_labels, test_img_dir, val_transform, label_dict)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    return val_loader, test_loader


def create_model(args, ema=False):
    num_classes = 10 if args.dataset == 'cifar10' else 100
    if args.architecture == 'cnn':
        model = CNN(num_classes=num_classes, isL2=args.isL2, double_output=args.isMT)
    elif args.architecture == 'resnet18':
        model = ResNet18(num_classes=num_classes, isL2=args.isL2, double_output=args.isMT)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def average_models(models, averaged_model):
    averaged_weights = OrderedDict()
    sd = averaged_model.state_dict()
    for key in sd:
        for model in models:
            try:
                averaged_weights[key] += model.state_dict()[key].cpu() / len(models)
            except KeyError:
                averaged_weights[key] = model.state_dict()[key].cpu() / len(models)
    averaged_model.load_state_dict(averaged_weights)


def update_ema_variables(model, ema_model, alpha_scal, global_step):
    # Use the true average until the exponential average is more correct
    alpha_scal = min(1 - 1 / (global_step + 1), alpha_scal)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_scal).add_(param.data, alpha=1 - alpha_scal)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train(dataloader, model, optimizer, device, args, epoch, global_step, ema_model=None):
    isMT = ema_model is not None

    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = symmetric_mse_loss

    num_batches = len(dataloader)
    losses = []

    model.train()
    accuracy = 0
    for i, (X, _, y_pseudo, p_weight, c_weight) in enumerate(dataloader):

        # in MT case unpack double input
        if isMT:
            X, ema_X = X
            ema_X = ema_X.to(device)

        X, y_pseudo = X.to(device), y_pseudo.to(device)
        p_weight, c_weight = p_weight.to(device), c_weight.to(device)

        minibatch_size = len(X)

        if isMT:
            class_logit, cons_logit, _ = model(X)
            with torch.no_grad():
                ema_logit, _, _ = ema_model(ema_X)

            if args.logit_distance_cost >= 0:
                res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            else:
                res_loss = 0

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            else:
                consistency_loss = 0

        else:
            class_logit, _ = model(X)

        loss = loss_fn(class_logit, y_pseudo) * p_weight * c_weight
        loss = loss.mean()

        if isMT:
            loss = loss + res_loss + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        if isMT:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        accuracy += (class_logit.argmax(1) == y_pseudo).type(torch.float).mean().item() / num_batches
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss, accuracy, global_step


def evaluate(dataloader, model, device):
    loss_fn = nn.CrossEntropyLoss()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    eval_loss, correct = 0, 0
    with torch.no_grad():
        for X, y, _, _, _ in dataloader:
            X, y = X.to(device), y.to(device)
            preds, *_ = model(X)
            eval_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y).type(torch.float).sum().item()

    eval_loss /= num_batches
    accuracy = correct / size

    return eval_loss, accuracy


def save_checkpoint(epoch, checkpoint_path, model, optimizer, ema_model=None):
    print(f"Saving Checkpoint at epoch {epoch}")
    save_path = os.path.join(checkpoint_path, f'epoch_{epoch}.ckpt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': torch.nn.CrossEntropyLoss(),
    }, save_path)
    if ema_model is not None:
        f"Saving EMA Checkpoint at epoch {epoch}"
        save_path = os.path.join(checkpoint_path, f'EMA_epoch_{epoch}.ckpt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': ema_model.state_dict(),
        }, save_path)
