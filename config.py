import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients")
    parser.add_argument('--num_labelled_clients', '-p', type=int, default=100, help="number of clients with labels")
    parser.add_argument('--client_frac', '-q', type=float, default=0.05, help="fraction of clients per round")
    parser.add_argument('--dataset', '-d', type=str, default="cifar10", help="dataset to train on")
    parser.add_argument('--batch_size', '-b', type=int, default=100, help="batch size")
    parser.add_argument('--labelled_batch_size', type=int, default=50, help="number of labelled in batch")
    parser.add_argument('--isL2', '-i', type=bool, default=True, help="are features L2 normalized")
    parser.add_argument('--fully_supervised', '-f', type=bool, default=False, help="Train only with labelled data")
    parser.add_argument('--label_split', '-s', type=int, default=0, help="Which data split to use")
    parser.add_argument('--num_labels', '-n', type=int, default=5000, help="number of labelled training points")
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, help="weight decay")
    parser.add_argument('--learning_rate', '-l', type=float, default=0.3, help="learning rate")
    parser.add_argument('--lr_rampdown', '-r', type=int, default=2000, help="num epochs till 0 lr")
    parser.add_argument('--rounds', '-m', type=int, default=1500, help="number of rounds to train for")
    parser.add_argument('--local_epochs', '-e', type=int, default=5, help="number of local epochs to train per round")
    parser.add_argument('--workers', '-j', default=2, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_checkpoint', '-c', default=10, type=int, help='how often to save model checkpoint')
    parser.add_argument('--dfs_k', '-k', default=10, type=int, help='number of neighbors for label propagation')
    parser.add_argument('--start_round', '-z', default=1500, type=int, help='number of rounds from stage 1 training')
    parser.add_argument('--gpu_id', '-g', default=0, type=int, help='which gpu to use')
    parser.add_argument('--iid', default='iid', type=str, help='labels are distributed iid or noniid')
    parser.add_argument('--lp_computation', default='global', type=str, help='is LP step global or local')
    parser.add_argument('--index_type', default='LSH', choices=['FlatIP', 'LSH'], help='which faiss index to use')
    parser.add_argument('--n_bits', default=4096, type=int, help='n_bits for LSH hashing')
    parser.add_argument('--plabel_network', default=0, type=int, help='use network for plabelling, 0:no, 1:per_round, 2:fedsem')
    parser.add_argument('--isMT', default=0, type=int, help='whether to combine with mean teacher')
    parser.add_argument('--logit_distance_cost', default=0.01, type=float, help='cost of double output mean teacher')
    parser.add_argument('--consistency', default=100, type=float, help='consistency loss weighting')
    parser.add_argument('--consistency_type', default='mse', type=str, help='which consistency loss to use')
    parser.add_argument('--consistency-rampup', default=5, type=int, help='length of the consistency loss ramp-up')
    parser.add_argument('--ema_decay', default=0.97, type=float, help='decay for ema of ema_model')
    args = parser.parse_args()
    return args


def set_args(args):
    args.architecture = 'cnn' if 'cifar' in args.dataset else 'resnet18'

    if args.dataset in ['cifar100', 'miniimagenet'] and args.num_labels == 5000:
        args.num_clients = 50
        args.num_labelled_clients = 50
        args.client_frac = 0.1

    args.rounds = 1500
    args.start_round = args.rounds

    args.labelled_batch_size = int(min(50, args.num_labels / args.num_labelled_clients))
    args.batch_size = 2 * args.labelled_batch_size

    args.labelled_client_frac = args.num_labelled_clients / args.num_clients

    if args.iid == 'iid':
        args.dist_num = 0
    else:
        args.dist_num = 5

    if args.labelled_batch_size >= 50:
        args.learning_rate = 0.3

    elif args.labelled_batch_size >= 20:
        args.learning_rate = 0.1

    else:
        args.learning_rate = 0.05

    if args.isMT:
        args.isL2 = False
