from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='semantic matching model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')  # use -1 for CPU
    parser.add_argument('--seed', type=int, default=42, help='the answer to life, the universe and everything.')
    parser.add_argument('--save_path', type=str, default='./snapshots/')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=128)

    # parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    return args
