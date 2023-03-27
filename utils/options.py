import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default="trimmed_mean", help="aggregation method")
    parser.add_argument('--n_clients', type=int, default=20, help="number of clients")
    parser.add_argument('--num_epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--dirichlet_alpha', type=int, default=0.1, help="")

    args = parser.parse_args()

    return args