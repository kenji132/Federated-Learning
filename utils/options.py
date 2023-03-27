import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default="trimmed_mean", help="aggregation method")
    parser.add_argument('--n_clients', type=int, default=20, help="number of clients")
    parser.add_argument('--n_epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help="dirichlet disrtibution")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--trim_factor', type=float, default=0.15, help="the proportion trimmed by trimmed mean")
    parser.add_argument('--mp_lambda', type=int, default=1000000, help="scaling factor in model poisoning attack" )
    parser.add_argument('--opt', type=str, default="Adam", help="optimization function")
    parser.add_argument('--fake_client_rate', type=float, default=0.3, help="fake client rate")

    args = parser.parse_args()

    return args