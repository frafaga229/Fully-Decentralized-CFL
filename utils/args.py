import argparse


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str
    )
    parser.add_argument(
        'method',
        help='the method to be used;'
             ' possible are `FedAvg`, `local` and `clustered`;',
        type=str
    )
    parser.add_argument(
        '--sampling_rate',
        help='proportion of clients to be used at each round; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--input_dimension',
        help='the dimension of one input sample; only used for synthetic dataset',
        type=int,
        default=None
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bz',
        help='batch_size; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='frequency of writing logs; defaults is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate; default is 1e-3',
        default=1e-3
    )
    parser.add_argument(
        "--lr_scheduler",
        help='learning rate decay scheme to be used;'
             ' possible are "sqrt", "linear", "cosine_annealing", "multi_step" and "constant" (no learning rate decay);'
             'default is "constant"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--verbose",
        help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
        type=int,
        default=0
    )
    parser.add_argument(
        "--logs_dir",
        help='directory to write logs; if not passed, it is set using arguments',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--save_path",
        help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--seed",
        help='random seed',
        type=int,
        default=1234
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
