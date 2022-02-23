import argparse
from utils.plots import *


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--logs_dir',
        help='path of the logs directory, should contain sub-directories corresponding to method '
             ' each sub-directory must contain a single tf events file',
        type=str
    )
    parser.add_argument(
        '--save_dir',
        help='path to save the plots',
        type=str
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    for tag in TAGS:
        make_plot(logs_dir_=args.logs_dir, tag_=tag, save_path=args.save_dir)

