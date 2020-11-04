import argparse

import common


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates crops satisfying objects proportion threshold")
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the crops should be saved")
    parser.add_argument('--threshs', nargs='+', default=[.01], type=float, help="Object proportion thresholds")
    parser.add_argument('--bg', default=0, type=int, help="Background mask code")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--level', default=0, type=int, help="nested level of class folders compared to args.data")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    args.threshs = sorted(args.threshs)

    if args.dest is None:
        args.dest = common.maybe_create(f'{args.data}_cropped_{"_".join(map(str, args.patch_sizes))}')

    common.time_method(main, args)
