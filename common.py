import os


def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def maybe_create(*d):
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def add_obj_detec_args(parser):
    parser.add_argument('--obj-detec', action='store_true', help="if dataset is obj detec dataset")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")

