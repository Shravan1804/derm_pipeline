import os


def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def maybe_create(*d):
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def list_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def set_seeds(seed, random=None, np=None, torch=None):
    if random:
        random.seed(seed)
    if np:
        np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)

def check_args(args):
    args.data = args.data.rstrip('/')
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Provided dataset dir {args.data} invalid."


def add_obj_detec_args(parser):
    parser.add_argument('--obj-detec', action='store_true', help="if dataset is obj detec dataset")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")


def load_fastai_model(model_path):
    import fastai.vision as fvision
    from radam import *
    return fvision.load_learner(os.path.dirname(model_path), os.path.basename(model_path))

