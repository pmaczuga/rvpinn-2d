import argparse
from src.run import run
from src.pinn_core import PINN

from src.utils import *
from src.params import Params

parser = argparse.ArgumentParser(
                    prog='RVPINN-2D',
                    description='Runs the training of RVPINN-2D')

parser.add_argument('--tag', type=str, 
                    help='Name of the folder for storing training results')

parser.add_argument('--epochs', type=int)
parser.add_argument('--layers', type=int)
parser.add_argument('--neurons-per-layer', type=int)
parser.add_argument('--learning-rate', type=float)

parser.add_argument('--equation', '-e', type=str, 
                    help=('Equation to use - either "sins" or "exp-sins"'))
parser.add_argument('--epsilon', type=float)

parser.add_argument("--compute-error", type=bool, action=argparse.BooleanOptionalAction,
                    help='Whether to compute error in each iteration will influence performance')
parser.add_argument("--n-points-x", type=int,
                    help='Number of integration nodes in x axis')
parser.add_argument("--n-points-t", type=int,
                    help='Number of integration nodes in t axis')
parser.add_argument("--n-points-x-error", type=int,
                    help=('Number of integration nodes for computing error in x axis. '
                         'Ideally greater or equal than n_points_x'))
parser.add_argument("--n-points-t-error", type=int,
                    help=('Number of integration nodes for computing error in t axis. '
                         'Ideally greater or equal than n_points_x'))
parser.add_argument("--n-test-x", type=int, 
                    help='Number of test functions in x axis')
parser.add_argument("--n-test-t", type=int, 
                    help='Number of test functions in t axis')
parser.add_argument("--integration-rule-loss", type=str,
                    help='Integration rule for loss function. Only "midpoint for now.')
parser.add_argument("--integration-rule-error", type=str,
                    help='Integration rule for calculating error. Only "midpoint for now.')

parser.add_argument("--params", type=str, default="params.ini", 
                    help=('Path to .ini file with parameters. '
                          'Defaults to "params.ini" in current directory'))


def get_params(args: argparse.Namespace) -> Params:
    kwargs = {key: value for key, value in vars(args).items() if value is not None}
    params = Params(filename=args.params, **kwargs)
    return params

def main():
    args = parser.parse_args()
    params = get_params(args)
    device = get_device()
    run(params, device)

if __name__ == '__main__':
    main()
