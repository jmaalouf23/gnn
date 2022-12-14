from argparse import ArgumentParser, Namespace
import torch


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file containing train/val/test split indices')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--task', type=str, default='regression',
                        help='Regression or classification task')
    parser.add_argument('--pytorch_seed', type=int, default=0,
                        help='Seed for PyTorch randomness (e.g., random initial weights).')
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=60,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers to use in dataloader')
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help='Whether or not to retain default ordering during training')

    # Model arguments
    parser.add_argument('--gnn_type', type=str,
                        choices=['mpn','mpn2'],
                        help='Type of gnn to use')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')
    parser.add_argument('--graph_pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'attn', 'set2set'],
                        help='How to aggregate atom representations to molecule representation')
    parser.add_argument('--message', type=str, default='sum',
                        choices=['sum', 'tetra_pd', 'tetra_permute', 'tetra_permute_concat'],
                        help='How to pass neighbor messages')
    parser.add_argument('--ffn_depth', type=int, default=0, help='FFN layers less the linear output and input layer')
    parser.add_argument('--ffn_hidden_size', type=int, default=300, help='Dimensionality of hidden layers in FFN')
    parser.add_argument('--add_feature_path', type=str,
                        help='Path to csv file containing rdkit features')
    parser.add_argument('--n_add_feature', type=int,
                        help='Number of rdkit features per smiles')
    parser.add_argument('--n_out', type=int, default=1,
                        help='Number of ouputs in a multi-target problem')
    parser.add_argument('--scaled_err', default=True)
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=1)
    parser.add_argument('--number_of_molecules', type=int, default=1)
    parser.add_argument('--distributed',action='store_true',default=False)
    parser.add_argument('--include_plots',action='store_true',default=False)


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args
