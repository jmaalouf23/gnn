import sys
import torch
from torch import nn
sys.path.insert(1,'/home/gridsan/jmaalouf/packages/chemprop_03')
from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features.featurization import set_extra_atom_fdim
from chemprop.features.featurization import atom_features
from chemprop.nn_utils import get_activation_function, initialize_weights
                
class MainModel(torch.nn.Module): 
    """Contains a message passing networks (MPN) for representing a molecule...
    followed by an MLP readout layer for predicting solubility."""

    def __init__(self, args,rank):
        """
        Arguments:
          hyper:'chemprop.args.TrainArgs' object containing model settings and hyperparameters.
        """
        super(MainModel,self).__init__()
        
        """
        Build Graph Convolutional neural network model Pytorch using ChemProp. 
        For details on Chemprop's MPN model, see: https://github.com/chemprop/chemprop/blob/master/chemprop/models/mpn.py
        For details on important model arguments: https://github.com/chemprop/chemprop/blob/736530ac2ca42069088069f38571dc7a23e7e1d9/chemprop/args.py#L215
        
        """
        
        hyper = TrainArgs()
        hyper.dataset_type=args.task # regression or classification
        hyper.hidden_size = args.hidden_size # Dim of hidden layers in MPN
        hyper.depth = args.depth # Number of message passing steps
        hyper.bias = False # Whether to add bias to MPN linear layers
        hyper.dropout = args.dropout # MPN Dropout probability
        hyper.activation = 'ReLU' # MPN Activation function.['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
        hyper.aggregation = args.graph_pool # MPN Aggregation scheme for atomic vectors into molecular vectors. ['mean', 'sum', 'norm']
        hyper.aggregation_norm = 100 # For MPN norm aggregation, number by which to divide summed up atomic features
        hyper.device=torch.device(rank%2)
        hyper.ffn_activation='ReLU' # FFN Activation function. ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
        hyper.ffn_hidden_size = args.ffn_hidden_size # Dim of hidden layer in readout FFN
        hyper.ffn_num_layers = args.ffn_depth + 1 # FFN layers, default args.ffn_depth=0, which translates to a ffn_num_depth of 1
        hyper.output_size=args.n_out
        
        
        #Define MPN Encoder
        self.hyper = hyper
        self.mpn = MPN(self.hyper)
        # Define FFN readout layer
        self.mlp_input_size = args.hidden_size 
        self.output_size=self.hyper.output_size
        self.create_ffn(self.hyper)

        
    def create_ffn(self,args: TrainArgs) -> None:
            """
            Creates the feed-forward layers for the model.
            :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
            """
            self.multiclass = args.dataset_type == 'multiclass'
            if self.multiclass:
                self.num_classes = args.multiclass_num_classes
            if args.features_only:
                first_linear_dim = args.features_size
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
                if args.use_input_features:
                    first_linear_dim += args.features_size

            if args.atom_descriptors == 'descriptor':
                first_linear_dim += args.atom_descriptors_size

            dropout = nn.Dropout(args.dropout)
            activation = get_activation_function(args.ffn_activation)

            # Create FFN layers
            if args.ffn_num_layers == 1:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, self.output_size)
                ]
            else:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, args.ffn_hidden_size)
                ]
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ])
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, self.output_size),
                ])

            #If spectra model, also include spectra activation
            if args.dataset_type == 'spectra':
                if args.spectra_activation == 'softplus':
                    spectra_activation = nn.Softplus()
                else: # default exponential activation which must be made into a custom nn module
                    class nn_exp(torch.nn.Module):
                        def __init__(self):
                            super(nn_exp, self).__init__()
                        def forward(self, x):
                            return torch.exp(x)
                    spectra_activation = nn_exp()
                ffn.append(spectra_activation)

            #Create FFN model
            self.ffn = nn.Sequential(*ffn)
            
    def forward(self, x):

        # --------------- MPN ---------------
        # A list of lists of SMILES in batch. Example: [ [smiles1], [smiles2], ..., [smilesN] ]
        smiles_list = [ [x[i][0]] for i in range(len(x)) ] 
       
        custom_atom_features = (None,)
        if self.hyper.atom_descriptors == 'feature':
            custom_atom_features = [ [x[i][1]] for i in range(len(x)) ] # A list of numpy arrays (N_atoms x N_features) containing custom atom features
        bond_features = (None,)
        
        vec = self.mpn(smiles_list, atom_features_batch = custom_atom_features, bond_features_batch=bond_features)
        output = self.ffn(vec)
        return output
    
    
    
class MainModel_2(torch.nn.Module):
    """Contains two separate message passing networks (MPN) for representing two molecules...
    followed by an MLP readout layer for predicting solubility."""

    def __init__(self, hyper):
        """
        Arguments:
          hyper:'chemprop.args.TrainArgs' object containing model settings and hyperparameters.
        """

        super().__init__()
        self.hyper = hyper

        # Define separate MPN modules for solute and solvent
        self.mpn_solute = MPN(self.hyper)
        self.mpn_solvent = MPN(self.hyper)

        # Define MLP readout layer (one hidden layer for now)
        mlp_input_size = self.hyper.hidden_size * 2 # input to MLP is concatenated solute and solvent representations
        self.mlp = nn.Sequential( nn.Linear(mlp_input_size, self.hyper.mlp_hidden_size),
                                  nn.ReLU(),
                                  nn.Linear( self.hyper.mlp_hidden_size, self.hyper.mlp_hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(self.hyper.mlp_hidden_size, 1))

    def forward(self, x):

        # --------------- MPN ---------------

        # Parse solute and solvent features from input x
        # SMILES

        smiles_list_solute = [ [x[i][0]] for i in range(len(x)) ] # A list of lists of SMILES in batch. Example: [ [smiles1], [smiles2], ..., [smilesN] ]
        smiles_list_solvent = [[x[i][1]] for i in range(len(x))] # A list of lists of SMILES in batch
        # Custom atom features
        custom_atom_features_solute = (None,)
        custom_atom_features_solvent = (None,)
        # If augmenting default atom features with custom atom features
        if self.hyper.atom_descriptors == 'feature':
            custom_atom_features_solute = x[2] # A list of numpy arrays (N_atoms x N_features) containing custom atom features
            custom_atom_features_solvent = x[3] # A list of numpy arrays (N_atoms x N_features) containing custom atom features

        # Bond features (use chemprop default)
        bond_features = (None,)

        # Apply MPNs to obtain solute and solvent vector representations
        vec_solute = self.mpn_solute(smiles_list_solute, atom_features_batch = custom_atom_features_solute, bond_features_batch=bond_features)
        vec_solvent = self.mpn_solvent(smiles_list_solvent, atom_features_batch = custom_atom_features_solvent, bond_features_batch = bond_features)

        # --------------- MLP ---------------

        # # Concatenate solute and solvent vectors
        vec_concat = torch.cat((vec_solute, vec_solvent), 1) # vector of size: 1 by 2*mpn_hidden_size

        # # Pass into MLP
        output = self.mlp(vec_concat)

        return output