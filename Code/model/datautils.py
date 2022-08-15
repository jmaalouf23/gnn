import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 X,
                 y):
        """ 
        GraphDataset object
        Args:
        X: Array where the each column is set of smiles. For example, when predicting solubility,
            the first column can be solute smiles, the second column can be the solvent smiles.
        y: array of target values of size N vs D, where D are is the number of targets and N is the dataset size.
        
        """

        self.X=X #Smile strings
        self.y=y #target value

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx,:],self.y[idx,:]
    
    
def collate(batch):
    
    """
    Batch multiple graphs into one batched graph
    
    Params:
    batch (tuple): tuple of X,y data where X are smiles and y are target values
    
    Return:
    (tuple): batched X and y
    """
    
    X_batch=[]
    y_batch = []

    for i in range(len(batch)):
        X,y=batch[i]
        X_batch.append(X)
        y_batch.append(y)

    return X_batch, y_batch
