import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 X,
                 y):
        """ 
        GraphDataset object
        Args:
        X: Array where the first column is solute smiles string and second column
        is solvent smiles strings
        y: 1d array of target values
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
    Args:
    batch (tuple): tuple of X,y data where X are smiles and y are target values
    Return:
    (tuple): Batched AtomicNum, Edge, Natom, y
    """
    
    X_batch=[]
    y_batch = []

    for i in range(len(batch)):
        X,y=batch[i]
        X_batch.append(X)
        y_batch.append(y)

    return X_batch, y_batch
