import torch
import scipy as sp
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

class ArrayDataset(Dataset):
    """Dataset wrapping arrays (tensor, numpy, or scipy sparse).

    Each sample will be retrieved by indexing arrays along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, convert_sparse = True):
        self.convert_sparse = convert_sparse
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]
    
    def collate_fn(self, batch):
        # Zip the batch
        zipped = list(map(list, zip(*batch)))

        if self.convert_sparse:
        # If data is CSR, convert to sparse tensor, else use default_collate
            result = tuple(
                default_collate(element) 
                if not isinstance(element[0], sp.sparse.csr_matrix)
                else self.csr_to_tensor(sp.sparse.vstack((element)))
                for element in zipped
                )
        else:
            # If data is CSR, return as CSR
            result = tuple(
                default_collate(element) 
                if not isinstance(element[0], sp.sparse.csr_matrix) 
                else sp.sparse.vstack((element))
                for element in zipped
                )


        return result

    def csr_to_tensor(self, x):
        x = x.tocoo()
        return torch.sparse.FloatTensor(torch.LongTensor([x.row, x.col]), 
                                        torch.FloatTensor(x.data), 
                                        torch.Size(x.shape)
                                        )