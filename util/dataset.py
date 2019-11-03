from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Dataset when it is possible and efficient to load all data items to memory
    '''

    ####
    def __init__(self, data_tensor, transform=None):
        '''
        data_tensor = actual data items; (N x C x H x W)
        '''

        self.data_tensor = data_tensor
        self.transform = transform

    ####
    def __getitem__(self, i):
        img = self.data_tensor[i]

        if self.transform is not None:
            img = self.transform(img)

        return img, i

    ####
    def __len__(self):
        return self.data_tensor.size(0)