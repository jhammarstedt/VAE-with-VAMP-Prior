"""

"""

import pickle
import numpy as np
import torch
import torch.utils.data as tud
from scipy.io import loadmat
import os
from math import ceil




def read_pickle(file_name):
    with open(file_name, 'rb') as fn:
        data = pickle.load(fn)
    return data


def load_omniglot(train_size=0.8):


    def reshape_omniglot():
        return 


    # data characteristics
    dataset_shape = [1, 28, 28]
    dataset_type = "binary"

    omni_mat = loadmat(os.path.join('data','chardata.mat'))

    train_data =  omni_mat['data'].T.astype('float32')
    test_data = 2

    # print(train_data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F').shape)
    # print((train_data.reshape(-1, 28, 28)).shape)

    np.random.shuffle(train_data)

    x_train, x_valu = np.split(train_data, [ceil(train_size * len(train_data)), len(train_data)])


    # Pytorch specifies
    train_td = tud.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    


def load_dynamic_mnist(batch_size, **kwargs):
    # set args
    dataset_shape = [1, 784]

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )
    print(x_train.shape)
    y_train = np.array( train_loader.dataset.targets.float().numpy(), dtype=int)

    x_test = test_loader.dataset.data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array(test_loader.dataset.targets.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # pytorch data loader
    train = tud.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = tud.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)

    validation = tud.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = tud.DataLoader(validation, batch_size=batch_size, shuffle=False, **kwargs)

    test = tud.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = tud.DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, dataset_shape


if __name__ == "__main__": 
    load_omniglot()



