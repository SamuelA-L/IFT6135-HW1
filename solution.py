# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE
        output['target'] = torch.from_numpy(self.outputs[idx].astype(np.float32))
        sequence = self.inputs[idx].astype(np.float32)
        sequence = np.moveaxis(sequence, 0, -1)
        output['sequence'] = torch.from_numpy(sequence)

        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        return self.inputs.shape[-1]

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        return True


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

        self.drop1 = nn.Dropout(p=self.dropout)
        self.drop2 = nn.Dropout(p=self.dropout)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = self.fc1(x.view(x.shape[0], -1))
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)


        # cnn = nn.Sequential(
        #     self.conv1,
        #     self.bn1,
        #     nn.ReLU(),
        #     self.maxpool1,
        #
        #     self.conv2,
        #     self.bn2,
        #     nn.ReLU(),
        #     self.maxpool2,
        #
        #     self.conv3,
        #     self.bn3,
        #     nn.ReLU(),
        #     self.maxpool3
        #
        # )
        # fcn = nn.Sequential(
        #
        #     self.fc1,
        #     self.bn4,
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #
        #     self.fc2,
        #     self.bn5,
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #
        #     self.fc3
        # )
        # x = cnn(x)
        # x = fcn(x.view(x.shape[0], -1))

        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE
    tp = 0
    fp = 0
    p = 0
    n = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1

        if y_true[i] == 1:
            p += 1
        elif y_true[i] == 0:
            n += 1

    output['tpr'] = 0 if p == 0 else tp / p
    output['fpr'] = 0 if n == 0 else fp/n

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    y_true = np.random.randint(low=0, high=2, size=1000)
    y_prob_pred = np.random.uniform(low=0, high=1, size=1000)

    for treshold in np.arange(0, 1., 0.05):
        y_pred = (y_prob_pred > treshold).astype(int)
        result = compute_fpr_tpr(y_true, y_pred)
        output['fpr_list'].append(result['fpr'])
        output['tpr_list'].append(result['tpr'])

    # WRITE CODE HERE

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    y_true = np.random.randint(low=0, high=2, size=1000)
    y_prob_pred = np.zeros(1000)
    for i, true in enumerate(y_true):
        if true == 1:
            y_prob_pred[i] = np.random.uniform(low=0.4, high=1.1)
        elif true == 0:
            y_prob_pred[i] = np.random.uniform(low=0, high=0.7)

    for treshold in np.arange(0, 1., 0.05):
        y_pred = (y_prob_pred > treshold).astype(int)
        result = compute_fpr_tpr(y_true, y_pred)
        output['fpr_list'].append(result['fpr'])
        output['tpr_list'].append(result['tpr'])

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE

    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)
    """
    output = {'auc': 0.}

    # WRITE CODE HERE

    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    # critereon = nn.BCELoss()
    # critereon = nn.CrossEntropyLoss()
    critereon = nn.NLLLoss


    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    return output['total_score'], output['total_loss']
