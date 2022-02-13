# # # **Question 5 (Interpreting the Model)**
# # (30 points) In real-world applications of deep learning, it is *crucial* that we verify that our models are learning what we expect them to learn. In this exercise, we will replicate a part of figure 3b from [Basset](https://pubmed.ncbi.nlm.nih.gov/27197224/).
# # In genetics, there exists well known DNA *motifs*: short sequences which appear throughtout our DNA, and whose function are well documented. We expect that the filters of the first convolution layer should learn to identify some of these motifs in order to solve this task.
# # **Please submit the answers to this exercise on a single paged PDF!**


import matplotlib.pyplot as plt
import torch
import warnings
import random
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import solution
from tqdm import tqdm
if not torch.cuda.is_available():
    warnings.warn('CUDA is not available.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 100
learning_rate = 0.002

basset_dataset_train = solution.BassetDataset(path='./content/A1', f5name='er.h5', split='train')
basset_dataset_valid = solution.BassetDataset(path='./content/A1', f5name='er.h5', split='valid')
basset_dataset_test = solution.BassetDataset(path='./content/A1', f5name='er.h5', split='test')


basset_dataloader_train = DataLoader(basset_dataset_train,
                                     batch_size=batch_size,
                                     drop_last=True,
                                     shuffle=True,
                                     num_workers=1)
basset_dataloader_valid = DataLoader(basset_dataset_valid,
                                     batch_size=batch_size,
                                     drop_last=True,
                                     shuffle=False,
                                     num_workers=1)
basset_dataloader_test = DataLoader(basset_dataset_test,
                                    batch_size=batch_size,
                                    drop_last=True,
                                    shuffle=False,
                                    num_workers=1)


# # 1. First, we need to ensure that our model has learned something. Plot the ROC curve and compute the AUC of your
# model after training. Compare the ROC curves and the AUC before and after training with your simulated models.
# What do you notice?


def plot_roc(tpr_list, fpr_list, model_name):

    plt.plot(fpr_list, tpr_list)
    plt.title(model_name + ' ROC')
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()

def compute_fpr_tpr_list(y_true, y_prob_pred):

    output = {'fpr_list': [], 'tpr_list': []}

    for treshold in np.arange(0, 1., 0.05):
        y_pred = (y_prob_pred > treshold).astype(int)
        result = solution.compute_fpr_tpr(y_true, y_pred)
        output['fpr_list'].append(result['fpr'])
        output['tpr_list'].append(result['tpr'])

    return output['tpr_list'], output['fpr_list']


def q1():
    untrained_model = solution.Basset().to(device)
    untrained_model.eval()
    trained_model = torch.load('model_params.pt').to(device)
    trained_model.eval()

    y_true = np.empty(0, dtype=int)
    y_trained_pred = np.empty(0)
    y_untrained_pred = np.empty(0)

    c = 0
    for batch in tqdm(basset_dataloader_test):
        c += 1
        x, y = batch.values()
        x = x.to(device)
        trained_output = torch.sigmoid(trained_model(x))
        untrained_output = torch.sigmoid(untrained_model(x))

        y_trained_pred = np.append(y_trained_pred, trained_output.cpu().detach().numpy())
        y_untrained_pred = np.append(y_untrained_pred, untrained_output.cpu().detach().numpy())
        y_true = np.append(y_true, y.numpy()).astype(int)

    trained_tpr, trained_fpr = compute_fpr_tpr_list(y_true, y_trained_pred)
    untrained_tpr, untrained_fpr = compute_fpr_tpr_list(y_true, y_untrained_pred)

    plot_roc(trained_tpr, trained_fpr, 'Trained Model')
    plot_roc(untrained_tpr, untrained_fpr, 'Untrained Model')

    print('trained_auc = ', solution.compute_auc(y_true, y_trained_pred))
    print('untrained_auc = ', solution.compute_auc(y_true, y_untrained_pred))


# # 2. We represent motifs as position weight matrices (PWMs). This is a matrix of size $4$ $\times$ the motif length,
# where the $(i,j)$th entry is a count of how often base-pair $i$ occurs at position $j$. Open the PWM for the CTCF
# motif, which can be found in `MA0139.1.jaspar`. Normalize this matrix so that each column sums to $1$.#

A = [ 87, 167, 281,  56,   8, 744,  40, 107, 851,   5, 333,  54,  12,  56, 104, 372,  82, 117, 402]
C = [291, 145,  49, 800, 903,  13, 528, 433,  11,   0,   3,  12,   0,   8, 733,  13, 482, 322, 181]
G = [ 76, 414, 449,  21,   0,  65, 334,  48,  32, 903, 566, 504, 890, 775,   5, 507, 307,  73, 266]
T = [459, 187, 134,  36,   2,  91,  11, 324,  18,   3,   9, 341,   8,  71,  67,  17,  37, 396,  59]


def q2():
    full_matrix = np.array([A, C, G, T])

    full_matrix = full_matrix/full_matrix.sum(axis=0, keepdims=1)
    plt.imshow(full_matrix, cmap='viridis')
    plt.title('Normalized PWM for the CTCF motif')
    plt.colorbar()
    plt.show()

# 3. In the
# methods section of the [paper](https://pubmed.ncbi.nlm.nih.gov/27197224/) (page 998), the authors describe how they
# converted each of the $300$ filters into normalized PWMs. First, for each filter, they determined the maximum\
# activated value across the *dataset*(you may use a subset of the test set here). Compute these values.


batch_size = 1
learning_rate = 0.002

basset_dataset_test = solution.BassetDataset(path='./content/A1', f5name='er.h5', split='test')

basset_dataloader_test = DataLoader(basset_dataset_test,
                                    batch_size=batch_size,
                                    drop_last=True,
                                    shuffle=False,
                                    num_workers=1)

test = next(iter(basset_dataloader_test))
x, y = test.values()
x = x.to(device)
model = torch.load('model_params.pt').to(device)
model.eval()
test = model.get_1st_conv_output_max(x)


# # 4. Next, they counted the base-pair occurrences in the set of sequences that activate the filter to a value that is
# more than half of its maximum value.
# #   Note: You should use `torch.functional.unfold`.

blocks = torch.nn.functional.unfold(x, (19, 4))
# outputs =
# for i in blocks:


print(blocks.shape)
# # 5. Given your 300 PWMs derived from your convolution filt


