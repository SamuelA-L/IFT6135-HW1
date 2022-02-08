#!/usr/bin/env python
# coding: utf-8

# # Genetics Application
# 
# For this set of questions, we will explore the use of Convolutional Neural Networks to solve a problem with Biological significance (don't worry, no background knowledge is expected). Every cell in your body contains Deoxyribonucleic acid (DNA), which is essentially the instructions for making all the proteins in your body. DNA can be thought of as a very long string where the alphabet is \{A,C,T,G\}.
# The physical attributes of the DNA string and its characters are not important here, except that ~98% of our DNA is physically inaccessible to external molecules. Understanding which regions of DNA are accessible and why is of great interest to scientists. This motivates learning predictive models which could accurately classify such regions. In particular, we will be implementing the deep network called [Basset](https://pubmed.ncbi.nlm.nih.gov/27197224/).
# 

# In[ ]:


#@title Mount your Google Drive
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
#
# from google.colab import drive
# drive.mount('./content/gdrive')


# In[ ]:


#@title Link your assignment folder & install requirements
#@markdown Enter the path to the assignment folder in your Google Drive
import sys
import os
import shutil
import warnings

# folder = "./content/gdrive/MyDrive/A1" #@param {type:"string"}
# get_ipython().system('ln -Ts $folder ./content/A1 2> /dev/null')

# Add the assignment folder to Python path
if './content/A1' not in sys.path:
    sys.path.insert(0, './content/A1')

# Install requirements
# get_ipython().system('pip install -qr ./content/A1/requirements.txt')

# Check if CUDA is available
import torch
if not torch.cuda.is_available():
    warnings.warn('CUDA is not available.')


# In[ ]:


# unzip data
if not os.path.exists('./content/A1/er.h5'):
    os.system('unzip ./content/A1/er.zip')
    os.system('mv ./content/er.h5 ./content/A1/er.h5')


# ## Running on GPU
# In Google Colab, you can run your code on GPU. This will be particularly important in CNN part of the assignment. To make sure the notebook is running on GPU, you can change the notebook settings with
# * (EN) `Edit > Notebook Settings`
# * (FR) `Modifier > Paramètres du notebook`
# 
# Be mindful not to use the GPU if your code does not need to run on GPU.

# In[ ]:


import random

import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import solution


# In[ ]:


# The hyperparameters we will use
batch_size = 64
learning_rate = 0.002


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Let's build the dataset that we will be using. Before doing any machine learning, you should always inspect the data you will be using (especially when that data is uncommon)!
#
# In our case, the input data are subsequences of DNA that come from the reference human genome HG19. HG19 can be thought of as the DNA of a prototypical human (but it does not come from any single person).
# The input data is one-hot encoded, such that:
#
# $$A = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, C = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, G = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, T = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$
#
# Each base-pair $\{A,C,G,T\}$ are concatenated along the length of the sequence. Each data point is equivalent to an image with a rectangular shape (sequence length, 4) and a single channel for data processing purposes.
#
# The target data for our sequence are binary strings encoded as $1$ if the DNA subsequence is accessible and $0$ if it is not. This data was collected across experiments. For each sample, the target is a vector of size 164, with $1$s in the indices of the experiments where the DNA is accessible and $0$ otherwise.

# # **Question 1 (Understanding the Data)**
# (10 points). You will notice that PyTorch is very object-oriented -- everything you use is the subclass of a PyTorch class. For datasets, we subclass `torch.utils.data.Dataset`.
#
# 1. Define the `__getitem__` and `__len__` methods for the `BassetDataset` class in `solution.py`
# 2. What is the length of each sequence we will feed into our model?  Make the `get_seq_len` method of the `BassetDataset` class to return it.
# 3. Is it true that each data point is also equivalent to an image with a rectangular shape (1, sequence length) with 4 channels? Write the method `is_equvalent` to return your answer (either return True or False).
#
# Note: this data has already been pre-processed, so do not include any additional data transformations!

# In[ ]:


# investigate your data
f = h5py.File('./content/A1/er.h5', 'r')
print(f.keys())
f.close()


# In[ ]:


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



# # # **Question 2 (Building the Network)**
# # (15 points). PyTorch also provides an abstraction for differentiable models: `torch.nn.Module`. In `solution.py`, we define Basset as a subclass of this class.
# #
# # 1. Fill in the missing values denoted as `?` within the class definition using [supplementary figure 13](https://genome.cshlp.org./content/suppl/2016/06/10/gr.200535.115.DC1/Supplementary_Figures.pdf) and convolution arithmetic.
# # 2. Write the `forward` pass function. Again, please refer to [supplementary figure 13](https://genome.cshlp.org./content/suppl/2016/06/10/gr.200535.115.DC1/Supplementary_Figures.pdf).
# #
# #   Do not include any output activation in your forward method! In practice, it is better to use a loss function that has the output activation built-in.
#
# # In[ ]:
#
#
model = solution.Basset().to(device)
#
#
#
# # # **Question 3 (Area Under the Curve)**
# # (25 points). Each DNA sequence is either exposed (we'll call this the positive case) or not (we'll call this the negative case). The output of our model should be used to return a binary decision about the sequence.
# #
# # For a given case, we say that our model made a positive prediction whenever its output value is above some threshold $k$, and a negative prediction otherwise.
# #
# # Define the *True Positive Rate* (TPR) as the number of correctly prediced positive cases divided by the number of positive cases. Define the *False Positive Rate* (FPR) as the  number of cases that were incorrectly predicted as positive divided by the number of negative cases.
# #
# # 1. Complete the function `compute_fpr_tpr` in `solution.py`. This code will compute the TPR and FPR given your models decisions and the true targets.
# #
# # 2. Your FPR and TPR change as a function of $k$. Specifically, we can plot the FPR on the x-axis and the TPR on the y-axis for different values of $k$. The shape of the resulting curve (ROC curve) tells us something about our classifiers performance. We will explore via simulation what this plot looks like when our model is just making random guesses. In `solution.py`, fill in the function `compute_fpr_tpr_dumb_model`. This function will generate 1000 binary random variables to use as targets and 1000 uniform random variables between 0 and 1 as our model predictions. It will then compute the fpr and tpr for $k\in \{0, 0.05,..., 0.95, 1\}$ You should plot what the ROC curves look like for your own knowledge.
#
# # In[ ]:
#
#
solution.compute_fpr_tpr_dumb_model()
#
#
# # 3. We will now simulate a better model. In `solution.py`, fill in the function `compute_fpr_tpr_smart_model`. This will simulate 1000 targets the same way as before. However, this will simulate model outputs as uniform random variables between 0.4 and 1 for the positive cases. For the negative cases, simulate uniform random variables between 0 and 0.6. Compute the tpr and fpr varying $k$ like before. You should also look at the ROC curve.
#
# # In[ ]:
#
#
solution.compute_fpr_tpr_smart_model()
#
#
# # 4. The Area Under the ROC Curve (AUC) summarizes the ROC plot as a single number. It is literally computed as the area under the the ROC curve (take the average of the left and right Reimann sums). Complete the function `utils.compute_auc` and use it in the function `compute_auc_both_models` to compute the AUC of the ROC curves you made in parts 2 and 3.
#
# # In[ ]:
#
#
# solution.compute_auc_both_models()
#
#
# # 5. Complete the function `compute_auc_untrained_model`, which will compute the AUC of your untrained model. Use the first 100 batches of the test set.
# #
# #     Note: since we are doing multi-task prediction, your model outputs and targets will have shape (Batch size, Num targets). Flatten these matrices so that they are two vectors, each of size Batch size * Num targets.
# #     Then, procede with AUC as if you were in the usual single task case...
#
# # In[ ]:
#
#
# solution.compute_auc_untrained_model(model, basset_dataloader_test, device)
#
#
# # # **Question 4 (Training the Network)**
# # (20 points) We will now write the training loop for Basset.
# #
# # 1. Notice that for each input, we have one target per experiment, and each target is binary. Write the function `get_critereon`, which will pick the appropriate `loss` function.
# #
# #   Hint: you may consult the [relevant PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
# #
# #
#
# # In[ ]:
#
#
# criterion = solution.get_critereon()
#
#
# # 2. Finish the training loop by filling in the missing code for the `train_loop`  pass and `valid_loop` functions in `solution.py.`
# #
# #     Both loops must return the loss and AUC (computed the same way as the previous question). They must be returned by each function (see the docstring for more details).
#
# # In[ ]:
#
#
# optimizer = optim.Adam(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999))
#
# valid_score_best = 0
# patience = 2
# num_epochs = 5  # you don't need to train this for that long!
#
# for e in range(num_epochs):
#     train_score, train_loss = solution.train_loop(model, basset_dataloader_train, device, optimizer, criterion)
#     valid_score, valid_loss = solution.valid_loop(model, basset_dataloader_valid, device, optimizer, criterion)
#
#     print('epoch {}: loss={:.3f} score={:.3f}'.format(e,
#                                                       valid_loss,
#                                                       valid_score))
#
#     if valid_score > valid_score_best:
#         print('Best score: {}. Saving model...'.format(valid_score))
#         torch.save(model, 'model_params.pt')
#         valid_score_best = valid_score
#     else:
#         patience -= 1
#         print('Score did not improve! {} <= {}. Patience left: {}'.format(valid_score,
#                                                                           valid_score_best,
#                                                                           patience))
#     if patience == 0:
#         print('patience reduced to 0. Training Finished.')
#         break
#
#
# # # **Question 5 (Interpreting the Model)**
# #
# # (30 points) In real-world applications of deep learning, it is *crucial* that we verify that our models are learning what we expect them to learn. In this exercise, we will replicate a part of figure 3b from [Basset](https://pubmed.ncbi.nlm.nih.gov/27197224/).
# #
# # In genetics, there exists well known DNA *motifs*: short sequences which appear throughtout our DNA, and whose function are well documented. We expect that the filters of the first convolution layer should learn to identify some of these motifs in order to solve this task.
# #
# # **Please submit the answers to this exercise on a single paged PDF!**
# #
# # 1. First, we need to ensure that our model has learned something. Plot the ROC curve and compute the AUC of your model after training. Compare the ROC curves and the AUC before and after training with your simulated models. What do you notice?
#
# # 2. We represent motifs as position weight matrices (PWMs). This is a matrix of size $4$ $\times$ the motif length, where the $(i,j)$th entry is a count of how often base-pair $i$ occurs at position $j$. Open the PWM for the CTCF motif, which can be found in `MA0139.1.jaspar`. Normalize this matrix so that each column sums to $1$.
#
# # 3. In the methods section of the [paper](https://pubmed.ncbi.nlm.nih.gov/27197224/) (page 998), the authors describe how they converted each of the $300$ filters into normalized PWMs. First, for each filter, they determined the maximum activated value across the *dataset* (you may use a subset of the test set here). Compute these values.
#
# #
# # 4. Next, they counted the base-pair occurrences in the set of sequences that activate the filter to a value that is more than half of its maximum value.
# #
# #   Note: You should use `torch.functional.unfold`.
#
# # 5. Given your 300 PWMs derived from your convolution filters, check to see if any of them are similar to the PWM for CTCF. You could quantify the similarity using *Pearson Correlation Coefficient*. Make a visualization of the PWM of the CTCF motif along with the most similar ones learned from the network.
#
# # In[ ]:
#



