#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastparquet import ParquetFile
import gc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shelve
import time

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAVE_DIR = ROOT / "results"

data_filename = 'CombinedHFOPSDData-BIDSh5art.parq'
data_path = DATA_DIR / data_filename

fullnn_shelve_path = SAVE_DIR / 'psdnn_fullmdl.out'
lounn_shelve_path = SAVE_DIR / 'psdnn_loumdls.out'

#############
#Remove all SOZ datasamples
pf = ParquetFile(data_path)

#Get desired columns
cols = pf.columns
feature_cols = cols[6:]
feat_names = feature_cols
meta_cols = ['pid','myChanIdx','chanType']
df = pf.to_pandas(columns=meta_cols + feature_cols)
# df = pf.to_pandas()

pids = pd.unique(df['pid'])

gc.collect()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define Neural Network Model/Parameters and Model Trainer Function

use_cuda = True
device = torch.device('cpu')
if(use_cuda and torch.cuda.is_available()):
    device = torch.device('cuda')

if device.type == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
class HFONN(nn.Module):
    # Constructor
    def __init__(self):
        # Call parent contructor
        super().__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(50, 40)  # 16 sono le colonne in input
        # self.linear1 = nn.Linear(84, 40)  # 16 sono le colonne in input
        self.linear2 = nn.Linear(40,15)
        self.linear3 = nn.Linear(15, 1)

    def forward(self, tab):      
        tab = self.linear1(tab)
        tab = self.relu(tab)
        tab = self.linear2(tab)
        tab = self.relu(tab)
        tab = self.linear3(tab)

        return tab
    
    def penlayer(self, tab):
        tab = self.linear1(tab)
        tab = self.relu(tab)
        tab = self.linear2(tab)
        tab = self.relu(tab)
        
        return tab

# Neural Network Trainer
def nnTrainer(model, train_loader, valid_loader, optimizer, criterion, epochs = 200):
    import time
    import copy
    
    st0 = time.time()
    
    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf
    min_valid_epochs = 0
    min_valid_model = None
    
    for e in range(epochs):
        train_loss = 0.0
        
        for data, labels in train_loader:
            
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            # target = model(data).squeeze(1)
            target = model(data).view(-1)
            labels = labels.view(-1)
            loss = criterion(target,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_loader))
        
        #Calculate Validation loss
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data,labels in valid_loader:
                target = model(data).view(-1)
                labels = labels.view(-1)
                # target = model(data).squeeze(1)
                loss = criterion(target,labels)
                valid_loss += loss.item()
            model.train()
            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)
        
        if(valid_loss < min_valid_loss):
            min_valid_loss = valid_loss
            min_valid_epochs = e
            min_valid_model = copy.deepcopy(model)
        
        print("E{}: Training Loss: {} \t\t Validation Loss: {}".format(
            e,train_loss/len(train_loader),valid_loss))
    
    ed0 = time.time()
    print(str(ed0-st0) + " seconds")
    
    return min_valid_model, train_losses, valid_losses

###############################################################################
###############################################################################
# %% Full Model Neural Network (All Patients)

norv_label_encoder = LabelEncoder()  
norv_label_encoder.fit(['REST','BOTH'])
norv_label_encoder.classes_ = np.array(['REST','BOTH'])

# Split data into train, test and validation (BOTH vs REST ONLY)
temp = df.index[(df['chanType'] != 'RV') & (df['chanType'] != 'SOZ')].tolist()
train,test = train_test_split(df.iloc[temp,:].reset_index(drop=True),
                              train_size = 0.9, stratify = df.pid[temp])
train,valid = train_test_split(train.reset_index(drop=True),
                               train_size = 0.9, stratify = train.pid)

# Create dataloader for mini-batch loading
train_dataset = TensorDataset(Tensor(train.loc[:,feature_cols].values).to(device),
                              Tensor(norv_label_encoder.transform(train.chanType.values)).to(device))
valid_dataset = TensorDataset(Tensor(valid.loc[:,feature_cols].values).to(device),
                              Tensor(norv_label_encoder.transform(valid.chanType.values)).to(device))

# %%

train_loader = DataLoader(train_dataset, batch_size = 512, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = 512, shuffle= True)

model = HFONN()
model = model.to(device)

# Declare Loss Criteron and Optimizer
optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=0.01)
criterion = nn.BCEWithLogitsLoss()
### lr = .001

epochs = 200

t0 = time.perf_counter()
min_valid_model, train_losses, valid_losses = nnTrainer(model, train_loader, 
                                                        valid_loader, optimizer, criterion, epochs = epochs)
t1= time.perf_counter()
print(f"Training time: {(t1-t0)/60:.2f} seconds")

min_valid_model.eval()
min_valid_model.requires_grad_(False)

# %% SAVE MODEL
my_shelf = shelve.open(fullnn_shelve_path,'n')
my_shelf['min_valid_model'] = min_valid_model
my_shelf.close()

del my_shelf
gc.collect()

# %% Predictions
test_data = torch.tensor(test.loc[:,feature_cols].values, dtype=torch.float32).to(device)
test_labels = norv_label_encoder.transform(test.chanType)
with torch.no_grad():
    test_preds = nn.Sigmoid()(min_valid_model(test_data))
test_preds = test_preds.data.cpu().numpy()

with torch.no_grad():
    train_preds = nn.Sigmoid()(min_valid_model(train_dataset.tensors[0])).data.cpu().numpy()
    valid_preds = nn.Sigmoid()(min_valid_model(valid_dataset.tensors[0])).data.cpu().numpy()
train_labels = train_dataset.tensors[1].data.cpu().numpy()
valid_labels = valid_dataset.tensors[1].data.cpu().numpy()

# Plot Training Loss & ROC curves

fig, ax = plt.subplots(1,2, figsize = (10,4))
fig.suptitle("HFO PSD (BIDS) BOTH vs REST NN")

ax[0].plot(train_losses, color = "k", label = "Training", alpha = .6, linewidth = 3 )
ax[0].plot(valid_losses, color = "blue", label = "Validation", alpha = .6, linewidth = 3 )
ax[0].set_title("Cross Entropy Loss", fontsize = 18)
ax[0].set_xlabel("Epochs (iterations)")
ax[0].legend(fontsize = 14)


r = RocCurveDisplay.from_predictions(train_labels,train_preds,
                        ax = ax[1], pos_label= 1, linewidth = 3, color = 'k',
                        name = "Training", alpha = .6)
r = RocCurveDisplay.from_predictions(valid_labels,valid_preds,
                        ax = ax[1], pos_label= 1, linewidth = 3, color = 'blue',
                        name = "Validation", alpha  =.6)
r = RocCurveDisplay.from_predictions(test_labels,test_preds,
                        ax = ax[1], pos_label= 1, linewidth = 3, color = 'r',
                        name="Test", alpha = .6)
ax[1].set_xfont = 18
ax[1].set_title("ROC",fontsize=18)

# %% FREE MEMORY RAM/GPU
train_dataset = None
train_loader = None
test_data = None
valid_dataset = None
valid_loader = None

gc.collect()
torch.cuda.empty_cache()
print(str(torch.cuda.memory_allocated(0)*1e-6) + "MB allocated on Cuda")

# %% Leave Patient Out Neural Network Model

# set_seed(13)
usebatch = 512
nnlr = 0.01
useepoch = 150

#models
nnmdls = []
nnmdls_cpu = []
#model outputs
temp = ['pid','elec','label','pred','full_idxs']
nnlou_df = pd.DataFrame(columns = temp)

norv_label_encoder = LabelEncoder()  
norv_label_encoder.fit(['REST','BOTH'])
norv_label_encoder.classes_ = np.array(['REST','BOTH'])

full_label_encoder = LabelEncoder()
full_label_encoder.classes_ = np.array(["REST","BOTH","RV","SOZ"])

for pid in pids:
    print('Leaving out UMID' + str(pid))
    
    #Model on BOTH vs REST only
    curidxs = df.index[(df['pid'] != pid) & 
                              (df['chanType'] != 'RV') &
                              (df['chanType'] != 'SOZ')].tolist()
    
    #Generate Training and Validation Set
    train,valid = train_test_split(df.loc[curidxs,:].reset_index(drop=True),
                                  train_size = 0.9, stratify = df.chanType[curidxs])
    
    # Create dataloader for mini-batch loading
    train_dataset = TensorDataset(Tensor(train.loc[:,feature_cols].values).to(device),
                                  Tensor(norv_label_encoder.transform(train.chanType.values)).to(device))
    valid_dataset = TensorDataset(Tensor(valid.loc[:,feature_cols].values).to(device),
                                  Tensor(norv_label_encoder.transform(valid.chanType.values)).to(device))

    train_loader = DataLoader(train_dataset, batch_size = usebatch, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = usebatch, shuffle= False)

    #Initialize Neural Network Model
    model = HFONN()
    model = model.to(device)
    
    # Declare Loss Criteron and Optimizer
    optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=nnlr)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting NN")
    t0= time.perf_counter()
    min_valid_model, train_losses, valid_losses = nnTrainer(model, train_loader, 
                                                            valid_loader, optimizer, criterion, epochs = useepoch)
    t1 = time.perf_counter()
    
    min_valid_model.eval()
    min_valid_model.requires_grad_(False)
    
    print("Neural Network Complete")
    print(f"UM:{pid} training time: {(t1-t0)/60:.2f} mins")

    #Predict on Left Out Patient###############################################
    curidxs = df.index[(df['pid'] == pid)].tolist()
    test_elecs = df.myChanIdx[curidxs].values
    
    test_data = torch.tensor(df.loc[curidxs,feature_cols].values, dtype=torch.float32).to(device)
    test_labels = full_label_encoder.transform(df.chanType[curidxs])
    with torch.no_grad():
        test_preds = nn.Sigmoid()(model(test_data))
    test_preds = test_preds.data.cpu().numpy()
    curn = len(test_labels)

    curdf = pd.DataFrame(
        np.hstack((
            np.full((curn,1),pid), 
            test_elecs.reshape(-1,1),
            test_labels.reshape(-1,1),
            test_preds.reshape(-1,1),
            np.array(curidxs).reshape(-1,1)
        )),
        columns = nnlou_df.columns)
    
    nnlou_df = pd.concat([nnlou_df,curdf],ignore_index=True)
    nnmdls_cpu.append(min_valid_model.to('cpu'))
    
    # GARBAGE COLLECTION ######################################################
    del train_dataset, train_loader
    del valid_dataset, valid_loader
    del test_data

    gc.collect()
    torch.cuda.empty_cache()
    
    torch.cuda.synchronize()
    
    # PREEMPTIVE SAVE
    my_shelf = shelve.open(lounn_shelve_path,'n')
    my_shelf['nnmdls_cpu'] = nnmdls_cpu
    my_shelf['nnlou_df'] = nnlou_df
    my_shelf.close()
    
    del my_shelf
    gc.collect()


# %%

my_shelf = shelve.open(lounn_shelve_path,'n')
my_shelf['nnmdls'] = nnmdls
my_shelf['nnlou_df'] = nnlou_df
my_shelf.close()
print('DATA SAVED!!!')

# %% Plot ROC Curve
fig, axes = plt.subplots(1,1,figsize = (12,7))

cmap = cm.get_cmap('tab20')

for pidx in range(len(pids)):
    pid = pids[pidx]
    
    curidxs = nnlou_df.index[(nnlou_df['pid'] == pid) &
                             (nnlou_df['label'] < 2)].tolist()
    
    r = RocCurveDisplay.from_predictions(nnlou_df.label[curidxs], 
                                         nnlou_df.pred[curidxs],
                                         ax = axes, pos_label= 1, linewidth = 4,
                                         name = str(pid),color = cmap(pidx/len(pids)))
    
plt.legend().set_visible(False)
plt.xlabel("FPR",fontsize=30)
plt.ylabel("TPR",fontsize = 30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(loc='lower right',fontsize = 10)



