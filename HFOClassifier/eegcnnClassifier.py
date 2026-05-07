#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastparquet import ParquetFile
import gc
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

data_filename = 'CombinedHFOEEGData-BIDSh5art-f32ver.parq'
data_path = DATA_DIR / data_filename

fullnn_shelve_path = SAVE_DIR / 'eegcnn_fullmdl.out'
lounn_shelve_path = SAVE_DIR / 'eegcnn_loumdls.out'

#Remove all SOZ datasamples
pf = ParquetFile(data_path)

#Get desired columns
cols = pf.columns
feature_cols = cols[6:]
meta_cols = ['pid','myChanIdx','chanType']
# %%
df = pf.to_pandas(columns=meta_cols + feature_cols)

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
        self.conv1 = nn.Conv1d(1,6,3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(6,16,3)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=50)
        self.linear1 = nn.Linear(800,300)
        # self.linear1 = nn.Linear(3248, 300)  # 16 sono le colonne in input
        self.linear2 = nn.Linear(300, 150)
        self.linear3 = nn.Linear(150, 75)
        self.linear4 = nn.Linear(75, 15)
        self.linear5 = nn.Linear(15,1)

    def forward(self, tab):
        tab = self.pool(self.relu(self.conv1(tab)))
        tab = self.pool(self.relu(self.conv2(tab)))
        tab = self.avgpool(tab)
        tab = torch.flatten(tab,1)
        tab = self.linear1(tab)
        tab = self.relu(tab)
        tab = self.linear2(tab)
        tab = self.relu(tab)
        tab = self.linear3(tab)
        tab = self.relu(tab)
        tab = self.linear4(tab)
        tab = self.relu(tab)
        tab = self.linear5(tab)

        return tab
    
    def penlayer(self, tab):
        tab = self.pool(self.relu(self.conv1(tab)))
        tab = self.pool(self.relu(self.conv2(tab)))
        tab = self.avgpool(tab)
        tab = torch.flatten(tab,1)
        tab = self.linear1(tab)
        tab = self.relu(tab)
        tab = self.linear2(tab)
        tab = self.relu(tab)
        tab = self.linear3(tab)
        tab = self.relu(tab)
        tab = self.linear4(tab)
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
            # data = data.unsqueeze(-1).permute(0,2,1)
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
                # data = data.unsqueeze(-1).permute(0,2,1)
                
                # target = model(data).squeeze(1)
                target = model(data).view(-1)
                labels = labels.view(-1)
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

# create dataset class for data loading
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return slices as-is; already tensors on device
        return self.X[idx], self.y[idx]
    

###############################################################################
###############################################################################
# %% Full Model Neural Network (All Patients)

norv_label_encoder = LabelEncoder()  
norv_label_encoder.fit(['REST','BOTH'])
norv_label_encoder.classes_ = np.array(['REST','BOTH'])

# Split data into train, test and validation (BOTH vs REST ONLY)
temp_idx = df.index[(df['chanType'] != 'RV') & (df['chanType'] != 'SOZ')]
train_idx,test_idx = train_test_split(temp_idx,
                              train_size = 0.9, stratify = df.pid[temp_idx])
train_idx,valid_idx = train_test_split(train_idx,
                               train_size = 0.9, stratify = df.loc[train_idx,'pid'])

# ****MEMORY SPIKE BUT TEMPORALLY FASTER
train_X = df.loc[train_idx,feature_cols].to_numpy(dtype=np.float32)
train_X = np.expand_dims(train_X, axis = 1)
train_X = torch.from_numpy(train_X).to(device)
train_Y = norv_label_encoder.transform(df.chanType[train_idx].values)
train_Y = torch.from_numpy(train_Y).float().to(device)

valid_X = df.loc[valid_idx,feature_cols].to_numpy(dtype=np.float32)
valid_X = np.expand_dims(valid_X, axis = 1)
valid_X = torch.from_numpy(valid_X).to(device)
valid_Y = norv_label_encoder.transform(df.chanType[valid_idx].values)
valid_Y = torch.from_numpy(valid_Y).float().to(device)

gc.collect()
    
# Create dataloader for mini-batch loading
train_dataset = EEGDataset(train_X, train_Y)
valid_dataset = EEGDataset(valid_X, valid_Y)

# %% Run CNN Model

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers = 0)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers = 0)

model = HFONN().to(device=device, dtype = torch.float32)

# Declare Loss Criteron and Optimizer
optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=0.00001)
criterion = nn.BCEWithLogitsLoss()

epochs = 150

t0 = time.perf_counter()
min_valid_model, train_losses, valid_losses = nnTrainer(model, train_loader, 
                                                        valid_loader, optimizer, criterion, epochs = epochs)
t1 = time.perf_counter()
print(f"Training time: {(t1-t0)/60:.2f} seconds")

min_valid_model.eval()
min_valid_model.requires_grad_(False)

# %% SAVE MODEL
my_shelf = shelve.open(fullnn_shelve_path,'n')
my_shelf['min_valid_model'] = min_valid_model
my_shelf.close()

del my_shelf
gc.collect()

# %% PREDICTIONS

#Train predictions
train_loader_noshuff = DataLoader(train_dataset, batch_size = 1024, shuffle = False)
train_labels = train_dataset.y.data.cpu().numpy()
train_preds = []

with torch.no_grad():
    for data, _ in train_loader_noshuff:
        curpreds = nn.Sigmoid()(min_valid_model(data))
        train_preds.append(curpreds.cpu())
train_preds = torch.cat(train_preds).numpy()


#TEST Predictions
test_X = df.loc[test_idx,feature_cols].to_numpy(dtype=np.float32)
test_X = np.expand_dims(test_X, axis=1)
test_X = torch.from_numpy(test_X).to(device)
gc.collect()
test_Y = norv_label_encoder.transform(df.chanType[test_idx].values)
test_dataset = EEGDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size = 1024, shuffle = False)


test_preds = []

with torch.no_grad():
    for data, _ in test_loader:
        curpreds = nn.Sigmoid()(min_valid_model(data))
        test_preds.append(curpreds.cpu())

test_preds = torch.cat(test_preds).numpy()

#VALIDATION Predictions
valid_labels = valid_dataset.y.data.cpu().numpy()
valid_preds = []

with torch.no_grad():
    for data, _ in valid_loader:
        curpreds = nn.Sigmoid()(min_valid_model(data))
        valid_preds.append(curpreds.cpu())

valid_preds = torch.cat(valid_preds).numpy()

# %%
# Plot Training Loss & ROC curves

fig, ax = plt.subplots(1,2, figsize = (10,4))
fig.suptitle("Full Model CNN Learning Performance", fontsize = 20)
fig.subplots_adjust(top = 0.8)

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
r = RocCurveDisplay.from_predictions(test_Y,test_preds,
                        ax = ax[1], pos_label= 1, linewidth = 3, color = 'r',
                        name="Test", alpha = .6)
ax[1].set_xfont = 18
ax[1].set_title("ROC",fontsize=18)

# %% FREE MEMORY RAM/GPU
train_X = None
train_Y = None
train_dataset = None
train_loader = None
train_loader_noshuff = None

test_X = None
test_Y = None
test_loader = None

valid_X = None
valid_Y = None
valid_dataset = None
valid_loader = None

gc.collect()
torch.cuda.empty_cache()
print(str(torch.cuda.memory_allocated(0)*1e-6) + "MB allocated on Cuda")

###############################################################################
###############################################################################
# %% Leave Patient Out Neural Network Model

usebatch = 512
nnlr = 0.00002
useepoch = 150


#models
nnmdls = []
nnmdls_cpu = []
#model outputs
temp = ['pid','elec','label','pred','full_idxs']

nnlou_df = pd.DataFrame(columns = temp)     #MEMORY PURPOSES SAVE IDXS THEN TRANSFER

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
                              (df['chanType'] != 'SOZ')]
    
    #Generate Training and Validation Set #####################################
    train_idx, valid_idx = train_test_split(curidxs, train_size = 0.9,
                                            stratify = df.chanType[curidxs])
    
    train_X_temp = df.loc[train_idx,feature_cols].to_numpy(dtype=np.float32) #to numpy
    train_X_temp = np.expand_dims(train_X_temp, axis = 1) #[data,1,features] format
    train_X = torch.from_numpy(train_X_temp).to(device).contiguous()  #to GPU
    del train_X_temp
    gc.collect()
    train_Y = norv_label_encoder.transform(df.chanType[train_idx].values)
    train_Y = torch.from_numpy(train_Y).float().to(device).contiguous()

    valid_X = df.loc[valid_idx,feature_cols].to_numpy(dtype=np.float32)
    valid_X = np.expand_dims(valid_X, axis = 1)
    valid_X = torch.from_numpy(valid_X).to(device).contiguous()
    valid_Y = norv_label_encoder.transform(df.chanType[valid_idx].values)
    valid_Y = torch.from_numpy(valid_Y).float().to(device).contiguous()
    gc.collect()
   
    # Create dataloader for mini-batch loading
    train_dataset = EEGDataset(train_X, train_Y)
    valid_dataset = EEGDataset(valid_X, valid_Y)

    train_loader = DataLoader(train_dataset, batch_size=usebatch, shuffle=True, num_workers = 0)
    valid_loader = DataLoader(valid_dataset, batch_size=usebatch, shuffle=False, num_workers = 0)

    #Initialize Neural Network Model###########################################
    model = HFONN().to(device=device, dtype = torch.float32)
    
    # Declare Loss Criteron and Optimizer
    optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=nnlr)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = useepoch
    
    print("Starting NN")
    t0 = time.perf_counter()
    min_valid_model, train_losses, valid_losses = nnTrainer(model, train_loader, 
                                                            valid_loader, optimizer, criterion, epochs = epochs)
    t1 = time.perf_counter()
    
    min_valid_model.eval()
    min_valid_model.requires_grad_(False)
    
    print("Neural Network Complete")
    print(f"UM:{pid} training time: {(t1-t0)/60:.2f} mins")

    #Predict on Left Out Patient###############################################
    test_idx = df.index[(df['pid'] == pid)]
    test_elecs = df.myChanIdx[test_idx].values

    test_X_temp = df.loc[test_idx,feature_cols].to_numpy(dtype=np.float32)
    test_X_temp = np.expand_dims(test_X_temp, axis = 1)
    test_X = torch.from_numpy(test_X_temp).to(device)
    del test_X_temp
    gc.collect()
    test_Y = full_label_encoder.transform(df.chanType[test_idx].values)
    test_Y = torch.from_numpy(test_Y).float().to(device)
    test_dataset = EEGDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size = 1024, shuffle = False)

    test_preds = [] 
    
    with torch.no_grad(): 
        for data, _ in test_loader: 
            curpreds = nn.Sigmoid()(min_valid_model(data)) 
            test_preds.append(curpreds.cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_labels = test_Y.cpu().numpy() if torch.is_tensor(test_Y) else test_Y
    curn = len(test_preds)
    
    curdf = pd.DataFrame(
        np.hstack((
            np.full((curn,1),pid), 
            test_elecs.reshape(-1,1),
            test_labels.reshape(-1,1),
            test_preds.reshape(-1,1),
            np.array(test_idx).reshape(-1,1)
        )),
        columns = nnlou_df.columns)
        
    nnlou_df = pd.concat([nnlou_df,curdf],ignore_index=True)
    nnmdls_cpu.append(min_valid_model.to('cpu'))
    
    # GARBAGE COLLECTION ######################################################
    del train_X, train_Y, train_dataset, train_loader
    del valid_X, valid_Y, valid_dataset, valid_loader
    del test_X, test_Y, test_dataset, test_loader

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
    

# %%SAVE LOU DATA

my_shelf = shelve.open(lounn_shelve_path,'n')
my_shelf['nnmdls'] = nnmdls
my_shelf['nnlou_df'] = nnlou_df
my_shelf.close()

del my_shelf
gc.collect()

# %% Plot ROC Curve
pids = np.unique(nnlou_df.pid)
allrocs = []

from sklearn.metrics import RocCurveDisplay
fig, axes = plt.subplots(1,1)


for pidx in range(len(pids)):
    pid = pids[pidx]
    
    curidxs = nnlou_df.index[(nnlou_df['pid'] == pid) & 
                             (nnlou_df['label']< 2)].tolist()
    
    r = RocCurveDisplay.from_predictions(nnlou_df.label[curidxs], 
                                         nnlou_df.pred[curidxs],
                                         ax = axes, pos_label= 1, linewidth = 4,
                                         name = str(pid))
    allrocs.append(r.roc_auc)
    
plt.title("Leave Patient Out Neural Net. ROC",fontsize = 18)
plt.xlabel("FPR",fontsize=14)
plt.ylabel("TPR",fontsize = 14)
plt.legend(bbox_to_anchor = (1.1,1))





