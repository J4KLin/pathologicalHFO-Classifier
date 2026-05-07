# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy.io as sio
import seaborn as sns
import shelve
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearnex import patch_sklearn
patch_sklearn()


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAVE_DIR = ROOT / "results"

data_filename = 'CombinedHFOPSDData-BIDSh5art.parq'
data_type = 'PSD'

# data_filename = 'CombinedSubsetHFOFEAT-BIDSh5art.parquet'
# main_type = "FEAT"

data_path = DATA_DIR / data_filename

fullnn_shelve_path = SAVE_DIR / data_type + 'lr_fullmdl.out'
lounn_shelve_path = SAVE_DIR / data_type + 'lr_loumdls.out'


df = pd.read_parquet(data_path)

pids = pd.unique(df['pid'])


######Adjust Data ticks/label
ignorecomb = False

if(data_type == "FEAT"):
    ignorefeats = np.array(['f_TKE_mean', 'f_TKE_var', 'f_curv_var', 'f_PSD_mean', 'f_LL_var',
           'f_curv_mean', 'f_PSD_power_quartile_3', 'f_PSD_power_median',
           'f_curv_kurt', 'f_TD_kurt', 'f_LL_mean', 'f_PSD_kurt', 'f_LL_kurt'])
    feat_idxs = (np.arange(6,len(df.columns)))
        
    dataticks = df.columns[feat_idxs].astype('string').to_numpy()
    temp = ~np.isin(dataticks,ignorefeats)
    feat_idxs = feat_idxs[temp]
    dataticks = dataticks[temp]
    dataticks = np.array([s[2:] for s in dataticks])
    
    idx = [idx for idx in range(len(dataticks)) if dataticks[idx] == 'TD_amplitude']
    dataticks[idx[0]] = 'Power'
    
    datalabel = 'Features'
    
    fname = "Derived Feature"

# %
if(data_type == "PSD"):
    dataticks = np.arange(0,4096/2,4096/819)
    dataticks = dataticks[(dataticks >= 80) & (dataticks <= 500)]

    feat_idxs = list(np.arange(6,len(df.columns)))
    if(ignorecomb):
        usedata = np.where((dataticks % 60) > .5)[0]
        dataticks = dataticks[usedata]
        feat_idxs = usedata + 6
    
    fname = "Frequency (Hz)"
        
norv_LabelEncoder = LabelEncoder().fit(["REST","BOTH"])
norv_LabelEncoder.classes_ = np.array(["REST","BOTH"])
norv_idxs = df.index[(df['chanType'] != "RV") & 
                     (df['chanType'] != "SOZ")].tolist()


full_LabelEncoder = LabelEncoder().fit(["REST","BOTH","RV","SOZ"])
full_LabelEncoder.classes_ = np.array(["REST","BOTH","RV","SOZ"])
###############################################################################
# %% Training and testing full logistic model #################################

train,test = train_test_split(df.iloc[norv_idxs,:].reset_index(drop=True),
                              train_size = 0.9, stratify = df.pid[norv_idxs])
    
norv_trainlabels = norv_LabelEncoder.transform(train.chanType)
norv_testlabels = norv_LabelEncoder.transform(test.chanType)

usemdl = 'lr' #lr
penalty = 'l2'

fullmdl = LogisticRegression(random_state = 0,penalty=penalty).fit(train.iloc[:,feat_idxs],norv_trainlabels)
norv_trainpreds = fullmdl.predict_proba(train.iloc[:,feat_idxs])
norv_testpreds = fullmdl.predict_proba(test.iloc[:,feat_idxs])

# %% Plot ROC and betas
fig, axes = plt.subplots(1,2,figsize = (12,4))
fig.suptitle("Logistic Regression (l2 penalty) of " + data_type, fontsize=20)
fig.subplots_adjust(top=.85)

r = RocCurveDisplay.from_predictions(norv_trainlabels, norv_trainpreds[:,1],
                        ax=axes[0], pos_label= 1, linewidth = 3, color = 'k',
                        name = "Training", alpha = .6)
r = RocCurveDisplay.from_predictions(norv_testlabels, norv_testpreds[:,1],
                        ax=axes[0], pos_label= 1, linewidth = 3, color = 'r',
                        name = "Testing", alpha = .6)
axes[0].set_title("ROC BOTH vs REST", fontsize = 18)

axes[1].plot(dataticks,fullmdl.coef_.flatten())
axes[1].set_xticklabels(dataticks, rotation=90)
axes[1].set_xlabel(datalabel)
axes[1].set_ylabel("/beta")
axes[1].set_title("Log. Reg. Coefficients",fontsize = 18)


# %%
my_shelf = shelve.open(fullnn_shelve_path,'n')
my_shelf['nnmdls'] = fullmdl
my_shelf.close()

print('DATA SAVED!!!')

###############################################################################
###############################################################################
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Perform Leave-one-out (patient) validated Log. Regression

lou_preds = []
lou_idxs = []
lou_labels = []
lou_mdls = []

temp = ['pid','elec','label','pred']
lrlou_df = pd.DataFrame(columns = temp)

for pid in pids:
    print("UMID" + str(pid))
    idxs = df.index[(df['chanType'] != "RV") & 
                    (df['chanType'] != "SOZ") & (df['pid'] != pid)].tolist()
    
    curmdl = LogisticRegression(random_state = 0).fit(df.iloc[idxs,feat_idxs],
                                                      norv_LabelEncoder.transform(df.chanType[idxs]))
    
    idxs = df.index[(df['pid'] == pid)].tolist()
    curtestlabels = full_LabelEncoder.transform(df.chanType[idxs])
    curtestpreds = curmdl.predict_proba(df.iloc[idxs,feat_idxs])

    lou_preds.append(curtestpreds)
    lou_labels.append(curtestlabels)
    lou_mdls.append(curmdl)
    lou_idxs.append(idxs)


    test_elecs = df.myChanIdx[idxs].values
    curn = len(idxs)
 
    curdf = pd.DataFrame(np.hstack((np.repeat(pid,curn).reshape(curn,1), test_elecs.reshape(curn,1),
                                    curtestlabels.reshape(curn,1),
                                    curtestpreds[:,1].reshape(curn,1))),
                         columns = lrlou_df.columns)
    
    lrlou_df = pd.concat([lrlou_df,curdf],ignore_index=True)
    

# %%

my_shelf = shelve.open(lounn_shelve_path,'n')
my_shelf['lou_mdls'] = lou_mdls
my_shelf['lrlou_df'] = lrlou_df
my_shelf.close()
print('DATA SAVED!!!')


# %% recompute lrlou labels

lou_labels = []

for pid in pids:
    print("UMID" + str(pid))
    
    idxs = df.index[(df['pid'] == pid)].tolist()
    curtestlabels = full_LabelEncoder.transform(df.chanType[idxs])

    lou_labels.append(curtestlabels)

# %% Plot ROC Curve
cmap = cm.get_cmap('tab20')
fig, axes = plt.subplots(1,1,figsize = (14,8))


for i in range(len(pids)):
    pid = pids[i]
    
    curidxs = np.where(lou_labels[i] < 2)[0]
    r = RocCurveDisplay.from_predictions(lou_labels[i][curidxs], lou_preds[i][curidxs,1],
                            pos_label= 1, linewidth = 4,
                            ax = axes,name = str(pid), alpha = .6,color = cmap(i/len(pids)))
    
    
plt.legend().set_visible(False)
plt.title("Leave Patient Out Log. Regression \n ROC " + data_type,fontsize = 24)
plt.xlabel("FPR",fontsize=14)
plt.ylabel("TPR",fontsize = 14)
fig.legend(loc='upper left',bbox_to_anchor=(.73,.89),fontsize=14)


# %% PLOT COEFFS FOR SAVING
# %
if (data_type == 'PSD'):
    usedata = np.where((dataticks % 60) > .5)[0]
else:
    usedata = np.arange(len(dataticks))
    
tfont = 16
labfont = 24
ticfont = 16
legfont = 16

fig,axes = plt.subplots(1,1,figsize = (8,6))
for i in range(len(pids)):
    pid = pids[i]
    axes.plot(dataticks[usedata],lou_mdls[i].coef_.flatten()[usedata],
                 alpha = .9,label = 'UM'+str(int(pid)), 
                 color = cmap(i/len(pids)),linewidth = 3)

if(data_type == 'FEAT'):
    axes.set_xticklabels(dataticks,rotation = 90,fontsize = ticfont)
axes.set_xlabel(fname,fontsize = labfont)
axes.set_ylabel("Coefficient",fontsize = labfont)
axes.tick_params(axis='x',labelsize = ticfont)
axes.tick_params(axis='y',labelsize = ticfont)

axes.axhline(y=0, color = "k", linestyle = "--",alpha = .5, linewidth =4)
lgd = axes.legend(loc='upper left', bbox_to_anchor=(1,1.02),
                  frameon = False,fontsize=legfont,ncol=2)


for lines in lgd.get_lines():
    lines.set_linewidth(6)




