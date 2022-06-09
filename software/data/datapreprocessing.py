#!/usr/bin/env python
# coding: utf-8

# In[1]:


from python_speech_features import mfcc
from CQCC.cqcc import cqcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse


# In[4]:


label_path = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
data_path = 'LA/ASVspoof2019_LA_train/flac/'
output_path = './data/train.pkl'


# In[5]:


# Extraction of CQCC features based on the implemented CQCC from matlab version
def extract_cqcc(x, fs):
    B = 96
    fmax = fs/2
    fmin = fmax/2**9
    d = 16
    cf = 19
    ZsdD = 'ZsdD'

    x = x.reshape(x.shape[0], 1)  
    
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
    return CQcc, fmax, fmin


# In[10]:


# read in labels
flabel = {}
for line in open(label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    flabel[filename] = label

feats = []
for filepath in os.listdir(data_path):
    filename = filepath.split('.')[0]
    if filename not in flabel: 
        continue
    label = flabel[filename]
    print("Filename:", os.path.join(data_path, filepath))
    sig, rate = sf.read(os.path.join(data_path, filepath))
    print("Rate:", rate)
    feat_cqcc, fmax, fmin = extract_cqcc(sig, rate)
    print("Feat CQCC:", feat_cqcc.shape)
    numframes = feat_cqcc.shape[0]
    winstep = 0.005
    winlen =  (len(sig) - winstep*rate*(numframes-1))/rate
    feat_mfcc = mfcc(sig,rate,winlen=winlen,winstep=winstep, lowfreq=fmin,highfreq=fmax)      
    print("Feat MFCC:", feat_mfcc.shape)
    feats.append((feat_cqcc, feat_mfcc, label))

print("number of instances:", len(feats))





# In[16]:


import pickle
file_name='train.pkl'
f = open(file_name,'wb')
pickle.dump(feats,f)
f.close()


# In[ ]:




