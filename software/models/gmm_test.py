#Import libraries
import os
import numpy as np
import pickle
import librosa
import soundfile as sf
import time
import gc

from sklearn import preprocessing
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture as GMM
from pdb import set_trace
from scipy import stats


#Define GMM feature
def gmmtest(testpath, dest, featuretype):
    
    gmm_sp  = pickle.load(open(dest + 'sp' + '.gmm','rb'))
    gmm_bon = pickle.load(open(dest + 'bon' + '.gmm','rb'))
    
    bonafide_data = []
    spoof_data = []
    
    predb = []
    preds = []
    
    
    with open(testpath, 'rb') as infile:
        
        data = pickle.load(infile)
        for feat_cqcc, feat_mfcc, label in data:
            
            if (label == 'bonafide'):
                j += 1
                bonafide_data.append(feats)
            elif(label == 'spoof'):
                spoof_data.append(feats)
                
            if featuretype == "cqcc":
                feats = feat_cqcc
            
            elif featuretype == "mfcc":
                feats = feat_mfcc
           
          
           

   
    j_bon = len(bonafide_data)
    k_sp  = len(spoof_data)

    for i in range(j_bon):
        if (i % 50 == 0):
            print('Evaluating Bonafide sample at',i/j_bon * 100, '%')
        
        X = bonafide_data[i]
        spoof_score = gmm_sp.score(X)
        bonafide_score = gmm_bon.score(X)
        
        predb.append(bonfaide_score-spoof_score)

    for i in range(k_sp):
        if (i % 50 == 0):
            print('Evaluating Spoof sample at',i/k_sp * 100, '%')
        X = spoof_data[i]
        bonafide_score = gmm_bon.score(X)
        spoof_score = gmm_sp.score(X)

        preds.append(bonafide_score-spoof_score)

    predb1 = np.asarray(predb)
    preds1 = np.asarray(preds)

    predb1[predb1 < 0] = 0
    predb1[predb1 > 0] = 1
    predbresult1 = np.sum(predb1)
    
    print(predbresult1, 'CORRECTLY evaluated Bonafide samples were out of', j_bon,'samples. Bon_Accuracy = ', predbresult1/j_bon)


    preds1[preds1 > 0] = 0
    preds1[preds1 < 0] = 1
    predsresult = np.sum(preds1)
    print(predsresult, 'CORRECTLY evaluated Spoofed samples were out of', k_sp, 'samples. Sp_Accuracy = ', predsresult/k_sp)

    print('Accuracy of the GMM classifier = ',(predbresult1 + predsresult)/(j_bon + k_sp))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str,  default='./data/dev.pkl', help='path to pickled file. For example, data/train.pkl')
    parser.add_argument("--model_path", required=True, type=str, default='./data/', help='path to pickled file. For example, data/train.pkl')
    parser.add_argument("--featuretype", required=True, type=str, default='cqcc', help='select the feature type. cqcc or mfcc')
    args = parser.parse_args()

    dev_path = args.data_path
    dest = args.model_path    
    gmmtest(dev_path, dest, args.featuretype)