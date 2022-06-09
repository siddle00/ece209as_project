#Library Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


from sklearn.svm import SVC


#Parse and obtain the pickle files
parser = argparse.ArgumentParser()
parser.add_argument("--feature_type", required=True, type=str, help='Select Feature CQCC or MFCC')
parser.add_argument("--data_path", required=True, type=str, help='Path to pickled files, e.g data/train.pkl')
args = parser.parse_args()


X = []
y = []
max_length = 50  # 1.25 seconds


with open(args.data_path, 'rb') as infile:
    data = pickle.load(infile)
    for feat_cqcc, feat_mfcc, label in data:
        if args.feature_type == "cqcc":
            features = feat_cqcc
        elif args.feature_type == "mfcc":
            features = feat_mfcc
        
        nd = features.shape[1]
        
        if len(features) > max_length:
            features = features[:max_length]
        elif len(features) < max_length:
            features = np.concatenate((features, np.array([[0.]*nd]*(max_length-len(features)))), axis=0)
        X.append(features.reshape(-1))
        y.append(label)

clf = SVC()
clf.fit(np.array(X), y)

print ('Training Accuracy:', clf.score(X, y))

with open('svm_{}.pkl'.format(max_length), 'wb') as outfile:
    pickle.dump(clf, outfile)
