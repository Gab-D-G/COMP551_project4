import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.svm import LinearSVC

GO_idx=int(sys.argv[1])

filename='data/NZ_del20.pkl' #deleted 20% images with lower NZ count
with open(filename, 'rb') as handle:
    subset_dict=pickle.load(handle)
images_list=subset_dict['images']
GO_terms=subset_dict['GO_terms']
del images_list

filename='data/enc_features.pkl'
with open(filename, 'rb') as handle:
    enc_features=pickle.load(handle)


'''
PICK THE GO TERMS THAT SHOW UP IN BETWEEN 15 and 500 Slices
'''
def get_15to500_GO(GO_terms):
    GO_annotations_nb = {}
    for GO_list in GO_terms:
        for GO_term in GO_list:
            if GO_term in GO_annotations_nb:
                GO_annotations_nb[GO_term] += 1
            else:
                GO_annotations_nb[GO_term] = 1

    list15to500GO = []
    list15to500values = []
    for GO in GO_annotations_nb.keys():
        if GO_annotations_nb[GO] >= 15 and GO_annotations_nb[GO] <= 50:
            list15to500GO.append(GO)
            list15to500values.append(GO_annotations_nb[GO])

    GO_list=[]
    for GO_term in list15to500GO:
        image_list=[]
        for image_GO in GO_terms:
            if GO_term in image_GO:
                image_list.append(1)
            else:
                image_list.append(0)
        GO_list.append(image_list)
    return np.asarray(GO_list), list15to500values, list15to500GO
[GO_term_array, list15to500values, list15to500GO]=get_15to500_GO(GO_terms)

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
def kfold(X,Y, model, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    k_fold_scores=[]
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train,Y_train)
        Y_pred=model.predict(X_test)
        score = metrics.roc_auc_score(Y_test, Y_pred)
        k_fold_scores.append(score)
    return np.asarray(k_fold_scores)

model = LinearSVC(penalty='l2', dual=False, C=10, class_weight="balanced", max_iter=1000)

filename='GO_chunk.pkl'
with open(filename, 'rb') as handle:
    chunk_list=pickle.load(handle)

idx=chunk_list[GO_idx]

X=enc_features
scores=[]
for i in idx:
    Y=GO_term_array[i,:]
    score=kfold(X,Y,model, 5)
    print(score)
    scores.append(score, list15to500values[i], list15to500GO[i])


filename='data/pred/pred_GO%s.pkl' % (str(GO_idx))
with open(filename, 'wb') as handle:
    pickle.dump(scores, handle)
