import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

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
        if GO_annotations_nb[GO] >= 15 and GO_annotations_nb[GO] <= 500:
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


pipe_LinearSVC = Pipeline([
    ('clf', LinearSVC(penalty='l1', dual=False, C=0.1, class_weight="balanced", max_iter=1000, random_state=0)),
])

'''
Searching for hyperparameters with Randomized search
'''

#19*100=900min, but could run it in parallel
params = {"clf__penalty": ['l1'],
          "clf__C": [0.1]} 

seed = 551 # Setting a constant seed for repeatability.
num_iter=1
cv=5 #number of cross-validation folds
pipe=pipe_LinearSVC 

filename='GO_chunk.pkl'
with open(filename, 'rb') as handle:
    chunk_list=pickle.load(handle)

idx=chunk_list[GO_idx]

X=enc_features
scores=[]
for i in idx:
    Y=GO_term_array[i,:]
    random_search = RandomizedSearchCV(pipe, param_distributions = params, scoring='roc_auc',cv=cv, verbose = 10, random_state = seed, n_iter = num_iter)
    random_search.fit(X, Y)

    #getting cross validation results
    results=random_search.cv_results_
    data = {"mean_test_score": list(results.get('mean_test_score').data),}

    print(data["mean_test_score"])
    scores.append([data, list15to500values[i], list15to500GO[i]])

filename='data/pred/pred_GO%s.pkl' % (str(GO_idx))
with open(filename, 'wb') as handle:
    pickle.dump(scores, handle)
