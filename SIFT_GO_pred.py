import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

GO_idx=int(sys.argv[1])
GO_term_array=np.load('sift_data/mat_Y.npy')
features=np.load('sift_data/500_mat_X.npy')
filename='sift_data/list15to500GO_GAB.pkl'
with open(filename, 'rb') as handle:
    GO_terms=pickle.load(handle)
    
pipe_LinearSVC = Pipeline([
    ('clf', LinearSVC(penalty='l1', dual=False, C=0.1, class_weight="balanced", max_iter=1000, random_state=0)),
])

'''
Searching for hyperparameters with Randomized search
'''

params = {"clf__penalty": ['l1'],
          "clf__C": [0.1]} 

seed = 551 # Setting a constant seed for repeatability.
num_iter=1
cv=5 #number of cross-validation folds
pipe=pipe_LinearSVC 

filename='sift_data/sift_GO_chunk.pkl'
with open(filename, 'rb') as handle:
    chunk_list=pickle.load(handle)

idx=chunk_list[GO_idx]

X=features
scores=[]
for i in idx:
    Y=GO_term_array[:,i]
    random_search = RandomizedSearchCV(pipe, param_distributions = params, scoring='roc_auc',cv=cv, verbose = 10, random_state = seed, n_iter = num_iter)
    random_search.fit(X, Y)

    #getting cross validation results
    results=random_search.cv_results_
    data = {"mean_test_score": list(results.get('mean_test_score').data),}

    print(data["mean_test_score"])
    scores.append([data, GO_terms[i]])

filename='data/sift_data/pred/pred_GO%s.pkl' % (str(GO_idx))
with open(filename, 'wb') as handle:
    pickle.dump(scores, handle)
