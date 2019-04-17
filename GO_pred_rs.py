from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC

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

X=enc_features
scores=[]
for i in idx_list:
    Y=GO_term_array[i,:]
    random_search = RandomizedSearchCV(pipe, param_distributions = params, scoring='roc_auc',cv=cv, verbose = 10, random_state = seed, n_iter = num_iter)
    random_search.fit(X, Y)

    #getting cross validation results
    results=random_search.cv_results_
    data = {"mean_test_score": list(results.get('mean_test_score').data),}

    print(data["mean_test_score"])
    scores.append([data, list15to500values[i], list15to500GO[i]])
