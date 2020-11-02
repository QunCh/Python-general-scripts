import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from sklearn.model_selection import KFold
import statsmodels.api as sm
%matplotlib inline
import seaborn as sns
import pandas as pd

# Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()

predictors = pd.read_csv(r"C:\Users\qunch\Documents\GitHub\Applied-Predictive-Modeling-with-Python\data\predictors.csv")
classes = pd.read_csv(r"C:\Users\qunch\Documents\GitHub\Applied-Predictive-Modeling-with-Python\data\classes.csv")

colors = ['r', 'b']
markers = ['o', 's']
c = ['Class1', 'Class2']
for k, m in enumerate(colors):
    i = np.where(classes.iloc[:,0] == c[k])[0]
    if k == 0:
        plt.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 1')
    else:
        plt.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 2')

plt.title('Original Data')
plt.xlabel('Predictor A')
plt.ylabel('Predictor B')
plt.legend(loc='upper center', ncol=2)
plt.show()

## basic settings for classification boundary
X = np.array(predictors)
y = np.ravel(classes)

h = .002  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() -0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

## K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

## 3 neighbors (use odd value to avoid ties)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
Z3 = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
Z3 = Z3.reshape(xx.shape)
Z3 = (Z3 == 'Class1').astype('int')

## optimal neighbors
knn_param = {
    'n_neighbors': range(1, 100),
}

gs_neigh = GridSearchCV(KNeighborsClassifier(), knn_param, cv = 2, n_jobs=-1)
gs_neigh.fit(X, y)
Z = gs_neigh.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
Z = (Z == 'Class1').astype('int')

colors = ['r', 'b']
markers = ['o', 's']
c = ['Class1', 'Class2']

fig, (ax1, ax2) = plt.subplots(1, 2)

for k, m in enumerate(colors):
    i = np.where(classes.iloc[:,0] == c[k])[0]
    if k == 0:
        ax1.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 1')
    else:
        ax1.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 2')

ax1.set_title('Model #1')
ax1.set_xlabel('Predictor A')
ax1.set_ylabel('Predictor B')
ax1.legend(loc='upper center', ncol=2)
# plot boundary
ax1.contour(xx, yy, Z3, cmap=plt.cm.Paired)


for k, m in enumerate(colors):
    i = np.where(classes.iloc[:,0] == c[k])[0]
    if k == 0:
        ax2.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 1')
    else:
        ax2.scatter(predictors.iloc[i, 0], predictors.iloc[i, 1], 
                    c=m, marker=markers[k], alpha=0.4, s=26, label='Class 2')

ax2.set_title('Model #2')
ax2.set_xlabel('Predictor A')
ax2.set_ylabel('Predictor B')
ax2.legend(loc='upper center', ncol=2)
# plot boundary
ax2.contour(xx, yy, Z, cmap=plt.cm.Paired)

############ Model Tuning

######### Data Splitting

# dissimilarity sampling
def group_dissimilarity(A, S):
    '''measure average dissimilarity between A and a group of samples S'''
    A = np.array(A, ndmin=2)
    A_extend = np.repeat(A, S.shape[0], axis=0)
    return np.mean(np.sqrt(np.sum((A_extend - S)**2, axis=1)))
    
def dissimilarity_sampling(data, train_prop = 0.5):
    '''dissimilarity sampling of a dataset into training and test set'''
    data = np.array(data, ndmin=2)
    n_data = data.shape[0]
    n_train = np.int(n_data*train_prop)
    rd_int = np.random.randint(n_data, size=1)
    train = np.array(data[rd_int, :], ndmin=2) # initial point
    data = np.delete(data, rd_int, axis=0) # remove selected sample from data
    
    # sampling
    counter = 1
    while counter < n_train:
        counter += 1
        dist_array = np.zeros(n_data - counter + 1)
        for idx, i in enumerate(data):
            dist_array[idx] = group_dissimilarity(i, train)
        dx = np.argmax(dist_array) # find the most dissimilar
        train = np.vstack([train, data[dx]])
        data = np.delete(data, dx, axis=0)
        
    return (train, data)

pred_train, pred_test = dissimilarity_sampling(predictors)
print "Dimensions of training set is {0}; test set is {1}".format(pred_train.shape, pred_test.shape)


############# Case Study

germancredit = pd.read_csv("../datasets/GermanCredit/GermanCredit.csv")
germancredit.head(5)

germancredit = germancredit.drop('Unnamed: 0', axis=1) # drop the first column
germancredit.shape

def stratified_sampling(data, target, prop_train):
    '''a stratified random sampling based on proportion of the target outcomes.'''
    from collections import Counter
    
    n_class = dict(Counter(data[target]).items()) # frequency of possible outcomes
    data.reindex(np.random.permutation(data.index)) # random shuffle
    for key, val in n_class.iteritems():
        n_train = np.int(val*prop_train)
        try:
            trainset = trainset.append(data[data[target] == key].iloc[:n_train,:], ignore_index=True)
            testset = testset.append(data[data[target] == key].iloc[n_train:, :], ignore_index=True)
        except NameError:
            trainset = data[data[target] == key].iloc[:n_train,:]
            testset = data[data[target] == key].iloc[n_train:, :]
        
    return trainset, testset
    
credit_train, credit_test = stratified_sampling(germancredit, 'Class', 0.8)
print "The size of training set is {0} and test set is {1}.".format(credit_train.shape[0], credit_test.shape[0])

#################### Choose Final Tuning Parameter

X_train = credit_train.drop('Class', axis=1)
y_train = credit_train['Class']
X_test = credit_test.drop('Class', axis=1)
y_test = credit_test['Class']

# Support Vector Machine with kernel 'rbf'
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

svc_param = {
    'C': np.logspace(-2, 7,num=10, base = 2),
}

# 10-fold CV for each model
from scipy.stats import sem

gs_svm = GridSearchCV(SVC(kernel = 'rbf'), svc_param, cv=10, n_jobs=-1)
%time _ = gs_svm.fit(X_train, y_train)

#cross-validated score
cv_score = np.zeros([len(svc_param['C']), 2])
for idx, i in enumerate(gs_svm.grid_scores_):
    cv_score[idx, 0] = np.mean(i[2])
    cv_score[idx, 1] = sem(i[2]) * 2 # two standard errors
    
gs_svm.grid_scores_

# apparent score
apparent_score = np.zeros(len(svc_param['C']))

# iterate over all models
for cdx, c in enumerate(svc_param['C']):
    svm = SVC(kernel = 'rbf', C = c)
    svm.fit(X_train, y_train)
    apparent_score[cdx] = svm.score(X_train, y_train)

plt.fill_between(svc_param['C'],
                cv_score[:,0] - cv_score[:,1],
                cv_score[:,0] + cv_score[:,1],
                color = 'b', alpha = .2)
plt.plot(svc_param['C'], cv_score[:,0], 'o-k', c='b', label='Cross-validated')

plt.plot(svc_param['C'], apparent_score, 'o-k', c='g', label='Apparent')

plt.xlabel('Cost')
plt.ylabel('Estimated Accuracy')
plt.ylim((None, 1.05))
plt.legend(loc = 'upper right', ncol=2)

# LOOCV (leave-one-out cross-validation)
from sklearn.cross_validation import LeaveOneOut

gs_svm = GridSearchCV(SVC(kernel = 'rbf'), svc_param, cv=LeaveOneOut(X_train.shape[0]), n_jobs=-1)
%time _ = gs_svm.fit(X_train, y_train)

#cross-validated score
loocv_score = np.zeros([len(svc_param['C']), 2])
for idx, i in enumerate(gs_svm.grid_scores_):
    loocv_score[idx, 0] = np.mean(i[2])
    loocv_score[idx, 1] = sem(i[2])*2 # two standard errors

# bootstrap (with/without 632 method)
bootstrap_score = np.zeros([len(svc_param['C']), 2])
bootstrap632_score = np.zeros([len(svc_param['C']), 2])

n_resamples = 50

def bootstrap_resample(X, y):
    '''generate bootstrap training/test set.'''
    sample_size = X.shape[0]
    boot_idx = np.random.choice(sample_size, sample_size)
    # find out-of-bag samples
    boot_outbag = list(set(range(sample_size)) - set(np.unique(boot_idx)))
    X_bs, X_outbag = X.iloc[boot_idx, :], X.iloc[boot_outbag, :]
    y_bs, y_outbag = y[boot_idx], y[boot_outbag]
    return X_bs, X_outbag, y_bs, y_outbag

# iterate over all models
for cdx, c in enumerate(svc_param['C']):
    scores = np.zeros(n_resamples)
    scores_632 = np.zeros(n_resamples)
    for r in range(n_resamples):
        X_bs, X_outbag, y_bs, y_outbag = bootstrap_resample(X_train, y_train)
        svm = SVC(kernel = 'rbf', C = c)
        svm.fit(X_bs, y_bs)
        scores[r] = svm.score(X_outbag, y_outbag)
        # 632 method
        scores_632[r] = 0.632*svm.score(X_outbag, y_outbag) + 0.368*svm.score(X_bs, y_bs)
    bootstrap_score[cdx, 0] = np.mean(scores)
    bootstrap_score[cdx, 1] = sem(scores)*2 # two standard errors
    bootstrap632_score[cdx, 0] = np.mean(scores_632)
    bootstrap632_score[cdx, 1] = sem(scores_632)*2 # two standard errors

# repeated training/test splits
repeated_score = np.zeros([len(svc_param['C']), 2])

n_resamples = 50
p_heldout = 0.2 # proportion of held-out

def random_splits(X, y, p_heldout):
    '''random split training/test set.'''
    sample_idx = range(X.shape[0])
    n_heldout = np.int(X.shape[0]*p_heldout)
    np.random.shuffle(sample_idx)
    heldout_idx = sample_idx[:n_heldout]
    sample_idx = sample_idx[n_heldout:]
    X_heldout, X_sample = X.iloc[heldout_idx, :], X.iloc[sample_idx, :]
    y_heldout, y_sample = y[heldout_idx], y[sample_idx]
    return X_sample, X_heldout, y_sample, y_heldout

# iterate over all models
for cdx, c in enumerate(svc_param['C']):
    scores = np.zeros(n_resamples)
    for r in range(n_resamples):
        X_sample, X_heldout, y_sample, y_heldout = random_splits(X_train, y_train, p_heldout)
        svm = SVC(kernel = 'rbf', C = c)
        svm.fit(X_sample, y_sample)
        scores[r] = svm.score(X_heldout, y_heldout)
    repeated_score[cdx, 0] = np.mean(scores)
    repeated_score[cdx, 1] = sem(scores)*2 # two standard erros
    
print repeated_score

fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)

# apparent score
#axarr[0, 0].plot(svc_param['C'], apparent_score, 'o-k', c='b')
#axarr[0, 0].set_title("Apparent")
#axarr[0, 0].set_ylim((None, 1.05))

# 10-fold cross-validation
axarr[1, 0].fill_between(svc_param['C'],
                 cv_score[:,0] - cv_score[:,1],
                 cv_score[:,0] + cv_score[:,1],
                 color = 'b', alpha = .2)
axarr[1, 0].plot(svc_param['C'], cv_score[:,0], 'o-k', c='b')
axarr[1, 0].set_title('10-fold Cross-Validation')

# LOOCV
axarr[1, 1].fill_between(svc_param['C'],
                 loocv_score[:,0] - loocv_score[:,1],
                 loocv_score[:,0] + loocv_score[:,1],
                 color = 'b', alpha = .2)
axarr[1, 1].plot(svc_param['C'], loocv_score[:,0], 'o-k', c='b')
axarr[1, 1].set_title('LOOCV')

# Bootstrap
axarr[2, 0].fill_between(svc_param['C'],
                 bootstrap_score[:,0] - bootstrap_score[:,1],
                 bootstrap_score[:,0] + bootstrap_score[:,1],
                 color = 'b', alpha = .2)
axarr[2, 0].plot(svc_param['C'], bootstrap_score[:,0], 'o-k', c='b')
axarr[2, 0].set_title('Bootstrap')

# Bootstrap with 632
axarr[2, 1].fill_between(svc_param['C'],
                 bootstrap632_score[:,0] - bootstrap632_score[:,1],
                 bootstrap632_score[:,0] + bootstrap632_score[:,1],
                 color = 'b', alpha = .2)
axarr[2, 1].plot(svc_param['C'], bootstrap632_score[:,0], 'o-k', c='b')
axarr[2, 1].set_title('Bootstrap 632')

#
axarr[0, 1].fill_between(svc_param['C'],
                 repeated_score[:,0] - repeated_score[:,1],
                 repeated_score[:,0] + repeated_score[:,1],
                 color = 'b', alpha = .2)
axarr[0, 1].plot(svc_param['C'], repeated_score[:,0], 'o-k', c='b', label='Repeated training/test splits')
axarr[0, 1].set_title('Repeated training/test splits')

fig.text(0.5, 0.04, 'Cost', ha='center', va='center')
fig.text(0.04, 0.5, 'Resampled Accuracy', va='center', rotation='vertical')
#fig.xlabel('Cost')
#fig.ylabel('Estimated Accuracy')
#fig.ylim('Estimated Accuracy')
#fig.legend(loc = 'upper right', ncol=2)







######################## Elements of Statistical learning

X = np.random.uniform(0, 1, size=(80, 20))
Y = (np.sum(X[:,:10], axis=1)>5) + 0
X = np.hstack((np.ones(shape=(80,1)), X))

# K-fold Cross-validation 
kf = KFold(n_splits=10) # 10折
p_errors = []
for p in range(21):
    errors = []
    stds = []
    for train_index, valid_index in kf.split(X):
        X_train, Y_train = X[train_index], Y[train_index]
        X_valid, Y_valid = X[valid_index], Y[valid_index]
        best_model, best_combination, best_beta, min_error = None, None, None, np.inf
        for predictors in combinations(list(range(1, 21)), p):
            predictors = [0] + list(predictors)
            X_ = X_train[:,predictors]
            beta = np.linalg.inv(X_.T @ X_) @ X_.T @ Y_train
            y_hat = np.round(X_ @ beta)
            error = np.mean(np.abs(Y_train - y_hat))
            if error < min_error:
                best_combination = predictors
                min_error = error
                best_beta = beta
        pred = np.round(X_valid[:,best_combination] @ best_beta)
        valid_error = np.mean(np.abs(Y_valid - pred))
        errors.append(valid_error)
        #print(best_combination, min_error, valid_error)
    print(p, np.mean(errors), np.std(errors, ddof=1)/np.sqrt(10))
    p_errors.append(errors)


y = [np.mean(e) for e in p_errors]
e = [np.std(e, ddof=1)/np.sqrt(10) for e in p_errors]
x = list(range(21))
plt.errorbar(x, y, e, marker='o')
plt.ylim(0, 0.6)