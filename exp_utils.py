import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

from utils import *
from exp_utils import *
from tests import *
from cd_models import *


def GenData(theta, lamb, n = 5000):
        
    Xs = np.random.normal(0,1,n)
    ys = np.random.normal(Xs,1)
    Xs = Xs.reshape(-1,1)
    ys = ys.reshape(-1)

    Xt = np.random.normal(lamb,1,n)
    yt = np.random.normal(theta + Xt,1)
    Xt = Xt.reshape(-1,1)
    yt = yt.reshape(-1)
    
    return Xs, ys, Xt, yt

def GenData2(gamma, delta, d=2, n = 5000):
    
    ys = np.random.binomial(1, .5, n)  
    yt = np.random.binomial(1, .5 + delta, n)
    
    Xs=[]
    Xt=[]
    
    for i in range(n):
        Xs.append(np.random.normal(ys[i],1,d).tolist())
        Xt.append(np.random.normal(yt[i]+gamma,1,d).tolist())
        
    Xs=np.array(Xs)
    Xt=np.array(Xt)
    Xs = Xs.reshape(n,-1)
    Xt = Xt.reshape(n,-1)
    
    ys = ys.reshape(-1)
    yt = yt.reshape(-1)
    
    return Xs, ys, Xt, yt

### cd_model
class cde_regH0:
    def sample(self, X):      
        return pd.DataFrame(np.random.normal(X, 1))
    
### TV
def tv(Zs,Zt):
    zvalues=np.unique(np.hstack((np.unique(Zs),np.unique(Zt))))
    tv=0
    for z in zvalues:
        pzt=np.mean(np.array(Zt).squeeze()==z)
        pzs=np.mean(np.array(Zs).squeeze()==z)
        tv+=.5*np.abs(pzt-pzs)
    return tv

def tv_conc(Xs,ys,Xt,yt):    
    xvalues=np.unique(np.hstack((np.unique(Xs),np.unique(Xt))))
    yvalues=np.unique(np.hstack((np.unique(ys),np.unique(yt))))
    tv=0
    for x in xvalues:
        pxt=np.mean(np.array(Xt).squeeze()==x)
        pxs=np.mean(np.array(Xs).squeeze()==x)
        tv_aux=0

        for y in yvalues:
            pyt=np.mean(yt[Xt==x].squeeze()==y)
            pys=np.mean(ys[Xs==x].squeeze()==y)

            tv_aux+=.5*np.abs(pyt-pys)

        tv+=.5*(pxt+pxs)*tv_aux
    return tv

### MMD
def ker(x, y, sigma):
    return np.exp(-np.sum((x-y)**2)/sigma)

def h(i, X, Y, sigma):
    return ker(X[2*i-1],X[2*i], sigma) + ker(Y[2*i-1],Y[2*i], sigma) - ker(X[2*i-1],Y[2*i], sigma) - ker(X[2*i],Y[2*i-1], sigma)

def MMD(X1, X2):
    n1,n2=X1.shape[0],X2.shape[0]
    assert n1==n2
    m=int(n1/2)
    sigma=np.median(euclidean_distances(np.vstack((X1, X2))))
    H=[h(i, X1, X2, sigma) for i in range(m)]
    return 1-norm.cdf(np.sqrt(m)*np.mean(H)/np.std(H))

### Z-test
def ztest(ys,yt):

    nt=yt.shape[0]
    ns=ys.shape[0]
    pt=np.mean(yt)
    ps=np.mean(ys)

    p=(nt*pt+ns*ps)/(nt+ns)
    pval=2*(1-norm.cdf(np.abs((pt-ps)/np.sqrt(p*(1-p)*(1/nt+1/ns)))))

    return pval
    
### Permutation test
def permut_compute(f,Xs,Xt,X):
    shuffle = np.random.choice(range(X.shape[0]), size=(X.shape[0],), replace=False)
    inds = shuffle[:Xs.shape[0]]
    indt = shuffle[Xs.shape[0]:]
    return f(X[inds], X[indt])

def permut_pval(value, perm, B):
    #Enforcing uniformity of p-values under H0 (adding a very small random number - we guarantee every statistic has a different value)
    s=10**-10
    perm+=np.random.normal(0,s,perm.shape[0])
    value+=np.random.normal(0,s,1)[0]
    return (1+np.sum(np.array(perm) >= value))/(B+1)

def permut_test(f,Xs,Xt,B):
    value=f(Xs, Xt)
    X = np.vstack((Xs.reshape(Xs.shape[0], -1), Xt.reshape(Xt.shape[0], -1)))
    perm = np.array([permut_compute(f,Xs,Xt,X) for b in range(B)])
    pval=permut_pval(value, perm, B)
    return pval

### Classifier test
class classifier:

    def __init__(self, boost=True, validation_split=.1, cat_features=None, cv=5):
        
        self.boost=boost
        self.validation_split=validation_split
        self.cat_features=cat_features
        self.cv=cv
    
    def fit(self, X, y, random_seed=None):

        n=X.shape[0]

        if self.boost:   
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, stratify=y, random_state=random_seed)

            self.model =  CatBoostClassifier(loss_function = 'MultiClass',
                                             cat_features=self.cat_features,
                                             thread_count=-1,
                                             random_seed=random_seed)

            self.model.fit(X_train, y_train,
                           verbose=False,
                           eval_set=(X_val, y_val),
                           early_stopping_rounds = 5)
         
        else:
            if self.cv==None:
                self.model = LogisticRegression(solver='liblinear', random_state=random_seed).fit(X, y)
            else: 
                self.model = LogisticRegressionCV(cv=self.cv, scoring='neg_log_loss', solver='liblinear', 
                                                  random_state=random_seed).fit(X, y)
               
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
    
    def accuracy(self, X, y):
        return np.mean(self.model.predict(X)==y)
    
    def pval(self, X, y):
        n = X.shape[0]
        p = self.accuracy(X, y)
        pval=1-norm.cdf((p-.5)/np.sqrt((p*(1-p))/n))
        return pval
 
    def auc(self, X, y):
        return roc_auc_score(y, self.model.predict(X))
    
def classifier_test(Xs_train, Xs_test, Xt_train, Xt_test, boost=False, cv=None):
    X_aux_train = np.vstack((Xt_train, Xs_train))
    y_aux_train = np.hstack((np.ones(Xt_train.shape[0]),np.zeros(Xs_train.shape[0])))
    X_aux_test = np.vstack((Xt_test, Xs_test))
    y_aux_test = np.hstack((np.ones(Xt_test.shape[0]),np.zeros(Xs_test.shape[0])))
    clf = classifier(boost=boost, cv=cv)
    clf.fit(X_aux_train, y_aux_train)
    pval=clf.pval(X_aux_test, y_aux_test)
    return pval

### Cond. Indep. Test 1
def get_cond_permut(X, y_d):
    y = get_classes(y_d)
    Y = np.unique(y)
    ind={}
    for j in Y:
        ind[j] = np.array(y==j).squeeze()

    X_perm = pd.DataFrame(np.zeros(X.shape))
    shuffle={}
    indt={}
    for j in Y:
        shuffle[j] = np.random.choice(range(np.sum(ind[j])), size=(np.sum(ind[j]),), replace=False)
        ind_perm=[i for i, x in enumerate(y.squeeze()==j) if x] #getting positions where True
        X_perm.iloc[ind_perm,:] = X.loc[ind[j],:].iloc[shuffle[j],:]
        
    return X_perm

def ci_test1(Xs_train, Xs_test, ys_train, ys_test, Xt_train, Xt_test, yt_train, yt_test, B, boost=True, cv=None):
   
    X_train = pd.concat([Xs_train, Xt_train], axis=0).reset_index(drop=True)
    y_train = pd.concat([ys_train, yt_train], axis=0).reset_index(drop=True)
    Z_train = pd.DataFrame(np.hstack((np.zeros(ys_train.shape[0]),np.ones(yt_train.shape[0]))).reshape((-1,1))).reset_index(drop=True).rename(columns={0:'z'})
    X_train_fake = get_cond_permut(X_train, y_train) 
    X_train_fake.columns = ['x'+str(j) for j in range(X_train_fake.shape[1])]
   
    X_aux_train = pd.concat([pd.concat([X_train, Z_train, y_train], axis=1), pd.concat([X_train_fake, Z_train, y_train], axis=1)], axis=0)
    y_aux_train = pd.DataFrame(np.hstack((np.ones(X_train.shape[0]),np.zeros(X_train.shape[0]))).reshape((-1,1))).reset_index(drop=True).values.ravel()

    clf = classifier(boost=boost, cv=cv)
    clf.fit(X_aux_train, y_aux_train)

    X_test = pd.concat([Xs_test, Xt_test], axis=0).reset_index(drop=True)
    y_test = pd.concat([ys_test, yt_test], axis=0).reset_index(drop=True)
    Z_test = pd.DataFrame(np.hstack((np.zeros(ys_test.shape[0]),np.ones(yt_test.shape[0]))).reshape((-1,1))).reset_index(drop=True).rename(columns={0:'z'})
    X_aux_test = pd.concat([X_test, Z_test, y_test], axis=1)

    s=10**-10
    p_class = np.mean(clf.predict(X_aux_test))+np.random.normal(0,s,1)[0]
    p_proba = np.mean(clf.predict_proba(X_aux_test))+np.random.normal(0,s,1)[0]

    pval_class=1
    pval_proba=1
    
    for b in range(B):
        X_test_fake = get_cond_permut(X_test, y_test)
        X_test_fake.columns = ['x'+str(j) for j in range(X_test_fake.shape[1])]
        X_aux_test_fake = pd.concat([X_test_fake, Z_test, y_test], axis=1)
        pval_class += 1*(p_class<np.mean(clf.predict(X_aux_test_fake))+np.random.normal(0,s,1)[0])
        pval_proba += 1*(p_proba<np.mean(clf.predict_proba(X_aux_test_fake))+np.random.normal(0,s,1)[0])

    pval_class/=(B+1)
    pval_proba/=(B+1)
    
    return pval_class, pval_proba

### Cond. Indep. Test 2
def ci_test2(Xs_train, Xs_test, ys_train, ys_test, Xt_train, Xt_test, yt_train, yt_test, cd_model, B, boost=True, cv=None):
   
    X_train = pd.concat([Xs_train, Xt_train], axis=0).reset_index(drop=True)
    y_train = pd.concat([ys_train, yt_train], axis=0).reset_index(drop=True)
    Z_train = pd.DataFrame(np.hstack((np.zeros(ys_train.shape[0]),np.ones(yt_train.shape[0]))).reshape((-1,1))).reset_index(drop=True).rename(columns={0:'z'})
    y_train_fake = cd_model.sample(X_train)
    y_train_fake.columns = ['y'+str(j) for j in range(y_train_fake.shape[1])]

    X_aux_train = pd.concat([pd.concat([X_train, Z_train, y_train], axis=1), pd.concat([X_train, Z_train, y_train_fake], axis=1)], axis=0)
    y_aux_train = pd.DataFrame(np.hstack((np.ones(X_train.shape[0]),np.zeros(X_train.shape[0]))).reshape((-1,1))).reset_index(drop=True).values.ravel()

    clf = classifier(boost=boost, cv=cv)
    clf.fit(X_aux_train, y_aux_train)

    X_test = pd.concat([Xs_test, Xt_test], axis=0).reset_index(drop=True)
    y_test = pd.concat([ys_test, yt_test], axis=0).reset_index(drop=True)
    Z_test = pd.DataFrame(np.hstack((np.zeros(ys_test.shape[0]),np.ones(yt_test.shape[0]))).reshape((-1,1))).reset_index(drop=True).rename(columns={0:'z'})
    X_aux_test = pd.concat([X_test, Z_test, y_test], axis=1)

    s=10**-10
    p_class= np.mean(clf.predict(X_aux_test))+np.random.normal(0,s,1)[0]
    p_proba= np.mean(clf.predict_proba(X_aux_test))+np.random.normal(0,s,1)[0]

    pval_class=1
    pval_proba=1
    
    for b in range(B):
        y_test_fake = cd_model.sample(X_test)
        y_test_fake.columns = ['y'+str(j) for j in range(y_test_fake.shape[1])]
        X_aux_test_fake = pd.concat([X_test, Z_test, y_test_fake], axis=1)
        pval_class += 1*(p_class<np.mean(clf.predict(X_aux_test_fake))+np.random.normal(0,s,1)[0])
        pval_proba += 1*(p_proba<np.mean(clf.predict_proba(X_aux_test_fake))+np.random.normal(0,s,1)[0])

    pval_class/=(B+1)
    pval_proba/=(B+1)
    
    return pval_class, pval_proba
   

### Plots
def exp_plots(theta, series, xlab, ylab, grid):
    names=['$P_{Y}$','$P_{\mathbf{X}}$',
           '$P_{\mathbf{X}|Y}$','$P_{Y|\mathbf{X}}$',
           '$P_{\mathbf{X},Y}$']
    colors=['#2F58EB', '#773BEB', '#12B8EB', '#EB9846', '#6D8AF1','#808080']
    
    plt.plot(theta, series[:,1], color=colors[4], marker="^", lw=1.5, label=names[0], alpha=.7, markersize=6)
    plt.plot(theta, series[:,2], color=colors[3], marker="v", lw=1.5, label=names[1], alpha=.7, markersize=6)
    plt.plot(theta, series[:,3], color=colors[2], marker="D", lw=1.5, label=names[2], alpha=.7, markersize=6)
    plt.plot(theta, series[:,4], color=colors[1], marker="s", lw=1.5, label=names[3], alpha=.7, markersize=6)
    plt.plot(theta, series[:,5], color=colors[0], marker="o", lw=1.5, label=names[4], alpha=.7, markersize=6)

    plt.legend(bbox_to_anchor=(.04, .975), loc='upper left', 
                   ncol = 3, prop={'size': 12}, borderaxespad=.0, frameon=False)
        
    plt.grid(alpha=.2, axis=grid)
    plt.axhline(y=0, color='k', linestyle='-', lw=1, alpha=.25)
    plt.ylabel(ylab, size=15)
    plt.xlabel(xlab, size=15)
    plt.tick_params(labelsize=11)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.locator_params(axis="y", nbins=4)
    
### deep learning exps
from tqdm import tqdm
def get_shifted_data(X,y,ds):

    Xs_dic, ys_dic, Xt_dic, yt_dic = {},{},{},{}

    for d in tqdm(ds):

      K=np.unique(y).shape[0]
      prop=np.linspace(d,1-d,K)
      Xt,yt,Xs,ys = np.zeros(shape=(0,X.shape[1])), np.zeros(shape=(0,y.shape[1])), np.zeros(shape=(0,X.shape[1])), np.zeros(shape=(0,y.shape[1]))

      shuffle = np.random.choice(range(X.shape[0]), size=(X.shape[0],), replace=False)
      X=X[shuffle]
      y=y[shuffle]

      for k in range(K):
        ind=(y==k).squeeze()
        nk=np.sum(ind)
        m=int(prop[k]*nk)
        Xs=np.vstack((Xs, X[ind][:m]))
        ys=np.vstack((ys, y[ind][:m]))
        Xt=np.vstack((Xt, X[ind][m:]))
        yt=np.vstack((yt, y[ind][m:]))

      Xs_dic[d], ys_dic[d], Xt_dic[d], yt_dic[d] = Xs, ys, Xt, yt
    return Xs_dic, ys_dic, Xt_dic, yt_dic

def perform_ours(Xs_dic, ys_dic, Xt_dic, yt_dic, ds, reps, task, B, test):
    pvals=[]
    kls=[]

    for rep in tqdm(range(reps)):

        pvals2=[]
        kls2=[]
        covshift_models=[]

        for d in ds:

            ### Setting-up data
            Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test, \
            Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test = prep_data(Xs_dic[d], ys_dic[d].squeeze(), Xt_dic[d], yt_dic[d].squeeze(), test=test, task=task, random_state=rep)

            ### Training models
            totshift_model = KL(boost=False, cv=None)
            totshift_model.fit(Zs_train, Zt_train)
            covshift_models.append(KL(boost=False, cv=None))
            covshift_models[-1].fit(Xs_train, Xt_train)

            cd_model = cde_class(boost=False, cv=None)
            cd_model.fit(pd.concat([Xs_train, Xt_train], axis=0), 
                         pd.concat([ys_train, yt_train], axis=0))


            ### Getting test statistics and p-vals
            out = ShiftDiagnostics(Xs_test, ys_test, Xt_test, yt_test,
                                   totshift_model=totshift_model, covshift_model=covshift_models[-1], labshift_model=None,
                                   cd_model=cd_model, task=task, B=B, verbose=False)

            ### Output
            pvals2.append([d, out['lab']['pval'], out['cov']['pval'], out['conc1']['pval'], out['conc2']['pval'], out['tot']['pval']])
            kls2.append([d, out['lab']['kl'], out['cov']['kl'], out['conc1']['kl'], out['conc2']['kl'], out['tot']['kl']])

        pvals.append(pvals2)
        kls.append(kls2)
    return np.array(kls), np.array(pvals)

def perform_bench(Xs_dic, ys_dic, Xt_dic, yt_dic, ds, reps, task, B, test):
    
    pvals=[]

    for rep in tqdm(range(reps)):

        pvals2=[]

        for d in ds:

            ### Setting-up data
            Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test, \
            Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test = prep_data(Xs_dic[d], ys_dic[d].squeeze(), Xt_dic[d], yt_dic[d].squeeze(), test=test, task=task, random_state=rep)

            ### XY
            pval_Xy = classifier_test(Zs_train, Zs_test, Zt_train, Zt_test, boost=False, cv=None)

            ### Y
            pval_y = classifier_test(ys_train, ys_test, yt_train, yt_test, boost=False, cv=None)

            ### X
            pval_X = classifier_test(Xs_train, Xs_test, Xt_train, Xt_test, boost=False, cv=None)

            ### X|Y
            pval_X_y, _ = ci_test1(Xs_train, Xs_test, ys_train, ys_test, Xt_train, Xt_test, yt_train, yt_test, B, boost=False, cv=None)

            ### Y|X
            cd_model = cde_class(boost=False, cv=None)
            cd_model.fit(pd.concat([Xs_train, Xt_train], axis=0), 
                         pd.concat([ys_train, yt_train], axis=0))
            pval_y_X, _ = ci_test2(Xs_train, Xs_test, ys_train, ys_test, Xt_train, Xt_test, yt_train, yt_test, cd_model, B, boost=False, cv=None)
            
            ### Output
            pvals2.append([d, pval_y, pval_X, pval_X_y, pval_y_X, pval_Xy])
    
        pvals.append(pvals2)
        
    return np.array(pvals)

def exp_plots2(theta, series, xlab, ylab, grid, legend=False):
    names=['$P_{Y}$','$P_{\mathbf{X}}$',
           '$P_{\mathbf{X}|Y}$','$P_{Y|\mathbf{X}}$',
           '$P_{\mathbf{X},Y}$']
    colors=['#2F58EB', '#773BEB', '#12B8EB', '#EB9846', '#6D8AF1','#808080']
    
    plt.plot(theta, series.mean(axis=0)[:,1], color=colors[4], marker="^", lw=.75, label=names[0], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,1], yerr=series.std(axis=0)[:,1], color=colors[4], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,2], color=colors[3], marker="v", lw=.75, label=names[1], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,2], yerr=series.std(axis=0)[:,2], color=colors[3], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,3], color=colors[2], marker="D", lw=.75, label=names[2], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,3], yerr=series.std(axis=0)[:,3], color=colors[2], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,4], color=colors[1], marker="s", lw=.75, label=names[3], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,4], yerr=series.std(axis=0)[:,4], color=colors[1], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,5], color=colors[0], marker="o", lw=.75, label=names[4], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,5], yerr=series.std(axis=0)[:,5], color=colors[0], lw=.75)
    
    if legend:
        plt.legend(bbox_to_anchor=(2.5, 1.2), loc='center', 
                       ncol = 1, prop={'size': 12}, borderaxespad=.0, frameon=False)

    plt.grid(alpha=.2, axis=grid)
    plt.axhline(y=0, color='k', linestyle='-', lw=1, alpha=.25)
    plt.ylabel(ylab, size=11)
    plt.xlabel(xlab, size=11)
    plt.tick_params(labelsize=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.locator_params(axis="y", nbins=4)
    
def exp_plots3(theta, series, xlab, ylab, grid, legend=False):
    names=['$P_{Y}$','$P_{\mathbf{X}}$',
           '$P_{\mathbf{X}|Y}$','$P_{Y|\mathbf{X}}$',
           '$P_{\mathbf{X},Y}$']
    colors=['#2F58EB', '#773BEB', '#12B8EB', '#EB9846', '#6D8AF1','#808080']
    
    plt.plot(theta, series.mean(axis=0)[:,1], color=colors[4], marker="^", lw=.75, label=names[0], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,1], yerr=series.std(axis=0)[:,1], color=colors[4], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,2], color=colors[3], marker="v", lw=.75, label=names[1], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,2], yerr=series.std(axis=0)[:,2], color=colors[3], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,3], color=colors[2], marker="D", lw=.75, label=names[2], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,3], yerr=series.std(axis=0)[:,3], color=colors[2], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,4], color=colors[1], marker="s", lw=.75, label=names[3], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,4], yerr=series.std(axis=0)[:,4], color=colors[1], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,5], color=colors[0], marker="o", lw=.75, label=names[4], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,5], yerr=series.std(axis=0)[:,5], color=colors[0], lw=.75)
    
    if legend:
        plt.legend(bbox_to_anchor=(2.5, .5), loc='center', 
                       ncol = 1, prop={'size': 12}, borderaxespad=.0, frameon=False)

    plt.grid(alpha=.2, axis=grid)
    plt.axhline(y=0, color='k', linestyle='-', lw=1, alpha=.25)
    plt.ylabel(ylab, size=11)
    plt.xlabel(xlab, size=11)
    plt.tick_params(labelsize=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.locator_params(axis="y", nbins=4)
    
def exp_plots4(theta, series, xlab, ylab, grid, legend=True):
    names=['$P_{Y}$','$P_{\mathbf{X}}$',
           '$P_{\mathbf{X}|Y}$','$P_{Y|\mathbf{X}}$',
           '$P_{\mathbf{X},Y}$']
    colors=['#2F58EB', '#773BEB', '#12B8EB', '#EB9846', '#6D8AF1','#808080']
    
    plt.plot(theta, series.mean(axis=0)[:,1], color=colors[4], marker="^", lw=.75, label=names[0], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,1], yerr=series.std(axis=0)[:,1], color=colors[4], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,2], color=colors[3], marker="v", lw=.75, label=names[1], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,2], yerr=series.std(axis=0)[:,2], color=colors[3], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,3], color=colors[2], marker="D", lw=.75, label=names[2], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,3], yerr=series.std(axis=0)[:,3], color=colors[2], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,4], color=colors[1], marker="s", lw=.75, label=names[3], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,4], yerr=series.std(axis=0)[:,4], color=colors[1], lw=.75)
    
    plt.plot(theta, series.mean(axis=0)[:,5], color=colors[0], marker="o", lw=.75, label=names[4], alpha=.8, markersize=4)
    plt.errorbar(theta, series.mean(axis=0)[:,5], yerr=series.std(axis=0)[:,5], color=colors[0], lw=.75)
    
    if legend:
        plt.legend(bbox_to_anchor=(.04, .975), loc='upper left', 
                       ncol = 3, prop={'size': 10}, borderaxespad=.0, frameon=False)

    plt.grid(alpha=.2, axis=grid)
    plt.axhline(y=0, color='k', linestyle='-', lw=1, alpha=.25)
    plt.ylabel(ylab, size=11)
    plt.xlabel(xlab, size=11)
    plt.tick_params(labelsize=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.locator_params(axis="y", nbins=4)
    