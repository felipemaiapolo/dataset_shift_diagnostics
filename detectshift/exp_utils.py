import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

def ztest(ys,yt):

    nt=yt.shape[0]
    ns=ys.shape[0]
    pt=np.mean(yt)
    ps=np.mean(ys)

    p=(nt*pt+ns*ps)/(nt+ns)
    pval=2*(1-norm.cdf(np.abs((pt-ps)/np.sqrt(p*(1-p)*(1/nt+1/ns)))))

    return pval

def exp_plots(theta, series, xlab, ylab, grid):
    names=['$\mathcal{P}_{Y}$','$\mathcal{P}_{X}$',
           '$\mathcal{P}_{X|Y}$','$\mathcal{P}_{Y|X}$',
           '$\mathcal{P}_{X,Y}$']
    colors=['#2F58EB', '#773BEB', '#12B8EB', '#EB9846', '#6D8AF1','#808080']
    
    plt.plot(theta, series[:,1], color=colors[4], marker="^", lw=2, label=names[0], alpha=.7, markersize=6)
    plt.plot(theta, series[:,2], color=colors[3], marker="v", lw=2, label=names[1], alpha=.7, markersize=6)
    plt.plot(theta, series[:,3], color=colors[2], marker="D", lw=2, label=names[2], alpha=.7, markersize=6)
    plt.plot(theta, series[:,4], color=colors[1], marker="s", lw=2, label=names[3], alpha=.7, markersize=6)
    plt.plot(theta, series[:,5], color=colors[0], marker="o", lw=2, label=names[4], alpha=.7, markersize=6)

    plt.legend(bbox_to_anchor=(.05, .975), loc='upper left', 
                   ncol = 3, prop={'size': 11}, borderaxespad=.0, frameon=False)
        
    plt.grid(alpha=.2, axis=grid)
    plt.axhline(y=0, color='k', linestyle='-', lw=1, alpha=.25)
    plt.ylabel(ylab, size=13)
    plt.xlabel(xlab, size=13)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    plt.locator_params(axis="y", nbins=4)