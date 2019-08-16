from numpy import *
import pickle, gzip
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import scipy.linalg as linalg
import scipy
import itertools
from scipy import signal as sig
import random
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
import math
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error as mse
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
import mdp
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15 


perf={}
for it in range(1):
    def rKN(x, fx, hs):
        xk = []

        k1=fx(x)*hs

        xk=x + k1*0.5
        k2=fx(xk)*hs

        xk = x + k2*0.5
        k3=fx(xk)*hs

        xk = x + k3
        k4=fx(xk)*hs

        x = x + (k1 + 2*(k2 + k3) + k4)/6
        return x

    param=1.2 #0.1*(it+1)
    cd=1.0
    def lorenz2(inp, s=10*param, r=28*param, b=2.667*param):
        x,y,z=inp[0],inp[1],inp[2]
        x_dot = (s*(y - x))
        y_dot = (r*x - y - x*z)
        z_dot = (x*y - b*z)
        return np.array([x_dot, y_dot, z_dot])

    '''def lorenz2(inp, a=0.2*param, b=0.2*param, c=5.7*param):
        x,y,z=inp[0],inp[1],inp[2]
        x_dot = -y - z
        y_dot = x+a*y
        z_dot = b+z*(x-c)
        return np.array([x_dot, y_dot, z_dot])'''

    def lorenz(inp, s=10, r=28, b=2.667):
        x,y,z=inp[0],inp[1],inp[2]
        x_dot = cd*(s*(y - x))
        y_dot = cd*(r*x - y - x*z)
        z_dot = cd*(x*y - b*z)
        return np.array([x_dot, y_dot, z_dot])


    trainLen = 100000
    testLen = 100000
    initLen = 100

    dt=0.01
    xo=np.arange(0,dt*(trainLen+testLen+10*initLen),dt)
    inp1=[]
    inp2=[]
    a=[2,3,4]
    b=[5,4,2]
    start1=300 #input signal transients
    for i in range(5*len(xo)):
        if i%5==0:
            inp1.append(a[0])      #use only the z signal for training
            inp2.append(b[0])
        a=rKN(a,lorenz,dt)
        b=rKN(b,lorenz2,dt)
    max1=max(inp1)
    max2=max(inp2)
    inp1=np.array(inp1)[start1:]
    inp2=np.array(inp2)[start1:]
    inp1=(inp1-np.mean(inp1))/np.std(inp1[500:1500])
    inp2=(inp2-np.mean(inp2))/np.std(inp2[500:1500])
    
    #inp1=sin(2*pi*x)
    #inp2=sin(4*pi*x)

    '''figure()
    plot(inp1[:1000])
    plot(inp2[:1000])
    plot(inp3[:1000])
    plot(inp4[:1000])
    show()'''


    from numpy import *
    from matplotlib.pyplot import *
    import scipy.linalg


    # generate the ESN reservoir
    inSize = 1
    outSize=1
    resSize = 2000
    a = 0.3 # leakage parameter
    den=1
    b=0.2 #bias

    data = inp1
    data_g=[]
    #al=[0.1,0.125,0.15,0.2,0.25,0.4,0.5,0.6,0.75,0.8,0.85,0.875,0.9]
    

    Win = (random.rand(resSize,1+inSize)-0.5)*den*2
    W = scipy.sparse.rand(resSize, resSize, density= 0.01, format='coo', random_state=100).A#random.rand(resSize,resSize)-0.5 
    # normalizing and setting spectral radius:
    print ('Computing spectral radius...'),
    rhoW = max(abs(linalg.eig(W)[0]))
    print ('done.')
    W *= 0.9 / rhoW
    random.seed(42)

    X = zeros((1+inSize+resSize,(trainLen-initLen)))
    Yt = zeros((1,(trainLen-initLen)))


    x = zeros((resSize,1))
    #sin_val = (np.sin(np.arange(np.pi/2,3*np.pi/2,(np.pi)/trainLen))+1)/2
    disc_a=np.linspace(0,1,11)
    tim_space=[int(trainLen/len(disc_a))+1]*len(disc_a)#[trainLen/5,trainLen/8,trainLen/10, trainLen/20,  trainLen/20,trainLen/20,trainLen/10,trainLen/8,trainLen/5] #[int(trainLen/11)+1]*11
    tim_space=[round(z) for z in tim_space]
    timtim=[]
    for j in range(len(tim_space)):
        timtim.extend([disc_a[j]]*int(tim_space[j]))
                
    sin_val=np.array(timtim)
    for t in range(trainLen):
        alph=sin_val[t]
        data_in=sqrt(alph)*inp1+np.sqrt(1-alph)*inp2
        u=data_in[t, np.newaxis]
        x = (1-a)*x + a*tanh( dot( Win, np.insert(u,0,1)[:,np.newaxis]) + dot( W, x ) +b)
        #print(x[0]), 
        if t >= initLen:
            X[:,(t-initLen)] = vstack((1,u,x))[:,0]
            Yt[:,(t-initLen)] = alph

    # train the output
    reg = 1e-6  # regularization coefficient
    X_T = X.T
    Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
        reg*eye(1+inSize+resSize) ) )
    print('done')

    orig=np.dot(Wout,X).T
    pred_tr=sin_val

    train_alp=[]
    tim_space[0]-=initLen
    s=0
    for l in range(len(disc_a)):
        train_alp.append(np.mean(orig[s:s+tim_space[l]]))
        s+=tim_space[l]
        
        

    '''figure()
    plot(orig, label='prediction')
    plot(pred_tr, label='actual_trainingalpha')
    legend()
    show()'''

    #total_err=np.linalg.norm(np.array(orig)-np.array(pred_tr))
    #print (total_err)



    # run the trained ESN in a generative mode. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    act=[]
    pred=[]     
    
    x = zeros((resSize,1))
    sin_val = (np.sin(np.arange(np.pi/2,3*np.pi/2,(np.pi)/testLen))+1)/2
    tim_space=[int(testLen/len(disc_a))+1]*len(disc_a)#[testLen/5,testLen/8,testLen/10, testLen/20,  testLen/20,testLen/20,testLen/10,testLen/8,testLen/5] #
    tim_space=[round(z) for z in tim_space]
    timtim=[]
    for j in range(len(tim_space)):
        timtim.extend([disc_a[j]]*int(tim_space[j]))
    sin_val=np.array(timtim)
    
    for t in range(testLen):
        alp=sin_val[t]
        datax=sqrt(alp)*inp1+np.sqrt(1-alp)*inp2
        if t==0:
            u = datax[trainLen,np.newaxis].T
        x = (1-a)*x + a*tanh( dot( Win, np.insert(u,0,1)[:,np.newaxis] ) + dot( W, x ) +b )
        y = dot( Wout, vstack((1,u,x)) ) [:,0]
        if t>initLen:
            act.append(alp)
            pred.append(y[0])   #predict alphas only
        u = datax[trainLen+t+1,np.newaxis].T
        
    '''figure()
    plot(pred, label='prediction')
    plot(act, label='actual_testingalpha')
    legend()
    show()'''

    test_alp=[]
    s=0
    tim_space[0]-=initLen
    for l in range(len(disc_a)):
        test_alp.append(np.mean(pred[s:s+tim_space[l]]))
        s+=tim_space[l]

    print (train_alp)
    print (test_alp)

    figure()
    scatter(disc_a,train_alp, label='train')
    scatter(disc_a,test_alp, label='test')
    legend()
    xlabel('actual_alpha')
    ylabel('predicted_alpha')
    show()

    coef=np.polyfit(train_alp, disc_a,deg=3)   #fitting the train alphas to the actual alphas with a third order polynomial
    p = np.poly1d(coef)
    test_alp_fit=p(test_alp)
    train_alp_fit=p(train_alp)
    #test_alp_fit[0]=0
    actual_alp_fit=p(disc_a)

    figure()
    subplot(2, 1, 1)
    scatter(disc_a,test_alp, label='test $\\alpha$', marker="v")
    #scatter(disc_a,actual_alp_fit, label='3deg polynomial fitted between 0 and 1')
    scatter(disc_a,train_alp, label='train $\\alpha$',marker="^")
    plot(actual_alp_fit, disc_a,label='fitting function', color='green')
    plot(disc_a,disc_a, '--', color='k')
    legend()
    xlabel('Actual $\\alpha$',fontsize=16)
    ylabel('Estimated $\\alpha$',fontsize=16)
    
    subplot(2, 1, 2)
    scatter(disc_a,test_alp_fit, label='test $\\alpha$ corrected',marker="v")
    #scatter(disc_a,actual_alp_fit, label='3deg polynomial fitted between 0 and 1')
    scatter(disc_a,train_alp_fit, label='train $\\alpha$ corrected',marker="^")
    plot(disc_a,disc_a, '--',color='k')
    legend()
    xlabel('Actual $\\alpha$', fontsize=16)
    ylabel('Estimated $\\alpha$ (fit)',fontsize=16)
    
    savefig('20pcparam_generalize_discrete_alpha_.pdf', bbox_inches='tight')

    print (np.linalg.norm(test_alp_fit-disc_a))

    with open('20pcparam_generalize_discrete_alpha.pickle', 'wb') as handle:
        pickle.dump([disc_a,test_alp_fit, train_alp_fit,p], handle, protocol=pickle.HIGHEST_PROTOCOL)

    file = open('20pcparam_generalize_discrete_alpha.pickle','rb')
    read_file=pickle.load(file)
    disc_a,test_alp_fit, train_alp_fit,p=read_file
    train_alp=[]
    for i in range(len(train_alp_fit)):
        train_alp.append((p-train_alp_fit[i]).roots[1])

    test_alp=[]
    for i in range(len(test_alp_fit)):
        test_alp.append((p-test_alp_fit[i]).roots[1])'''
    
    
    


