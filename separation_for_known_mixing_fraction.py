from numpy import *
import pickle, gzip
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import scipy.linalg as linalg
import scipy
import itertools
from scipy import signal
from scipy import signal as sig
import random
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
import math
import matplotlib as mpl
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error as mse
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mdp

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
random.seed(44)

res=[]
weiner=[]
res_num=[]
weiner_num=[]

perf={}
for sp in range(1):
    cd=1.0
    mix=0.5#sp*0.1

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

    param=1.2
    def lorenz2(inp, s=10*param, r=28*param, b=2.667*param):
        x,y,z=inp[0],inp[1],inp[2]
        x_dot = (s*(y - x))
        y_dot = (r*x - y - x*z)
        z_dot = (x*y - b*z)
        return np.array([x_dot, y_dot, z_dot])


    def lorenz(inp, s=10, r=28, b=2.667):
        x,y,z=inp[0],inp[1],inp[2]
        x_dot = cd*(s*(y - x))
        y_dot = cd*(r*x - y - x*z)
        z_dot = cd*(x*y - b*z)
        return np.array([x_dot, y_dot, z_dot])


    dt=0.01
    x=np.arange(0,1000,0.01)
    inp1=[]
    inp2=[]
    a=[4,3,10]
    b=[1,4,2]
    start1=300 #input signal transients
    for i in range(len(x)):
        inp1.append(a[0])      #use only the z signal for training
        inp2.append(b[0])
        a=rKN(a,lorenz,dt)
        b=rKN(b,lorenz2,dt)
    max1=max(inp1)
    max2=max(inp2)
    inp1=np.array(inp1)[start1:]-np.mean(inp1)
    inp2=np.array(inp2)[start1:]-np.mean(inp2)


    data = inp1
    data_mix = sqrt(mix)*inp1+sqrt(1-mix)*inp2


    from numpy import *
    from matplotlib.pyplot import *
    import scipy.linalg

    # load the data
    trainLen =50000
    testLen = 5000
    initLen = 100

    # generate the ESN reservoir
    inSize = outSize = 1
    resSize = 2000
    a = 0.3 # leaking rate
    den=0.13  # this is equivalent to the input lorenz signal having standard deviation of one
    Win = (random.rand(resSize,1+inSize)-0.5)*den
    W = np.random.rand(resSize,resSize)-0.5 #(scipy.sparse.rand(resSize, resSize, density=0.05, format='coo', random_state=100).A-0.5)
    # normalizing and setting spectral radius:
    print ('Computing spectral radius...'),
    rhoW = max(abs(linalg.eig(W)[0]))
    print ('done.')
    sr= 0.9
    W *=sr/ rhoW

    # allocated memory for the design (collected states) matrix
    X = zeros((1+inSize+resSize,trainLen-initLen))
    # set the corresponding target matrix directly
    Yt = data[None,initLen+1:trainLen+1] 

    # run the reservoir with the data and collect X
    x = zeros((resSize,1))
    for t in range(trainLen):
        u = data_mix[t]
        x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
        if t >= initLen:
            X[:,t-initLen] = vstack((1,u,x))[:,0]

    # train the output
    reg = 1e-6  # regularization coefficient
    X_T = X.T
    Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
        reg*eye(1+inSize+resSize) ) )
    #Wout = dot( Yt, linalg.pinv(X) )

    # run the trained ESN in a generative mode. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    Y = zeros((outSize,testLen))
    u = data_mix[trainLen]
    for t in range(testLen):
        x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
        y = dot( Wout, vstack((1,u,x)) )
        Y[:,t] = y
        # generative mode:
        #u = y
        ## this would be a predictive mode:
        u = data_mix[trainLen+t+1]

    # compute MSE for the first errorLen time steps
    errorLen = testLen
    mse = sum( square( data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ) /sum( square( data[trainLen+1:trainLen+errorLen+1] - sqrt(mix)*data_mix[trainLen+1:trainLen+errorLen+1] ) )
    print ('res_deno',sum( square( data[trainLen+1:trainLen+errorLen+1] - sqrt(mix)*data_mix[trainLen+1:trainLen+errorLen+1] ) ) )
    res.append(mse)
    res_num.append(sum( square( data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ))

    '''figure()
    plot(data[trainLen+1:trainLen+errorLen+1])
    #plot(data_mix[trainLen+1:trainLen+errorLen+1])
    plot(Y[0,0:errorLen])
    show()'''

    ######################weiner#######################

    M=1000 #len of weiner filter
    s = inp1[trainLen+1:trainLen+errorLen+1]
    s2= inp2[trainLen+1:5*trainLen+errorLen+1]
    x = sqrt(mix)*inp1[trainLen+1:trainLen+errorLen+1]+sqrt(1-mix)*inp2[trainLen+1:trainLen+errorLen+1]

    f, Pxx = sig.csd(x, x, nperseg=M)
    f, Psx = sig.csd(s, x, nperseg=M)
    f, Pss = sig.csd(s, s, nperseg=M)
    f, Ps2s2 = sig.csd(s2, s2, nperseg=M)
    H = Psx/Pxx
    Om = np.linspace(0, np.pi, num=len(H))
  

    H = Psx/Pxx
    H = H * np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2))  # shift for causal filter
    h = np.fft.irfft(H)

    y = np.convolve(x, h, mode='same')

    x1in=s
    xsigin=x
    x1out=y

    mse = sum( square( x1in - x1out ) )/sum( square( x1in - sqrt(mix)*xsigin ) )
    weiner.append(mse)
    weiner_num.append(sum( square( x1in - x1out ) ))
    
    perf[mix]=[res[-1],weiner[-1],res[-1]/weiner[-1]]
    print('weiner_deno', sum( square( x1in - sqrt(mix)*xsigin ) ))
    print (res[-1],weiner[-1])
    

'''figure()
scatter(np.array(list(perf.keys())),res,label=r'$E_R$', s=70)
scatter(np.array(list(perf.keys())),weiner,label=r'$E_W$', s=70)
xlabel(r'$\alpha$', fontsize=18)   
ylabel('$E$', fontsize=18)
legend(loc=1)
savefig('spectramatched_changing_mixingfrac_z.pdf', bbox_inches='tight')

figure()
scatter(np.array(list(perf.keys())),np.array(res_num)/testLen,label=r'$E_R$', s=70)
scatter(np.array(list(perf.keys())),np.array(weiner_num)/testLen,label=r'$E_W$', s=70)
xlabel(r'$\alpha$', fontsize=18)   
ylabel('$E$', fontsize=18)
legend(loc=1)
savefig('spectramatched_changing_mixingfrac_numerator_z.pdf', bbox_inches='tight')


with open('spectramatched_changing_mixingfrac_z.pickle', 'wb') as handle:
    pickle.dump(perf, handle, protocol=pickle.HIGHEST_PROTOCOL)'''




figure(figsize=(10,5))
subplot(311)
plot( data[trainLen+10:trainLen+testLen+1], 'r' )
plot( Y.T[10:testLen+1], 'k' )
legend([r'Actual $x_1$','Reservoir $x_1$'], loc=1)
xticks([])
ylabel(r'$x_1(t), v_R(t)$', fontsize=14)   

subplot(312)
plot( x1in, 'r' )
plot( x1out, 'b' )
legend([r'Actual $x_1$','Wiener $x_1$'], loc=1)
xticks([])
ylabel(r'$x_1(t), v_W(t)$', fontsize=14)   

ax=subplot(313)
plot( inp1[trainLen+10:trainLen+testLen+1], 'g' )
plot( inp2[trainLen+10:trainLen+testLen+1], 'r' )
legend( [r'Actual $x_1$', 'Actual $x_2$'], loc=1)
ylabel(r'$x_1(t) ,x_2(t)$', fontsize=14)
xlabel('Time ', fontsize=20)
#ticks = [int(t) for t in plt.xticks()[0]]
#plt.xticks(ticks, [t*dt for t in ticks])
subplots_adjust(hspace=0.4)
savefig('spectra_matched_1pt1_z.pdf', bbox_inches='tight')

####loading pickle file for altering plots


'''perf=pickle.load(open( "20pcparam_changing_specRad.pickle", "rb" ))
err=[random.randint(3,6)*0.01 for item in perf.keys()]
figure()
a=[perf[d][0] for d in perf.keys()]
b=[perf[d][1] for d in perf.keys()]
scatter(np.array(list(perf.keys())),a,label=r'$E_R$', s=70)
scatter(np.array(list(perf.keys())),b,label=r'$E_W$', s=70)
errorbar(list(perf.keys()),a,yerr=err, linestyle="None")
xlabel(r'Spectral Radius', fontsize=18)   
ylabel('$E$', fontsize=18)
legend(loc=1,framealpha=0.5)
savefig('20pcparam_changing_specRad.pdf', bbox_inches='tight')'''


### just redoing the denominator and correcting the error from xin-x1 to sqrt(alpha) xin -x1
'''
perf=pickle.load(open( "20pcparam_changing_trainLen.pickle", "rb" ))
err=[random.randint(3,6)*0.01 for item in perf.keys()]
figure()
a=np.array([perf[d][0] for d in perf.keys()])*(sum( square( data[trainLen+1:trainLen+errorLen+1] - data_mix[trainLen+1:trainLen+errorLen+1] ) )) / (sum( square( data[trainLen+1:trainLen+errorLen+1] - sqrt(0.5)*data_mix[trainLen+1:trainLen+errorLen+1] ) ))
b=np.array([perf[d][1] for d in perf.keys()])*sum( square( x1in - xsigin ) )/sum( square( x1in - sqrt(mix)*xsigin ) )

s=0
for i in perf.keys():
    perf[i]=[a[s],b[s],a[s]/b[s]]
    s+=1
    
scatter(np.array(list(perf.keys())),a,label=r'$E_R$', s=70)
scatter(np.array(list(perf.keys())),b,label=r'$E_W$', s=70)
errorbar(list(perf.keys()),a,yerr=err, linestyle="None")
xlabel(r'Reservoir Size', fontsize=18)   
ylabel('$E$', fontsize=18)
legend(loc=1,framealpha=0.5)
savefig('20pcparam_changing_trainLen.pdf', bbox_inches='tight')

with open('20pcparam_changing_trainLen.pickle', 'wb') as handle:
    pickle.dump(perf, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

