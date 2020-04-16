#######################################################################

############         Code by Sanjukta Krishnagopal         ############

### If you use all or part of this code please cite the paper below ###

#######################################################################



from numpy import *
import pickle, gzip
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import random
import matplotlib as mpl
from scipy.sparse import coo_matrix
import numpy as np

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
random.seed(42)


    
#fourth order Runge Kutta
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


#defining the two lorenz systems
cd=0.9
param=1.1
def lorenz2(inp, s=10*param, r=28*param, b=2.667*param):
    x,y,z=inp[0],inp[1],inp[2]
    x_dot = cd*(s*(y - x))
    y_dot = cd*(r*x - y - x*z)
    z_dot = cd*(x*y - b*z)
    return np.array([x_dot, y_dot, z_dot])


def lorenz(inp, s=10, r=28, b=2.667):
    x,y,z=inp[0],inp[1],inp[2]
    x_dot = (s*(y - x))
    y_dot = (r*x - y - x*z)
    z_dot = (x*y - b*z)
    return np.array([x_dot, y_dot, z_dot])


#initial condiitons
a=[4,3,10]
b=[1,4,2]

inp1=[]
inp2=[]

dt=0.1
total_t=2000
#removing input signal transients
start1=300 
for i in range(int(total_t*dt)):
    #using z signal of Lorenz system
    inp1.append(a[0]) 
    inp2.append(b[0])
    a=rKN(a,lorenz,dt)
    b=rKN(b,lorenz2,dt)
max1=max(inp1)
max2=max(inp2)
inp1=np.array(inp1)[start1:]
inp2=np.array(inp2)[start1:]

sig1 = inp1
#mixing fraction in which the two signals are mixed
mix=0.5       
mixed_s = mix*inp1+sqrt(1-mix**2)*inp2

trainLen = 50000
testLen = 5000
initLen = 100


# generate the ESN reservoir
inSize = outSize = 1
resSize = 500
a = 0.3  #leakage rate
den=0.1
Win = (random.rand(resSize,1+inSize)-0.5)*den
W = (scipy.sparse.rand(resSize, resSize, density=0.05, format='coo', random_state=100).A-0.5)*2.0

print ('Computing spectral radius...'),
rhoW = max(abs(linalg.eig(W)[0]))
sr= 0.8  #spectral radius
W *=sr/ rhoW

# allocated memory for the states matrix
X = zeros((1+inSize+resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = sig1[None,initLen+1:trainLen+1] 

# run the reservoir with the data and collect X
x = zeros((resSize,1))
for t in range(trainLen):
    u = mixed_s[t,np.newaxis].T
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    if t >= initLen:
        X[:,t-initLen] = vstack((1,u,x))[:,0]

# train the output using Ridge regression
reg = 1e-8  #regularization coefficient
X_T = X.T
Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(1+inSize+resSize) ) )


# run the trained ESN in a generative mode. reservoir state x is initialized with last timestep of training data.
Y = zeros((outSize,testLen))
u = mixed_s[trainLen,np.newaxis].T
for t in range(testLen):
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    y = dot( Wout, vstack((1,u,x)) )
    Y[:,t] = y
    #generative mode:
    #u = y
    #predictive mode:
    u = mixed_s[trainLen+t+1,np.newaxis].T 

# compute MSE for the first errorLen time steps
mse_r = sum( square( data[trainLen+1:trainLen+testLen+1] - Y[0,0:testLen] ) ) /sum( square( data[trainLen+1:trainLen+testLen+1] - data1[trainLen+1:trainLen+testLen+1] ) )
print ('Reservoir test signal separation performance - MSE:',mse_r)


#####################Signal Separation using the Wiener filter for comparison#################

M=1000 #length of weiner filter
s = inp1[trainLen+1:trainLen+errorLen+1]
s2= inp2[trainLen+1:5*trainLen+errorLen+1]
x = mix*inp1[trainLen+1:trainLen+errorLen+1]+sqrt(1-mix**2)*inp2[trainLen+1:trainLen+errorLen+1]

#Estimate power spectral density using Welch’s method.
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

mse_w = sum( square( x1in - x1out ) ) /sum( square( x1in - xsigin ) )
print ('Wiener test signal separation performance - MSE:',mse_w)



# plotting
figure()
subplot(311)
plot( sig1[trainLen+10:trainLen+testLen+1], 'g' )
plot( Y.T[10:], 'b' )
legend(['Actual x1','Predicted signal y'], loc=1)
xticks([])
ylabel('x1(t), y(t)', fontsize=16)   

subplot(312)
plot( inp1[trainLen+10:trainLen+testLen+1], 'g' )
plot( inp2[trainLen+10:trainLen+testLen+1], 'r' )
legend( ['Actual x1', 'Actual x2'], loc=1)
ylabel('x1(t) ,x2(t) ', fontsize=16)
subplots_adjust(hspace=0.4)

subplot(313)
plot( X[2:22,0:200].T )
ylabel('Some X(n)', fontsize=16)
ylim(1,-1)
xlabel('Time', fontsize=18)   
show()
