import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb
import pickle
import statistics

dfall = pd.read_csv('./MLLResults.csv')
dfall['epochs']=dfall['step']/100


##--------- MNIST32: final LL values at different epochs
dataset='mnist32repeat'
imageSize=32
nc=1
epoch=45
batch=100

df=dfall[dfall['dataset']==dataset]
fig, axs = plt.subplots(2,3)
fig.suptitle('MNIST ImageSize=32x32')
fig.tight_layout()

axs[0,0].plot(df.epochs,df.DCGAN_train_G1_G2, 'ob:', linewidth=2, label='DCGAN')
axs[0,0].plot(df.epochs,df.OGAN_train_G1_G2, 'sr-', linewidth=2, label='OGAN')
axs[0,0].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G1 and G2')
axs[0,0].legend()
axs[0,0].grid()

axs[0,1].plot(df.epochs,df.DCGAN_train_G1, 'ob:', linewidth=2, label='DCGAN')
axs[0,1].plot(df.epochs,df.OGAN_train_G1, 'sr-', linewidth=2, label='OGAN')
axs[0,1].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G1')
axs[0,1].legend()
axs[0,1].grid()

axs[0,2].plot(df.epochs,df.DCGAN_train_G2, 'ob:', linewidth=2, label='DCGAN')
axs[0,2].plot(df.epochs,df.OGAN_train_G2, 'sr-', linewidth=2, label='OGAN')
axs[0,2].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G2')
axs[0,2].legend()
axs[0,2].grid()

axs[1,0].plot(df.epochs,df.DCGAN_test_G1_G2, 'ob:', linewidth=2, label='DCGAN')
axs[1,0].plot(df.epochs,df.OGAN_test_G1_G2, 'sr-', linewidth=2, label='OGAN')
axs[1,0].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G1 and G2')
axs[1,0].legend()
axs[1,0].grid()

axs[1,1].plot(df.epochs,df.DCGAN_test_G1, 'ob:', linewidth=2, label='DCGAN')
axs[1,1].plot(df.epochs,df.OGAN_test_G1, 'sr-', linewidth=2, label='OGAN')
axs[1,1].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G1')
axs[1,1].legend()
axs[1,1].grid()

axs[1,2].plot(df.epochs,df.DCGAN_test_G2, 'ob:', linewidth=2, label='DCGAN')
axs[1,2].plot(df.epochs,df.OGAN_test_G2, 'sr-', linewidth=2, label='OGAN')
axs[1,2].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G2')
axs[1,2].legend()
axs[1,2].grid()

#plt.show()

##--------- cifar32: final LL values at different epochs
dataset='cifar32'
imageSize=32
nc=1
epoch=45
batch=100
df=dfall[dfall['dataset']==dataset]

fig, axs = plt.subplots(2,3)
fig.suptitle('CIFAR10 ImageSize=32x32')
fig.tight_layout()

axs[0,0].plot(df.epochs,df.DCGAN_train_G1_G2, 'ob:', linewidth=2, label='DCGAN')
axs[0,0].plot(df.epochs,df.OGAN_train_G1_G2, 'sr-', linewidth=2, label='OGAN')
axs[0,0].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G1 and G2')
axs[0,0].legend()
axs[0,0].grid()

axs[0,1].plot(df.epochs,df.DCGAN_train_G1, 'ob:', linewidth=2, label='DCGAN')
axs[0,1].plot(df.epochs,df.OGAN_train_G1, 'sr-', linewidth=2, label='OGAN')
axs[0,1].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G1')
axs[0,1].legend()
axs[0,1].grid()

axs[0,2].plot(df.epochs,df.DCGAN_train_G2, 'ob:', linewidth=2, label='DCGAN')
axs[0,2].plot(df.epochs,df.OGAN_train_G2, 'sr-', linewidth=2, label='OGAN')
axs[0,2].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Training: G2')
axs[0,2].legend()
axs[0,2].grid()

axs[1,0].plot(df.epochs,df.DCGAN_test_G1_G2, 'ob:', linewidth=2, label='DCGAN')
axs[1,0].plot(df.epochs,df.OGAN_test_G1_G2, 'sr-', linewidth=2, label='OGAN')
axs[1,0].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G1 and G2')
axs[1,0].legend()
axs[1,0].grid()

axs[1,1].plot(df.epochs,df.DCGAN_test_G1, 'ob:', linewidth=2, label='DCGAN')
axs[1,1].plot(df.epochs,df.OGAN_test_G1, 'sr-', linewidth=2, label='OGAN')
axs[1,1].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G1')
axs[1,1].legend()
axs[1,1].grid()

axs[1,2].plot(df.epochs,df.DCGAN_test_G2, 'ob:', linewidth=2, label='DCGAN')
axs[1,2].plot(df.epochs,df.OGAN_test_G2, 'sr-', linewidth=2, label='OGAN')
axs[1,2].set(xlabel='epochs', ylabel='Log-likelihood (nats)', title='Testing: G2')
axs[1,2].legend()
axs[1,2].grid()

#plt.show()

##--------- mnist32: moving average of LL values per batch
##-- DCGAN - G2
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G1_E2.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

DCGAN_LL_G2 = GAN_LL
DCGAN_LL_G2_MovingAverage = GAN_LL_MovingAverage

#-- DCGAN - G1
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G2_E1.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

DCGAN_LL_G1 = GAN_LL
DCGAN_LL_G1_MovingAverage = GAN_LL_MovingAverage

#-- DCGAN - test G2
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G1test_E2.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

DCGAN_LL_test_G2 = GAN_LL
DCGAN_LL_test_G2_MovingAverage = GAN_LL_MovingAverage

#-- DCGAN - test G1
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G2test_E1.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

DCGAN_LL_test_G1 = GAN_LL
DCGAN_LL_test_G1_MovingAverage = GAN_LL_MovingAverage

##-- OGAN - G2
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G1_E2.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

OGAN_LL_G2 = GAN_LL
OGAN_LL_G2_MovingAverage = GAN_LL_MovingAverage

#-- OGAN - G1
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G2_E1.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

OGAN_LL_G1 = GAN_LL
OGAN_LL_G1_MovingAverage = GAN_LL_MovingAverage

#-- OGAN - test G2
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G1test_E2.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

OGAN_LL_test_G2 = GAN_LL
OGAN_LL_test_G2_MovingAverage = GAN_LL_MovingAverage

#-- OGAN - test G1
Path='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step'
filename='likelihood_G2test_E1.pkl';
Steps=[2000,3000,4000,5000]

with open(Path+str(Steps[0])+'Repeat/'+filename, 'rb') as f:
   GAN_LL=np.array(pickle.load(f)).reshape(1,-1)
for Step in Steps[1:]:
   with open(Path+str(Step)+'Repeat/'+filename, 'rb') as f:
      GAN_LL=np.vstack((GAN_LL,np.array(pickle.load(f)).reshape(1,-1)))
GAN_LL_MovingAverage = np.cumsum(GAN_LL, axis=1)/np.arange(1,GAN_LL.shape[1]+1)

OGAN_LL_test_G1 = GAN_LL
OGAN_LL_test_G1_MovingAverage = GAN_LL_MovingAverage

## ---- G1 and G2
DCGAN_LL_G1_G2 = np.logaddexp(DCGAN_LL_G1,DCGAN_LL_G2)
DCGAN_LL_test_G1_G2 = np.logaddexp(DCGAN_LL_test_G1,DCGAN_LL_test_G2)
OGAN_LL_G1_G2 = np.logaddexp(OGAN_LL_G1,OGAN_LL_G2)
OGAN_LL_test_G1_G2 = np.logaddexp(OGAN_LL_test_G1,OGAN_LL_test_G2)


## --- plot DCGAN 
DCGAN_LL = DCGAN_LL_test_G2_MovingAverage
plt.figure()
ax = plt.gca()
color = next(ax._get_lines.prop_cycler)['color']
plt.plot(np.arange(10,DCGAN_LL.shape[1]),DCGAN_LL[0][10:], linestyle='--', linewidth=2, color=color, label='epoch20')
plt.plot(np.arange(1000,DCGAN_LL.shape[1],1000),DCGAN_LL[0][1000::1000], linestyle='', marker='o', color=color)

color = next(ax._get_lines.prop_cycler)['color']
plt.plot(np.arange(10,DCGAN_LL.shape[1]),DCGAN_LL[1][10:], linestyle='--', linewidth=2, color=color, label='eppch30')
plt.plot(np.arange(1000,DCGAN_LL.shape[1],1000),DCGAN_LL[1][1000::1000], linestyle='', marker='o', color=color)

color = next(ax._get_lines.prop_cycler)['color']
plt.plot(np.arange(10,DCGAN_LL.shape[1]),DCGAN_LL[2][10:], linestyle='--', linewidth=2, color=color, label='epoch40')
plt.plot(np.arange(1000,DCGAN_LL.shape[1],1000),DCGAN_LL[2][1000::1000], linestyle='', marker='o', color=color)

color = next(ax._get_lines.prop_cycler)['color']
plt.plot(np.arange(10,DCGAN_LL.shape[1]),DCGAN_LL[3][10:], linestyle='--', linewidth=2, color=color, label='epoch50')
plt.plot(np.arange(1000,DCGAN_LL.shape[1],1000),DCGAN_LL[3][1000::1000], linestyle='', marker='o', color=color, label='epoch50')

plt.legend()
plt.grid()
plt.title('DCGAN: Log-likelihood of G1 during Testing')
plt.xlabel('Samples')
plt.ylabel('Log-likelihood (nats)')

plt.show()

