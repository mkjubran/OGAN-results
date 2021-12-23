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
dataset='mnist32'
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
# DCGAN_Step=3000
#fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step2000Repeat/likelihood_G1_E2.pkl'
#with open(fname, 'rb') as f:
#   DCGAN_LL_G2_Step2000=pickle.load(f)

#fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step3000Repeat/likelihood_G1_E2.pkl'
#with open(fname, 'rb') as f:
#   DCGAN_LL_G2_Step3000=pickle.load(f)

fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step4000Repeat/likelihood_G1_E2.pkl'
with open(fname, 'rb') as f:
   DCGAN_LL_G2_Step4000=pickle.load(f)

fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020_ImSampling/Step5000Repeat/likelihood_G1_E2.pkl'
with open(fname, 'rb') as f:
   DCGAN_LL_G2_Step5000=pickle.load(f)

fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step4000Repeat/likelihood_G1_E2.pkl'
with open(fname, 'rb') as f:
   OGAN_LL_G2_Step4000=pickle.load(f)

fname='./MeasureLL_tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.005_W20.005_valbatches100_S2000_GS2019_GS2020_ImSampling/Step5000Repeat/likelihood_G1_E2.pkl'
with open(fname, 'rb') as f:
   OGAN_LL_G2_Step5000=pickle.load(f)

#DCGAN_LL_G2_Step2000_MovingAverage=[]
#DCGAN_LL_G2_Step3000_MovingAverage=[]
DCGAN_LL_G2_Step4000_MovingAverage=[]
DCGAN_LL_G2_Step5000_MovingAverage=[]
OGAN_LL_G2_Step4000_MovingAverage=[]
OGAN_LL_G2_Step5000_MovingAverage=[]
for cnt in range(1,len(DCGAN_LL_G2_Step5000)):
   #DCGAN_LL_G2_Step2000_MovingAverage.append(statistics.mean(DCGAN_LL_G2_Step2000[0:cnt]))
   #DCGAN_LL_G2_Step3000_MovingAverage.append(statistics.mean(DCGAN_LL_G2_Step3000[0:cnt]))
   DCGAN_LL_G2_Step4000_MovingAverage.append(statistics.mean(DCGAN_LL_G2_Step4000[0:cnt]))
   DCGAN_LL_G2_Step5000_MovingAverage.append(statistics.mean(DCGAN_LL_G2_Step5000[0:cnt]))
   OGAN_LL_G2_Step4000_MovingAverage.append(statistics.mean(OGAN_LL_G2_Step4000[0:cnt]))
   OGAN_LL_G2_Step5000_MovingAverage.append(statistics.mean(OGAN_LL_G2_Step5000[0:cnt]))


plt.figure()
plt.plot(range(1,len(DCGAN_LL_G2_Step4000)),DCGAN_LL_G2_Step4000_MovingAverage,'b:', linewidth=2, label='Step4000')
plt.plot(range(1,len(DCGAN_LL_G2_Step5000)),DCGAN_LL_G2_Step5000_MovingAverage,'r:', linewidth=2, label='Step5000')
plt.plot(range(1,len(OGAN_LL_G2_Step4000)),OGAN_LL_G2_Step4000_MovingAverage,'b-', linewidth=2, label='Step4000')
plt.plot(range(1,len(OGAN_LL_G2_Step5000)),OGAN_LL_G2_Step5000_MovingAverage,'r-', linewidth=2, label='Step5000')
plt.legend()

#print(statistics.mean(DCGAN_LL_G2_Step2000))
#print(statistics.mean(DCGAN_LL_G2_Step3000))
print(statistics.mean(DCGAN_LL_G2_Step4000))
print(statistics.mean(DCGAN_LL_G2_Step5000))
plt.show()

#pdb.set_trace()
