import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb

dfall = pd.read_csv('./MLLResults.csv')
dfall['epochs']=dfall['step']/100


##--------- MNIST32
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

plt.show()

##--------- cifar32
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

plt.show()

