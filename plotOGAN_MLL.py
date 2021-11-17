import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb

##--------- MNIST32
df = pd.read_csv('./MLLResults.csv')
df['epochs']=df['step']/100
imageSize=32
nc=1
epoch=45
batch=100

plt.plot(df.epochs,df.DCGAN_train, 'dr-', linewidth=2, label='DCGAN_train')
plt.plot(df.epochs,df.DCGAN_train, 'or:', linewidth=2, label='DCGAN_test')
plt.plot(df.epochs,df.OGAN_train, 'xb-', linewidth=2, label='OGAN_train')
plt.plot(df.epochs,df.OGAN_train, '*b:', linewidth=2, label='OGAN_test')
plt.xlabel('epochs')
plt.ylabel('LL')
plt.legend()
plt.grid()
plt.title('MNIST ImageSize=32')
plt.show()
