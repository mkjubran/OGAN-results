import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

##--------- MNIST32
df = pd.read_csv('./output-mnist32.csv')
df=df[1:]
imageSize=32
epoch=45
batch=100


VLL = [col for col in df if col.startswith('Validation LL/Sample')]
df_VLLb = df[VLL]
df_VLLb = df_VLLb.dropna()
df_VLLb = df_VLLb.astype('float')
df_VLLb = df_VLLb[0:epoch*batch]
ax = plt.gca()
df_VLLb['Validation LL/Sample: G2(testset)-->(E1,G1)'].plot(kind='line',ax=ax)
df_VLLb['Validation LL/Sample: G1(testset)-->(E2,G2)'].plot(kind='line', color='red', ax=ax)
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('MNIST ImageSize=32')
plt.show()

df_VLLb['min_Validation LL/Sample'] = df_VLLb[['Validation LL/Sample: G2(testset)-->(E1,G1)','Validation LL/Sample: G1(testset)-->(E2,G2)']].min(axis=1)

ax = plt.gca()
df_VLLb['Validation LL/Sample: G2(testset)-->(E1,G1)'].plot(kind='line',ax=ax)
df_VLLb['Validation LL/Sample: G1(testset)-->(E2,G2)'].plot(kind='line', color='red', ax=ax)
df_VLLb['min_Validation LL/Sample'].plot(kind='line', color='green', ax=ax)
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('MNIST ImageSize=32')
plt.show()

VLL = [col for col in df if col.startswith('Validation LL/epoch')]
df_VLLe = df[VLL]
df_VLLe = df_VLLe.dropna()
df_VLLe = df_VLLe.astype('float')


VLL = [col for col in df if col.startswith('Validation LL/Sample')]
df_VLLb = df[VLL]
df_VLLb = df_VLLb.dropna()
df_VLLb = df_VLLb.astype('float')
df_VLLb = df_VLLb[0:epoch*batch]

df_VLLb['min_Validation LL/Sample'] = df_VLLb[['Validation LL/Sample: G2(testset)-->(E1,G1)','Validation LL/Sample: G1(testset)-->(E2,G2)']].min(axis=1)

df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['min_Validation LL/Sample']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G1(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G2(testset)-->(E1,G1)']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G2(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G1(testset)-->(E2,G2)']/(math.log(2)*(imageSize**2))

ax = plt.gca()
df_VLLe['Validation LL/epoch G1(bits/dim)'].plot(kind='line', ax=ax, linestyle='--', linewidth=2, label='OGAN-G1')
df_VLLe['Validation LL/epoch G2(bits/dim)'].plot(kind='line', color='k', ax=ax, linestyle=':', linewidth=3, label='OGAN-G2')
df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)'].plot(kind='line', color='r', ax=ax, linewidth=2, label='Gated(OGAN-G1,OGAN-G2)')
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('MNIST ImageSize=32')
plt.show()



##--------- cifar32
df = pd.read_csv('./output-cifar32.csv')
df=df[1:]
imageSize=32
epoch=45
batch=100

VLL = [col for col in df if col.startswith('Validation LL/epoch')]
df_VLLe = df[VLL]
df_VLLe = df_VLLe.dropna()
df_VLLe = df_VLLe.astype('float')


VLL = [col for col in df if col.startswith('Validation LL/Sample')]
df_VLLb = df[VLL]
df_VLLb = df_VLLb.dropna()
df_VLLb = df_VLLb.astype('float')
df_VLLb = df_VLLb[0:epoch*batch]

df_VLLb['min_Validation LL/Sample'] = df_VLLb[['Validation LL/Sample: G2(testset)-->(E1,G1)','Validation LL/Sample: G1(testset)-->(E2,G2)']].min(axis=1)

df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['min_Validation LL/Sample']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G1(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G2(testset)-->(E1,G1)']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G2(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G1(testset)-->(E2,G2)']/(math.log(2)*(imageSize**2))

ax = plt.gca()
df_VLLe['Validation LL/epoch G1(bits/dim)'].plot(kind='line', ax=ax, linestyle='--', linewidth=2, label='OGAN-G1')
df_VLLe['Validation LL/epoch G2(bits/dim)'].plot(kind='line', color='k', ax=ax, linestyle=':', linewidth=3, label='OGAN-G2')
df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)'].plot(kind='line', color='r', ax=ax, linewidth=2, label='Gated(OGAN-G1,OGAN-G2)')
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('CIFAR10 ImageSize=32')
plt.show()


##--------- celeba32
df = pd.read_csv('./output-celeba32.csv')
df=df[1:]
imageSize=32
epoch=45
batch=100

VLL = [col for col in df if col.startswith('Validation LL/epoch')]
df_VLLe = df[VLL]
df_VLLe = df_VLLe.dropna()
df_VLLe = df_VLLe.astype('float')


VLL = [col for col in df if col.startswith('Validation LL/Sample')]
df_VLLb = df[VLL]
df_VLLb = df_VLLb.dropna()
df_VLLb = df_VLLb.astype('float')
df_VLLb = df_VLLb[0:epoch*batch]

df_VLLb['min_Validation LL/Sample'] = df_VLLb[['Validation LL/Sample: G2(testset)-->(E1,G1)','Validation LL/Sample: G1(testset)-->(E2,G2)']].min(axis=1)

df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['min_Validation LL/Sample']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G1(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G2(testset)-->(E1,G1)']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G2(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G1(testset)-->(E2,G2)']/(math.log(2)*(imageSize**2))

ax = plt.gca()
df_VLLe['Validation LL/epoch G1(bits/dim)'].plot(kind='line', ax=ax, linestyle='--', linewidth=2, label='OGAN-G1')
df_VLLe['Validation LL/epoch G2(bits/dim)'].plot(kind='line', color='k', ax=ax, linestyle=':', linewidth=3, label='OGAN-G2')
df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)'].plot(kind='line', color='r', ax=ax, linewidth=2, label='Gated(OGAN-G1,OGAN-G2)')
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('CELEBA ImageSize=32')
plt.show()


##--------- MNIST64
df = pd.read_csv('./output-mnist64.csv')
df=df[1:]
imageSize=64
epoch=45
batch=100

VLL = [col for col in df if col.startswith('Validation LL/epoch')]
df_VLLe = df[VLL]
df_VLLe = df_VLLe.dropna()
df_VLLe = df_VLLe.astype('float')


VLL = [col for col in df if col.startswith('Validation LL/Sample')]
df_VLLb = df[VLL]
df_VLLb = df_VLLb.dropna()
df_VLLb = df_VLLb.astype('float')
df_VLLb = df_VLLb[0:epoch*batch]

df_VLLb['min_Validation LL/Sample'] = df_VLLb[['Validation LL/Sample: G2(testset)-->(E1,G1)','Validation LL/Sample: G1(testset)-->(E2,G2)']].min(axis=1)

df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['min_Validation LL/Sample']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G1(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G2(testset)-->(E1,G1)']/(math.log(2)*(imageSize**2))
df_VLLe['Validation LL/epoch G2(bits/dim)']=df_VLLb.groupby(np.arange(len(df_VLLb))//100).mean()['Validation LL/Sample: G1(testset)-->(E2,G2)']/(math.log(2)*(imageSize**2))

ax = plt.gca()
df_VLLe['Validation LL/epoch G1(bits/dim)'].plot(kind='line', ax=ax, linestyle='--', linewidth=2, label='OGAN-G1')
df_VLLe['Validation LL/epoch G2(bits/dim)'].plot(kind='line', color='k', ax=ax, linestyle=':', linewidth=3, label='OGAN-G2')
df_VLLe['Validation LL/epoch: min(G1+G2) (bits/dim)'].plot(kind='line', color='r', ax=ax, linewidth=2, label='Gated(OGAN-G1,OGAN-G2)')
plt.xlabel('epochs')
plt.ylabel('LL (bits/dim)')
plt.legend()
plt.title('MNIST ImageSize=64')
plt.show()
