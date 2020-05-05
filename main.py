import sys, math, random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.tensor as ts
import torch.nn as nn
import torch.optim as optim
import mdn

print("Program: main\n")

ONE_OVER_SQRT_2PI = 1/np.sqrt(2*math.pi)

# input  - time: t
# output - position: x, y

n = int(5e4)
t_data = np.random.sample(n)[:, np.newaxis].astype(np.float32) # newaxis - extend the axis

t_data1 = t_data[:int(n/9)]
x_data1 = 5 + np.exp(t_data1) + np.multiply(t_data1**2/5, np.random.standard_normal(t_data1.shape))
y_data1 = 5 * t_data1         + np.multiply(t_data1**2/5, np.random.standard_normal(t_data1.shape))
x_mean1 = 5 + np.exp(np.linspace(0,1,num=21))
y_mean1 = 5 * np.linspace(0,1,num=21)
label1 = np.concatenate((x_data1, y_data1), axis=1)

t_data2 = t_data[int(n/9):]
x_data2 = 7 - np.exp(t_data2) + np.multiply(t_data2**2/5, np.random.standard_normal(t_data2.shape))
y_data2 = 3 * t_data2         + np.multiply(t_data2**2/5, np.random.standard_normal(t_data2.shape))
x_mean2 = 7 - np.exp(np.linspace(0,1,num=21))
y_mean2 = 3 * np.linspace(0,1,num=21)
label2 = np.concatenate((x_data2, y_data2), axis=1)

# t_data3 = t_data[int(2*n/3):]
# x_data3 = 6                   + np.multiply(t_data3**2/5, np.random.standard_normal(t_data3.shape))
# y_data3 = 4 * t_data3         + np.multiply(t_data3**2/5, np.random.standard_normal(t_data3.shape))
# label3 = np.concatenate((x_data3, y_data3), axis=1)

x_data = np.concatenate((x_data1,x_data2), axis=0)
y_data = np.concatenate((y_data1,y_data2), axis=0)
labels = np.concatenate((label1,label2), axis=0)
# print(labels)
# plt.scatter(x_data1, y_data1, marker='.')
# plt.scatter(x_data2, y_data2, marker='.')
# plt.scatter(x_data3, y_data3, marker='.')
# plt.show()
# sys.exit()

train_set = list(zip(ts(t_data), ts(labels)))
random.shuffle(train_set)
# train_set = train_set[0]
# print(train_set)
# for minibatch, label in train_set:
#     print(label)
#     sys.exit()
print("Data prepared.")


# initialize the model
model = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6, 32),
    nn.ReLU(),
    mdn.MDN(32, 2, 2) # dim_fea, dim_prob, num_gaus
)
optimizer = optim.Adam(model.parameters())

model0 = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6, 32),
    nn.ReLU(),
    # nn.Linear(32,2)
    mdn.MDN(32, 2, 1)
)
optimizer0 = optim.Adam(model0.parameters())

# train the model
cnt = 0
for minibatch, label in train_set:
    if len(np.shape(minibatch)) < len(np.shape(t_data)):
        minibatch = minibatch[np.newaxis, :] # add batch size
    if len(np.shape(label)) < len(np.shape(labels)):
        label = label[np.newaxis, :] # add batch size
    cnt += 1
    if cnt%2000 == 0:
        print("\r{}k/{}k".format(cnt/1000,n/1000), end='')
    model.zero_grad()
    alp, mu, sigma = model(minibatch)
    loss = mdn.loss_NLL_MDN(alp, mu, sigma, label)
    loss.backward()
    optimizer.step()

    model0.zero_grad()

    alp0, mu0, sigma0 = model0(minibatch)
    loss0 = mdn.loss_NLL_MDN(alp0, mu0, sigma0, label)

    # mse_loss = nn.MSELoss()
    # loss0 = mse_loss(model0(minibatch), labels.float())
    loss0.backward()
    optimizer0.step()
print('\n')

# sample new points from the trained model
t_test = ts(np.random.sample(int(5e3))[:, np.newaxis].astype(np.float32))
t_test0 = ts(np.random.sample(int(2e3))[:, np.newaxis].astype(np.float32))

alp, mu, sigma = model(t_test)
samples = mdn.sample(alp, mu, sigma).detach().numpy()

alp0, mu0, sigma0 = model0(t_test0)
samples0 = mdn.sample(alp0, mu0, sigma0).detach().numpy()

# samples0 = model0(t_test0).detach().numpy()


plt.scatter(x_data, y_data, marker='.')
plt.scatter(samples0[:,0], samples0[:,1], marker='.')
plt.scatter(samples[:,0], samples[:,1], marker='.')
plt.scatter(np.concatenate((x_mean1,x_mean2),axis=0), 
            np.concatenate((y_mean1,y_mean2),axis=0), marker='x')

plt.arrow(6,0, 0,5, head_width=0.05, head_length=0.1)
plt.xlabel("x")
plt.ylabel("y")
plt.text(6.1,5,'time')
plt.legend(['data','comp','pred','mean'])
plt.show()