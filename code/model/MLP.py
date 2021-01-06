import os
import pandas as pd
import sys
pd.set_option('display.max_columns', None)
import pickle
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import math
from sklearn.model_selection import train_test_split



# data with one strain
def load_data_strain(plane, strain):
    '''
     :param plane: choose plane P123-P27
     :param strain: choose strain 30-35
    '''

    print('[***] Plane ' + plane + ', Strain ' + strain)

    dir = '../../data_per_plane/'
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')
    # M = pd.read_csv(dir+plane+'_mean.csv', encoding='utf-8')
    # S = pd.read_csv(dir+plane+'_sd.csv', encoding='utf-8')

    X = np.asarray(data.get(data.columns.values.tolist()[1:31]))
    y = np.asarray(data.get(strain))

    print('[***] Data is loaded')

    return X, y

# data with six strains
def load_data(plane):
    '''
     :param plane: choose plane P123-P27
    '''

    print('[***] Plane ' + plane)

    dir = '../../data_per_plane/'
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')
    # M = pd.read_csv(dir+plane+'_mean.csv', encoding='utf-8')
    # S = pd.read_csv(dir+plane+'_sd.csv', encoding='utf-8')

    X = np.asarray(data.get(data.columns.values.tolist()[1:31]))
    Y = np.asarray(data.get(data.columns.values.tolist()[31:]))

    print('[***] Data is loaded')

    return X, Y

#
# class MLP(pt.nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = pt.nn.Linear(30, 128)
#         self.fc2 = pt.nn.Linear(128, 256)
#         self.fc3 = pt.nn.Linear(256, 128)
#         self.fc4 = pt.nn.Linear(128, 6)
#         self.dropout = pt.nn.Dropout(p=0.5)
#
#     def forward(self, input):
#         dout = pt.sigmoid(self.fc1(input))
#         dout = self.dropout(dout)
#         dout = pt.sigmoid(self.fc2(dout))
#         dout = self.dropout(dout)
#         dout = pt.sigmoid(self.fc3(dout))
#         dout = self.dropout(dout)
#         return self.fc4(dout)

class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(30, 256)
        self.fc2 = pt.nn.Linear(256, 64)
        self.fc3 = pt.nn.Linear(64, 6)
        self.dropout = pt.nn.Dropout(p=0.1)

    def forward(self, input):
        dout = pt.sigmoid(self.fc1(input))
        dout = self.dropout(dout)
        dout = pt.sigmoid(self.fc2(dout))
        dout = self.dropout(dout)
        return self.fc3(dout)

#  序列模型 SHAP解释用
# class MLP(pt.nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc=pt.nn.Sequential(
#             pt.nn.Linear(30, 256),
#             pt.nn.Sigmoid(),
#             pt.nn.Dropout(0.1),
#             pt.nn.Linear(256, 64),
#             pt.nn.Sigmoid(),
#             pt.nn.Dropout(0.1),
#             pt.nn.Linear(64, 6),
#         )
#
#     def forward(self, input):
#         dout = self.fc(input)
#         return dout

def train_model(plane, epoch,batch_size,early_stop,continue_train):
    # load data
    X, Y = load_data(plane)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_test = pt.tensor(X_test).float()
    Y_test = pt.tensor(Y_test).float()

    # load model
    if(continue_train==0):
        model = MLP()
        print(model)
        loss_list=[]
        best_acc = 100

    else:
        model=pt.load("save_MLP_model/mlp_model_plane" + str(plane) + ".pt")
        print(model)
        with open('save_MLP_model/plane'+str(plane)+'loss_list.seqs', 'rb') as f:
            previous_acc = pickle.load(f)
            best_acc=previous_acc[-1]
            loss_list=previous_acc

    optimizer = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossfunc = pt.nn.MSELoss()

    # train
    early_stop_count=0
    for i in range(epoch):
        if (early_stop_count==early_stop):
            print("[***] Early stop here")
            break
        for j in range(int(len(X_train) / batch_size) + 1):
            X_train_batch = X_train[j * batch_size:(j + 1) * batch_size]
            Y_train_batch = Y_train[j * batch_size:(j + 1) * batch_size]

            X_train_batch = pt.tensor(X_train_batch).float()
            Y_train_batch = pt.tensor(Y_train_batch).float()

            optimizer.zero_grad()
            Y_pre = model(X_train_batch)
            loss = lossfunc(Y_pre, Y_train_batch)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print("loss in batch " + str(j) + ": " + str(loss.item()))

        # val
        Y_pre = model(X_test)
        loss = lossfunc(Y_pre, Y_test)
        acc=loss.item()
        loss_list.append(acc)

        if(best_acc>acc):
            best_acc=acc
            early_stop_count=0
            # save model
            pt.save(model, "save_MLP_model/mlp_model_plane" + str(plane) + ".pt")
        else:
            early_stop_count=early_stop_count+1
        print("*** Epoch "+str(i)+" val MES: " + str(acc))

    pickle.dump(loss_list, open('save_MLP_model/plane'+str(plane)+'loss_list.seqs', 'wb'), -1)
    print("[***] Best MES is " + str(best_acc))

    # loss fig
    x_epoch=[i for i in range(len(loss_list))]
    plt.plot(x_epoch, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('MLP_train_loss'+str(plane)+'.jpg', dpi=300)

    return best_acc

def train_model_uncertainty(plane, epoch,batch_size,early_stop,continue_train,uncertainty_model_number):
    # load data
    X, Y = load_data(plane)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_test = pt.tensor(X_test).float()
    Y_test = pt.tensor(Y_test).float()

    # load model
    if (continue_train == 0):
        model = MLP()
        print(model)
        loss_list = []
        best_acc = 100
        uncertainty_list=[]

    else:
        model = pt.load("save_MLP_model/mlp_model_plane" + str(plane) + ".pt")
        print(model)
        with open('save_MLP_model/plane' + str(plane) + 'loss_list.seqs', 'rb') as f:
            previous_acc = pickle.load(f)
            best_acc = previous_acc[-1]
            loss_list = previous_acc
        with open('save_MLP_model/plane' + str(plane) + 'uncertainty_list.seqs', 'rb') as f:
            uncertainty_list = pickle.load(f)


    optimizer = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossfunc = pt.nn.MSELoss()

    # train
    early_stop_count=0
    for i in range(epoch):
        if (early_stop_count==early_stop):
            print("[***] Early stop here")
            break
        for j in range(int(len(X_train) / batch_size) + 1):
            X_train_batch = X_train[j * batch_size:(j + 1) * batch_size]
            Y_train_batch = Y_train[j * batch_size:(j + 1) * batch_size]

            X_train_batch = pt.tensor(X_train_batch).float()
            Y_train_batch = pt.tensor(Y_train_batch).float()

            optimizer.zero_grad()
            Y_pre = model(X_train_batch)
            loss = lossfunc(Y_pre, Y_train_batch)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print("loss in batch " + str(j) + ": " + str(loss.item()))

        # val with uncertainty
        model_acc = []
        for u in range(uncertainty_model_number):
            Y_pre = model(X_test)
            loss = lossfunc(Y_pre, Y_test)
            acc=loss.item()
            model_acc.append(acc)

        avg_mode_acc = np.mean(model_acc)
        loss_list.append(avg_mode_acc)

        # total uncertainty
        total_uncertainty = (-avg_mode_acc) * math.log(avg_mode_acc, 2)
        # expected data uncertainty
        expected_data_uncertainty=sum((-a) * math.log(a, 2) for a in model_acc)
        # model uncertainty
        model_uncertainty = total_uncertainty - expected_data_uncertainty
        uncertainty_list.append(model_uncertainty)
        print("*** Model uncertainty: " + str(model_uncertainty))

        if(best_acc>avg_mode_acc):
            best_acc=avg_mode_acc
            early_stop_count=0
            # save model
            pt.save(model, "save_MLP_model/mlp_model_plane" + str(plane) + ".pt")
        else:
            early_stop_count=early_stop_count+1
        print("*** Epoch "+str(i)+" val MES: " + str(avg_mode_acc))

    pickle.dump(loss_list, open('save_MLP_model/plane'+str(plane)+'loss_list.seqs', 'wb'), -1)
    pickle.dump(uncertainty_list, open('save_MLP_model/plane' + str(plane) + 'uncertainty_list.seqs', 'wb'), -1)
    print("[***] Best MES is " + str(best_acc))

    # loss fig
    x_epoch=[x for x in range(len(loss_list))]
    plt.plot(x_epoch, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('MLP_train_loss'+str(plane)+'.jpg', dpi=300)
    plt.clf()
    # uncertainty fig
    x_epoch=[x for x in range(len(uncertainty_list))]
    plt.plot(x_epoch, uncertainty_list)
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    plt.savefig('MLP_train_uncertainty'+str(plane)+'.jpg', dpi=300)
    plt.clf()
    return best_acc

def test(X,Y,plane):
    # load data
    X=pt.tensor(X).float()
    Y= pt.tensor(Y).float()

    # load model
    load_model=pt.load("save_MLP_model/mlp_model_plane" + str(plane) + ".pt")
    lossfunc = pt.nn.MSELoss()

    # test
    Y_pre = load_model(X)
    loss = lossfunc(Y_pre, Y)
    print("[***] test MES: " + str(loss.item()))

if __name__ == "__main__":
    # param
    plane=['P123','P124','P125','P126','P127']
    epoch=500
    batch_size=128
    early_stop=5
    continue_train=0
    uncertainty=1
    uncertainty_model_number=5

    PLANE_MSE=[]
    for p in plane:
        # train
        if(uncertainty==1):
            # train with uncertainty
            beast_acc=train_model_uncertainty(p, epoch, batch_size, early_stop, continue_train, uncertainty_model_number)
        else:
            beast_acc=train_model(p, epoch, batch_size, early_stop, continue_train)
        PLANE_MSE.append(beast_acc)

    print("[***] MES of 7 planes is: " + str(PLANE_MSE))

    # test
    # X, Y = load_data('P123')
    # test(X,Y,'P123')





