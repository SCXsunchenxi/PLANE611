import os
import pandas as pd
import sys
pd.set_option('display.max_columns', None)
import pickle
import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import random
from sklearn.manifold import TSNE


def load_data(plane):
    '''
     :param plane: choose plane P123-P27
    '''

    print('[***] Data from Plane ' + plane)
    dir = '../../data_per_plane/'
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')

    X = np.asarray(data.get(data.columns.values.tolist()[1:31]))
    print('[***] Data is loaded')

    return X

def random_data(size):
    '''
     :param size: number of random data
    '''

    print('[***] Data is random')
    M = pd.read_csv('../../data_per_plane/data_mean.csv', encoding='utf-8')
    S = pd.read_csv('../../data_per_plane/data_std.csv', encoding='utf-8')

    random_DF = pd.DataFrame()
    for i in range(size):
        random_list = []
        for i in range(36):
            random_list.append(S.iloc[0, i] * random.uniform(-1, 1) + M.iloc[0, i])
        random_df = pd.DataFrame(np.array(random_list).reshape(1, 36))
        random_DF = pd.concat([random_DF, random_df], ignore_index=True)

    X = np.asarray(random_DF.get(random_DF.columns.values.tolist()[:30]))
    print('[***]'+ str(size) +' random data is loaded')

    return X

def random_data_with_center(size,cluster_range,cluster_number):
    '''
     :param size: number of random data
    '''

    print('[***] Data is random')
    data = pd.read_csv('../../data_per_plane/P123_data.csv', encoding='utf-8')
    S = pd.read_csv('../../data_per_plane/data_std.csv', encoding='utf-8')

    for d in range(cluster_number):

        center = pd.DataFrame(np.array(data.iloc[random.randint(0, 1347320), 1:].tolist()).reshape(1, 36))

        random_DF = pd.DataFrame()
        for i in range(size):
            random_list = []
            for i in range(36):
                random_list.append(S.iloc[0, i] * random.uniform(-cluster_range, cluster_range) + center.iloc[0, i])
            random_df = pd.DataFrame(np.array(random_list).reshape(1, 36))
            random_DF = pd.concat([random_DF, random_df], ignore_index=True)
        if (d == 0):
            X = np.asarray(random_DF.get(random_DF.columns.values.tolist()[0:30]))
        else:
            X = np.concatenate((X, np.asarray(random_DF.get(random_DF.columns.values.tolist()[0:30]))), axis=0)

    print('[***]'+ str(cluster_number) +' clusters random data are loaded')

    tsne = TSNE(n_components=2)
    newData2 = tsne.fit_transform(X)
    plt.scatter(newData2[:, 0], newData2[:, 1], s=1)
    plt.savefig("random_with_center", dpi=400)

    return X

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

def get_pre(X,plane):
    # load data
    X=pt.tensor(X).float()


    # load model
    load_model=pt.load("save_MLP_model/mlp_model_plane" + str(plane) + ".pt")

    # pre
    Y_pre = load_model(X)

    return Y_pre




if __name__ == "__main__":

    size, cluster_range, cluster_number =100,0.3,30
    X = random_data_with_center(size, cluster_range, cluster_number)
    # pre data
    for i in range(cluster_number):
        Y_P123 = get_pre(X[i*size:size*(i+1)], 'P123')
        Y_P124 = get_pre(X[i*size:size*(i+1)], 'P124')
        Y_P125 = get_pre(X[i*size:size*(i+1)], 'P125')
        Y_P126 = get_pre(X[i*size:size*(i+1)], 'P126')
        Y_P127 = get_pre(X[i*size:size*(i+1)], 'P127')

        pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata_center'+str(i)+'.seqs', 'wb'), -1)
        pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata_center'+str(i)+'.seqs', 'wb'), -1)
        pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata_center'+str(i)+'.seqs', 'wb'), -1)
        pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata_center'+str(i)+'.seqs', 'wb'), -1)
        pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata_center'+str(i)+'.seqs', 'wb'), -1)

    # size = 100
    # X = random_data(size)
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata1.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata1.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata1.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata1.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata1.seqs', 'wb'), -1)
    #
    # size = 100
    # X = random_data(size)
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata2.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata2.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata2.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata2.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata2.seqs', 'wb'), -1)
    #
    #
    # size = 100
    # X = random_data(size)
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata3.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata3.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata3.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata3.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata3.seqs', 'wb'), -1)
    #
    #
    # size = 100
    # X = random_data(size)
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata4.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata4.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata4.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata4.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata4.seqs', 'wb'), -1)
    #
    #
    # size = 100
    # X = random_data(size)
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_randomdata5.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_randomdata5.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_randomdata5.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_randomdata5.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_randomdata5.seqs', 'wb'), -1)
    #
    #
    # X = load_data('P123')
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_P123data.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_P123data.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_P123data.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_P123data.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_P123data.seqs', 'wb'), -1)
    #
    #
    # X = load_data('P124')
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_P124data.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_P124data.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_P124data.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_P124data.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_P124data.seqs', 'wb'), -1)
    #
    #
    # X = load_data('P125')
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_P125data.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_P125data.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_P125data.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_P125data.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_P125data.seqs', 'wb'), -1)
    #
    #
    # X = load_data('P126')
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_P126data.seqs', 'wb'), -1)
    #
    #
    # X = load_data('P127')
    # # pre data
    # Y_P123 = get_pre(X, 'P123')
    # Y_P124 = get_pre(X, 'P124')
    # Y_P125 = get_pre(X, 'P125')
    # Y_P126 = get_pre(X, 'P126')
    # Y_P127 = get_pre(X, 'P127')
    # pickle.dump(Y_P123.tolist(), open('coefficient_calibration_data/planeP123_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P124.tolist(), open('coefficient_calibration_data/planeP124_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P125.tolist(), open('coefficient_calibration_data/planeP125_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P126.tolist(), open('coefficient_calibration_data/planeP126_P126data.seqs', 'wb'), -1)
    # pickle.dump(Y_P127.tolist(), open('coefficient_calibration_data/planeP127_P126data.seqs', 'wb'), -1)