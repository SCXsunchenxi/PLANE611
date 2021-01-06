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
import shap

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

if __name__ == "__main__":

    plane='P123'
    X, Y = load_data(plane)
    X=pt.tensor(X).float()

    model=pt.load("save_MLP_model/mlp_model_plane" + str(plane) + ".pt")

    # shap.initjs()
    explainer = shap.DeepExplainer(model,X)

    shap_values = explainer.shap_values(X)

