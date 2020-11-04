import sys
import os
import pandas as pd

pd.set_option('display.max_columns', None)
import pickle
import numpy as np
import math
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import random
from LSTM import LSTM
from sklearn import metrics


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj


def load_data(plane, strain, dir):
    print('Plane ' + plane + ', Strain ' + strain)
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')
    # M = pd.read_csv(dir+plane+'_mean.csv', encoding='utf-8')
    # S = pd.read_csv(dir+plane+'_sd.csv', encoding='utf-8')

    print('*** data is loaded')

    return data


def data_extract(data, strain, test_number):
    # length of a series
    sequence_length = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # length of a test of a plane
    lengths = data.groupby('number')['0'].count().tolist()
    # batch size
    batch_size = 64
    # 30 feature name
    feature_name = data.columns.values.tolist()[1:31]
    # label name
    label_name = strain

    batch_data = []
    batch_label = []
    # random series length
    index_sequence_length = sequence_length[random.randint(0, len(sequence_length) - 1)]
    # which test
    if test_number == None:
        number = random.randint(0, len(lengths) - 1)
    else:
        number = test_number

    for i in range(batch_size):
        index_row = random.randint(0, lengths[number] - index_sequence_length - 1)
        d = data[data['number'] == number].reset_index(drop=True).loc[index_row:index_row + index_sequence_length - 1,
            feature_name].values.tolist()
        label = data[data['number'] == number].reset_index(drop=True).loc[
            index_row + index_sequence_length - 1, label_name]
        batch_data.append(d)
        batch_label.append([label])

    return batch_data, batch_label


def training(plane, strain, test_number, dir, fold, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key,
             model_path,
             learning_rate=[1e-5, 2e-2], lr_decay=2000, earlystopping=10):
    # load data
    data = load_data(plane=plane, strain=strain, dir=dir)
    if test_number == None:
        number_train_batches = 20000
    else:
        number_train_batches = 2000
    input_dim = 30
    output_dim = 1

    # model built
    lstm = LSTM(input_dim, output_dim, hidden_dim, fc_dim, key)
    loss, y_pre, y_label = lstm.get_cost_acc()
    lr = learning_rate[0] + tf.train.exponential_decay(learning_rate[1],
                                                       lstm.step,
                                                       lr_decay,
                                                       1 / np.e)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    best_valid_loss = 1e10

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    EP=0
    # train
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(number_train_batches):
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys = data_extract(data, strain, test_number)
                step = epoch * number_train_batches + i
                sess.run(optimizer,
                         feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys, lstm.keep_prob: train_dropout_prob,
                                    lstm.step: step})
                print('Training epoch ' + str(epoch) + ' batch ' + str(i) + ' done')

            # valid

            batch_xs, batch_ys = data_extract(data, strain, test_number)
            loss, y_pre, y_true = sess.run(lstm.get_cost_acc(), feed_dict={lstm.input: batch_xs,
                                                                           lstm.labels: batch_ys,
                                                                           lstm.keep_prob: train_dropout_prob})

            print("validation MSE = {:.3f}".format(loss))
            MAE = metrics.mean_absolute_error(y_pre, y_true)
            print("validation MAE = {:.3f}".format(MAE))
            print('epoch ' + str(epoch) + ' done........................')
            if (loss <= best_valid_loss):
                EP = 0
                best_valid_loss=loss
                print("[*] Best loss so far! ")
                saver.save(sess, model_path + 'model' + str(fold) + '/')
                print("[*] Model saved at", model_path + 'model' + str(fold) + '/', flush=True)
            else:
                EP=EP+1
                if EP>earlystopping:
                    print("Early stopping! Fold " + str(fold) + " training is over!")
                    saver.save(sess, model_path + 'model' + str(fold) + '/')
                    print("[******] Model saved at", model_path + 'model' + str(fold) + '/', flush=True)

        print("Fold " + str(fold) + " training is over!")
        saver.save(sess, model_path + 'model' + str(fold) + '/')
        print("[******] Model saved at", model_path + 'model' + str(fold) + '/', flush=True)


def testing(plane, strain, test_number, dir, hidden_dim, fc_dim, key, model_path,fold):
    data = load_data(plane=plane, strain=strain, dir=dir)

    input_dim = 30
    output_dim = 1

    test_dropout_prob = 1.0
    lstm_load = LSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path + 'model'+str(fold)+'/')

        batch_xs, batch_ys = data_extract(data, strain, test_number)
        loss, y_pre, y_true = sess.run(lstm_load.get_cost_acc(), feed_dict={lstm_load.input: batch_xs,
                                                                            lstm_load.labels: batch_ys, \
                                                                            lstm_load.keep_prob: test_dropout_prob})

        MAE = metrics.mean_absolute_error(y_pre, y_true)
        print("Test Loss = {:.3f}".format(loss))
        print("validation MAE = {:.3f}".format(MAE))


def main(training_mode, plane, strain, test_number, data_path, fold, learning_rate, lr_decay, training_epochs,
         dropout_prob, hidden_dim, fc_dim,
         model_path,earlystopping):
    """
    :param training_mode:  1 train，0 test，
    :param plane:  P123-P127，
    :param strain:  30-35，
    :param test_number:  which test，
    :param data_path:
    :param fold: 5-fold
    :param learning_rate:
    :param training_epochs:
    :param dropout_prob: dropout，1 keep all
    :param hidden_dim:
    :param fc_dim:
    :param model_path: save/load model path
    """
    training_mode = int(training_mode)
    path = str(data_path)

    # train
    if training_mode == 1:
        learning_rate = learning_rate
        lr_decay = lr_decay
        training_epochs = int(training_epochs)
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        training(plane, strain, test_number, path, fold, training_epochs, dropout_prob, hidden_dim, fc_dim,
                 training_mode, model_path,
                 learning_rate, lr_decay,earlystopping)

    # test
    elif training_mode == 0:
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        testing(plane, strain, test_number, path, hidden_dim, fc_dim, training_mode, model_path,fold)


if __name__ == "__main__":
    main(training_mode=1, plane='P123', strain='30', test_number=0, data_path='../../data_per_plane/', fold=1,
         learning_rate=[1e-5, 2e-2], lr_decay=2000,
         training_epochs=15, dropout_prob=0.25, hidden_dim=64, fc_dim=32, model_path='save_LSTM/',earlystopping=10)
