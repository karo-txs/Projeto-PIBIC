import numpy as np
import time
import csv
from sklearn.model_selection import RandomizedSearchCV, KFold
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow
import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import pandas as pd
import models.basictools as bt


def model_adamax(hidden_size, act, lr, dropout, input_shape):
    model = tensorflow.keras.models.Sequential()
    model.add(Dense(hidden_size, input_dim=input_shape, activation=act))
    model.add(Dropout(dropout))

    model.add(Dense(11, activation='softmax'))

    callbacks_list = []

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
    callbacks_list.append(es)

    optimizer = keras.optimizers.Adamax(learning_rate=lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def model_rmsprop(hidden_size, act, lr, dropout, input_shape):
    model = tensorflow.keras.models.Sequential()
    model.add(Dense(hidden_size, input_dim=input_shape, activation=act))
    model.add(Dropout(dropout))

    model.add(Dense(11, activation='softmax'))

    callbacks_list = []

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
    callbacks_list.append(es)

    optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def model_sgdm(hidden_size, act, lr, dropout, input_shape):
    model = tensorflow.keras.models.Sequential()
    model.add(Dense(hidden_size, input_dim=input_shape, activation=act))
    model.add(Dropout(dropout))

    model.add(Dense(11, activation='softmax'))

    callbacks_list = []

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
    callbacks_list.append(es)

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


class Hyper:
    def __init__(self):
        self.best_params = {}
        self.best_score = {}
        self.duration = 0

    def run_(self, dataset, resampling, optimizer_name, iter):
        dataset.define_dataset_single(resampling)

        x_train, y_train, x_test, y_test = dataset.get_data(resampling)

        batch_size = [32, 64, 128, 256]
        lr = [1e-1, 1e-2, 3e-2, 1e-3]
        activation = ['relu', 'tanh', 'elu']
        hidden_size = [20, 50, 100, 150, 200]
        dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

        param_grid = dict(batch_size=batch_size, hidden_size=hidden_size, act=activation, lr=lr, dropout=dropout,
                          input_shape=[x_train.shape[1]])

        model = self.get_model(optimizer_name)

        random = RandomizedSearchCV(estimator=model, cv=KFold(10), param_distributions=param_grid,
                                    verbose=True, n_iter=iter, n_jobs=-1, scoring='f1_macro')
        inicio = time.time()
        random_result = random.fit(np.concatenate((x_train, x_test), axis=0), np.argmax(
            np.concatenate((y_train, y_test), axis=0), axis=1))
        fim = time.time()
        self.duration = bt.get_time(fim-inicio)
        self.best_params = random_result.best_params_
        self.best_score = random_result.best_score_

    def get_model(self, optimizer_name):
        if optimizer_name == 'adamax':
            return KerasClassifier(build_fn=model_adamax, epochs=50, verbose=False)
        elif optimizer_name == 'rmsprop':
            return KerasClassifier(build_fn=model_rmsprop, epochs=50, verbose=False)
        elif optimizer_name == 'sgdm':
            return KerasClassifier(build_fn=model_sgdm, epochs=50, verbose=False)

    def update_hyperparameter_list(self, combinacao):
        path = '../results/hyperparameters.csv'
        lines = {'combinacao': combinacao,
                 'params': self.best_params,
                 'f1': self.best_score,
                 'duration': self.duration}
        try:
            open(path, 'r')
            with open(path, 'a') as arq:
                writer = csv.writer(arq)
                writer.writerow(lines.values())
        except IOError:
            data = pd.DataFrame(columns=lines.keys())
            data = data.append(lines, ignore_index=True)
            data.to_csv(path, index=False)

