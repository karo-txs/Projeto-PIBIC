import numpy as np
import pandas as pd
import time
import csv
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
import tensorflow
from bayes_opt import BayesianOptimization  # url = https://github.com/fmfn/BayesianOptimization
import models.basictools as bt


class Hyper:
    def __init__(self):
        self.best = {'acuracia': 0,
                     'acuracia_balanceada': 0,
                     'precisao': 0,
                     'recall': 0,
                     'f1': 0,
                     'gmean': 0,
                     'auc': 0}
        self.max = []
        self.duration = 0

    def run_(self, dataset, resampling, optimizer_name, iter):
        dataset.define_dataset_single(resampling)

        x_train, y_train, x_test, y_test = dataset.get_data(resampling)
        act = self.get_act(resampling)

        def fitness_funcc(hidden_size, lr):
            results = {'acuracia': [],
                       'acuracia_balanceada': [],
                       'precisao': [],
                       'recall': [],
                       'f1': [],
                       'gmean': [],
                       'auc': []}

            kf = KFold(n_splits=10, shuffle=True)
            for train_index, test_index in kf.split(x_train):
                K.clear_session()

                X_train, X_valid = x_train[train_index], x_train[test_index]
                Y_train, Y_valid = y_train[train_index], y_train[test_index]

                model = tensorflow.keras.models.Sequential()
                model.add(Dense(int(hidden_size), input_dim=x_train.shape[1], activation=act))
                model.add(Dropout(0.5))

                model.add(Dense(11, activation='softmax'))

                callbacks_list = []

                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
                callbacks_list.append(es)

                optimizer = self.get_optimizer(optimizer_name, float('{:.5f}'.format(lr)))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy']
                )
                model.fit(X_train, Y_train,
                          epochs=50,
                          verbose=0,
                          validation_data=(X_valid, Y_valid),
                          callbacks=callbacks_list)

                y_test_bool = np.argmax(y_test, axis=1)

                y_pred = model.predict(x_test)
                y_pred_bool = np.argmax(y_pred, axis=1)

                ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
                self.define_unit_results(results, {'acuracia': ac,
                                                   'acuracia_balanceada': ba,
                                                   'precisao': precisao,
                                                   'recall': recall,
                                                   'f1': f1,
                                                   'gmean': geo,
                                                   'auc': auc})
            self.return_mean(results)
            if self.best['f1'] < results['f1']:
                self.set_best(results)

            return results['f1']

        # Bounded region of parameter space
        pbounds = {'hidden_size': (14, 250), 'lr': (0.0001, 0.01)}

        optimizer = BayesianOptimization(
            f=fitness_funcc,
            pbounds=pbounds,
            random_state=1,
        )
        inicio = time.time()
        optimizer.maximize(init_points=2, n_iter=iter)
        fim = time.time()
        self.duration = bt.get_time(fim-inicio)

        self.max = optimizer.max
        self.max['params']['act'] = act

    def get_act(self, dataname):
        if dataname == 'origin' or dataname == 'tomek':
            return 'tanh'
        elif dataname == 'smote' or dataname == 'bdsmote':
            return 'elu'
        elif dataname == 'smoteenn' or dataname == 'smotetomek':
            return 'relu'

    def get_optimizer(self, optimizer_name, learn_rate):
        opt = None
        if optimizer_name == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == 'adamax':
            opt = keras.optimizers.Adamax(learning_rate=learn_rate)
        elif optimizer_name == 'adagrad':
            opt = keras.optimizers.Adagrad(learning_rate=learn_rate)
        elif optimizer_name == 'adadelta':
            opt = keras.optimizers.Adadelta(learning_rate=learn_rate)
        elif optimizer_name == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learn_rate)
        elif optimizer_name == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.0)
        elif optimizer_name == 'sgdm':
            opt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.9)
        return opt

    def define_unit_results(self, results, dict_):
        results['acuracia'].append(dict_['acuracia'])
        results['acuracia_balanceada'].append(dict_['acuracia_balanceada'])
        results['precisao'].append(dict_['precisao'])
        results['recall'].append(dict_['recall'])
        results['f1'].append(dict_['f1'])
        results['gmean'].append(dict_['gmean'])
        results['auc'].append(dict_['auc'])

    def return_mean(self, results):
        _len = len(results['acuracia'])

        results['acuracia'] = sum(results['acuracia']) / _len
        results['acuracia_balanceada'] = sum(results['acuracia_balanceada']) / _len
        results['precisao'] = sum(results['precisao']) / _len
        results['recall'] = sum(results['recall']) / _len
        results['f1'] = sum(results['f1']) / _len
        results['gmean'] = sum(results['gmean']) / _len
        results['auc'] = sum(results['auc']) / _len

    def set_best(self, results):
        self.best = results

    def update_hyperparameter_list(self, combinacao):
        path = '../results/hyperparameters.csv'
        lines = {'combinacao': combinacao,
                 'params': self.max['params'],
                 'f1': self.max['target'],
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

