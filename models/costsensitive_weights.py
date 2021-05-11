from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time
import ast
import csv
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
import tensorflow
from bayes_opt import BayesianOptimization  # url = https://github.com/fmfn/BayesianOptimization
import models.basictools as bt


class Weigths:
    def get_class_weight(self, data):
        y = data['Class']
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        # balanced -> n_samples / (n_classes * np.bincount(y))
        # The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001
        vect = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.ravel())

        class_weight = {}
        index = 0
        for i in vect:
            class_weight[index] = i
            index += 1

        return class_weight


class WeigthsV2:
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
        self.iter = 0

    def run_(self, dataset, resampling, iter):
        dataset.define_dataset_single(resampling)

        x_train, y_train, x_test, y_test = dataset.get_data(resampling)

        def fitness_funcc(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
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
                model.add(Dense(100, input_dim=x_train.shape[1], activation='tanh'))
                model.add(Dropout(0.5))

                model.add(Dense(11, activation='softmax'))

                callbacks_list = []

                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
                callbacks_list.append(es)

                optimizer = self.get_optimizer('adamax', 0.001)

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy']
                )
                model.fit(X_train, Y_train,
                          epochs=50,
                          verbose=0,
                          validation_data=(X_valid, Y_valid),
                          callbacks=callbacks_list,
                          class_weight={0: c0, 1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 7: c7, 8: c8, 9: c9, 10: c10})

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
        pbounds = {'c0': (0, 1), 'c1': (0, 1), 'c2': (0, 1), 'c3': (0, 1), 'c4': (0, 1), 'c5': (0, 1), 'c6': (0, 1),
                   'c7': (0, 1), 'c8': (0, 1), 'c9': (0, 1), 'c10': (0, 1)}

        optimizer = BayesianOptimization(
            f=fitness_funcc,
            pbounds=pbounds,
            random_state=1,
        )
        inicio = time.time()
        self.iter = iter
        optimizer.maximize(init_points=2, n_iter=self.iter)
        fim = time.time()
        self.duration = bt.get_time(fim - inicio)
        self.max = optimizer.max
        self.max['params']['act'] = 'tanh'

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

    def update_weigths_list(self, combinacao):
        path = '../results/weigths.csv'
        lines = {'combinacao': combinacao,
                 'params': self.max['params'],
                 'f1': self.max['target'],
                 'iter': self.iter,
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

    def get_class_weight(self):
        data = pd.read_csv('../results/weigths.csv')
        params = data.loc[data['combinacao'] == 'adamax+origin']['params']
        params = ast.literal_eval(params[0])

        weights = {0: params['c0'], 1: params['c1'], 2: params['c2'], 3: params['c3'], 4: params['c4'], 5: params['c5'],
                   6: params['c6'], 7: params['c7'], 8: params['c8'], 9: params['c9'], 10: params['c10']}

        return weights
