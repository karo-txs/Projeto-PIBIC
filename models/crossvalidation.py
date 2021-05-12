import numpy as np
import pandas as pd
import time
import ast
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
import tensorflow
import models.basictools as bt
from models.costsensitive_weights import Weigths, WeigthsV2


def cross_val_single(dataset, resampling, export_results=False):
    results_final = get_structure_results()
    inicio = time.time()
    for i in range(30):
        print(f'-- {i+1} --')
        dataset.define_dataset_single(resampling)

        hyper = pd.read_csv('../results/hyperparameters.csv')
        params_adamax = hyper.loc[hyper['combinacao'] == 'adamax+'+str(resampling)]['params']
        params_adamax = ast.literal_eval(params_adamax[get_index('adamax', resampling)])

        params_rmsprop = hyper.loc[hyper['combinacao'] == 'rmsprop+'+str(resampling)]['params']
        params_rmsprop = ast.literal_eval(params_rmsprop[get_index('rmsprop', resampling)])

        params_sgdm = hyper.loc[hyper['combinacao'] == 'sgdm+'+str(resampling)]['params']
        params_sgdm = ast.literal_eval(params_sgdm[get_index('sgdm', resampling)])

        x_train, y_train, x_test, y_test = dataset.get_data(resampling)

        params_adamax['vocab_size'] = x_train.shape[1]
        params_rmsprop['vocab_size'] = x_train.shape[1]
        params_sgdm['vocab_size'] = x_train.shape[1]

        params = {'adamax': params_adamax, 'rmsprop': params_rmsprop, 'sgdm': params_sgdm}

        weights = Weigths().get_class_weight()

        results = run_cross_validation(x_train, y_train, x_test, y_test, params, weights)
        if export_results:
            bt.update_results_per_cross(results, (i+1), resampling)

        define_results(results_final, results)

    fim = time.time()
    duration = bt.get_time(fim - inicio)
    results_final = {resampling: define_average(results_final)}
    bt.print_table_single(results_final, resampling, export_results)
    print('Duration per execution: ' + duration)


def run_cross_validation(x_train,
                         y_train,
                         x_test,
                         y_test,
                         params,
                         weights):

    results = get_structure_results()
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(x_train):
        K.clear_session()

        X_train, X_valid = x_train[train_index], x_train[test_index]
        Y_train, Y_valid = y_train[train_index], y_train[test_index]

        # ---------------------- #
        #         ADAMAX         #
        # ---------------------- #
        optimizer_name = 'adamax'

        model_adamax = get_model(params['adamax'], optimizer_name)
        model_adamax_cs = get_model(params['adamax'], optimizer_name)

        callbacks_list = []

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
        callbacks_list.append(es)

        y_test_bool = np.argmax(y_test, axis=1)

        # Blind-sensitive

        model_adamax.fit(X_train, Y_train,
                         epochs=50,
                         verbose=0,
                         validation_data=(X_valid, Y_valid),
                         callbacks=callbacks_list)

        y_pred = model_adamax.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, optimizer_name, {'acuracia': ac,
                                                      'acuracia_balanceada': ba,
                                                      'precisao': precisao,
                                                      'recall': recall,
                                                      'f1': f1,
                                                      'gmean': geo,
                                                      'auc': auc}, True)

        # Cost-sensitive

        model_adamax_cs.fit(X_train, Y_train,
                            epochs=50,
                            verbose=0,
                            validation_data=(X_valid, Y_valid),
                            callbacks=callbacks_list,
                            class_weight=weights)

        y_pred = model_adamax_cs.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, str(optimizer_name)+'-'+str('cs'), {'acuracia': ac,
                                                                         'acuracia_balanceada': ba,
                                                                         'precisao': precisao,
                                                                         'recall': recall,
                                                                         'f1': f1,
                                                                         'gmean': geo,
                                                                         'auc': auc}, True)
        # ---------------------- #
        #         RMSPROP        #
        # ---------------------- #
        optimizer_name = 'rmsprop'

        model_rmsprop = get_model(params['rmsprop'], optimizer_name)
        model_rmsprop_cs = get_model(params['rmsprop'], optimizer_name)

        callbacks_list = []

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
        callbacks_list.append(es)

        y_test_bool = np.argmax(y_test, axis=1)

        # Blind-sensitive

        model_rmsprop.fit(X_train, Y_train,
                          epochs=50,
                          verbose=0,
                          validation_data=(X_valid, Y_valid),
                          callbacks=callbacks_list)

        y_pred = model_rmsprop.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, optimizer_name, {'acuracia': ac,
                                                      'acuracia_balanceada': ba,
                                                      'precisao': precisao,
                                                      'recall': recall,
                                                      'f1': f1,
                                                      'gmean': geo,
                                                      'auc': auc}, True)

        # Cost-sensitive

        model_rmsprop_cs.fit(X_train, Y_train,
                             epochs=50,
                             verbose=0,
                             validation_data=(X_valid, Y_valid),
                             callbacks=callbacks_list,
                             class_weight=weights)

        y_pred = model_rmsprop_cs.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, str(optimizer_name)+'-'+str('cs'), {'acuracia': ac,
                                                                         'acuracia_balanceada': ba,
                                                                         'precisao': precisao,
                                                                         'recall': recall,
                                                                         'f1': f1,
                                                                         'gmean': geo,
                                                                         'auc': auc}, True)

        # ---------------------- #
        #          SGDM          #
        # ---------------------- #
        optimizer_name = 'sgdm'

        model_sgdm = get_model(params[optimizer_name], optimizer_name)
        model_sgdm_cs = get_model(params[optimizer_name], optimizer_name)

        callbacks_list = []

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
        callbacks_list.append(es)

        y_test_bool = np.argmax(y_test, axis=1)

        # Blind-sensitive

        model_sgdm.fit(X_train, Y_train,
                       epochs=50,
                       verbose=0,
                       validation_data=(X_valid, Y_valid),
                       callbacks=callbacks_list)

        y_pred = model_sgdm.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, optimizer_name, {'acuracia': ac,
                                                      'acuracia_balanceada': ba,
                                                      'precisao': precisao,
                                                      'recall': recall,
                                                      'f1': f1,
                                                      'gmean': geo,
                                                      'auc': auc}, True)

        # Cost-sensitive

        model_sgdm_cs.fit(X_train, Y_train,
                          epochs=50,
                          verbose=0,
                          validation_data=(X_valid, Y_valid),
                          callbacks=callbacks_list,
                          class_weight=weights)

        y_pred = model_sgdm_cs.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred_bool, y_test_bool)
        define_unit_results(results, str(optimizer_name)+'-'+str('cs'), {'acuracia': ac,
                                                                         'acuracia_balanceada': ba,
                                                                         'precisao': precisao,
                                                                         'recall': recall,
                                                                         'f1': f1,
                                                                         'gmean': geo,
                                                                         'auc': auc}, True)
    define_average(results)

    return results


def define_results(results_final, results):
    define_unit_results(results_final, 'adamax', results)
    define_unit_results(results_final, 'adamax-cs', results)

    define_unit_results(results_final, 'rmsprop', results)
    define_unit_results(results_final, 'rmsprop-cs', results)

    define_unit_results(results_final, 'sgdm', results)
    define_unit_results(results_final, 'sgdm-cs', results)


def define_average(results):
    _len = len(results['adamax']['acuracia'])

    results['adamax']['acuracia'] = sum(results['adamax']['acuracia']) / _len
    results['adamax']['acuracia_balanceada'] = sum(results['adamax']['acuracia_balanceada']) / _len
    results['adamax']['precisao'] = sum(results['adamax']['precisao']) / _len
    results['adamax']['recall'] = sum(results['adamax']['recall']) / _len
    results['adamax']['f1'] = sum(results['adamax']['f1']) / _len
    results['adamax']['gmean'] = sum(results['adamax']['gmean']) / _len
    results['adamax']['auc'] = sum(results['adamax']['auc']) / _len

    results['adamax-cs']['acuracia'] = sum(results['adamax-cs']['acuracia']) / _len
    results['adamax-cs']['acuracia_balanceada'] = sum(results['adamax-cs']['acuracia_balanceada']) / _len
    results['adamax-cs']['precisao'] = sum(results['adamax-cs']['precisao']) / _len
    results['adamax-cs']['recall'] = sum(results['adamax-cs']['recall']) / _len
    results['adamax-cs']['f1'] = sum(results['adamax-cs']['f1']) / _len
    results['adamax-cs']['gmean'] = sum(results['adamax-cs']['gmean']) / _len
    results['adamax-cs']['auc'] = sum(results['adamax-cs']['auc']) / _len

    results['rmsprop']['acuracia'] = sum(results['rmsprop']['acuracia']) / _len
    results['rmsprop']['acuracia_balanceada'] = sum(results['rmsprop']['acuracia_balanceada']) / _len
    results['rmsprop']['precisao'] = sum(results['rmsprop']['precisao']) / _len
    results['rmsprop']['recall'] = sum(results['rmsprop']['recall']) / _len
    results['rmsprop']['f1'] = sum(results['rmsprop']['f1']) / _len
    results['rmsprop']['gmean'] = sum(results['rmsprop']['gmean']) / _len
    results['rmsprop']['auc'] = sum(results['rmsprop']['auc']) / _len

    results['rmsprop-cs']['acuracia'] = sum(results['rmsprop-cs']['acuracia']) / _len
    results['rmsprop-cs']['acuracia_balanceada'] = sum(results['rmsprop-cs']['acuracia_balanceada']) / _len
    results['rmsprop-cs']['precisao'] = sum(results['rmsprop-cs']['precisao']) / _len
    results['rmsprop-cs']['recall'] = sum(results['rmsprop-cs']['recall']) / _len
    results['rmsprop-cs']['f1'] = sum(results['rmsprop-cs']['f1']) / _len
    results['rmsprop-cs']['gmean'] = sum(results['rmsprop-cs']['gmean']) / _len
    results['rmsprop-cs']['auc'] = sum(results['rmsprop-cs']['auc']) / _len

    results['sgdm']['acuracia'] = sum(results['sgdm']['acuracia']) / _len
    results['sgdm']['acuracia_balanceada'] = sum(results['sgdm']['acuracia_balanceada']) / _len
    results['sgdm']['precisao'] = sum(results['sgdm']['precisao']) / _len
    results['sgdm']['recall'] = sum(results['sgdm']['recall']) / _len
    results['sgdm']['f1'] = sum(results['sgdm']['f1']) / _len
    results['sgdm']['gmean'] = sum(results['sgdm']['gmean']) / _len
    results['sgdm']['auc'] = sum(results['sgdm']['auc']) / _len

    results['sgdm-cs']['acuracia'] = sum(results['sgdm-cs']['acuracia']) / _len
    results['sgdm-cs']['acuracia_balanceada'] = sum(results['sgdm-cs']['acuracia_balanceada']) / _len
    results['sgdm-cs']['precisao'] = sum(results['sgdm-cs']['precisao']) / _len
    results['sgdm-cs']['recall'] = sum(results['sgdm-cs']['recall']) / _len
    results['sgdm-cs']['f1'] = sum(results['sgdm-cs']['f1']) / _len
    results['sgdm-cs']['gmean'] = sum(results['sgdm-cs']['gmean']) / _len
    results['sgdm-cs']['auc'] = sum(results['sgdm-cs']['auc']) / _len

    return results


def get_structure_results():
    return {'adamax': {'acuracia': [],
                       'acuracia_balanceada': [],
                       'precisao': [],
                       'recall': [],
                       'f1': [],
                       'gmean': [],
                       'auc': []},
            'adamax-cs': {'acuracia': [],
                          'acuracia_balanceada': [],
                          'precisao': [],
                          'recall': [],
                          'f1': [],
                          'gmean': [],
                          'auc': []},
            'rmsprop': {'acuracia': [],
                        'acuracia_balanceada': [],
                        'precisao': [],
                        'recall': [],
                        'f1': [],
                        'gmean': [],
                        'auc': []},
            'rmsprop-cs': {'acuracia': [],
                           'acuracia_balanceada': [],
                           'precisao': [],
                           'recall': [],
                           'f1': [],
                           'gmean': [],
                           'auc': []},
            'sgdm': {'acuracia': [],
                     'acuracia_balanceada': [],
                     'precisao': [],
                     'recall': [],
                     'f1': [],
                     'gmean': [],
                     'auc': []},
            'sgdm-cs': {'acuracia': [],
                        'acuracia_balanceada': [],
                        'precisao': [],
                        'recall': [],
                        'f1': [],
                        'gmean': [],
                        'auc': []}
            }


def define_unit_results(results, optimizer_name, dict_, dict_simple=False):
    if not dict_simple:
        results[optimizer_name]['acuracia'].append(dict_[optimizer_name]['acuracia'])
        results[optimizer_name]['acuracia_balanceada'].append(dict_[optimizer_name]['acuracia_balanceada']),
        results[optimizer_name]['precisao'].append(dict_[optimizer_name]['precisao'])
        results[optimizer_name]['recall'].append(dict_[optimizer_name]['recall'])
        results[optimizer_name]['f1'].append(dict_[optimizer_name]['f1'])
        results[optimizer_name]['gmean'].append(dict_[optimizer_name]['gmean'])
        results[optimizer_name]['auc'].append(dict_[optimizer_name]['auc'])
    else:
        results[optimizer_name]['acuracia'].append(dict_['acuracia'])
        results[optimizer_name]['acuracia_balanceada'].append(dict_['acuracia_balanceada']),
        results[optimizer_name]['precisao'].append(dict_['precisao'])
        results[optimizer_name]['recall'].append(dict_['recall'])
        results[optimizer_name]['f1'].append(dict_['f1'])
        results[optimizer_name]['gmean'].append(dict_['gmean'])
        results[optimizer_name]['auc'].append(dict_['auc'])


def get_optimizer(optimizer_name, learn_rate):
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


def get_model(params, optimizer_name):
    model = tensorflow.keras.models.Sequential()
    model.add(Dense(params['hidden_size'], input_dim=params['vocab_size'], activation=params['act']))
    model.add(Dropout(0.5))

    model.add(Dense(11, activation='softmax'))

    optimizer = get_optimizer(optimizer_name, float('{:.5f}'.format(params['lr'])))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def get_index(optimizer_name, resampling):

    def index(init):
        if optimizer_name == 'adamax':
            return init
        elif optimizer_name == 'rmsprop':
            return init+1
        elif optimizer_name == 'sgdm':
            return init+2

    if resampling == 'origin':
        return index(0)
    elif resampling == 'tomek':
        return index(3)
    elif resampling == 'smote':
        return index(6)
    elif resampling == 'bdsmote':
        return index(9)
    elif resampling == 'smoteenn':
        return index(12)
    elif resampling == 'smotetomek':
        return index(15)

