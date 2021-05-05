import numpy as np
import os
from sklearn.model_selection import KFold
import models.basictools as bt


def cross_val_complete(hyper, dataset):
    print('-- ORIGIN start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('origin')
    params = hyper.hiper_origin
    results_origin = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    print('-- TOMEK start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('dataTomek')
    params = hyper.hiper_tomek
    results_tomek = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    print('-- SMOTE start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('dataSmote')
    params = hyper.hiper_smote
    results_smote = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    print('-- BD-SMOTE start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('dataBoderlineSmote')
    params = hyper.hiper_bd_smote
    results_bd_smote = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    print('-- SMOTEENN start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('dataSmoteEnn')
    params = hyper.hiper_smote_enn
    results_smote_enn = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    print('-- SMOTETOMEK start --')
    vocab_size, x_train, y_train, x_test, y_test = dataset.get_data('dataSmoteTomek')
    params = hyper.hiper_smote_tomek
    results_smote_tomek = run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

    results = {'origin': results_origin, 'tomek': results_tomek, 'smote': results_smote, 'bd-smote': results_bd_smote,
               'smoteenn': results_smote_enn, 'smotetomek': results_smote_tomek}

    bt.print_table(results, True)


def run_cross_validation(x_train,
                         y_train,
                         x_test,
                         y_test,
                         params,
                         vocab_size,
                         return_dict_results=True):
    models_adamax = []
    models_adamax_cs = []
    models_rmsprop = []
    models_rmsprop_cs = []
    models_sgdm = []
    models_sgdm_cs = []

    inputs = x_train
    targets = y_train

    kfold = KFold(n_splits=10, shuffle=True)

    results = {'adamax': {'acuracia': [],
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

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, val in kfold.split(inputs, targets):
        adamax = params['adamax'].clone_params(vocab_size)
        adamax_cs = params['adamax'].clone_params(vocab_size)
        rmsprop = params['rmsprop'].clone_params(vocab_size)
        rmsprop_cs = params['rmsprop'].clone_params(vocab_size)
        sgdm = params['rmsprop'].clone_params(vocab_size)
        sgdm_cs = params['rmsprop'].clone_params(vocab_size)

        adamax.create_net()
        adamax_cs.create_net()
        rmsprop.create_net()
        rmsprop_cs.create_net()
        sgdm.create_net()
        sgdm_cs.create_net()

        weights = {0: 0.128, 1: 0.128, 2: 0.128, 3: 0.0464, 4: 0.128, 5: 0.0464, 6: 0.0464, 7: 0.128, 8: 0.128,
                   9: 0.0464, 10: 0.0464}

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        adamax_cs.fit(inputs[train], targets[train], inputs[val], targets[val], 5, weights)
        rmsprop_cs.fit(inputs[train], targets[train], inputs[val], targets[val], 5, weights)
        sgdm_cs.fit(inputs[train], targets[train], inputs[val], targets[val], 5, weights)
        adamax.fit(inputs[train], targets[train], inputs[val], targets[val], 5)
        rmsprop.fit(inputs[train], targets[train], inputs[val], targets[val], 5)
        sgdm.fit(inputs[train], targets[train], inputs[val], targets[val], 5)

        y_val_bool = np.argmax(targets[val], axis=1)
        # ----------- ADAMAX ----------- #

        y_pred = adamax.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['adamax']['acuracia'].append(ac)
        results['adamax']['acuracia_balanceada'].append(ba),
        results['adamax']['precisao'].append(precisao)
        results['adamax']['recall'].append(recall)
        results['adamax']['f1'].append(f1)
        results['adamax']['gmean'].append(geo)
        results['adamax']['auc'].append(auc)

        # ----------- ADAMAX+COST-SENSITIVE ----------- #

        y_pred = adamax_cs.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['adamax-cs']['acuracia'].append(ac)
        results['adamax-cs']['acuracia_balanceada'].append(ba),
        results['adamax-cs']['precisao'].append(precisao)
        results['adamax-cs']['recall'].append(recall)
        results['adamax-cs']['f1'].append(f1)
        results['adamax-cs']['gmean'].append(geo)
        results['adamax-cs']['auc'].append(auc)

        # ----------- RMSPROP ----------- #

        y_pred = rmsprop.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['rmsprop']['acuracia'].append(ac)
        results['rmsprop']['acuracia_balanceada'].append(ba),
        results['rmsprop']['precisao'].append(precisao)
        results['rmsprop']['recall'].append(recall)
        results['rmsprop']['f1'].append(f1)
        results['rmsprop']['gmean'].append(geo)
        results['rmsprop']['auc'].append(auc)

        # ----------- RMSPROP+COST-SENSITIVE ----------- #

        y_pred = rmsprop_cs.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['rmsprop-cs']['acuracia'].append(ac)
        results['rmsprop-cs']['acuracia_balanceada'].append(ba),
        results['rmsprop-cs']['precisao'].append(precisao)
        results['rmsprop-cs']['recall'].append(recall)
        results['rmsprop-cs']['f1'].append(f1)
        results['rmsprop-cs']['gmean'].append(geo)
        results['rmsprop-cs']['auc'].append(auc)

        # ----------- SGDM ----------- #

        y_pred = sgdm.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['sgdm']['acuracia'].append(ac)
        results['sgdm']['acuracia_balanceada'].append(ba),
        results['sgdm']['precisao'].append(precisao)
        results['sgdm']['recall'].append(recall)
        results['sgdm']['f1'].append(f1)
        results['sgdm']['gmean'].append(geo)
        results['sgdm']['auc'].append(auc)

        # ----------- SGDM+COST-SENSITIVE ----------- #

        y_pred = sgdm_cs.predict(inputs[val])

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_val_bool)

        results['sgdm-cs']['acuracia'].append(ac)
        results['sgdm-cs']['acuracia_balanceada'].append(ba),
        results['sgdm-cs']['precisao'].append(precisao)
        results['sgdm-cs']['recall'].append(recall)
        results['sgdm-cs']['f1'].append(f1)
        results['sgdm-cs']['gmean'].append(geo)
        results['sgdm-cs']['auc'].append(auc)

        fold_no = fold_no + 1

        models_adamax.append(adamax)
        models_adamax_cs.append(adamax_cs)
        models_rmsprop.append(rmsprop)
        models_rmsprop_cs.append(rmsprop_cs)
        models_sgdm.append(sgdm)
        models_sgdm_cs.append(sgdm_cs)

    # ----------Test Final-----------
    results = {'adamax': {'acuracia': [],
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

    for model_ada, model_ada_cs, model_rms, model_rms_cs, model_sgdm, model_sgdm_cs in \
            zip(models_adamax, models_adamax_cs, models_rmsprop, models_rmsprop_cs, models_sgdm, models_sgdm_cs):
        y_test_bool = np.argmax(y_test, axis=1)

        # ----------- ADAMAX ----------- #

        y_pred = model_ada.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['adamax']['acuracia'].append(ac)
        results['adamax']['acuracia_balanceada'].append(ba),
        results['adamax']['precisao'].append(precisao)
        results['adamax']['recall'].append(recall)
        results['adamax']['f1'].append(f1)
        results['adamax']['gmean'].append(geo)
        results['adamax']['auc'].append(auc)

        # ----------- ADAMAX+COST-SENSITIVE ----------- #

        y_pred = model_ada_cs.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['adamax-cs']['acuracia'].append(ac)
        results['adamax-cs']['acuracia_balanceada'].append(ba),
        results['adamax-cs']['precisao'].append(precisao)
        results['adamax-cs']['recall'].append(recall)
        results['adamax-cs']['f1'].append(f1)
        results['adamax-cs']['gmean'].append(geo)
        results['adamax-cs']['auc'].append(auc)

        # ----------- RMSPROP ----------- #

        y_pred = model_rms.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['rmsprop']['acuracia'].append(ac)
        results['rmsprop']['acuracia_balanceada'].append(ba),
        results['rmsprop']['precisao'].append(precisao)
        results['rmsprop']['recall'].append(recall)
        results['rmsprop']['f1'].append(f1)
        results['rmsprop']['gmean'].append(geo)
        results['rmsprop']['auc'].append(auc)

        # ----------- RMSPROP+COST-SENSITIVE ----------- #

        y_pred = model_rms_cs.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['rmsprop-cs']['acuracia'].append(ac)
        results['rmsprop-cs']['acuracia_balanceada'].append(ba),
        results['rmsprop-cs']['precisao'].append(precisao)
        results['rmsprop-cs']['recall'].append(recall)
        results['rmsprop-cs']['f1'].append(f1)
        results['rmsprop-cs']['gmean'].append(geo)
        results['rmsprop-cs']['auc'].append(auc)

        # ----------- SGDM ----------- #

        y_pred = model_sgdm.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['sgdm']['acuracia'].append(ac)
        results['sgdm']['acuracia_balanceada'].append(ba),
        results['sgdm']['precisao'].append(precisao)
        results['sgdm']['recall'].append(recall)
        results['sgdm']['f1'].append(f1)
        results['sgdm']['gmean'].append(geo)
        results['sgdm']['auc'].append(auc)

        # ----------- SGDM+COST-SENSITIVE ----------- #

        y_pred = model_sgdm_cs.predict(x_test)

        ac, ba, precisao, recall, f1, geo, auc = bt.get_results(y_pred, y_test_bool)

        results['sgdm-cs']['acuracia'].append(ac)
        results['sgdm-cs']['acuracia_balanceada'].append(ba),
        results['sgdm-cs']['precisao'].append(precisao)
        results['sgdm-cs']['recall'].append(recall)
        results['sgdm-cs']['f1'].append(f1)
        results['sgdm-cs']['gmean'].append(geo)
        results['sgdm-cs']['auc'].append(auc)

    _len = len(results['adamax']['acuracia'])

    if return_dict_results:

        final_results = {'adamax': {'acuracia': sum(results['adamax']['acuracia']) / _len,
                                    'acuracia_balanceada': sum(results['adamax']['acuracia_balanceada']) / _len,
                                    'precisao': sum(results['adamax']['precisao']) / _len,
                                    'recall': sum(results['adamax']['recall']) / _len,
                                    'f1': sum(results['adamax']['f1']) / _len,
                                    'gmean': sum(results['adamax']['gmean']) / _len,
                                    'auc': sum(results['adamax']['auc']) / _len},
                         'adamax-cs': {'acuracia': sum(results['adamax-cs']['acuracia']) / _len,
                                       'acuracia_balanceada': sum(results['adamax-cs']['acuracia_balanceada']) / _len,
                                       'precisao': sum(results['adamax-cs']['precisao']) / _len,
                                       'recall': sum(results['adamax-cs']['recall']) / _len,
                                       'f1': sum(results['adamax-cs']['f1']) / _len,
                                       'gmean': sum(results['adamax-cs']['gmean']) / _len,
                                       'auc': sum(results['adamax-cs']['auc']) / _len},
                         'rmsprop': {'acuracia': sum(results['rmsprop']['acuracia']) / _len,
                                     'acuracia_balanceada': sum(results['rmsprop']['acuracia_balanceada']) / _len,
                                     'precisao': sum(results['rmsprop']['precisao']) / _len,
                                     'recall': sum(results['rmsprop']['recall']) / _len,
                                     'f1': sum(results['rmsprop']['f1']) / _len,
                                     'gmean': sum(results['rmsprop']['gmean']) / _len,
                                     'auc': sum(results['rmsprop']['auc']) / _len},
                         'rmsprop-cs': {'acuracia': sum(results['rmsprop-cs']['acuracia']) / _len,
                                        'acuracia_balanceada': sum(results['rmsprop-cs']['acuracia_balanceada']) / _len,
                                        'precisao': sum(results['rmsprop-cs']['precisao']) / _len,
                                        'recall': sum(results['rmsprop-cs']['recall']) / _len,
                                        'f1': sum(results['rmsprop-cs']['f1']) / _len,
                                        'gmean': sum(results['rmsprop-cs']['gmean']) / _len,
                                        'auc': sum(results['rmsprop-cs']['auc']) / _len},
                         'sgdm': {'acuracia': sum(results['sgdm']['acuracia']) / _len,
                                     'acuracia_balanceada': sum(results['sgdm']['acuracia_balanceada']) / _len,
                                     'precisao': sum(results['sgdm']['precisao']) / _len,
                                     'recall': sum(results['sgdm']['recall']) / _len,
                                     'f1': sum(results['sgdm']['f1']) / _len,
                                     'gmean': sum(results['sgdm']['gmean']) / _len,
                                     'auc': sum(results['sgdm']['auc']) / _len},
                         'sgdm-cs': {'acuracia': sum(results['sgdm-cs']['acuracia']) / _len,
                                        'acuracia_balanceada': sum(results['rmsprop-cs']['acuracia_balanceada']) / _len,
                                        'precisao': sum(results['sgdm-cs']['precisao']) / _len,
                                        'recall': sum(results['sgdm-cs']['recall']) / _len,
                                        'f1': sum(results['sgdm-cs']['f1']) / _len,
                                        'gmean': sum(results['sgdm-cs']['gmean']) / _len,
                                        'auc': sum(results['sgdm-cs']['auc']) / _len}
                         }
        return final_results

    else:
        acuracia_adamax = sum(results['adamax']['acuracia']) / _len
        acuracia_balanceada_adamax = sum(results['adamax']['acuracia_balanceada']) / _len
        precisao_adamax = sum(results['adamax']['precisao']) / _len
        recall_adamax = sum(results['adamax']['recall']) / _len
        f1_adamax = sum(results['adamax']['f1']) / _len
        gmean_adamax = sum(results['adamax']['gmean']) / _len
        auc_adamax = sum(results['adamax']['auc']) / _len

        acuracia_adamax_cs = sum(results['adamax-cs']['acuracia']) / _len
        acuracia_balanceada_adamax_cs = sum(results['adamax-cs']['acuracia_balanceada']) / _len
        precisao_adamax_cs = sum(results['adamax-cs']['precisao']) / _len
        recall_adamax_cs = sum(results['adamax-cs']['recall']) / _len
        f1_adamax_cs = sum(results['adamax-cs']['f1']) / _len
        gmean_adamax_cs = sum(results['adamax-cs']['gmean']) / _len
        auc_adamax_cs = sum(results['adamax-cs']['auc']) / _len

        acuracia_rmsprop = sum(results['rmsprop']['acuracia']) / _len
        acuracia_balanceada_rmsprop = sum(results['rmsprop']['acuracia_balanceada']) / _len
        precisao_rmsprop = sum(results['rmsprop']['precisao']) / _len
        recall_rmsprop = sum(results['rmsprop']['recall']) / _len
        f1_rmsprop = sum(results['rmsprop']['f1']) / _len
        gmean_rmsprop = sum(results['rmsprop']['gmean']) / _len
        auc_rmsprop = sum(results['rmsprop']['auc']) / _len

        acuracia_rmsprop_cs = sum(results['rmsprop-cs']['acuracia']) / _len
        acuracia_balanceada_rmsprop_cs = sum(results['rmsprop-cs']['acuracia_balanceada']) / _len
        precisao_rmsprop_cs = sum(results['rmsprop-cs']['precisao']) / _len
        recall_rmsprop_cs = sum(results['rmsprop-cs']['recall']) / _len
        f1_rmsprop_cs = sum(results['rmsprop-cs']['f1']) / _len
        gmean_rmsprop_cs = sum(results['rmsprop-cs']['gmean']) / _len
        auc_rmsprop_cs = sum(results['rmsprop-cs']['auc']) / _len

        acuracia_sgdm = sum(results['sgdm']['acuracia']) / _len
        acuracia_balanceada_sgdm = sum(results['sgdm']['acuracia_balanceada']) / _len
        precisao_sgdm = sum(results['sgdm']['precisao']) / _len
        recall_sgdm = sum(results['sgdm']['recall']) / _len
        f1_sgdm = sum(results['sgdm']['f1']) / _len
        gmean_sgdm = sum(results['sgdm']['gmean']) / _len
        auc_sgdm = sum(results['sgdm']['auc']) / _len

        acuracia_sgdm_cs = sum(results['sgdm-cs']['acuracia']) / _len
        acuracia_balanceada_sgdm_cs = sum(results['sgdm-cs']['acuracia_balanceada']) / _len
        precisao_sgdm_cs = sum(results['sgdm-cs']['precisao']) / _len
        recall_sgdm_cs = sum(results['sgdm-cs']['recall']) / _len
        f1_sgdm_cs = sum(results['sgdm-cs']['f1']) / _len
        gmean_sgdm_cs = sum(results['sgdm-cs']['gmean']) / _len
        auc_sgdm_cs = sum(results['sgdm-cs']['auc']) / _len

        return acuracia_adamax, acuracia_balanceada_adamax, precisao_adamax, recall_adamax, f1_adamax, gmean_adamax, auc_adamax, \
            acuracia_adamax_cs, acuracia_balanceada_adamax_cs, precisao_adamax_cs, recall_adamax_cs, f1_adamax_cs, gmean_adamax_cs, auc_adamax_cs, \
            acuracia_rmsprop, acuracia_balanceada_rmsprop, precisao_rmsprop, recall_rmsprop, f1_rmsprop, gmean_rmsprop, auc_rmsprop, \
            acuracia_rmsprop_cs, acuracia_balanceada_rmsprop_cs,  precisao_rmsprop_cs, recall_rmsprop_cs, f1_rmsprop_cs,gmean_rmsprop_cs, auc_rmsprop_cs, \
            acuracia_sgdm, acuracia_balanceada_sgdm, precisao_sgdm, recall_sgdm, f1_sgdm, gmean_sgdm, auc_sgdm,\
            acuracia_sgdm_cs, acuracia_balanceada_sgdm_cs, precisao_sgdm_cs, recall_sgdm_cs, f1_sgdm_cs, gmean_sgdm_cs, auc_sgdm_cs


def run_30x(x_train,
            y_train,
            x_test,
            y_test,
            params,
            vocab_size,
            _range=30):
    results = {'adamax': {'acuracia': [],
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

    for i in range(_range):
        print(f'-- {i} --')
        acuracia_adamax,  acuracia_balanceada_adamax, precisao_adamax, recall_adamax, f1_adamax, gmean_adamax, auc_adamax, \
        acuracia_adamax_cs,  acuracia_balanceada_adamax_cs, precisao_adamax_cs, recall_adamax_cs, f1_adamax_cs, gmean_adamax_cs, auc_adamax_cs, \
        acuracia_rmsprop,  acuracia_balanceada_rmsprop, precisao_rmsprop, recall_rmsprop, f1_rmsprop, gmean_rmsprop, auc_rmsprop, \
        acuracia_rmsprop_cs,  acuracia_balanceada_rmsprop_cs, precisao_rmsprop_cs, recall_rmsprop_cs, f1_rmsprop_cs, gmean_rmsprop_cs, auc_rmsprop_cs, \
        acuracia_sgdm, acuracia_balanceada_sgdm, precisao_sgdm, recall_sgdm, f1_sgdm, gmean_sgdm, auc_sgdm, \
        acuracia_sgdm_cs, acuracia_balanceada_sgdm_cs, precisao_sgdm_cs, recall_sgdm_cs, f1_sgdm_cs, gmean_sgdm_cs, auc_sgdm_cs = \
            run_cross_validation(x_train, y_train, x_test, y_test, params, vocab_size)

        results['adamax']['acuracia'].append(acuracia_adamax)
        results['adamax']['acuracia_balanceada'].append(acuracia_balanceada_adamax)
        results['adamax']['precisao'].append(precisao_adamax)
        results['adamax']['recall'].append(recall_adamax)
        results['adamax']['f1'].append(f1_adamax)
        results['adamax']['gmean'].append(gmean_adamax)
        results['adamax']['auc'].append(auc_adamax)

        results['adamax-cs']['acuracia'].append(acuracia_adamax_cs)
        results['adamax-cs']['acuracia_balanceada'].append(acuracia_balanceada_adamax_cs)
        results['adamax-cs']['precisao'].append(precisao_adamax_cs)
        results['adamax-cs']['recall'].append(recall_adamax_cs)
        results['adamax-cs']['f1'].append(f1_adamax_cs)
        results['adamax-cs']['gmean'].append(gmean_adamax_cs)
        results['adamax-cs']['auc'].append(auc_adamax_cs)

        results['rmsprop']['acuracia'].append(acuracia_rmsprop)
        results['rmsprop']['acuracia_balanceada'].append(acuracia_balanceada_rmsprop)
        results['rmsprop']['precisao'].append(precisao_rmsprop)
        results['rmsprop']['recall'].append(recall_rmsprop)
        results['rmsprop']['f1'].append(f1_rmsprop)
        results['rmsprop']['gmean'].append(gmean_rmsprop)
        results['rmsprop']['auc'].append(auc_rmsprop)

        results['rmsprop-cs']['acuracia'].append(acuracia_rmsprop_cs)
        results['rmsprop-cs']['acuracia_balanceada'].append(acuracia_balanceada_rmsprop_cs)
        results['rmsprop-cs']['precisao'].append(precisao_rmsprop_cs)
        results['rmsprop-cs']['recall'].append(recall_rmsprop_cs)
        results['rmsprop-cs']['f1'].append(f1_rmsprop_cs)
        results['rmsprop-cs']['gmean'].append(gmean_rmsprop_cs)
        results['rmsprop-cs']['auc'].append(auc_rmsprop_cs)

        results['sgdm']['acuracia'].append(acuracia_rmsprop)
        results['sgdm']['acuracia_balanceada'].append(acuracia_balanceada_rmsprop)
        results['sgdm']['precisao'].append(precisao_rmsprop)
        results['sgdm']['recall'].append(recall_rmsprop)
        results['sgdm']['f1'].append(f1_rmsprop)
        results['sgdm']['gmean'].append(gmean_rmsprop)
        results['sgdm']['auc'].append(auc_rmsprop)

        results['sgdm-cs']['acuracia'].append(acuracia_rmsprop_cs)
        results['sgdm-cs']['acuracia_balanceada'].append(acuracia_balanceada_rmsprop_cs)
        results['sgdm-cs']['precisao'].append(precisao_rmsprop_cs)
        results['sgdm-cs']['recall'].append(recall_rmsprop_cs)
        results['sgdm-cs']['f1'].append(f1_rmsprop_cs)
        results['sgdm-cs']['gmean'].append(gmean_rmsprop_cs)
        results['sgdm-cs']['auc'].append(auc_rmsprop_cs)

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
    results['rsgdm-cs']['precisao'] = sum(results['sgdm-cs']['precisao']) / _len
    results['sgdm-cs']['recall'] = sum(results['sgdm-cs']['recall']) / _len
    results['sgdm-cs']['f1'] = sum(results['sgdm-cs']['f1']) / _len
    results['sgdm-cs']['gmean'] = sum(results['sgdm-cs']['gmean']) / _len
    results['sgdm-cs']['auc'] = sum(results['sgdm-cs']['auc']) / _len

    return results
