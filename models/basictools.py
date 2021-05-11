import pandas as pd
import warnings
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score)
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from keras.preprocessing.text import Tokenizer
from models.resampling import Resampling


def prepare_data(data, test_size, resampling='origin'):
    if resampling == 'origin':
        X_train, X_test, Y_train, Y_test = train_test_split(data['RequirementText'], data['Class'],
                                                            test_size=test_size,
                                                            stratify=data['Class'])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        x_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
        x_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')

        vocab_size = len(tokenizer.word_index) + 1

        bin = LabelBinarizer()
        bin.fit(Y_train)

        y_train = bin.transform(Y_train)
        y_test = bin.transform(Y_test)

    else:
        strategie = Resampling(resampling)
        X_train, X_test, Y_train, Y_test = train_test_split(data['RequirementText'], data['Class'],
                                                            test_size=test_size,
                                                            stratify=data['Class'])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        x_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
        x_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')

        vocab_size = len(tokenizer.word_index) + 1

        encoder = LabelBinarizer()
        encoder.fit(Y_train)

        y_train = encoder.transform(Y_train)
        y_test = encoder.transform(Y_test)

        x_train, y_train = strategie.fit_resample(x_train, y_train)

    return x_train, y_train, x_test, y_test


def display_results(pred_mlp, y_test):
    precisao = precision_score(y_test, pred_mlp, average='macro')
    recall = recall_score(y_test, pred_mlp, average='macro')
    geo = geometric_mean_score(y_test, pred_mlp, average='macro')
    f1 = f1_score(y_test, pred_mlp, average='macro')
    ac = accuracy_score(y_test, pred_mlp)
    # kp = cohen_kappa_score(y_test, le_pred_mlp)

    bin = LabelBinarizer()
    bin.fit(y_test)
    lb_y_test = bin.transform(y_test)
    lb_pred_mlp = bin.transform(pred_mlp)

    auc = roc_auc_score(lb_y_test, lb_pred_mlp, average='macro', multi_class='ovo')

    print('--------------------------------------RESULTADOS--------------------------------------------')
    print(
        f'Accuracy: {ac:.2f}, Precision: {precisao:.2f}, Recall: {recall:.2f}, F1_score: {f1:.2f}, Geo: {geo:.2f}, AUC: {auc:.2f}')
    print('')
    print('----------------------------------------REPORT----------------------------------------------')
    report = classification_report(y_test, pred_mlp)
    print(report)
    print('')
    print('-----------------------------------REPORT_IMBALANCED----------------------------------------')
    target_names = ['A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US']
    report_imbalanced = classification_report_imbalanced(y_test, pred_mlp, target_names=target_names)
    print(report_imbalanced)
    print('')
    print('-----------------------------------CONFUSION MATRIX------------------------------------------')
    cnf_matrix = confusion_matrix(y_test, pred_mlp)
    print(cnf_matrix)


def get_results(pred_mlp, y_test):
    warnings.filterwarnings('ignore')
    precisao = precision_score(y_test, pred_mlp, average='macro')
    recall = recall_score(y_test, pred_mlp, average='macro')
    geo = geometric_mean_score(y_test, pred_mlp, average='macro')
    f1 = f1_score(y_test, pred_mlp, average='macro')
    ac = accuracy_score(y_test, pred_mlp)
    ba = balanced_accuracy_score(y_test, pred_mlp)
    # kp = cohen_kappa_score(y_test, le_pred_mlp)

    bin = LabelBinarizer()
    bin.fit(y_test)
    lb_y_test = bin.transform(y_test)
    lb_pred_mlp = bin.transform(pred_mlp)

    auc = roc_auc_score(lb_y_test, lb_pred_mlp, average='macro', multi_class='ovo')

    return ac, ba, precisao, recall, f1, geo, auc


def count_classes(y_train, y_test, is_smoteenn=False):
    count = []

    for i in range(11):
        if i == 0:
            class_ = y_train.groupby(i).count()[1][1] + y_test.groupby(i).count()[1][1]
        else:
            class_ = y_train.groupby(i).count()[0][1] + y_test.groupby(i).count()[0][1]
        count.append(class_)

    total = sum(count)
    count.append(total)

    return count


def dataset_detail(datasets):
    classes = ['A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US', 'total']

    y_train = pd.DataFrame(datasets['origin']['y_train'])
    y_test = pd.DataFrame(datasets['origin']['y_test'])

    count_origin = count_classes(y_train, y_test)

    y_train = pd.DataFrame(datasets['tomek']['y_train'])
    y_test = pd.DataFrame(datasets['tomek']['y_test'])

    count_tomek = count_classes(y_train, y_test)

    y_train = pd.DataFrame(datasets['smote']['y_train'])
    y_test = pd.DataFrame(datasets['smote']['y_test'])

    count_smote = count_classes(y_train, y_test)

    y_train = pd.DataFrame(datasets['bdsmote']['y_train'])
    y_test = pd.DataFrame(datasets['bdsmote']['y_test'])

    count_bd_smote = count_classes(y_train, y_test)

    y_train = pd.DataFrame(datasets['smoteenn']['y_train'])
    y_test = pd.DataFrame(datasets['smoteenn']['y_test'])

    count_smoteenn = count_classes(y_train, y_test, True)

    y_train = pd.DataFrame(datasets['smotetomek']['y_train'])
    y_test = pd.DataFrame(datasets['smotetomek']['y_test'])

    count_smotetomek = count_classes(y_train, y_test)

    df = pd.DataFrame(list(zip(classes, count_origin, count_tomek, count_smote, count_bd_smote, count_smoteenn,
                               count_smotetomek)),
                      columns=['classes', 'original', 'tomek', 'smote', 'bdsmote', 'smoteenn', 'smotetomek'])
    df.to_excel('../results/details_datasets.xlsx')


def plot_training_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def plot_division_dataset(train, test, validation):
    division = [train, test, validation]
    labels = ['Train', 'Test', 'Validation']
    explode = (0, 0, 0)
    plt.pie(division, labels=labels, autopct='%1.1f%%', shadow=True, explode=explode)
    plt.legend(labels, loc=3)
    plt.axis('equal')
    plt.show()


def plot_requirements_by_class(y_train, y_test):
    print("Number of Requirements per class (train):")
    train_set = pd.DataFrame(y_train)
    print(train_set.shape)
    print(train_set.value_counts())
    print("Number of Requirements per class (test):")
    test_set = pd.DataFrame(y_test)
    print(test_set.shape)
    print(test_set.value_counts())


def print_results_cv(results, indent=0):
    for key, value in results.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_results_cv(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def print_table(results, export_table=False):
    info = {'Combinações': ['adamax+origin',
                            'rmsprop+origin',
                            'sgdm+origin',
                            'adamax+origin+cs',
                            'rmsprop+origin+cs',
                            'sgdm+origin+cs',

                            'adamax+tomek',
                            'rmsprop+tomek',
                            'sgdm+tomek',
                            'adamax+tomek+cs',
                            'rmsprop+tomek+cs',
                            'sgdm+tomek+cs',

                            'adamax+smote',
                            'rmsprop+smote',
                            'sgdm+smote',
                            'adamax+smote+cs',
                            'rmsprop+smote+cs',
                            'sgdm+smote+cs',

                            'adamax+bd-smote',
                            'rmsprop+bd-smote',
                            'sgdm+bd-smote',
                            'adamax+bd-smote+cs',
                            'rmsprop+bd-smote+cs',
                            'sgdm+bd-smote+cs',

                            'adamax+smoteenn',
                            'rmsprop+smoteenn',
                            'sgdm+smoteenn',
                            'adamax+smoteenn+cs',
                            'rmsprop+smoteenn+cs',
                            'sgdm+smoteenn+cs',

                            'adamax+smotetomek',
                            'rmsprop+smotetomek',
                            'sgdm+smotetomek',
                            'adamax+smotetomek+cs',
                            'rmsprop+smotetomek+cs',
                            'sgdm+smotetomek+cs'
                            ],
            'ba': [results['origin']['adamax']['acuracia_balanceada'],
                   results['origin']['rmsprop']['acuracia_balanceada'],
                   results['origin']['sgdm']['acuracia_balanceada'],
                   results['origin']['adamax-cs']['acuracia_balanceada'],
                   results['origin']['rmsprop-cs']['acuracia_balanceada'],
                   results['origin']['sgdm-cs']['acuracia_balanceada'],

                   results['tomek']['adamax']['acuracia_balanceada'],
                   results['tomek']['rmsprop']['acuracia_balanceada'],
                   results['tomek']['sgdm']['acuracia_balanceada'],
                   results['tomek']['adamax-cs']['acuracia_balanceada'],
                   results['tomek']['rmsprop-cs']['acuracia_balanceada'],
                   results['tomek']['sgdm-cs']['acuracia_balanceada'],

                   results['smote']['adamax']['acuracia_balanceada'],
                   results['smote']['rmsprop']['acuracia_balanceada'],
                   results['smote']['sgdm']['acuracia_balanceada'],
                   results['smote']['adamax-cs']['acuracia_balanceada'],
                   results['smote']['rmsprop-cs']['acuracia_balanceada'],
                   results['smote']['sgdm-cs']['acuracia_balanceada'],

                   results['bd-smote']['adamax']['acuracia_balanceada'],
                   results['bd-smote']['rmsprop']['acuracia_balanceada'],
                   results['bd-smote']['sgdm']['acuracia_balanceada'],
                   results['bd-smote']['adamax-cs']['acuracia_balanceada'],
                   results['bd-smote']['rmsprop-cs']['acuracia_balanceada'],
                   results['bd-smote']['sgdm-cs']['acuracia_balanceada'],

                   results['smoteenn']['adamax']['acuracia_balanceada'],
                   results['smoteenn']['rmsprop']['acuracia_balanceada'],
                   results['smoteenn']['sgdm']['acuracia_balanceada'],
                   results['smoteenn']['adamax-cs']['acuracia_balanceada'],
                   results['smoteenn']['rmsprop-cs']['acuracia_balanceada'],
                   results['smoteenn']['sgdm-cs']['acuracia_balanceada'],

                   results['smotetomek']['adamax']['acuracia_balanceada'],
                   results['smotetomek']['rmsprop']['acuracia_balanceada'],
                   results['smotetomek']['sgdm']['acuracia_balanceada'],
                   results['smotetomek']['adamax-cs']['acuracia_balanceada'],
                   results['smotetomek']['rmsprop-cs']['acuracia_balanceada'],
                   results['smotetomek']['sgdm-cs']['acuracia_balanceada'],
                   ],
            'f1': [results['origin']['adamax']['f1'],
                   results['origin']['rmsprop']['f1'],
                   results['origin']['sgdm']['f1'],
                   results['origin']['adamax-cs']['f1'],
                   results['origin']['rmsprop-cs']['f1'],
                   results['origin']['sgdm-cs']['f1'],

                   results['tomek']['adamax']['f1'],
                   results['tomek']['rmsprop']['f1'],
                   results['tomek']['sgdm']['f1'],
                   results['tomek']['adamax-cs']['f1'],
                   results['tomek']['rmsprop-cs']['f1'],
                   results['tomek']['sgdm-cs']['f1'],

                   results['smote']['adamax']['f1'],
                   results['smote']['rmsprop']['f1'],
                   results['smote']['sgdm']['f1'],
                   results['smote']['adamax-cs']['f1'],
                   results['smote']['rmsprop-cs']['f1'],
                   results['smote']['sgdm-cs']['f1'],

                   results['bd-smote']['adamax']['f1'],
                   results['bd-smote']['rmsprop']['f1'],
                   results['bd-smote']['sgdm']['f1'],
                   results['bd-smote']['adamax-cs']['f1'],
                   results['bd-smote']['rmsprop-cs']['f1'],
                   results['bd-smote']['sgdm-cs']['f1'],

                   results['smoteenn']['adamax']['f1'],
                   results['smoteenn']['rmsprop']['f1'],
                   results['smoteenn']['sgdm']['f1'],
                   results['smoteenn']['adamax-cs']['f1'],
                   results['smoteenn']['rmsprop-cs']['f1'],
                   results['smoteenn']['sgdm-cs']['f1'],

                   results['smotetomek']['adamax']['f1'],
                   results['smotetomek']['rmsprop']['f1'],
                   results['smotetomek']['sgdm']['f1'],
                   results['smotetomek']['adamax-cs']['f1'],
                   results['smotetomek']['rmsprop-cs']['f1'],
                   results['smotetomek']['sgdm-cs']['f1'],
                   ],
            'gmean': [results['origin']['adamax']['gmean'],
                      results['origin']['rmsprop']['gmean'],
                      results['origin']['sgdm']['gmean'],
                      results['origin']['adamax-cs']['gmean'],
                      results['origin']['rmsprop-cs']['gmean'],
                      results['origin']['sgdm-cs']['gmean'],

                      results['tomek']['adamax']['gmean'],
                      results['tomek']['rmsprop']['gmean'],
                      results['tomek']['sgdm']['gmean'],
                      results['tomek']['adamax-cs']['gmean'],
                      results['tomek']['rmsprop-cs']['gmean'],
                      results['tomek']['sgdm-cs']['gmean'],

                      results['smote']['adamax']['gmean'],
                      results['smote']['rmsprop']['gmean'],
                      results['smote']['sgdm']['gmean'],
                      results['smote']['adamax-cs']['gmean'],
                      results['smote']['rmsprop-cs']['gmean'],
                      results['smote']['sgdm-cs']['gmean'],

                      results['bd-smote']['adamax']['gmean'],
                      results['bd-smote']['rmsprop']['gmean'],
                      results['bd-smote']['sgdm']['gmean'],
                      results['bd-smote']['adamax-cs']['gmean'],
                      results['bd-smote']['rmsprop-cs']['gmean'],
                      results['bd-smote']['sgdm-cs']['gmean'],

                      results['smoteenn']['adamax']['gmean'],
                      results['smoteenn']['rmsprop']['gmean'],
                      results['smoteenn']['sgdm']['gmean'],
                      results['smoteenn']['adamax-cs']['gmean'],
                      results['smoteenn']['rmsprop-cs']['gmean'],
                      results['smoteenn']['sgdm-cs']['gmean'],

                      results['smotetomek']['adamax']['gmean'],
                      results['smotetomek']['rmsprop']['gmean'],
                      results['smotetomek']['sgdm']['gmean'],
                      results['smotetomek']['adamax-cs']['gmean'],
                      results['smotetomek']['rmsprop-cs']['gmean'],
                      results['smotetomek']['sgdm-cs']['gmean'],
                      ],
            'auc': [results['origin']['adamax']['auc'],
                    results['origin']['rmsprop']['auc'],
                    results['origin']['sgdm']['auc'],
                    results['origin']['adamax-cs']['auc'],
                    results['origin']['rmsprop-cs']['auc'],
                    results['origin']['sgdm-cs']['auc'],

                    results['tomek']['adamax']['auc'],
                    results['tomek']['rmsprop']['auc'],
                    results['tomek']['sgdm']['auc'],
                    results['tomek']['adamax-cs']['auc'],
                    results['tomek']['rmsprop-cs']['auc'],
                    results['tomek']['sgdm-cs']['auc'],

                    results['smote']['adamax']['auc'],
                    results['smote']['rmsprop']['auc'],
                    results['smote']['sgdm']['auc'],
                    results['smote']['adamax-cs']['auc'],
                    results['smote']['rmsprop-cs']['auc'],
                    results['smote']['sgdm-cs']['auc'],

                    results['bd-smote']['adamax']['auc'],
                    results['bd-smote']['rmsprop']['auc'],
                    results['bd-smote']['sgdm']['auc'],
                    results['bd-smote']['adamax-cs']['auc'],
                    results['bd-smote']['rmsprop-cs']['auc'],
                    results['bd-smote']['sgdm-cs']['auc'],

                    results['smoteenn']['adamax']['auc'],
                    results['smoteenn']['rmsprop']['auc'],
                    results['smoteenn']['sgdm']['auc'],
                    results['smoteenn']['adamax-cs']['auc'],
                    results['smoteenn']['rmsprop-cs']['auc'],
                    results['smoteenn']['sgdm-cs']['auc'],

                    results['smotetomek']['adamax']['auc'],
                    results['smotetomek']['rmsprop']['auc'],
                    results['smotetomek']['sgdm']['auc'],
                    results['smotetomek']['adamax-cs']['auc'],
                    results['smotetomek']['rmsprop-cs']['auc'],
                    results['smotetomek']['sgdm-cs']['auc'],
                    ],
            }
    if export_table:
        df = pd.DataFrame(list(zip(info['Combinações'], info['ba'], info['f1'], info['gmean'], info['auc'])),
                          columns=['combinações', 'ba', 'f1', 'gmean', 'auc'])
        df.to_excel('results/Results.xlsx')
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))


def print_table_single(results, dataname, export_results):
    info = {'combinacoes': ['adamax+'+str(dataname),
                            'rmsprop+'+str(dataname),
                            'sgdm+'+str(dataname),
                            'adamax+'+str(dataname)+'+cs',
                            'rmsprop+'+str(dataname)+'+cs',
                            'sgdm+'+str(dataname)+'+cs'
                            ],
            'ba': [results[dataname]['adamax']['acuracia_balanceada'],
                   results[dataname]['rmsprop']['acuracia_balanceada'],
                   results[dataname]['sgdm']['acuracia_balanceada'],
                   results[dataname]['adamax-cs']['acuracia_balanceada'],
                   results[dataname]['rmsprop-cs']['acuracia_balanceada'],
                   results[dataname]['sgdm-cs']['acuracia_balanceada']
                   ],
            'f1': [results[dataname]['adamax']['f1'],
                   results[dataname]['rmsprop']['f1'],
                   results[dataname]['sgdm']['f1'],
                   results[dataname]['adamax-cs']['f1'],
                   results[dataname]['rmsprop-cs']['f1'],
                   results[dataname]['sgdm-cs']['f1'],
                   ],
            'gmean': [results[dataname]['adamax']['gmean'],
                      results[dataname]['rmsprop']['gmean'],
                      results[dataname]['sgdm']['gmean'],
                      results[dataname]['adamax-cs']['gmean'],
                      results[dataname]['rmsprop-cs']['gmean'],
                      results[dataname]['sgdm-cs']['gmean']
                      ],
            'auc': [results[dataname]['adamax']['auc'],
                    results[dataname]['rmsprop']['auc'],
                    results[dataname]['sgdm']['auc'],
                    results[dataname]['adamax-cs']['auc'],
                    results[dataname]['rmsprop-cs']['auc'],
                    results[dataname]['sgdm-cs']['auc'],
                    ],
            }

    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    if export_results:
        df = pd.DataFrame(list(zip(info['combinacoes'], info['ba'], info['f1'], info['gmean'], info['auc'])),
                          columns=['combinacoes', 'ba', 'f1', 'gmean', 'auc'])
        df.to_csv('../results/results_'+str(dataname)+'.csv')


def get_time(valor):
    duration = '{:.2f}s'.format(valor)
    if valor > 60:
        valor = valor / 60
        duration = '{:.2f}m'.format(valor)
        if valor > 60:
            valor = valor / 60
            duration = '{:.2f}h'.format(valor)
    return duration


def update_results_per_cross(results, iter, resampling):
    path = '../results/results_'+str(resampling)+'_detail.csv'
    lines = {'iter': iter,
             'combinacoes': 'adamax',
             'acuracia': results['adamax']['acuracia'],
             'ba': results['adamax']['acuracia_balanceada'],
             'precisao': results['adamax']['precisao'],
             'recall': results['adamax']['recall'],
             'f1': results['adamax']['f1'],
             'gmean': results['adamax']['gmean'],
             'auc': results['adamax']['auc']}

    try:
        open(path, 'r')
        with open(path, 'a') as arq:
            writer = csv.writer(arq)
            writer.writerow(lines.values())

            write_lines(iter, writer, results, 'rmsprop')
            write_lines(iter, writer, results, 'sgdm')
            write_lines(iter, writer, results, 'adamax-cs')
            write_lines(iter, writer, results, 'rmsprop-cs')
            write_lines(iter, writer, results, 'sgdm-cs')

    except IOError:
        data = pd.DataFrame(columns=lines.keys())
        data = data.append(lines, ignore_index=True)
        data.to_csv(path, index=False)

        open(path, 'r')
        with open(path, 'a') as arq:
            writer = csv.writer(arq)

            write_lines(iter, writer, results, 'rmsprop')
            write_lines(iter, writer, results, 'sgdm')
            write_lines(iter, writer, results, 'adamax-cs')
            write_lines(iter, writer, results, 'rmsprop-cs')
            write_lines(iter, writer, results, 'sgdm-cs')


def write_lines(iter, writer, results, alg):
    lines = {'iter': iter,
             'combinacoes': alg,
             'acuracia': results[alg]['acuracia'],
             'ba': results[alg]['acuracia_balanceada'],
             'precisao': results[alg]['precisao'],
             'recall': results[alg]['recall'],
             'f1': results[alg]['f1'],
             'gmean': results[alg]['gmean'],
             'auc': results[alg]['auc']}

    writer.writerow(lines.values())

