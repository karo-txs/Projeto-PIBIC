import pandas as pd
from models.dataset import DataSet
from models.randomized_search import Hyper


def run_hyper(resamplings, algs):
    hyper = Hyper()

    for resampling in resamplings:
        for alg in algs:
            hyper.run_(dataset, resampling, alg, 600)
            hyper.update_hyperparameter_list(str(alg)+'+'+str(resampling))


if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    resamplings = ['origin', 'tomek', 'smote', 'smotetomek', 'smoteenn']
    algs = ['adamax', 'rmsprop', 'sgdm']

    run_hyper(resamplings, algs)

