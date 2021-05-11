import pandas as pd
from models.dataset import DataSet
from models.bayesian_optimization import Hyper


def run_hyper(resamplings, algs):
    hyper = Hyper()

    for resampling in resamplings:
        for alg in algs:
            hyper.run_(dataset, resampling, alg, 200)
            hyper.update_hyperparameter_list(str(alg)+'+'+str(resampling))


if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    resamplings = ['origin', 'bdsmote', 'smotetomek']
    algs = ['adamax', 'rmsprop', 'sgdm']

    run_hyper(resamplings, algs)

