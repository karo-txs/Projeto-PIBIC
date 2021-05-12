import pandas as pd
from models.dataset import DataSet
import models.crossvalidation as cross_valid


def run_cross_valid(dataset, resamplings):
    for resampling in resamplings:
        cross_valid.cross_val_single(dataset, resampling, export_results=True)


if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    resamplings = ['origin']
    run_cross_valid(dataset, resamplings)
