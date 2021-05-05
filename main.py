import pandas as pd
from models.dataset import DataSet
from models.hyperparameterization_params import Hyper
import models.crossvalidation as cv

if __name__ == '__main__':
    # Criação dos datasets
    dataset = DataSet(data=pd.read_csv('dataset/PROMISE_exp_preprocessed.csv'), generate_table=True)

    # Definição dos hiperparametros
    hyper = Hyper()

    # Cross Validation | Testes
    cv.cross_val_complete(hyper, dataset)

