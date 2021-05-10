import pandas as pd
from models.dataset import DataSet


if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    dataset.define_datasets(generate_table=True)

