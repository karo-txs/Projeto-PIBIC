import pandas as pd
from models.dataset import DataSet
from models.costsensitive_weights import Weigths, WeigthsV2

if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    weights = Weigths()
    print(weights.get_class_weight(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8')))

    weights = WeigthsV2()
    weights.run_(dataset, 'origin', 500)
    weights.update_weigths_list('adamax+origin')
    # 0.611 | 0.8839 | 0.9953 | 0.371 | 0.4918 | 0.5149 | 0.3797 | 0.5314 | 0.3008 | 0.7847 | 0.5489 | 0.2379
