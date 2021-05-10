import pandas as pd
from models.dataset import DataSet
from models.costsensitive_weights import WeigthsV1, WeigthsV2, WeigthsV3

if __name__ == '__main__':
    dataset = DataSet(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

    # hyper = WeigthsV1()
    # hyper.run_(dataset, 'origin', 'adamax')
    # # hyper.update_weights_list('adamax+origin+cs')
    #
    # hyper = WeigthsV2()
    # hyper.run_(dataset, 'origin', 'adamax')
    # hyper.update_weights_list('adamax+origin+cs')

    # weights = {0: 0.128, 1: 0.128, 2: 0.128, 3: 0.0464, 4: 0.128, 5: 0.0464, 6: 0.0464, 7: 0.128, 8: 0.128,
    #            9: 0.0464, 10: 0.0464}

    weights = WeigthsV3()
    print(weights.get_class_weight(data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8')))
