import models.basictools as bt


class DataSet:
    def __init__(self, data, generate_table=False):
        self.origin = {}
        self.tomek = {}
        self.smote = {}
        self.borderline_smote = {}
        self.smote_enn = {}
        self.smote_tomek = {}
        self.generate_table = generate_table

        self.definir_datasets(data)

    def definir_datasets(self, data):
        # ORIGIN
        vocab_size_origin, x_train_origin, y_train_origin, x_test_origin, y_test_origin = bt.prepare_data(data=data,
                                                                                                          test_size=0.2,
                                                                                                          random_state=42,
                                                                                                          resampling=None)
        self.set_data('origin', vocab_size_origin, x_train_origin, y_train_origin, x_test_origin, y_test_origin)


        # TOMEKLINKS
        vocab_size_tomek, x_train_tomek, y_train_tomek, x_test_tomek, y_test_tomek = bt.prepare_data(data=data,
                                                                                                     test_size=0.2,
                                                                                                     random_state=42,
                                                                                                     resampling='TomekLinks')
        self.set_data('dataTomek', vocab_size_tomek, x_train_tomek, y_train_tomek, x_test_tomek, y_test_tomek)

        # SMOTE
        vocab_size_smote, x_train_smote, y_train_smote, x_test_smote, y_test_smote = bt.prepare_data(data=data,
                                                                                                     test_size=0.2,
                                                                                                     random_state=42,
                                                                                                     resampling='SMOTE')
        self.set_data('dataSmote', vocab_size_smote, x_train_smote, y_train_smote, x_test_smote, y_test_smote)

        # BORDERLINE SMOTE
        vocab_size_bd_smote, x_train_bd_smote, y_train_bd_smote, x_test_bd_smote, y_test_bd_smote = bt.prepare_data(
            data=data, test_size=0.2, random_state=42, resampling='BorderlineSMOTE')
        self.set_data('dataBoderlineSmote', vocab_size_bd_smote, x_train_bd_smote, y_train_bd_smote, x_test_bd_smote,
                      y_test_bd_smote)

        # SMOTEENN
        vocab_size_smoteenn, x_train_smoteenn, y_train_smoteenn, x_test_smoteenn, y_test_smoteenn = bt.prepare_data(
            data=data, test_size=0.2, random_state=42, resampling='SMOTEENN')
        self.set_data('dataSmoteEnn', vocab_size_smoteenn, x_train_smoteenn, y_train_smoteenn, x_test_smoteenn,
                      y_test_smoteenn)


        # SMOTETOMEK
        vocab_size_smotetomek, x_train_smotetomek, y_train_smotetomek, x_test_smotetomek, y_test_smotetomek= bt.prepare_data(
            data=data, test_size=0.2,  random_state=42, resampling='SMOTETomek')
        self.set_data('dataSmoteTomek', vocab_size_smotetomek, x_train_smotetomek, y_train_smotetomek, x_test_smotetomek,
                      y_test_smotetomek)

        if self.generate_table:
            datasets = {'origin': {'y_train': y_train_origin,
                                   'y_test': y_test_origin},
                        'tomek': {'y_train': y_train_tomek,
                                  'y_test': y_test_tomek},
                        'smote': {'y_train': y_train_smote,
                                  'y_test': y_test_smote},
                        'bdsmote': {'y_train': y_train_bd_smote,
                                    'y_test': y_test_bd_smote},
                        'smoteenn': {'y_train': y_train_smoteenn,
                                     'y_test': y_test_smoteenn},
                        'smotetomek': {'y_train': y_train_smotetomek,
                                       'y_test': y_test_smotetomek},
                        }

            bt.dataset_detail(datasets)

    def set_data(self, dataname, vocab_size, x_train, y_train, x_test, y_test):
        data = {'vocab_size': vocab_size, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
        if dataname == 'origin':
            self.origin = data
        elif dataname == 'dataTomek':
            self.tomek = data
        elif dataname == 'dataSmote':
            self.smote = data
        elif dataname == 'dataBoderlineSmote':
            self.borderline_smote = data
        elif dataname == 'dataSmoteEnn':
            self.smote_enn = data
        elif dataname == 'dataSmoteTomek':
            self.smote_tomek = data

    def get_data(self, dataname):
        if dataname == 'origin':
            return self.origin['vocab_size'], self.origin['x_train'], self.origin['y_train'], self.origin['x_test'],\
                   self.origin['y_test']
        elif dataname == 'dataTomek':
            return self.tomek['vocab_size'], self.tomek['x_train'], self.tomek['y_train'], self.tomek['x_test'], \
                   self.tomek['y_test']
        elif dataname == 'dataSmote':
            return self.smote['vocab_size'], self.smote['x_train'], self.smote['y_train'], self.smote['x_test'], \
                   self.smote['y_test']
        elif dataname == 'dataBoderlineSmote':
            return self.borderline_smote['vocab_size'], self.borderline_smote['x_train'], \
                   self.borderline_smote['y_train'], self.borderline_smote['x_test'], self.borderline_smote['y_test']
        elif dataname == 'dataSmoteEnn':
            return self.smote_enn['vocab_size'], self.smote_enn['x_train'], self.smote_enn['y_train'], \
                   self.smote_enn['x_test'], self.smote_enn['y_test']
        elif dataname == 'dataSmoteTomek':
            return self.smote_tomek['vocab_size'], self.smote_tomek['x_train'], self.smote_tomek['y_train'], \
                   self.smote_tomek['x_test'], self.smote_tomek['y_test']

    def info_data(self, dataname):
        y_train, y_test = 0, 0
        if dataname == 'origin':
            y_train, y_test = self.origin['y_train'], self.origin['y_test']
        elif dataname == 'dataTomek':
            y_train, y_test = self.tomek['y_train'], self.tomek['y_test']
        elif dataname == 'dataSmote':
            y_train, y_test = self.smote['y_train'], self.smote['y_test']
        elif dataname == 'dataBorderlineSmote':
            y_train, y_test = self.borderline_smote['y_train'], self.borderline_smote['y_test']
        elif dataname == 'dataSmoteEnn':
            y_train, y_test = self.smote_enn['y_train'], self.smote_enn['y_test']
        elif dataname == 'dataSmoteTomek':
            y_train, y_test = self.smote_tomek['y_train'], self.smote_tomek['y_test']

        bt.plot_requirements_by_class(y_train, y_test)

