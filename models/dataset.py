import models.basictools as bt


class DataSet:
    def __init__(self, data):
        self.origin = {}
        self.tomek = {}
        self.smote = {}
        self.bdsmote = {}
        self.smoteenn = {}
        self.smotetomek = {}
        self.data = data

    def define_datasets(self, generate_table=False):
        # ORIGIN
        x_train_origin, y_train_origin, x_test_origin, y_test_origin = bt.prepare_data(data=self.data,
                                                                                       test_size=0.2,
                                                                                       resampling='origin')
        self.set_data('origin', x_train_origin, y_train_origin, x_test_origin, y_test_origin)

        # TOMEKLINKS
        x_train_tomek, y_train_tomek, x_test_tomek, y_test_tomek = bt.prepare_data(data=self.data,
                                                                                   test_size=0.2,
                                                                                   resampling='tomek')
        self.set_data('tomek', x_train_tomek, y_train_tomek, x_test_tomek, y_test_tomek)

        # SMOTE
        x_train_smote, y_train_smote, x_test_smote, y_test_smote = bt.prepare_data(data=self.data,
                                                                                   test_size=0.2,
                                                                                   resampling='smote')
        self.set_data('smote', x_train_smote, y_train_smote, x_test_smote, y_test_smote)

        # BORDERLINE SMOTE
        x_train_bd_smote, y_train_bd_smote, x_test_bd_smote, y_test_bd_smote = bt.prepare_data(data=self.data,
                                                                                               test_size=0.2,
                                                                                               resampling='bdsmote')
        self.set_data('bdsmote', x_train_bd_smote, y_train_bd_smote, x_test_bd_smote, y_test_bd_smote)

        # SMOTEENN
        x_train_smoteenn, y_train_smoteenn, x_test_smoteenn, y_test_smoteenn = bt.prepare_data(data=self.data,
                                                                                               test_size=0.2,
                                                                                               resampling='smoteenn')
        self.set_data('smoteenn', x_train_smoteenn, y_train_smoteenn, x_test_smoteenn, y_test_smoteenn)

        # SMOTETOMEK
        x_train_smotetomek, y_train_smotetomek, x_test_smotetomek, y_test_smotetomek= bt.prepare_data(data=self.data,
                                                                                                      test_size=0.2,
                                                                                                      resampling='smotetomek')
        self.set_data('smotetomek', x_train_smotetomek, y_train_smotetomek, x_test_smotetomek, y_test_smotetomek)

        if generate_table:
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

    def define_dataset_single(self, resampling):
        x_train_origin, y_train_origin, x_test_origin, y_test_origin = bt.prepare_data(
            data=self.data, test_size=0.2, resampling=resampling)
        self.set_data(resampling, x_train_origin, y_train_origin, x_test_origin, y_test_origin)

    def set_data(self, resampling, x_train, y_train, x_test, y_test):
        data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
        if resampling == 'origin':
            self.origin = data
        elif resampling == 'tomek':
            self.tomek = data
        elif resampling == 'smote':
            self.smote = data
        elif resampling == 'bdsmote':
            self.bdsmote = data
        elif resampling == 'smoteenn':
            self.smoteenn = data
        elif resampling == 'smotetomek':
            self.smotetomek = data

    def get_data(self, resampling):
        if resampling == 'origin':
            return self.origin['x_train'], self.origin['y_train'], self.origin['x_test'], self.origin['y_test']
        elif resampling == 'tomek':
            return self.tomek['x_train'], self.tomek['y_train'], self.tomek['x_test'], self.tomek['y_test']
        elif resampling == 'smote':
            return self.smote['x_train'], self.smote['y_train'], self.smote['x_test'], self.smote['y_test']
        elif resampling == 'bdsmote':
            return self.bdsmote['x_train'], self.bdsmote['y_train'], self.bdsmote['x_test'], self.bdsmote['y_test']
        elif resampling == 'smoteenn':
            return self.smoteenn['x_train'], self.smoteenn['y_train'], self.smoteenn['x_test'], self.smoteenn['y_test']
        elif resampling == 'smotetomek':
            return self.smotetomek['x_train'], self.smotetomek['y_train'], self.smotetomek['x_test'], self.smotetomek['y_test']

    def info_data(self, resampling):
        y_train, y_test = 0, 0
        if resampling == 'origin':
            y_train, y_test = self.origin['y_train'], self.origin['y_test']
        elif resampling == 'tomek':
            y_train, y_test = self.tomek['y_train'], self.tomek['y_test']
        elif resampling == 'smote':
            y_train, y_test = self.smote['y_train'], self.smote['y_test']
        elif resampling == 'bdsmote':
            y_train, y_test = self.bdsmote['y_train'], self.bdsmote['y_test']
        elif resampling == 'smoteenn':
            y_train, y_test = self.smoteenn['y_train'], self.smoteenn['y_test']
        elif resampling == 'smotetomek':
            y_train, y_test = self.smotetomek['y_train'], self.smotetomek['y_test']

        bt.plot_requirements_by_class(y_train, y_test)

    def get_tam(self, resampling):
        if resampling == 'origin':
            return self.origin['x_train'].shape[0]+self.origin['x_test'].shape[0]
        elif resampling == 'tomek':
            return self.tomek['x_train'].shape[0] + self.tomek['x_test'].shape[0]
        elif resampling == 'smote':
            return self.smote['x_train'].shape[0] + self.smote['x_test'].shape[0]
        elif resampling == 'bdsmote':
            return self.bdsmote['x_train'].shape[0] + self.bdsmote['x_test'].shape[0]
        elif resampling == 'smoteenn':
            return self.smoteenn['x_train'].shape[0] + self.smoteenn['x_test'].shape[0]
        elif resampling == 'smotetomek':
            return self.smotetomek['x_train'].shape[0] + self.smotetomek['x_test'].shape[0]

