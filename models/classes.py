import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

from keras.wrappers.scikit_learn import KerasClassifier


class KerasModel:
    def __init__(self,
                 input_dim,
                 nn1,
                 nn2,
                 output_dim,
                 dropout,
                 l1,
                 l2,
                 act,
                 act_out,
                 optimizer_name,
                 learn_rate,
                 batch_size,
                 verbose,
                 n_epochs):

        self.input_dim = input_dim
        self.nn1 = nn1
        self.nn2 = nn2
        self.output_dim = output_dim

        self.act = act
        self.act_out = act_out

        self.optimizer_name = optimizer_name
        self.learn_rate = learn_rate

        self.batch_size = batch_size
        self.verbose = verbose

        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2

        self.loss_fn = 'categorical_crossentropy'
        self.n_epochs = n_epochs

        self.model = KerasClassifier(build_fn=self.create_net, verbose=0)

    def create_net(self):

        opt = None

        if self.optimizer_name == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.learn_rate, beta_1=0.9, beta_2=0.999)
        elif self.optimizer_name == 'adamax':
            opt = keras.optimizers.Adamax(learning_rate=self.learn_rate, beta_1=0.9, beta_2=0.999)
        elif self.optimizer_name == 'adagrad':
            opt = keras.optimizers.Adagrad(learning_rate=self.learn_rate)
        elif self.optimizer_name == 'adadelta':
            opt = keras.optimizers.Adadelta(learning_rate=self.learn_rate)
        elif self.optimizer_name == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=self.learn_rate)
        elif self.optimizer_name == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.learn_rate, momentum=0.0)
        elif self.optimizer_name == 'sgdm':
            opt = keras.optimizers.SGD(learning_rate=self.learn_rate, momentum=0.9)
        else:
            print('ERROR: Invalid name!')

        reg = keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)

        model = Sequential()
        model.add(Dense(self.nn1, input_dim=self.input_dim, activation=self.act, kernel_regularizer=reg))
        model.add(Dropout(self.dropout))

        if self.nn2 != 0:
            model.add(Dense(self.nn2, activation=self.act, kernel_regularizer=reg))
            model.add(Dropout(self.dropout))

        model.add(Dense(self.output_dim, activation=self.act_out))

        model.compile(loss=self.loss_fn, optimizer=opt, metrics=['accuracy'])

        self.model = model

        return model

    def summary(self):
        self.model.summary()

    def fit(self, x_train, y_train, x_val, y_val, patience=5, weights=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=patience)
        #  mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=self.verbose,
        #  save_best_only=True)
        if weights is None:
            history = self.model.fit(x_train, y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.n_epochs,
                                     verbose=self.verbose,
                                     validation_data=(x_val, y_val),
                                     callbacks=[es])
        else:
            history = self.model.fit(x_train, y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.n_epochs,
                                     verbose=self.verbose,
                                     validation_data=(x_val, y_val),
                                     callbacks=[es],
                                     class_weight=weights)

            # saved_model = load_model('best_model.h5')

        return history

    def evaluate(self, x_test, y_test):
        _, ac = self.model.evaluate(x_test, y_test, verbose=self.verbose)
        return _, ac

    def predict(self, x_test):
        y_pred = self.model.predict(x_test, batch_size=self.batch_size, verbose=self.verbose)
        y_pred_bool = np.argmax(y_pred, axis=1)
        return y_pred_bool

    def clone_params(self, input_dim):
        return KerasModel(input_dim=input_dim,
                          nn1=self.nn1,
                          nn2=self.nn2,
                          output_dim=self.output_dim,
                          dropout=self.dropout,
                          l1=self.l1,
                          l2=self.l2,
                          act=self.act,
                          act_out=self.act_out,
                          optimizer_name=self.optimizer_name,
                          learn_rate=self.learn_rate,
                          batch_size=self.batch_size,
                          verbose=self.verbose,
                          n_epochs=self.n_epochs)


class Resampling:

    def __init__(self, name):
        self.strategie = None
        self.name = name

        if name == "ENN":
            self.strategie = EditedNearestNeighbours(sampling_strategy='auto',
                                                     n_neighbors=3,
                                                     kind_sel='all',
                                                     n_jobs=-1)
        elif name == "AllKnn":
            self.strategie = AllKNN(sampling_strategy='auto',
                                    n_neighbors=3,
                                    kind_sel='all',
                                    allow_minority=False,
                                    n_jobs=-1)
        elif name == "RENN":
            self.strategie = RepeatedEditedNearestNeighbours(sampling_strategy='auto',
                                                             n_neighbors=3,
                                                             max_iter=100,
                                                             kind_sel='all',
                                                             n_jobs=-1)

        elif name == "TomekLinks":
            self.strategie = TomekLinks(sampling_strategy='auto',
                                        n_jobs=-1)

        elif name == "SMOTE":
            self.strategie = SMOTE(sampling_strategy='auto',
                                   k_neighbors=5,
                                   n_jobs=-1,
                                   random_state=42)

        elif name == "BorderlineSMOTE":
            self.strategie = BorderlineSMOTE(random_state=42, n_jobs=-1,)

        elif name == "ADASYN":
            self.strategie = ADASYN(sampling_strategy='auto',
                                    n_neighbors=5,
                                    n_jobs=-1,
                                    random_state=42)

        elif name == "SMOTEENN":
            self.strategie = SMOTEENN(sampling_strategy='auto',
                                      smote=None,
                                      enn=None,
                                      n_jobs=-1,
                                      random_state=42)

        elif name == "SMOTETomek":
            self.strategie = SMOTETomek(sampling_strategy='auto',
                                        smote=None,
                                        tomek=None,
                                        n_jobs=-1,
                                        random_state=42)

    def fit_resample(self, x, y):
        x_res, y_res = self.strategie.fit_resample(x, y)
        return x_res, y_res

