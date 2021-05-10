import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


class Weigths:
    def get_class_weight(self, data):
        y = data['Class']
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        # balanced -> n_samples / (n_classes * np.bincount(y))
        # The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001
        vect = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.ravel())

        class_weight = {}
        index = 0
        for i in vect:
            class_weight[index] = i
            index += 1

        return class_weight

