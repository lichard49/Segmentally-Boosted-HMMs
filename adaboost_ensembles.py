from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaboostEnsembles:
    # creates n_classes number of ensembles
    # X : numpy array of size n_samples, n_features
    # Y: numpy array of size n_samples,
    def fit(self,X, y,n_estimators = 20):
        # create n_classes number of ensembles
        le = LabelEncoder()
        y_new = le.fit_transform(y)
        self.ensemble_list = []
        for class_label in le.classes_:
            y_class_label = y_new == le.transform([class_label])
            y_class_label *= 1 # hack to convert boolean array into numeric array
            # train ensemble for this
            ensemble = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=DecisionTreeClassifier(max_depth=1))
            ensemble.fit(X,y_class_label)
            self.ensemble_list.append(ensemble)

    def ensemble_scores(self,X):
        scores = np.zeros(len(self.ensemble_list))
        for ensemble,i in enumerate(self.ensemble_list):
            scores[i] = ensemble.decision_function(X)
        return scores





