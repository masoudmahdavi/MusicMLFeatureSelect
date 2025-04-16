from ml_models.regression_models import LinearRegressionModel, FeatureSelectionContext
import pandas as pd
import numpy as np

class BestFeatures:
    def __init__(self, data:dict):
        self.data = data
        self.model = self.select_method()

    def select_best_feature(self):
        pass
    
    def each_feature_one_model(self):
        modelـstrategy = LinearRegressionModel()
        context = FeatureSelectionContext(modelـstrategy)
        context.train()
        return self.model.feature_importances_

    def sklearn_feature_selection(self):
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.data['X_train'], np.ravel(self.data['y_train']))
        print(selector.scores_)
        exit()
        return selector.scores_

    def select_method(self):
        method = input("Choose feature selection method (func1/func2): ").strip()
        if method == "func1":
            return self.each_feature_one_model()
        elif method == "func2":
            return self.sklearn_feature_selection()
        else:
            raise ValueError("Invalid method selected. Choose either 'func1' or 'func2'.")
       