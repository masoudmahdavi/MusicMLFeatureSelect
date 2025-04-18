from ml_models.regression_models import LinearRegressionModel, FeatureSelectionContext
import pandas as pd
import numpy as np
from tabulate import tabulate
import logging


class BestFeatures:
    def __init__(self, data:dict, logger:logging.Logger):
        self.data = data
        self.logger = logger
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
        self.log_tabulate_data_shapes(selector, self.logger, self.data['X_train'].columns)
        return selector.scores_

    def select_method(self):
        # method = input("Choose feature selection method (func1/func2): ").strip()
        method = 'func2'
        if method == "func1":
            return self.each_feature_one_model()
        elif method == "func2":
            return self.sklearn_feature_selection()
        else:
            raise ValueError("Invalid method selected. Choose either 'func1' or 'func2'.")
    
    @staticmethod
    def log_tabulate_data_shapes(feature_selector, logger:logging.Logger, features):
        """Log the shapes of the dataframes in a tabular format."""
        print(feature_selector)
        df = pd.DataFrame()
        
        # df.columns = ["Feature", "Feature Score"]
        for feature, score in zip(features, feature_selector.scores_):  
            new_df = pd.DataFrame({'Feature': [feature], 'Feature Score': [score],})
            df = pd.concat([df, new_df], ignore_index=True)
        logger.info('\n'+tabulate(df, headers="keys", tablefmt="pretty"))
        del df, new_df