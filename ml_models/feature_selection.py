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
        from sklearn.metrics import balanced_accuracy_score
        modelـstrategy = LinearRegressionModel()
        context = FeatureSelectionContext(modelـstrategy)
        best_predictive_feature = context.predictive_feature(self.data['X_train'])
        columns = self.data['X_train'].columns
        for column in columns:
            context.fit(self.data['X_train'][column].to_frame(), self.data['y_train'])
            
            y_pred = context.predict(self.data['X_train'][column].to_frame())
            exit()
            bal_accuracy = balanced_accuracy_score(pd.Series(self.data['y_train']['Class']), y_pred)
            print(column, ':',bal_accuracy)
            best_predictive_feature[column] = bal_accuracy
        max_key = max(best_predictive_feature, key=best_predictive_feature.get)
        best_predictive_feature[max_key] = best_predictive_feature[max_key]
        exit()
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
        method = 'func1'
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


        from sklearn.metrics import balanced_accuracy_score
        best_predictive_feature = {}
        for column in column_names:
            logistic_model = LogisticRegression(max_iter=100)    
            logistic_model.fit(X_train[column].to_frame(), y_train)
            y_pred = logistic_model.predict(X_test[column].to_frame())
            bal_accuracy = balanced_accuracy_score(y_test, y_pred)
            print(column, ':',bal_accuracy)
            predictive_feature[column] = bal_accuracy
        max_key = max(predictive_feature, key=predictive_feature.get)
        best_predictive_feature[max_key] = predictive_feature[max_key]