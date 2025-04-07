
class BestFeatures:
    def __init__(self, data:dict):
        self.model = self.select_method()
        self.data = data

    def find_best_features(self):
        pass
    def fit(self):
        self.model.fit(self.X, self.y)
        return self.model.feature_importances_
    
    def each_feature_one_model(self):
        return self.model.feature_importances_

    def sklearn_feature_selection(self):
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.X, self.y)
        return selector.scores_
    
    def select_method(self):
        method = input("Choose feature selection method (func1/func2): ").strip()
        if method == "func1":
            return self.each_feature_one_model()
        elif method == "func2":
            return self.sklearn_feature_selection()
        else:
            raise ValueError("Invalid method selected. Choose either 'func1' or 'func2'.")
       