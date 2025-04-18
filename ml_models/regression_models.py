from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from abc import ABC, abstractmethod

class RegressionModelStrategy(ABC):
    """
    Abstract base class for regression model strategies.
    """

    def predictive_feature(self, features):
        """
        Predictive feature selection for regression models.
        """
        column_names = features.columns
        predictive_feature = {}
        for column in column_names:
            predictive_feature[column] = ''

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the regression model on the training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict the target variable for the test data.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the performance of the regression model.
        """
        pass

class LinearRegressionModel(RegressionModelStrategy):
    """
    Concrete implementation of a linear regression model.
    """

    def __init__(self):
        
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    

class FeatureSelectionContext:
    def __init__(self, strategy: RegressionModelStrategy):
        self.strategy = strategy

    def predictive_feature(self, features):
        return self.strategy.predictive_feature(features)

    def fit(self, X_train, y_train):
        return self.strategy.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.strategy.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        return self.strategy.evaluate(y_true, y_pred)