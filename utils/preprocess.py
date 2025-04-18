import pandas as pd
from model.model import Model
import logging
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from ml_models.feature_selection import BestFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



class Preprocess:
    def __init__(self, model:Model, logger:logging.Logger):
        self.model = model
        self.logger = logger
        self.logger.info("Preprocess class initialized.")
        self.raw_data = self.load_csv()
        self._describe_data()
    
    def _describe_data(self):
        """Print the description of the data."""
        null_counts = self.raw_data.isnull().sum().reset_index()
        null_counts.columns = ["Column Name", "Null Rows"]
        null_counts['type'] = null_counts['Column Name'].apply(lambda x: self.raw_data[x].dtype)
        self.logger.info('\n'+tabulate(null_counts, headers="keys", tablefmt="pretty"))
        
    def preprocess_raw_data(self):
        """Preprocess the raw data.
        This includes loading the data, handling missing values, normalizing data, and splitting into train/test sets.
        """
        self.logger.info("Starting data preprocessing...")
        
        self.logger.info("Removeing duplicate rows")
        self.raw_data = self.raw_data.drop_duplicates()
        
        self.logger.info("Splitting data into features and target variable.")
        features, target = self.split_featres_target()
       
        self.logger.info("Splitting data into train and test sets.")
        preprocessed_data = self.split_train_test(features, target)
      
        self.logger.info("Handling missing values.")
        preprocessed_data = self.fill_miss_data(preprocessed_data)
        
        self.logger.info("Normalizing data.")
        preprocessed_data['X_train'] = self.norm_num_data(preprocessed_data['X_train'], norm_method='Standard') #'min_max' or 'Standard'
        preprocessed_data['X_test'] = self.norm_num_data(preprocessed_data['X_train'], norm_method='Standard') #'min_max' or 'Standard'

        # combined_normiaized_text_df = combine_norm_and_text(normalized_df, handled_text_df)
        best_feature_selection_obj = BestFeatures(preprocessed_data, self.logger)
        

        self.logger.info("Data preprocessing completed.")
        return preprocessed_data

    def split_featres_target(self):
        """Split the data into features and target variable.

        Returns:
            tuple: A tuple containing features and target variable.
        """
        features = self.raw_data.drop(columns=['Class'])
        target = self.raw_data['Class']
        self.logger.info("Data split into features and target variable.")
        target = self.one_hot_encode_data(target)
        
        return features, target
    
    def one_hot_encode_data(self, data:pd.Series) -> pd.DataFrame:
        """This is used to convert categorical data into numerical data.

        Args:
            data (pd.Series): The data to be converted

        Returns:
            pd.DataFrame: Dataframe without text format columns.
        """
        data = data.to_frame(name='Class')
        handled_text_df = self.text_encoder(data,
                                method='ordinal_encoder', # 'one_hot_encoder' or 'ordinal_encoder'
                          )

        return handled_text_df
        
        
    def text_encoder(self, dataframe:pd.DataFrame, method='one_hot_encoder') -> pd.DataFrame:
        """Handels texts in dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe with text columns
            method (str, optional): Method of how to counter with texts. 
                                    Defaults to 'one_hot_encoder'.

        Returns:
            pd.DataFrame: Dataframe withouadd_encoded_to_dft text format columns.
        """
        music_class = dataframe[["Class"]]

        if method == 'one_hot_encoder':
                encoded_df = self.one_hot_encoder(music_class)
                encoded_df = self.add_encoded_to_df(encoded_df, dataframe)
              
        elif method == 'ordinal_encoder':
                encoded_df = self.ordinal_encoder(music_class)
                encoded_df = self.add_encoded_to_df(encoded_df, dataframe)
        return encoded_df

    def norm_num_data(self, num_dataframe:pd.DataFrame, norm_method:str):    
        # data_labels = num_dataframe["median_house_value"].copy()
        # num_dataframe = self._drop_income_cat(num_dataframe, "median_house_value")
        if norm_method == "min_max":
            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
            norm_df = min_max_scaler.fit_transform(num_dataframe)
    
        elif norm_method == "Standard":
            std_scaler = StandardScaler()
            norm_df = std_scaler.fit_transform(num_dataframe)
        
        df = pd.DataFrame(norm_df, columns=num_dataframe.columns)
        # df["median_house_value"] = data_labels
        return df
    
    def one_hot_encoder(self, music_class):
        one_hot_encoder = OneHotEncoder()
        encoded_cat = one_hot_encoder.fit_transform(music_class)
        csr_encoded_cat = pd.DataFrame(encoded_cat, columns=music_class.columns,
                                        index=music_class.index)
        return csr_encoded_cat

    def add_encoded_to_df(self, encoded_df:pd.DataFrame, dataframe:pd.DataFrame) -> pd.DataFrame:
        dataframe['Class'] = encoded_df
        # dataframe['dense_matrix'] = dataframe['ocean_proximity'].apply(lambda x: x.toarray())
        return dataframe
    
    def ordinal_encoder(self, music_class):
        ordinal_encoder = OrdinalEncoder()
        encoded_cat = ordinal_encoder.fit_transform(music_class)
        csr_encoded_cat = pd.DataFrame(encoded_cat, columns=music_class.columns,
                                        index=music_class.index)
        return csr_encoded_cat
    
    def fill_miss_data(self, data):
        """Fill missing data in the dataframe.

        Args:
            data (pd.DataFrame): Dataframe containing data

        Returns:
            pd.DataFrame: Dataframe with missing values filled
        """
        for key in data.keys():
            for column in data[key].columns:
                if data[key][column].isnull().sum() > 0:
                    data[key][column].fillna(data[key][column].mean(), inplace=True)

        self.logger.info("Missing values filled.")
        return data


    def split_train_test(self, features:pd.DataFrame, target:pd.DataFrame) -> dict:
        def log_tabulate_data_shapes(dict_data):
            """Log the shapes of the dataframes in a tabular format."""
            df = pd.DataFrame()
            for key, value in dict_data.items():
                new_df = pd.DataFrame({'shape': [value.shape]}, index=[key])
                df = pd.concat([df, new_df], ignore_index=False)
            self.logger.info('\n'+tabulate(df, headers="keys", tablefmt="pretty"))
            del df, new_df

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.logger.info("Data split into train and test sets.")
        dict_data = {'X_train':X_train,
                     'X_test':X_test,
                     'y_train':y_train, 
                     'y_test':y_test}
        log_tabulate_data_shapes(dict_data)
        self.logger.info("Data split completed.")
        return dict_data

    def load_csv(self) -> pd.DataFrame:
        """Read data from csv file.

        Raises:
            ValueError: If data is empty

        Returns:
            pd.DataFrame: Dataframe containing data
        """
        data = pd.read_csv(self.model.data_path)
        if data.empty:
            raise ValueError("Data is empty")
        return data