import pandas as pd
import numpy as np

class SalesDataLoader:
    def __init__(self, train_path='data/df_stock_sales_train.xlsx', test_path='data/df_stock_sales_test.xlsx', 
                 date_col='week', target_col='sales', 
                 id_cols=['store_number', 'product_number']):
        # Private attributes
        self.__train_path = train_path
        self.__test_path = test_path
        self.__date_col = date_col
        self.__target_col = target_col
        self.__id_cols = id_cols

    def get_train_path(self): return self.__train_path
    def get_test_path(self): return self.__test_path
    def get_date_col(self): return self.__date_col
    def get_target_col(self): return self.__target_col
    def get_id_cols(self): return self.__id_cols

    def load_raw_data(self):
        """Loads data and identifies categorical features."""
        train_df = pd.read_excel(self.__train_path, parse_dates=[self.__date_col])
        test_df = pd.read_excel(self.__test_path, parse_dates=[self.__date_col])

        return {'train': train_df, 'test': test_df}
        
    def preprocessing(self, df):
        """Handles time feature extraction, missing values, and sorting."""
        df = df.copy()
        df[self.__date_col] = pd.to_datetime(df[self.__date_col])
        df = df.sort_values(by=self.__id_cols + [self.__date_col])

        df['month'] = df[self.__date_col].dt.month
        df['week_of_year'] = df[self.__date_col].dt.isocalendar().week.astype(int)
        df['year'] = df[self.__date_col].dt.year

        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        grouper = df.groupby(['store_number', 'product_number'])['sales']
        df['sales_last_week'] = grouper.shift(1) 
        df['sales_15_weeks'] = grouper.shift(15)

        df[self.__target_col] = df[self.__target_col].fillna(0)
        

        return df.reset_index(drop=True)