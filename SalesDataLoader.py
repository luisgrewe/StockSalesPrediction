import pandas as pd
import numpy as np

class SalesDataLoader:
    def __init__(self, train_path='data/df_stock_sales_train.xlsx', test_path='data/df_stock_sales_test.xlsx', 
                 date_col='week', target_col='sales', 
                 id_cols=['store_number', 'product_number']):

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
        """Loads data."""
        train_df = pd.read_excel(self.__train_path, parse_dates=[self.__date_col])
        test_df = pd.read_excel(self.__test_path, parse_dates=[self.__date_col])

        return {'train': train_df, 'test': test_df}

    def preprocessing(self, df):
        """Handles time feature extraction, missing values, and sorting."""
        df = df.copy()
        df[self.__date_col] = pd.to_datetime(df[self.__date_col])

        # Sort to ensure time-series consistency
        df = df.sort_values(by=self.__id_cols + [self.__date_col])

        #Time-Series Consistency
        df[self.__date_col] = df[self.__date_col] - pd.to_timedelta(df[self.__date_col].dt.weekday, unit='D')

        # Time Features
        df['month'] = df[self.__date_col].dt.month
        df['week_of_year'] = df[self.__date_col].dt.isocalendar().week.astype(int)
        df['year'] = df[self.__date_col].dt.isocalendar().year.astype(int)

        df[self.__target_col] = df[self.__target_col].fillna(0)

        return df.reset_index(drop=True)

    def feature_engineering(self, df):
        """Generates features that are viable for a 15-week forecast horizon."""
        df = df.copy()

        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        # Global Time Trend
        min_date = df[self.__date_col].min()
        df['weeks_since_start'] = ((df[self.__date_col] - min_date).dt.days // 7)

        # Product Level Lags
        grouper = df.groupby(['store_number', 'product_number'])[self.__target_col]
        df['sales_lag_15'] = grouper.shift(15)
        df['sales_lag_16'] = grouper.shift(16)
        df['sales_lag_52'] = grouper.shift(52)

        # Rolling Window
        df['rolling_mean_4_wks'] = grouper.transform(lambda x: x.shift(15).rolling(window=4).mean())
        df['rolling_std_4_wks'] = grouper.transform(lambda x: x.shift(15).rolling(window=4).std())

        # Momentum compares 4-week trend to 12-week baseline
        df['rolling_mean_12_wks'] = grouper.transform(lambda x: x.shift(15).rolling(window=12).mean())
        df['velocity'] = df['rolling_mean_4_wks'] / (df['rolling_mean_12_wks'] + 1e-6)

        # Seasonal Averages
        df['store_seasonal_avg'] = df.groupby(['store_number', 'week_of_year'])[self.__target_col].transform(
            lambda x: x.shift(15).mean()
        )
        df['cat_seasonal_avg'] = df.groupby(['product_category', 'week_of_year'])[self.__target_col].transform(
            lambda x: x.shift(15).mean()
        )

        # Department-level seasonality
        df['dept_seasonal_avg'] = df.groupby(['department_group', 'week_of_year'])[self.__target_col].transform(
            lambda x: x.shift(15).mean()
        )

        df['sales_log_return'] = grouper.transform(
            lambda x: np.log1p(x.shift(15)) - np.log1p(x.shift(16))
        )

        # Sales Product Share per Store in a Week
        store_weekly_total = df.groupby(['store_number', self.__date_col])[self.__target_col].transform(
            lambda x: x.shift(15).sum())

        df['prod_store_share'] = df['sales_lag_15'] / (store_weekly_total + 1e-6)

        # Relative Performance
        df['perf_vs_store'] = df['sales_lag_15'] / (df['store_seasonal_avg'] + 1e-6)

        # Year-over-Year Growth Rate
        df['yoy_growth'] = df['sales_lag_15'] / (df['sales_lag_52'] + 1e-6)

        return df

    def prepare_test_features(self, df_train, df_test):
        """
        Safely generates features for the test set by looking back
        into the training history for lags and rolling windows.
        """
        # Get the raw data
        df_train = df_train.copy()
        df_test = df_test.copy()

        # Add marker and Combine
        df_train['is_test'] = False
        df_test['is_test'] = True

        df_test[self.__target_col] = np.nan

        full_df = pd.concat([df_train, df_test], axis=0)

        full_df = self.preprocessing(full_df)

        full_df_engineered = self.feature_engineering(full_df)

        test_final = full_df_engineered[full_df_engineered['is_test'] == True].copy()

        test_final = test_final.drop(columns=['is_test', 'sales'])

        return test_final

                