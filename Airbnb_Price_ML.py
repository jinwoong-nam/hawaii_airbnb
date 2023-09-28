import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

class ML_regression:

    def __init__(self, df):
        self.df = df
        
    def generate_X_y(self):
        ''' Generate predictor variables 'X' and the target variable 'y' for subsequent ML models
        Args:
        None

        Returns:
        X - matrix of predictor variables
        y - vector of the response variable

        This function follows the steps with the dataframe below to produce X and y:
        1. Categorical variables that have more than 30 categories will NOT be considered for X
        2. For quantitative variables, fill in missing values with the mean values of corresponding "accommodates" (use groupby)
            e.g.) if a row has accommodates=4 and a missing value for 'bathrooms', the mean of 'bathrooms' when accommodates=4 will be filled in.
        3. Perform one-hot encoding for categorical variables including an addition of encoded column for missing values
        4. Extract all categorical and quantitative variables to X matrix
        5. Remove predictor variables from X if they have non-zeros less than cutoff number (=10% of the total)
        6. Take price column as Y vector
        '''

        # 1. Take out the column with categorical variables of more than 30
        cat_vars = df.select_dtypes(include=['object']).copy().columns # select the columns of categorical variables
        for col in cat_vars:
            if df[col].nunique() > 30:
                df = df.drop(col, axis=1) # remove the column if the number of unique values is higher than 30  

        # 2. For quantitative variables, fill in missing values with the mean values of corresponding "accommodates" (use groupby)
        num_vars = df.select_dtypes(include=['float', 'int']).columns # take only quantitative variables columns
        accommodates_groupby = df.groupby('accommodates') # make a groupby object of accommodates
        for col in num_vars:
            null_idx = df[df[col].isnull()].index # for each column taken, take the indexes of null values
            for idx in null_idx:
                df.loc[idx, col] = accommodates_groupby[col].mean()[df.loc[idx, 'accommodates']] # fill in missing values with the mean value of the chosen column with the same "accommodates"   
        
        # 3. Perform one-hot encoding for categorical variables including an addition of encoded column for missing values and drop the original columns
        updated_cat_vars = df.select_dtypes(include=['object']).copy().columns
        for var in updated_cat_vars:
            df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True, dummy_na=True)], axis=1)
        
        # 4. Extract all categorical and quantitative variables to X matrix
        X = df.drop(columns='price').copy()

        # 5. Remove predictor variables with too many zeros
        reduced_X = X.iloc[:, np.where((X.sum() > len(X)*0.1) == True)[0]]

        # 6. Take price column as Y vector
        y = df['price'].copy()

        # Define the attributes
        self.X = reduced_X
        self.y = y

        return reduced_X, y
    

    def perform_rfr(self):
        ''' Do regression with random forest model with X and y obtained from generate_X_y
        Args:
        None

        Returns:
        X_train, X_test, y_train, y_test -  output from sklearn train test split
        rfr_model - model object from sklearn
        error_metrics_dict - dictionary. r2_scores_train, r2_scores_test, mae_train, mae_test
    
        Steps:
        1. Split the data into train and test sets
        2. Initiate a random forest regression model object
        3. Fit the model and obtain the prediction results
        4. Present the error metrics, r-squared and MAE, for the train and test sets
        '''

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 192)

        # Initiate a random forest regression model object
        ## Candidate parameters to be examined 
        params = {
            'rfr__max_depth': [10, 12, 15],
            'rfr__min_samples_leaf': [2, 3, 4],
            'rfr__min_samples_split': [4, 6, 8],
            'rfr__n_estimators': [200]
            }   
        ## Create pipelines for feature scaling and a regressor
        pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('rfr', RandomForestRegressor(random_state=149))
            ])
        ## Grid search cross validation
        rfr_model = GridSearchCV(pipeline, param_grid=params, cv=KFold(5, shuffle=True, random_state = 149))
        
        # Fit the model and obtain the prediction results
        rfr_model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_train_pred = rfr_model.predict(X_train.to_numpy())
        y_test_pred = rfr_model.predict(X_test.to_numpy())

        # Present the error metrics, r-squared and MAE, for the train and test sets
        error_metrics_dict = {'r2_scores_train':r2_score(y_train, y_train_pred),\
                            'r2_scores_test':r2_score(y_test, y_test_pred),\
                            'mae_train': mean_absolute_error(y_train, y_train_pred),\
                            'mae_test': mean_absolute_error(y_test, y_test_pred)}

        # Define the attributes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.rfr_model = rfr_model
        self.error_metrics_dict = error_metrics_dict

        # return X_train, X_test, y_train, y_test, rfr_model, error_metrics_dict
     

def feature_importance(self):
    ''' Analyze features importance and present the results
        Args:
        None

        Returns:
       
    
        Steps:
       
        '''
    df_feature_importance = pd.DataFrame(X_testing.columns, columns=['feature'])