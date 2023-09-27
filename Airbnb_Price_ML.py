import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def generate_X_y(df):
    ''' Generate predictor variables 'X' and the target variable 'y' for subsequent ML models
    Args:
    df - pandas dataframe. The data to be used for following predictive models

    Returns:
    X - matrix of predictor variables
    y - vector of the response variable

    This function follows the steps with the dataframe below to produce X and y:
    1. Categorical variables that have more than 30 categories will NOT be considered for X
    2. For quantitative variables, fill in missing values with the mean values of corresponding "accommodates" (use groupby)
        e.g.) if a row has accommodates=4 and a missing value for 'bathrooms', the mean of 'bathrooms' when accommodates=4 will be filled in.
    3. Perform one-hot encoding for categorical variables including an addition of encoded column for missing values
    4. Extract all categorical and quantitative variables to X matrix
    5. Take price column as Y vector
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

    # 5. Take price column as Y vector
    y = df['price'].copy()

    return X, y
    

def do_rfr(X, y):
     ''' Do regression with random forest model with given X and y
    Args:
    X - matrix of predictor variables
    y - vector of the response variable

    Returns:
    r2_train - float. r-squared value for train set
    r2_test - float. r-squared value for test set
    

    Steps:
    1. Remove predictor variables from X if they have non-zeros less than cutoff number (=10% of the total)
    2. Split the data into train and test sets
    3. Fit the model and obtain the prediction results
    4. Calculate the error metric, r-squared, for the train and test sets
    '''
    
    # # Candidate parameters to be examined 
    # params = {
    #     'rfr__max_depth': [10, 12, 15],
    #     'rfr__min_samples_leaf': [2, 3, 4],
    #     'rfr__min_samples_split': [4, 6, 8],
    #     'rfr__n_estimators': [200]
    #     }
    
    # # Create pipelines for feature scaling and a regressor
    # pipeline = Pipeline([
    #         ('rescale', StandardScaler()),
    #         ('rfr', RandomForestRegressor(random_state=149))
    #     ])

    # # Grid search cross validation
    # cv = GridSearchCV(pipeline, param_grid=params, cv=KFold(5, shuffle=True, random_state = 149))
        
    # return cv
        








def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test

def main():
    df = pd.read_csv('../Part1/stackoverflow/survey_results_public.csv')
    X, y = clean_data(df)
    #cutoffs here pertains to the number of missing values allowed in the used columns.
    #Therefore, lower values for the cutoff provides more predictors in the model.
    cutoffs = [5000, 3500, 2500, 1000, 100, 50, 30, 20, 10, 5]

    r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_mod(X, y, cutoffs, plot=False)
    print('Finished Finding the Best Model')
    return lm_model


if __name__ == '__main__':
    best_model = main()