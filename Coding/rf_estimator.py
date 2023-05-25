# Basic imports
import pandas as pd
import numpy as np

#ML-model imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Evaluation metrics imports
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error



# method to calculate average of scores
def average(lst):
    return sum(lst) / len(lst)


def prep_onehot(Dataset):
    #ascents_df_rf_model = ascents_method_pred_df.copy()
    ascents_df_rf_model = Dataset.copy()

    # prepare one-hot encoding
    X = ascents_df_rf_model.drop('Ascent Type', axis=1)
    y = pd.Series(ascents_df_rf_model['Ascent Type'])

    # one hot encoding
    cat_columns =['Country','Route Grade','Ascent Grade','Crag Name','Crag Path','Quality']
    X_encoded = pd.get_dummies(X, drop_first = True, columns = cat_columns)
    return X_encoded, y

def hyperopt_RF(X_encoded, y_labels):
    RANDOM_STATE = 12345 
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [60, 70, 80],
        'max_features': [7,8,9],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [12, 14, 16],
        'n_estimators': [350, 400, 450, 500, 550 ]
    }
    # Create a based model
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_

def train_eval_RF(X_encoded, y_labels, best_params):
    RANDOM_STATE = 12345 
    np.random.seed(RANDOM_STATE)
    print(best_params)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_encoded, y_labels)

    rf = RandomForestClassifier(random_state=RANDOM_STATE,bootstrap = True,max_depth=best_params['max_depth'],max_features=best_params['max_features'],min_samples_leaf=best_params['min_samples_leaf'],min_samples_split=best_params['min_samples_split'],n_estimators=best_params['n_estimators'])
    scores_accuracy = []
    scores_precision = []
    scores_recall = []
    scores_f1 = []

    for train_index, test_index in sss.split(X_encoded, y_labels):
        X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
        y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        scores_accuracy.append(accuracy_score(y_test, pred))
        scores_precision.append(precision_score(y_test,pred,pos_label = 'Red point'))
        scores_recall.append(recall_score(y_test,pred,pos_label = 'Red point'))
        scores_f1.append(f1_score(y_test,pred,pos_label = 'Red point'))

    # If time available output should generally be reworked to be in a better format.
    # Like this bringing it into a nice table form is a bit of manual work.
    # print the scores
    return scores_accuracy,scores_precision,scores_recall,scores_f1


    

def prep_onehot_reg(Dataset):
    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)

    ascents_df_RF_reg_model = Dataset.copy()
    #print(ascents_df_RF_reg_model.dtypes)

    # We only take the rows that are not nan
    ascents_df_RF_reg_model = ascents_df_RF_reg_model[ascents_df_RF_reg_model['Quality'].notna()]

    # For regression model to work we have to change the quality into a continious variable
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Crap', 'Quality'] = 1
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Don\'t Bother', 'Quality'] = 2
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Average', 'Quality'] = 3
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Good', 'Quality'] = 4
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Very Good', 'Quality'] = 5
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Classic', 'Quality'] = 6
    ascents_df_RF_reg_model.loc[ascents_df_RF_reg_model['Quality'] == 'Mega Classic', 'Quality'] = 7

    X = ascents_df_RF_reg_model.drop('Quality', axis=1)
    y = pd.Series(ascents_df_RF_reg_model['Quality'])

    cat_columns =['Route Grade','Ascent Grade','Ascent Type','Country','Crag Name','Crag Path']

    X_onehot = pd.get_dummies(data=X, drop_first=True,columns = cat_columns)
    return X_onehot, y

def hyperopt_RF_reg(X_encoded, y_labels):
    RANDOM_STATE = 12345 
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [60, 70, 80],
        'max_features': [7,8,9],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [12, 14, 16],
        'n_estimators': [350, 400, 450, 500, 550 ]
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_




def train_eval_reg_RF(X_onehot, y_labels, best_params):
    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)
    # RF Regressor approach (Quality rating)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_onehot, y_labels)

    reg_RF = RandomForestRegressor(random_state=RANDOM_STATE,bootstrap =best_params['bootstrap'], max_depth= best_params['max_depth'],max_features = best_params['max_features'],min_samples_leaf = best_params['min_samples_leaf'],min_samples_split = best_params['min_samples_split'],n_estimators = best_params['n_estimators'])
    scores_MAE = []
    scores_MAPE = []
    # training model
    
   
    try:
        for train_index, test_index in sss.split(X_onehot, y_labels):
            X_train, X_test = X_onehot.iloc[train_index], X_onehot.iloc[test_index]
            y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]
            reg_RF.fit(X_train, y_train)
            y_pred = reg_RF.predict(X_test)
            scores_MAE.append(mean_absolute_error(y_true=y_test,y_pred=y_pred))
            scores_MAPE.append(mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred))
        return scores_MAE, average(scores_MAE),min(scores_MAE), scores_MAPE, average(scores_MAPE),min(scores_MAPE)
    except:
        scores_NAN = [0] * 5
        scores_MAPE = [0] * 5
        nan = 'NAN'
        return scores_NAN,nan,nan,scores_MAPE,nan,nan
    


    