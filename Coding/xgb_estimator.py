# Basic imports
import pandas as pd
import numpy as np

#ML-model imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Evaluation metrics imports
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


# imports for mutual information feature selection
from sklearn.preprocessing import LabelEncoder

# method to calculate average of scores
def average(lst):
    return sum(lst) / len(lst)


# prepare target
def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return y_enc

def prep_onehot(Dataset):
    ascents_df_XGBoost_model = Dataset.copy()

    # prepare one-hot encoding
    X = ascents_df_XGBoost_model.drop('Ascent Type', axis=1)
    y = pd.Series(ascents_df_XGBoost_model['Ascent Type'])

    # one hot encoding
    cat_columns =['Country','Route Grade','Ascent Grade','Crag Name','Crag Path','Quality']
    X_encoded = pd.get_dummies(X, drop_first = True, columns = cat_columns)
    y_enc = prepare_targets(y)
    return X_encoded, y_enc

def hyperopt_XGB(X_encoded, y_labels):
    RANDOM_STATE = 12345 
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2],
        'max_depth': [2,4,5,6,7,8,9,10],
        'n_estimators': [50,80,100,130,150],
        }

    # Create a based model
    xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, objective='binary:logistic',eval_metric='logloss')
    #
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = xgb_model, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_


def train_eval_XGB(X_encoded, y_labels, best_params):
    #XGBoost approach for classfication

    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)
    
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_encoded, y_labels)

    xgb_cl = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs = -1, objective='binary:logistic', eval_metric = 'logloss',learning_rate = best_params['learning_rate'],max_depth = best_params['max_depth'],n_estimators =best_params['n_estimators'],)
    #xgb_cl = xgb.XGBClassifier(random_state=RANDOM_STATE,n_estimators=250, n_jobs=-1, eval_metric='logloss', subsample = best_params['subsample'],colsample_bytree =best_params['colsample_bytree'], max_depth =best_params['max_depth'], min_child_weight = best_params['min_child_weight'], learning_rate = best_params['learning_rate'])
    scores_accuracy = []
    scores_precision = []
    scores_recall = []
    scores_f1 = []

    for train_index, test_index in sss.split(X_encoded, y_labels):
        X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]
        xgb_cl.fit(X_train, y_train)
        pred = xgb_cl.predict(X_test)
        scores_accuracy.append(accuracy_score(y_test, pred))
        scores_precision.append(precision_score(y_test,pred,pos_label = 1))
        scores_recall.append(recall_score(y_test,pred,pos_label = 1))
        scores_f1.append(f1_score(y_test,pred,pos_label = 1))

    return scores_accuracy,scores_precision,scores_recall,scores_f1


def prep_onehot_reg(Dataset):
    # Simple XGBoost Approach for regression (Quality rating)

    RANDOM_STATE = 12345 #Do not change it!
    np.random.seed(RANDOM_STATE) #Do not change it!

    #ascents_df_MLP_reg_model = cleaned_ascents_df.copy()
    ascents_df_XGB_reg_model = Dataset.copy()

    # We only take the rows that are not nan
    ascents_df_XGB_reg_model = ascents_df_XGB_reg_model[ascents_df_XGB_reg_model['Quality'].notna()]

    # For regression model to work we have to change the quality into a continious variable
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Crap', 'Quality'] = 1
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Don\'t Bother', 'Quality'] = 2
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Average', 'Quality'] = 3
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Good', 'Quality'] = 4
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Very Good', 'Quality'] = 5
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Classic', 'Quality'] = 6
    ascents_df_XGB_reg_model.loc[ascents_df_XGB_reg_model['Quality'] == 'Mega Classic', 'Quality'] = 7

    X = ascents_df_XGB_reg_model.drop('Quality', axis=1)
    y = pd.Series(ascents_df_XGB_reg_model['Quality'])

    X_onehot = pd.get_dummies(data=X, drop_first=True)
    return X_onehot, y

def hyperopt_XGB_reg(X_encoded, y_labels):
    RANDOM_STATE = 12345 
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2],
        'max_depth': [2,4,5,6,7,8,9,10],
        'n_estimators': [50,80,100,130,150],
        }
    # Create a based model
    xgb_model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective='reg:squarederror',eval_metric='logloss')
    
    # Instantiate the grid search model
    #model = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=2, scoring="neg_log_loss")
    grid_search = GridSearchCV(estimator = xgb_model, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_

def train_eval_reg_XGB(X_onehot, y_labels, best_params):
    
    RANDOM_STATE = 12345 #Do not change it!
    np.random.seed(RANDOM_STATE) #Do not change it!

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_onehot, y_labels)

    xgb_r = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective ='reg:squarederror',eval_metric='logloss', learning_rate = best_params['learning_rate'], max_depth = best_params['max_depth'], n_estimators = best_params['n_estimators'])

    scores_MAE = []
    scores_MAPE = []
    try:
        # training model
        for train_index, test_index in sss.split(X_onehot, y_labels):
            X_train, X_test = X_onehot.iloc[train_index], X_onehot.iloc[test_index]
            y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]
            xgb_r.fit(X_train, y_train)
            y_pred = xgb_r.predict(X_test)
            scores_MAE.append(mean_absolute_error(y_true=y_test,y_pred=y_pred))
            scores_MAPE.append(mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred))
        return scores_MAE, average(scores_MAE),min(scores_MAE), scores_MAPE, average(scores_MAPE),min(scores_MAPE)
    except:
        scores_NAN = [0] * 5
        scores_MAPE = [0] * 5
        nan = 'NAN'
        return scores_NAN,nan,nan,scores_MAPE,nan,nan