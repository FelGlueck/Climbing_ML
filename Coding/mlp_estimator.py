# Basic imports
import pandas as pd
import numpy as np

#ML-model imports
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

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
    # copy the data frame either with the cons grade or without it
    ascents_df_MLP_model = Dataset.copy()
    #ascents_df_MLP_model = ascents_method_pred_df.copy()

    # prepare one-hot encoding
    X = ascents_df_MLP_model.drop('Ascent Type', axis=1)
    y_enc = pd.Series(ascents_df_MLP_model['Ascent Type'])

    # one line for split up dates, one without
    cat_columns =['Country','Route Grade','Ascent Grade','Crag Name','Crag Path','Quality']
    X_encoded = pd.get_dummies(X, drop_first = True, columns= cat_columns)
    return X_encoded, y_enc 


def hyperopt_MLP(X_encoded, y_labels):
    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search
    param_grid = {
        "hidden_layer_sizes": [(80,),(100,),(120,),(140,),(160,),(180,),(200,)],
        "activation": ['tanh', 'relu'],
        "solver": ['sgd', 'adam'],
        "alpha": [0.0001, 0.05, 0.07, 0.09],
        "learning_rate": ['constant','adaptive'],
    }
    mlp_gs = MLPClassifier(random_state=RANDOM_STATE,max_iter=2000)
    grid_search = GridSearchCV(estimator = mlp_gs,param_grid = param_grid, n_jobs=-1, cv=2, verbose = 2)
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_

def train_eval_MLP(X_encoded, y_labels, best_params):
    # simple MLP approach for classification (flash/redpoint)

    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_encoded, y_labels)

    #MLP_classifier = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(9,6,2), random_state=RANDOM_STATE, max_iter = 2000)
    MLP_classifier = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=best_params['hidden_layer_sizes'],solver=best_params['solver'], activation=best_params['activation'], learning_rate= best_params['learning_rate'],alpha = best_params['alpha'], max_iter = 2000)
    scores_accuracy = []
    scores_precision = []
    scores_recall = []
    scores_f1 = []

    for train_index, test_index in sss.split(X_encoded, y_labels):
        X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
        y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]
        MLP_classifier.fit(X_train, y_train)
        pred = MLP_classifier.predict(X_test)
        scores_accuracy.append(accuracy_score(y_test, pred))
        scores_precision.append(precision_score(y_test,pred,pos_label = 'Red point'))
        scores_recall.append(recall_score(y_test,pred,pos_label = 'Red point'))
        scores_f1.append(f1_score(y_test,pred,pos_label = 'Red point'))

    return scores_accuracy,scores_precision,scores_recall,scores_f1

def prep_onehot_reg(Dataset):
    # Simple MLP Approach for regression (Quality rating)

    RANDOM_STATE = 12345 #Do not change it!
    np.random.seed(RANDOM_STATE) #Do not change it!

    #ascents_df_MLP_reg_model = cleaned_ascents_df.copy()
    ascents_df_MLP_reg_model = Dataset.copy()

    # We only take the rows that are not nan
    ascents_df_MLP_reg_model = ascents_df_MLP_reg_model[ascents_df_MLP_reg_model['Quality'].notna()]

    # For regression model to work we have to change the quality into a continious variable
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Crap', 'Quality'] = 1
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Don\'t Bother', 'Quality'] = 2
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Average', 'Quality'] = 3
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Good', 'Quality'] = 4
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Very Good', 'Quality'] = 5
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Classic', 'Quality'] = 6
    ascents_df_MLP_reg_model.loc[ascents_df_MLP_reg_model['Quality'] == 'Mega Classic', 'Quality'] = 7

    X = ascents_df_MLP_reg_model.drop('Quality', axis=1)
    y = pd.Series(ascents_df_MLP_reg_model['Quality'])

    X_onehot = pd.get_dummies(data=X, drop_first=True)
    return X_onehot, y

def hyperopt_MLP_reg(X_encoded, y_labels):
    RANDOM_STATE = 12345
    np.random.seed(RANDOM_STATE)
    # Create the parameter grid based on the results of random search
    param_grid = {
        "hidden_layer_sizes": [(80,),(100,),(120,),(140,),(160,),(180,),(200,)],
        "activation": ['tanh', 'relu'],
        "solver": ['sgd', 'adam'],
        "alpha": [0.0001, 0.05, 0.07, 0.09],
        "learning_rate": ['constant','adaptive'],
    }
    mlp_gs = MLPRegressor(random_state=RANDOM_STATE,max_iter=2000)
    grid_search = GridSearchCV(mlp_gs, param_grid, n_jobs=-1, cv=2, verbose = 2)
    grid_search.fit(X_encoded, y_labels)
    return grid_search.best_params_


def train_eval_reg_MLP(X_onehot, y_labels, best_params):
    RANDOM_STATE = 12345 #Do not change it!
    np.random.seed(RANDOM_STATE) #Do not change it!


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,random_state=RANDOM_STATE)
    sss.get_n_splits(X_onehot, y_labels)

    MLP_regressor = MLPRegressor(random_state=RANDOM_STATE,max_iter=2000, solver= best_params['solver'], alpha=best_params['alpha'],hidden_layer_sizes=best_params['hidden_layer_sizes'], activation = best_params['activation'], learning_rate = best_params['learning_rate'])
    #MLP_regressor = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=best_params['hidden_layer_sizes'], random_state=1, max_iter = 1000)
    scores_MAE = []
    scores_MAPE = []
    
    try:
    # training model
        for train_index, test_index in sss.split(X_onehot, y_labels):
            X_train, X_test = X_onehot.iloc[train_index], X_onehot.iloc[test_index]
            y_train, y_test = y_labels.iloc[train_index], y_labels.iloc[test_index]
            MLP_regressor.fit(X_train, y_train)
            y_pred = MLP_regressor.predict(X_test)
            scores_MAE.append(mean_absolute_error(y_true=y_test,y_pred=y_pred))
            scores_MAPE.append(mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred))
        return scores_MAE, average(scores_MAE),min(scores_MAE), scores_MAPE, average(scores_MAPE),min(scores_MAPE)
    except:
        scores_NAN = [0] * 5
        scores_MAPE = [0] * 5
        nan = 'NAN'
        return scores_NAN,nan,nan,scores_MAPE,nan,nan
    
