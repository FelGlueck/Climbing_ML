# Basic imports
import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import glob
from pathlib import Path
from datetime import datetime
import logging
import argparse
import csv

#ML-model imports
# Own python files with model methods
import rf_estimator as rf_model
import xgb_estimator as xgb_model
import mlp_estimator as mlp_model


# method to calculate average of scores
def average(lst):
    return sum(lst) / len(lst)

# Set up the argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, required=True)
args = parser.parse_args()

# We set the variable for the folder with the Cleaned Datasets for classification task
class_clean_path = "clean_datasets/class_clean_datasets"
class_files = glob.glob(class_clean_path + "/*.csv")

#********************************Classification************************************
# We iterate through the files in the Cleaned Datasets folder
# First we run all the files through the Random Forest and log the results
if args.experiment == 'RF_class':
    i=0
    logging.basicConfig(filename='Logs/RF_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/RF_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'max'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = rf_model.prep_onehot (csv_load_cleaned_dataset_df)
            best_params = rf_model.hyperopt_RF(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_accuracy,scores_precision,scores_recall,scores_f1 = rf_model.train_eval_RF(X_encoded, y_labels, best_params)
            writer.writerow([class_files[i], 'Accuracy', scores_accuracy[0],scores_accuracy[1],scores_accuracy[2],scores_accuracy[3],scores_accuracy[4],average(scores_accuracy),max(scores_accuracy)])
            writer.writerow([class_files[i], 'Precision', scores_precision[0],scores_precision[1],scores_precision[2],scores_precision[3],scores_precision[4],average(scores_precision),max(scores_precision)])
            writer.writerow([class_files[i], 'Recall', scores_recall[0],scores_recall[1],scores_recall[2],scores_recall[3],scores_recall[4],average(scores_recall),max(scores_recall)])
            writer.writerow([class_files[i], 'F1-score', scores_f1[0],scores_f1[1],scores_f1[2],scores_f1[3],scores_f1[4],average(scores_f1),max(scores_f1)])
            logging.info('***********************next dataset****************************')
        file.close()


# Second we run all the files through XGBoost and log the results
if args.experiment == 'XGB_class':
    i = 0
    logging.basicConfig(filename='Logs/XGB_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/XGB_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'max'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = xgb_model.prep_onehot (csv_load_cleaned_dataset_df)
            best_params = xgb_model.hyperopt_XGB(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_accuracy,scores_precision,scores_recall,scores_f1 = xgb_model.train_eval_XGB(X_encoded, y_labels, best_params)
            writer.writerow([class_files[i], 'Accuracy', scores_accuracy[0],scores_accuracy[1],scores_accuracy[2],scores_accuracy[3],scores_accuracy[4],average(scores_accuracy),max(scores_accuracy)])
            writer.writerow([class_files[i], 'Precision', scores_precision[0],scores_precision[1],scores_precision[2],scores_precision[3],scores_precision[4],average(scores_precision),max(scores_precision)])
            writer.writerow([class_files[i], 'Recall', scores_recall[0],scores_recall[1],scores_recall[2],scores_recall[3],scores_recall[4],average(scores_recall),max(scores_recall)])
            writer.writerow([class_files[i], 'F1-score', scores_f1[0],scores_f1[1],scores_f1[2],scores_f1[3],scores_f1[4],average(scores_f1),max(scores_f1)])
            logging.info('***********************next dataset****************************')
        file.close()

    
# Third we run all the files through the MLP and log the results
if args.experiment == 'MLP_class':
    i = 0
    logging.basicConfig(filename='Logs/MLP_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/MLP_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'max'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = mlp_model.prep_onehot(csv_load_cleaned_dataset_df)
            best_params = mlp_model.hyperopt_MLP(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_accuracy,scores_precision,scores_recall,scores_f1 = mlp_model.train_eval_MLP(X_encoded, y_labels, best_params)
            writer.writerow([class_files[i], 'Accuracy', scores_accuracy[0],scores_accuracy[1],scores_accuracy[2],scores_accuracy[3],scores_accuracy[4],average(scores_accuracy),max(scores_accuracy)])
            writer.writerow([class_files[i], 'Precision', scores_precision[0],scores_precision[1],scores_precision[2],scores_precision[3],scores_precision[4],average(scores_precision),max(scores_precision)])
            writer.writerow([class_files[i], 'Recall', scores_recall[0],scores_recall[1],scores_recall[2],scores_recall[3],scores_recall[4],average(scores_recall),max(scores_recall)])
            writer.writerow([class_files[i], 'F1-score', scores_f1[0],scores_f1[1],scores_f1[2],scores_f1[3],scores_f1[4],average(scores_f1),max(scores_f1)])
            logging.info('***********************next dataset****************************')
        file.close()


#**********************************Regression***********************************************
# We run all the files through the RF regressor and log the results
if args.experiment == 'RF_reg':
    i = 0
    logging.basicConfig(filename='Logs/RF_reg_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/RF_reg_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'min'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = rf_model.prep_onehot_reg(csv_load_cleaned_dataset_df)
            best_params = rf_model.hyperopt_RF_reg(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_MAE, average_MAE, min_MAE,scores_MAPE, average_MAPE, min_MAPE = rf_model.train_eval_reg_RF(X_encoded, y_labels,best_params)
            writer.writerow([class_files[i], 'MAE', scores_MAE[0],scores_MAE[1],scores_MAE[2],scores_MAE[3],scores_MAE[4],average_MAE,min_MAE])
            writer.writerow([class_files[i], 'MAPE', scores_MAPE[0],scores_MAPE[1],scores_MAPE[2],scores_MAPE[3],scores_MAPE[4],average_MAPE,min_MAPE])
            logging.info('***********************next dataset****************************')
        file.close()


# We run all the files through the XGB regressor and log the results
if args.experiment == 'XGB_reg':
    i = 0
    logging.basicConfig(filename='Logs/XGB_reg_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/XGB_reg_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'min'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = xgb_model.prep_onehot_reg(csv_load_cleaned_dataset_df)
            best_params = xgb_model.hyperopt_XGB_reg(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_MAE, average_MAE, min_MAE,scores_MAPE, average_MAPE, min_MAPE = xgb_model.train_eval_reg_XGB(X_encoded, y_labels,best_params)
            writer.writerow([class_files[i], 'MAE', scores_MAE[0],scores_MAE[1],scores_MAE[2],scores_MAE[3],scores_MAE[4],average_MAE,min_MAE])
            writer.writerow([class_files[i], 'MAPE', scores_MAPE[0],scores_MAPE[1],scores_MAPE[2],scores_MAPE[3],scores_MAPE[4],average_MAPE,min_MAPE])
            logging.info('***********************next dataset****************************')
        file.close()


# We run all the files through the MLP Regressor and log the results
if args.experiment == 'MLP_reg':
    i = 0
    logging.basicConfig(filename='Logs/MLP_reg_log.txt',filemode='w', encoding='utf-8', level=logging.DEBUG)
    with open("Logs/MLP_reg_log_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Metric', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'average', 'min'])
        for i in range(len(class_files)):
            csv_load_cleaned_dataset_df = pd.read_csv(class_files[i],quotechar = "\"",low_memory=False,delimiter =',')
            logging.info('now: %s', datetime.now())
            logging.info(class_files[i])
            X_encoded, y_labels = mlp_model.prep_onehot_reg(csv_load_cleaned_dataset_df)
            best_params = mlp_model.hyperopt_MLP_reg(X_encoded, y_labels)
            logging.info('Hyperopptimization done: %s',datetime.now())
            logging.info('Best parameters: %s', best_params)
            scores_MAE, average_MAE, min_MAE,scores_MAPE, average_MAPE, min_MAPE = mlp_model.train_eval_reg_MLP(X_encoded, y_labels,best_params)
            writer.writerow([class_files[i], 'MAE', scores_MAE[0],scores_MAE[1],scores_MAE[2],scores_MAE[3],scores_MAE[4],average_MAE,min_MAE])
            writer.writerow([class_files[i], 'MAPE', scores_MAPE[0],scores_MAPE[1],scores_MAPE[2],scores_MAPE[3],scores_MAPE[4],average_MAPE,min_MAPE])
            logging.info('***********************next dataset****************************')
        file.close()
