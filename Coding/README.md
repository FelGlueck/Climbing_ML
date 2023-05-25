## Main coding

The coding is split into several parts

- Data preparation
- Training and evaluation procedure
- ML-models

### Data preparation

The data preparation can be executed with the following python command:

    py data_prep.py

It prepares and cleanes the coded datasets and stores them in the folder 'clean_datasets/class_clean_datasets'
It also prepares a data prep log in 'clean_datasets' which contains information about the dataset sizes and eventuall deleted ascents for which no grade conversion was possible.
    
### Training and evaluation

The training and evaluation procedure has to be executed for each model and experiment manually.
This can be done by adding the experiment to the python execution such as:

    py train_eval.py --experiment 'RF_class'
    py train_eval.py --experiment 'XGB_class'
    py train_eval.py --experiment 'MLP_class'

    py train_eval.py --experiment 'RF_reg'
    py train_eval.py --experiment 'XGB_reg'
    py train_eval.py --experiment 'MLP_reg'

When executed, the procedure will process all dataset within 'clean_datasets/class_clean_datasets'.
It will perform a hyperparameter optimization for each dataset and then evaluate the model with the best found parameters.

For each experiment a result file (.csv) and a log file (.txt) is created in the 'Logs' folder

### ML models

These python files contain the actual ML models as well as encoding methods, called by the traing and evaluation procedure.