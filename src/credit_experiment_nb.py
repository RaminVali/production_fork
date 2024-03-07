from sacred import Experiment
from sacred.observers import SqlObserver # allows connection to db

import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, train_test_split
import os
from dotenv import load_dotenv
# importing module to load environment variables
from logger import get_logger


from credit_preproc_ingredient import preproc_ingredient, get_column_transformer
from credit_data_ingredient import data_ingredient, load_data
from credit_db_ingredient import db_ingredient, df_to_sql
load_dotenv() # loading envrionment variables in a python script as opposed ot a notebook


db_url = os.getenv('DB_URL') # from envrionment file

_logs = get_logger(__name__) # our own standard logger
ex  = Experiment("Credit Experiment",
                 ingredients=[data_ingredient, preproc_ingredient, db_ingredient])

ex.logger = _logs
ex.observers.append(SqlObserver(db_url)) #appending the database observer

@ex.config # decorators come from sacred
def cfg():
    '''
    Config function: all variables here will be avialable in captured functions.
    '''
    preproc_pipe = "power"
    folds = 5
    scoring = ['neg_log_loss', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']

    

@ex.capture
def get_pipe(preproc_pipe):
    '''
    Returns an NB pipeline.
    '''
    _logs.info(f'Getting Naive Bayes Pipeline')
    ct = get_column_transformer(preproc_pipe)
    pipe = Pipeline(
        steps  = [
            ('preproc', ct),
            ('clf', GaussianNB())
        ]
    )
    return pipe


@ex.capture
def evaluate_model(pipe, X, Y, folds, scoring, _run): #_run is a sacred variable  
    '''Evaluate model using corss validation.'''
    _logs.info(f'Evaluating model')
    res_dict = cross_validate(pipe, X, Y, cv = folds, scoring = scoring)
    res = (pd.DataFrame(res_dict)
           .reset_index()
           .rename(columns={'index': 'fold'})
           .assign(run_id = _run._id))
    return res

@ex.capture
def res_to_sql(res):
    '''Write results to db.'''
    _logs.info(f'Writing results to db')
    df_to_sql(res, "model_cv_fold_results")
    
    df_to_sql(res.groupby('run_id', group_keys=False).mean(), "model_cv_results")

@ex.automain # not the capture function anymore. but the automain
def run():
    '''Main experiment run.'''
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    res = evaluate_model(pipe, X_train, Y_train)   
    res_to_sql(res)
   
if __name__=="__main__":
    ex.run_commandline()