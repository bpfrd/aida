import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
import time
import joblib
import os
import argparse
import sklearn
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(os.getenv('ROOT', '.')).expanduser()

# import  sys
# sys.exit(0)

def load_data(filepath, return_groups=False):
    with open(filepath, 'rb') as f:
        data = np.load(f)
        if return_groups:
            return data[:, :-2], data[:, -2], data[:, -1]
        else:
            return data[:, :-1], data[:, -1]

def get_param_grids():
    return {
        'dt': {
            'model': DecisionTreeClassifier(),
            'decisiontreeclassifier__criterion': ['gini', 'entropy'],
            'decisiontreeclassifier__splitter': ['best', 'random'],
            'decisiontreeclassifier__max_depth': [None, 2, 4, 8, 16],
            'decisiontreeclassifier__min_samples_split': [2, 5, 10],
            'decisiontreeclassifier__min_samples_leaf': [1, 2, 4],
            'decisiontreeclassifier__max_features': [None, 'sqrt', 'log2'],
         },
        # 'rf': {
        #     'model': RandomForestClassifier(),
        #     'randomforestclassifier__n_estimators': [100, 200, 400],
        #     'randomforestclassifier__max_depth': [None, 2, 4, 8, 16],
        #     'randomforestclassifier__min_samples_split': [2, 5, 10],
        #     'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        #     'randomforestclassifier__max_features': ['sqrt', 'log2'],
        #     'randomforestclassifier__class_weight': [None, 'balanced', 'balanced_subsample']
        # },
        # 'mlp': {
        #     'model': MLPClassifier(max_iter=1000),
        #     'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        #     'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
        #     'mlpclassifier__solver': ['adam', 'sgd'],
        #     'mlpclassifier__alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        #     'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
        #     'mlpclassifier__early_stopping': [True, False],
        # },
        'lr': {
            'model': LogisticRegression(),
            'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0],  
            'logisticregression__penalty': ['l2'], 
            'logisticregression__class_weight': [None, 'balanced'],    
        },
        # 'nb': {
        #     'model': GaussianNB(),
        #     'gaussiannb__priors': [None, (0.1, 0.9), (0.9, 0.1)],
        #     'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
        # },
        # 'knn': {
        #     'model': KNeighborsClassifier(),
        #     'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
        #     'kneighborsclassifier__weights': ['uniform', 'distance'],
        #     'kneighborsclassifier__p': [1, 2],
        #     'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        #     'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        # },
        # 'gbm': {
        #     'model': GradientBoostingClassifier(random_state=42),
        #     'gradientboostingclassifier__n_estimators': [100, 200],
        #     'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
        #     'gradientboostingclassifier__max_depth': [3, 5, 7],
        #     'gradientboostingclassifier__min_samples_split': [2, 5, 10],
        #     'gradientboostingclassifier__min_samples_leaf': [1, 2, 4],
        #     'gradientboostingclassifier__subsample': [0.8, 0.9, 1.0],
        #     'gradientboostingclassifier__max_features': [None, 'sqrt', 'log2'],
        # },
        # 'xgb': {
        #     'model': XGBClassifier(),
        #     'xgbclassifier__n_estimators': [100, 500, 1000],
        #     'xgbclassifier__max_depth': [3, 5, 7],
        #     'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
        #     'xgbclassifier__gamma': [0, 0.1, 0.2],
        #     'xgbclassifier__subsample': [0.8, 0.9, 1.0],
        #     'xgbclassifier__colsample_bytree': [0.8, 0.9, 1.0],
        #     'xgbclassifier__scale_pos_weight': [9],
        #     'xgbclassifier__reg_alpha': [0.1, 0.5, 1.0],
        #     'xgbclassifier__reg_lambda': [0.1, 0.5, 1.0]
        # },
        # 'svc': {
        #     'model': SVC(probability=True),
        #     'svc__kernel': ['rbf'],
        #     'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        #     'svc__class_weight': [None, 'balanced'],
        #     'svc__C':[0.1, 1.0, 10, 100]
        # },
    }

def save_model(model, name, root=ROOT):
    models_dir = os.path.join(root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f'{name}.pkl')
    joblib.dump(model, filepath)
    logging.info(f'Saved the trained model "{name}" to file "{filepath}".')
   
def report_performance(model, X_train, y_train, X_test, y_test, name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    logging.info(f'Metrics for {name}')
    logging.info(f'TRAIN:\n{classification_report(y_true=y_train, y_pred=y_train_pred)}')
    logging.info(f'TEST:\n{classification_report(y_true=y_test, y_pred=y_test_pred)}')
    logging.info(f'Balanced accuracy for test set: {balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)}')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f'sklearn version: {sklearn.__version__}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""training machine learning models DT, RF, SVC, MLP, NB, LR, KNN, GBM, XGB, and voting classifier... """)
    parser.add_argument('--cv', type=int, default=4, help='sklearn GridSearchCV param: cv')
    parser.add_argument('--n_jobs', type=int, default=-1, help='sklearn GridSearchCV param: n_jobs')
    parser.add_argument('--verbose', type=int, default=5, help='sklearn GridSearchCV param: verbose')
    parser.add_argument('--scoring', default='roc_auc', help='sklearn GridSearchCV param: scoring')
    args = parser.parse_args()

    # Reading data
    logging.info('Reading dataset...')
    X_train, y_train, groups = load_data(os.path.join(ROOT, 'data', 'preprocessed', 'train-data.npy'), return_groups=True)
    X_test, y_test = load_data(os.path.join(ROOT, 'data', 'preprocessed', 'test-data.npy'))

    logging.info(f'{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}')

    param_grids = get_param_grids()
    best_estimators = []
    for clf_name, param_grid in param_grids.items():
        start = time.time()
        logging.info(f'Training started for {clf_name}...')
        
        clf = param_grid.pop('model')
        estimator = make_pipeline(StandardScaler(), clf)
        cv = StratifiedGroupKFold(n_splits=args.cv, shuffle=True, random_state=42)
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=args.scoring, verbose=args.verbose, n_jobs=args.n_jobs)
        gs.fit(X_train, y_train, groups=groups)

        best_estimator = gs.best_estimator_
        best_estimators.append((clf_name, best_estimator))
        logging.info(f'Best params for {clf_name}: {best_estimator}')

        save_model(best_estimator, f'{clf_name}', ROOT)

        report_performance(best_estimator, X_train, y_train, X_test, y_test, clf_name)

        end = time.time()
        logging.info(f'Training finished in {end - start:.2f} seconds.')

    voting_classifier = VotingClassifier(estimators=best_estimators)
    voting_classifier.fit(X_train, y_train)
    report_performance(voting_classifier, X_train, y_train, X_test, y_test, 'voting_classifier')
    save_model(voting_classifier, 'voting_classifier', ROOT)
    
    logging.info('All done.')