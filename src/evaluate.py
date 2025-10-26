
import os
import numpy as np
import json
import joblib
from sklearn.metrics import roc_auc_score
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
ROOT = Path(os.getenv('ROOT', '.')).expanduser()


def calculate_metrics_binary_classes(y_true, y_pred, pos_label=1, as_tuple=True):
        
    neg_label = 1 - pos_label
    tp = np.sum(np.logical_and(y_true == pos_label, y_pred == pos_label))  # True Positives
    tn = np.sum(np.logical_and(y_true == neg_label, y_pred == neg_label))  # True Negatives
    fp = np.sum(np.logical_and(y_true == neg_label, y_pred == pos_label))  # False Positives
    fn = np.sum(np.logical_and(y_true == pos_label, y_pred == neg_label))  # False Negatives
    p = tp + fn # Positives
    n = tn + fp # Negatives

    ppv = tp / (tp + fp)  # Positive Predictive Value
    npv = tn / (tn + fn)  # Negative Predictive Value
    tpr = tp / (tp + fn)  # True Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    tnr = tn / (tn + fp)  # True Negative Rate
    fpr = fp / (fp + tn)  # False Positive Rate

    f1_tp = 2 * (ppv * tpr) / (ppv + tpr)  # F1-score for true positive
    f1_tn = 2 * (npv * tnr) / (npv + tnr)  # F1-score for true negative

    f1_weighted_avg = (f1_tp * p + f1_tn * n) / (p + n) # Weighted F1-score
    balanced_accuracy = (tpr + tnr) / 2 # Balanced Accuracy

    if as_tuple:
        return ppv, npv, tpr, fnr, tnr, fpr, f1_tp, f1_tn, f1_weighted_avg, balanced_accuracy, p, n
    return {
            'ppv': ppv,
            'npv': npv,
            'tpr': tpr,
            'fnr': fnr,
            'tnr': tnr,
            'fpr': fpr,
            'f1_tp': f1_tp,
            'f1_tn': f1_tn,
            'f1_weighted_avg': f1_weighted_avg,
            'balanced_accuracy': balanced_accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'p': p,
            'n': n
        }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float):
            return round(obj, 4)
        return super(NpEncoder, self).default(obj)

def load_model(model_name):
    print(f'loading model "{model_name}.pkl"...', end='')
    clf = joblib.load(os.path.join(ROOT, 'models', f'{model_name}.pkl'))
    print(f'\rmodel "{model_name}.pkl" loaded.')
    return clf


def load_data(filepath, return_groups=False):
    with open(filepath, 'rb') as f:
        data = np.load(f)
        if return_groups:
            return data[:, :-2], data[:, -2], data[:, -1]
        else:
            return data[:, :-1], data[:, -1]

def ev_to_latex(ev,
            drop_columns=['f1_weighted_avg','tp','tn','fp','fn','p','n','fnr','fpr'],
            caption='Evaluation Metrics',
            label='tab:evaluation_metrics',
            round_decimals=2):
    
    ev_df = pd.DataFrame(ev).round(round_decimals).T
    ev_df.drop(columns=drop_columns, inplace=True)
    ev_df.columns = [c.upper() for c in ev_df.columns]
    ev_df.index = [i.upper() for i in ev_df.index]
    latex_tab = ev_df.to_latex(
        caption=caption, 
        float_format='{:0.2f}'.format,
        label=label)
    return latex_tab

if __name__ == '__main__':

    X_train, y_train, groups = load_data(os.path.join(ROOT, 'data', 'preprocessed', 'train-data.npy'), return_groups=True)
    X_test, y_test = load_data(os.path.join(ROOT, 'data', 'preprocessed', 'test-data.npy'))
    
    model_names = ['dt', 'rf', 'svc', 'mlp', 'nb', 'lr', 'knn', 'gbm', 'xgb']
    models = {}
    for model_name in model_names:
        try:
            model = load_model(model_name)
            models[model_name] = model
            print(f'Model "{model_name}" loaded successfully.')
        except Exception as e:
            print(f'Error loading model "{model_name}": {e}')

    print("Loaded models:", {k: type(v).__name__ for k, v in models.items()})

    ev = {}
    for model_name in models:
        y_pred_proba = models[model_name].predict_proba(X_test)[:,1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        ev[model_name] = {**calculate_metrics_binary_classes(y_test, y_pred, as_tuple=False), "auc": auc}
    
    print(ev)
    latex_table = ev_to_latex(ev, caption='Evaluation Metrics for Different Models', round_decimals=4)
    print(latex_table)







    