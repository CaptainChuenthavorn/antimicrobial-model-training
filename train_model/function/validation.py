from pandas import DataFrame, get_dummies
from numpy import array, zeros
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score

def cross_validation(X, y, cv_method, algorithms: Dict, preprocessing: list = [], imbalance_handler: list = [], measures: Dict = None, threshold: float = .5) -> DataFrame:
    if measures is None:
        measures = {'accuracy': lambda true, pred: accuracy_score(true, [p >= threshold for p in pred]),
                    'precision': lambda true, pred: precision_score(true, [p >= threshold for p in pred]),
                    'recall': lambda true, pred: recall_score(true, [p >= threshold for p in pred]),
                    'f1': lambda true, pred: f1_score(true, [p >= threshold for p in pred]),
                    'roc-auc': lambda true, pred: roc_auc_score(true, pred),
                    # 'g-mean': lambda true, pred: geometric_mean_score(true, [p >= threshold for p in pred]),
                    # 'avg-precison': lambda true, pred: average_precision_score(true, pred)
                    }
    _df = DataFrame(columns=measures)
    scores = zeros((len(algorithms), len(measures)))
    t = 0
    for train_index, test_index in cv_method.split(X, y):
        _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
        _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
        for pre in preprocessing:
            _X_train, _X_test, _y_train, _y_test = pre(
                _X_train, _X_test, _y_train, _y_test)
        for imb in imbalance_handler:
            _X_train, _y_train = imb(_X_train, _y_train)
        algos = [al.fit(_X_train, _y_train) for al in algorithms.values()]
        i = 0
        for algo in algos:
            scores[i] += array([value(_y_test, [ctrue for _, ctrue in algo.predict_proba(_X_test)])
                               for value in measures.values()])
            i += 1
        t += 1
    scores /= t
    return _df.append(DataFrame(scores, columns=measures.keys(), index=[str(algo) for algo in algorithms.keys()]))


def evaluation(X, y, models: Dict, measures: Dict = None, threshold: float = .5) -> DataFrame:
    if measures is None:
        measures = {'accuracy': lambda true, pred: accuracy_score(true, [p >= threshold for p in pred]),
                    'percision': lambda true, pred: precision_score(true, [p >= threshold for p in pred]),
                    'recall': lambda true, pred: recall_score(true, [p >= threshold for p in pred]),
                    'f1': lambda true, pred: f1_score(true, [p >= threshold for p in pred]),
                    # 'roc-auc': lambda true, pred: roc_auc_score(true, pred),
                    # 'g-mean': lambda true, pred: geometric_mean_score(true, [p >= threshold for p in pred]),
                    # 'avg-precison': lambda true, pred: average_precision_score(true, pred)
                    }
    _df = DataFrame(columns=measures)
    scores = zeros((len(models), len(measures)))
    i = 0
    for model in models.values():
        scores[i] += array([value(y, [ctrue for _, ctrue in model.predict_proba(X)])
                           for value in measures.values()])
        i += 1
    return _df.append(DataFrame(scores, columns=measures.keys(), index=[str(model) for model in models.keys()]))

def binning_top_k(df: DataFrame, column, k: int , other = "other") -> DataFrame:
    indices = df[column].value_counts().nlargest(k + 1).index if other in df[column].value_counts().nlargest(k).index else df[column].value_counts().nlargest(k).index
    newdf = df.copy(deep=True)
    newdf[column] = newdf[column].where(newdf[column].isin(indices), other=other)
    return newdf

def binning_less_than(df: DataFrame, column, k: int , other = "other") -> DataFrame:
    newdf = df.copy()
    values = newdf[column].value_counts()
    index = values[values < k].index
    newdf.loc[newdf[column].isin(index),column] = other
    return newdf

def get_dummies_dataframe_columns(df_dummies: DataFrame, old_df: DataFrame) -> DataFrame:
    old_df = get_dummies(old_df).filter(df_dummies.columns)
    new_df = DataFrame(columns=list(df_dummies.columns)).append(old_df)
    new_df.fillna(0, inplace=True)
    return new_df


"""
cross_validation(X_train, y_train, skf,
                 {str(svm): svm, str(xgb): xgb},
                 [lambda _X_train, _X_test, _y_train, _y_test: (pd.get_dummies(
                     _X_train), get_dummies_dataframe_columns(pd.get_dummies(_X_train), _X_test), _y_train, _y_test)],
                 [lambda _X, _y: smote.fit_resample(_X.astype(float), _y), lambda _X, _y: (np.round(_X), _y)])

evaluation(get_dummies_dataframe_columns(pd.get_dummies(X_train), X_test), y_test, {'SVM': svm.fit(
    pd.get_dummies(X_train), y_train), 'XGBoost': xgb.fit(pd.get_dummies(X_train), y_train)})
"""
