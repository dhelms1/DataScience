import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fit and evaluate multiple machine learning models
    models: dictionary of different models to train and score
    NOTE: data is normalized before passing into function
    """
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

def data_normalize(X_train, X_test):
    """
    Normalize data using a MinMaxScaler (all data in range [0,1])
    returns a tuple in the form (train set, test set)
    """
    # Normalize Train Data
    train_scaler = MinMaxScaler()
    train_scaler.fit(X_train)
    X_train_norm = train_scaler.transform(X_train)
    # Normalize Test Data
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_norm = test_scaler.transform(X_test)
    
    return (X_train_norm, X_test_norm)

def correlation_heatmap(corr_matrix):
    """
    Plots a heatmap of a correlation matrix
    """
    fig, ax = plt.subplots(figsize=(15,10))
    ax = sns.heatmap(corr_matrix, annot=True, linewidth=0.5, fmt='.2f', cmap='YlGnBu')
    plt.title('Correlation Matrix')
    plt.show()
    return None
    
def confusion_heatmap(y_test, y_preds):
    """
    Plots a heatmap of a confusion matrix
    pass in y_test (actual) and y_preds (predicted) labels
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()
    return None

def cross_validation_metrics(model, X, y):
    """
    Return a dataframe of 5-fold cross validated metrics
    only works for binary classfication problems (no multiclass)
    """
    cv_acc = cross_val_score(model, X, y, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, scoring='precision')
    cv_recall = cross_val_score(model, X, y, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, scoring='f1')
    cv_metrics = pd.DataFrame({'Accuracy': np.mean(cv_acc), 'Precision': np.mean(cv_precision),
                               'Recall': np.mean(cv_recall), 'F1': np.mean(cv_f1)}, index=[0])
    return cv_metrics