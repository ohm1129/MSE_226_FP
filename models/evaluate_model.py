import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X, y, cv, model_name):
    """Evaluate a model using cross-validation"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"\n5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
    print(f"  F1-Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})")
    print(f"  Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
    print(f"  ROC-AUC:   {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std():.4f})")

    # Fit on full training data for detailed report
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\nFull Training Set Performance:")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred):.4f}")
    if y_pred_proba is not None:
        print(f"  ROC-AUC:   {roc_auc_score(y, y_pred_proba):.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

    return {
        'model_name': model_name,
        'cv_accuracy': cv_accuracy.mean(),
        'cv_f1': cv_f1.mean(),
        'cv_precision': cv_precision.mean(),
        'cv_recall': cv_recall.mean(),
        'cv_roc_auc': cv_roc_auc.mean(),
        'model': model
    }