"""
Module to train and evaluate Gaussian Naive Bayes classifier.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

def run_naive_bayes(X_train, y_train, X_val, y_val):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_val)
    y_prob = gnb.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_val, y_pred)
    return {'accuracy': acc, 'roc_auc': roc_auc, 'classification_report': report}

