"""
Module to train and evaluate Decision Tree classifier with depth control.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

def run_decision_tree(X_train, y_train, X_val, y_val, random_state=123):
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_val, y_pred)
    return {'model': clf, 'accuracy': acc, 'roc_auc': roc_auc, 'classification_report': report}

