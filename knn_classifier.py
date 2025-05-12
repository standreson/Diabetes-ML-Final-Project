"""
Module to train and evaluate KNN classifier.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def run_knn(X_train, y_train, X_val, y_val, k_values):
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        y_prob = knn.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        results[k] = {
            'accuracy': acc,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_val, y_pred)
        }
    return results
