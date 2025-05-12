
"""
Main script to run data loading, feature selection, and all classifiers.
"""
from data_utils import load_and_split
from knn_classifier import run_knn
from naive_bayes_classifier import run_naive_bayes
from decision_tree_classifier import run_decision_tree

def summarize_best_model(name, result):
    print(f"\n=== Best Model: {name} ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")
    print("Classification Report:")
    print(result['classification_report'])

if __name__ == '__main__':
    DATA_PATH = 'data/diabetes.csv'
    TOP_FEATURES = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure']
    K_VALUES = [1, 3, 5, 7, 9]

    # Load and split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH, TOP_FEATURES)

    # KNN
    knn_results = run_knn(X_train, y_train, X_test, y_test, K_VALUES)
    print("KNN Results:")
    for k, res in knn_results.items():
        print(f"k={k}: acc={res['accuracy']:.4f}, AUC={res['roc_auc']:.4f}")

    # Naive Bayes
    nb_res = run_naive_bayes(X_train, y_train, X_test, y_test)
    print("\nNaive Bayes Results:")
    print(f"acc={nb_res['accuracy']:.4f}, AUC={nb_res['roc_auc']:.4f}")

    # Decision Tree with depth constraint
    dt_res = run_decision_tree(X_train, y_train, X_test, y_test, max_depth=4)
    print("\nDecision Tree Results (max_depth=4):")
    print(f"acc={dt_res['accuracy']:.4f}, AUC={dt_res['roc_auc']:.4f}")

    # Select best
    all_aucs = {f'knn_k_{k}': res['roc_auc'] for k, res in knn_results.items()}
    all_aucs['naive_bayes'] = nb_res['roc_auc']
    all_aucs['decision_tree'] = dt_res['roc_auc']
    best_model = max(all_aucs, key=all_aucs.get)
    print(f"\nBest model: {best_model} with AUC={all_aucs[best_model]:.4f}")

    # Summarize best model
    if best_model.startswith('knn'):
        k = int(best_model.split('_')[-1])
        summarize_best_model(f"KNN (k={k})", knn_results[k])
    elif best_model == 'naive_bayes':
        summarize_best_model("Naive Bayes", nb_res)
    elif best_model == 'decision_tree':
        summarize_best_model("Decision Tree", dt_res)
