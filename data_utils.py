"""
Module for loading data and splitting into train, validation, and test sets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(path, top_features, random_state=123):
    # Load dataset
    df = pd.read_csv(path)
    X = df[top_features]
    y = df['Outcome']

    # 70/20/10 split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state, stratify=y)
    val_ratio = 20/90
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=random_state, stratify=y_tv)
    return X_train, X_val, X_test, y_train, y_val, y_test
