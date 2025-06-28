import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from util import load_and_preprocess_data

def evaluate_model(model, X, y, cv, dataset_name, model_name):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score

def compute_confusion_matrix(model, X, y, cv):
    cm = np.zeros((len(np.unique(y)), len(np.unique(y))))
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm += confusion_matrix(y_test, y_pred)
    cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized

def main():
    datasets = {
        'iris': load_and_preprocess_data('iris'),
        'wine': load_and_preprocess_data('wine'),
        'breast_cancer': load_and_preprocess_data('breast_cancer')
    }
    
    dt_params = {'max_depth': [3, 5, 7, 10, None]}
    rf_params = {'n_estimators': [10, 50, 100, 200]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    confusion_matrices = {}
    
    for dataset_name, (X, y) in datasets.items():
        # Hyperparameter tuning - Decision Tree
        best_dt_score = 0
        best_dt_param = None
        for max_depth in dt_params['max_depth']:
            dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            score, _ = evaluate_model(dt, X, y, cv, dataset_name, 'DecisionTree')
            if score > best_dt_score:
                best_dt_score = score
                best_dt_param = max_depth
        
        # Hyperparameter tuning - Random Forest
        best_rf_score = 0
        best_rf_param = None
        for n_estimators in rf_params['n_estimators']:
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            score, _ = evaluate_model(rf, X, y, cv, dataset_name, 'RandomForest')
            if score > best_rf_score:
                best_rf_score = score
                best_rf_param = n_estimators
        
        # Eval best models with best parameters
        dt = DecisionTreeClassifier(max_depth=best_dt_param, random_state=42)
        rf = RandomForestClassifier(n_estimators=best_rf_param, random_state=42)
        
        dt_score, dt_std = evaluate_model(dt, X, y, cv, dataset_name, 'DecisionTree')
        rf_score, rf_std = evaluate_model(rf, X, y, cv, dataset_name, 'RandomForest')
        
        # Compute confusion matrices - DONE
        confusion_matrices[(dataset_name, 'DecisionTree')] = compute_confusion_matrix(dt, X, y, cv)
        confusion_matrices[(dataset_name, 'RandomForest')] = compute_confusion_matrix(rf, X, y, cv)
        
        # Compute confidence intervals and p-values
        dt_scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')
        rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        diff_scores = rf_scores - dt_scores
        mean_diff = np.mean(diff_scores)
        std_diff = np.std(diff_scores)
        ci = 1.96 * std_diff / np.sqrt(len(diff_scores))
        t_stat, p_value = ttest_rel(rf_scores, dt_scores)
        
        results.append({
            'dataset': dataset_name,
            'dt_score': dt_score,
            'dt_std': dt_std,
            'rf_score': rf_score,
            'rf_std': rf_std,
            'dt_param': best_dt_param,
            'rf_param': best_rf_param,
            'ci_lower': mean_diff - ci,
            'ci_upper': mean_diff + ci,
            'p_value': p_value
        })
    
    #confusion matrices - DONE  
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    for i, dataset_name in enumerate(datasets.keys()):
        for j, model_name in enumerate(['DecisionTree', 'RandomForest']):
            cm = confusion_matrices[(dataset_name, model_name)]
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i, j])
            axes[i, j].set_title(f'{dataset_name} - {model_name}')
            axes[i, j].set_xlabel('Predicted')
            axes[i, j].set_ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    #print
    for result in results:
        print(f"Dataset: {result['dataset']}")
        print(f"Decision Tree: {result['dt_score']:.3f} ± {result['dt_std']:.3f} (max_depth={result['dt_param']})")
        print(f"Random Forest: {result['rf_score']:.3f} ± {result['rf_std']:.3f} (n_estimators={result['rf_param']})")
        print(f"Confidence Interval (RF - DT): [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p-value: {result['p_value']:.3f}")
        print()

if __name__ == "__main__":
    main()