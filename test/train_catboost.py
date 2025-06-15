import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from datasets import load_from_disk
from sklearn.model_selection import KFold

# Charger les données d'entraînement
dataset = load_from_disk("../datasets/diabetes-readmission")
train_data = dataset["train"].to_pandas()

# Séparer les features et la target
X_train = train_data.drop("readmitted", axis=1)
y_train = train_data["readmitted"]

# Initialiser la validation croisée
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listes pour stocker les scores
cv_accuracy = []
cv_auc = []

# Validation croisée
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nEntraînement du fold {fold}/{n_splits}")
    
    # Séparer les données d'entraînement et de validation
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # Créer et entraîner le modèle CatBoost
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=10,
        random_seed=42,
        eval_metric='AUC'
    )
    
    # Entraînement
    model.fit(
        X_fold_train, 
        y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        early_stopping_rounds=10
    )
    
    # Prédictions sur l'ensemble de validation
    y_pred = model.predict(X_fold_val)
    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculer les métriques
    fold_accuracy = accuracy_score(y_fold_val, y_pred)
    fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
    
    cv_accuracy.append(fold_accuracy)
    cv_auc.append(fold_auc)
    
    print(f"Fold {fold} - Accuracy: {fold_accuracy:.4f}, AUC: {fold_auc:.4f}")

# Afficher les résultats moyens
print("\nRésultats de la validation croisée :")
print(f"Accuracy moyenne: {np.mean(cv_accuracy):.4f} (±{np.std(cv_accuracy):.4f})")
print(f"AUC moyenne: {np.mean(cv_auc):.4f} (±{np.std(cv_auc):.4f})")

# Entraîner le modèle final sur toutes les données
print("\nEntraînement du modèle final sur toutes les données...")
final_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=10,
    random_seed=42,
    eval_metric='AUC'
)

final_model.fit(X_train, y_train)

# Sauvegarder le modèle final
final_model.save_model("diabetes_catboost_model.cbm")

# Afficher l'importance des features du modèle final
feature_importance = final_model.get_feature_importance()
print("\nImportance des features :")
for feature, importance in zip(X_train.columns, feature_importance):
    print(f"{feature}: {importance:.4f}") 