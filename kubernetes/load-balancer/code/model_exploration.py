import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from cluster_data_client import ClusterApiClient
import pandas as pd


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((f"{new_key}.start", v[0]))
            items.append((f"{new_key}.end", v[-1]))
        else:
            items.append((new_key, v))
    return dict(items)

def feature_engineer(df):
    df["states.duration.end"] = df["states.process_uptime_seconds.end"] - df["states.process_uptime_seconds.start"]
    df = df.drop(columns=["states.process_uptime_seconds.start", "states.process_uptime_seconds.end"])

    df["states.cpu_duration_second.end"] = df["states.process_cpu_seconds_total.end"] - df["states.process_cpu_seconds_total.start"]
    df = df.drop(columns=["states.process_cpu_seconds_total.start", "states.process_cpu_seconds_total.end"])

    df = df.loc[:, df.nunique() > 1]
    df = df.dropna(axis=1)
    return df

cluster_data_client = ClusterApiClient(api_name="localhost")
input_data = pd.DataFrame([flatten_dict(e) for e in cluster_data_client.getHistory()])
formated_data = feature_engineer(input_data)


# model training
cols = formated_data.columns
X_cols = [c for c in cols if "input." in c or ".start" in c or c in ["pod", "function"] ]
Y_cols = [c for c in cols if ".end" in c or c in ["exec_duration_ns"]]

X = formated_data[X_cols]
y = formated_data[Y_cols]

X_encoded = pd.get_dummies(X, columns=['pod', 'function'])



# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
)

# Entraînement
print("\n🔄 Training model...")
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# ===== ÉVALUATION GLOBALE =====
print("\n" + "="*50)
print("OVERALL PERFORMANCE")
print("="*50)

mae_global = mean_absolute_error(y_test, y_pred)
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred))
r2_global = r2_score(y_test, y_pred)

print(f"Global MAE:  {mae_global:.4f}")
print(f"Global RMSE: {rmse_global:.4f}")
print(f"Global R²:   {r2_global:.4f}")

# ===== ÉVALUATION PAR MÉTRIQUE =====
print("\n" + "=" * 50)
print("PER-METRIC PERFORMANCE")
print("=" * 50)

results = []
for i, col in enumerate(y.columns):
    y_true_col = y_test.iloc[:, i]
    y_pred_col = y_pred[:, i]

    mae = mean_absolute_error(y_true_col, y_pred_col)
    rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
    r2 = r2_score(y_true_col, y_pred_col)

    # Calculer l'erreur relative moyenne
    mape = np.mean(np.abs((y_true_col - y_pred_col) / (y_true_col + 1e-10))) * 100

    results.append({
        'Metric': col,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape
    })

    print(f"\n📊 {col}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

# Créer un DataFrame des résultats
results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print("SUMMARY TABLE")
print("=" * 50)
print(results_df.to_string(index=False))

# ===== ANALYSE DES ERREURS =====
print("\n" + "=" * 50)
print("ERROR ANALYSIS")
print("=" * 50)

for i, col in enumerate(y.columns):
    y_true_col = y_test.iloc[:, i].values
    y_pred_col = y_pred[:, i]

    errors = y_true_col - y_pred_col

    print(f"\n{col}:")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std error:  {np.std(errors):.4f}")
    print(f"  Max overestimation:  {np.min(errors):.4f}")
    print(f"  Max underestimation: {np.max(errors):.4f}")

# ===== VISUALISATION =====
print("\n📈 Generating visualizations...")

n_metrics = len(y.columns)
fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

if n_metrics == 1:
    axes = [axes]

for i, col in enumerate(y.columns):
    y_true_col = y_test.iloc[:, i].values
    y_pred_col = y_pred[:, i]

    # Scatter plot: Prédictions vs Réel
    axes[i].scatter(y_true_col, y_pred_col, alpha=0.5, s=20)

    # Ligne parfaite
    min_val = min(y_true_col.min(), y_pred_col.min())
    max_val = max(y_true_col.max(), y_pred_col.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    axes[i].set_xlabel('Actual', fontsize=12)
    axes[i].set_ylabel('Predicted', fontsize=12)
    axes[i].set_title(f'{col}\nR² = {r2_score(y_true_col, y_pred_col):.3f}', fontsize=14)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to 'prediction_results.png'")
plt.show()

# ===== FEATURE IMPORTANCE =====
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 50)

# Moyenne de l'importance sur tous les estimateurs
feature_importance = np.zeros(X_train.shape[1])
for estimator in model.estimators_:
    feature_importance += estimator.feature_importances_
feature_importance /= len(model.estimators_)

importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df.head(10).to_string(index=False))

# Visualisation de l'importance
plt.figure(figsize=(10, 6))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved feature importance to 'feature_importance.png'")
plt.show()

# ===== EXEMPLES DE PRÉDICTIONS =====
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS (First 5 test samples)")
print("=" * 50)

sample_df = pd.DataFrame()
for i, col in enumerate(y.columns):
    sample_df[f'{col}_actual'] = y_test.iloc[:5, i].values
    sample_df[f'{col}_predicted'] = y_pred[:5, i]
    sample_df[f'{col}_error'] = y_test.iloc[:5, i].values - y_pred[:5, i]

print(sample_df.to_string(index=False))

print("\n✅ Model evaluation complete!")