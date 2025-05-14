import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Đánh giá
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'Accuracy': report['accuracy'],
            'Recall (Class 1)': report['1']['recall'],
            'F1-score (Class 1)': report['1']['f1-score'],
            'ROC-AUC': roc_auc,
            'Training Time': model_info['training_time']
        }

    results_df = pd.DataFrame(results).T
    return results_df

def visualize_resampling_comparison(comparison_df, metric):
    if metric not in comparison_df.columns:
        print(f"⚠️ Metric '{metric}' not found in DataFrame. Available columns: {list(comparison_df.columns)}")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y=comparison_df.index, data=comparison_df, palette='viridis')
    plt.title(f'Comparison of {metric} Across Resampling Methods')
    plt.xlabel(metric)
    plt.ylabel('Resampling Method')
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs("figures", exist_ok=True)
    file_name = f'figures/resampling_{metric.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(file_name)
    plt.show()
