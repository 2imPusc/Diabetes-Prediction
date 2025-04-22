import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score

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

    return pd.DataFrame(results).T

def visualize_results(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Recall (Class 1)', y=results_df.index, data=results_df)
    plt.title('Comparison of Recall (Class 1) Across Models')
    plt.savefig('model_comparison_recall.png')
    plt.show()