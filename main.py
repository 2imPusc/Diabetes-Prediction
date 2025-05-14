import joblib
import pandas as pd
import os
from src.data_preprocessing import load_and_preprocess_data, split_and_balance_data
from src.train_model import train_models
from src.evaluate_model import evaluate_models, visualize_resampling_comparison

def main():
    # Tạo thư mục figures nếu chưa tồn tại
    os.makedirs('figures', exist_ok=True)

    # Đường dẫn đến dữ liệu
    data_path = 'data/diabetes_prediction_dataset.csv'

    # Tiền xử lý dữ liệu
    data = load_and_preprocess_data(data_path)

    # Các phương pháp resampling cần so sánh
    resampling_methods = ['smote', 'smote-enn', 'borderline-smote']
    all_results = {}

    for method in resampling_methods:
        print(f"\n=== Xử Lý Dữ Liệu Với Phương Pháp: {method.upper()} ===")
        
        # Chia và cân bằng dữ liệu với phương pháp hiện tại
        X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler = split_and_balance_data(
            data, resampling_method=method
        )

        # Lưu scaler
        joblib.dump(scaler, f'models/scaler_{method}.pkl')
        print(f"Đã lưu scaler tại models/scaler_{method}.pkl")

        # Huấn luyện tất cả mô hình
        models = train_models(X_train_res, y_train_res, y_train)

        # Đánh giá mô hình
        results_df = evaluate_models(models, X_test, y_test)
        print(f"\nKết Quả Với Phương Pháp {method.upper()}:")
        print(results_df)

        # Lưu kết quả của phương pháp này
        all_results[method] = results_df

    # Tạo bảng so sánh tổng hợp cho từng chỉ số
    metrics = ['Accuracy', 'Recall (Class 1)', 'F1-score (Class 1)', 'ROC-AUC']
    comparison_dfs = {}

    for metric in metrics:
        comparison_data = {}
        for method, results_df in all_results.items():
            comparison_data[method] = results_df[metric]
        comparison_dfs[metric] = pd.DataFrame(comparison_data)

    # In bảng so sánh
    for metric, df in comparison_dfs.items():
        print(f"\n=== So Sánh {metric} Giữa Các Mô Hình Và Phương Pháp Resampling ===")
        print(df)

    # Lưu bảng so sánh
    for metric, df in comparison_dfs.items():
        df.to_csv(f'figures/comparison_{metric.lower().replace(" ", "_")}.csv')

    # Trực quan hóa
    for metric in ['Recall (Class 1)', 'F1-score (Class 1)']:
        visualize_resampling_comparison(comparison_dfs[metric], metric)

if __name__ == "__main__":
    main()