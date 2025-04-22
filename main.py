from src.data_preprocessing import load_and_preprocess_data, split_and_balance_data
from src.train_model import train_models
from src.evaluate_model import evaluate_models

def main():
    # Đường dẫn đến dữ liệu
    data_path = 'data/diabetes_prediction_dataset.csv'

    # Tiền xử lý dữ liệu
    data = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler = split_and_balance_data(data)

    # Huấn luyện mô hình
    models = train_models(X_train_res, y_train_res, y_train)

    # Đánh giá mô hình
    results_df = evaluate_models(models, X_test, y_test)
    print("\n=== Kết Quả So Sánh Các Mô Hình ===")
    print(results_df)

if __name__ == "__main__":
    main()