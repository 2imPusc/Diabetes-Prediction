import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Thêm import cho Decision Tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # Thêm import cho GridSearchCV
import joblib
import time
import os

def train_models(X_train_res, y_train_res, y_train):
    # Tạo thư mục models nếu chưa tồn tại
    if not os.path.exists('models'):
        os.makedirs('models')

    # Khởi tạo các mô hình
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        # 'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')  # Thêm Decision Tree
    }

    # Tính scale_pos_weight cho XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    models['XGBoost'] = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss'
    )

    # Định nghĩa các tham số cần tinh chỉnh cho từng mô hình
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        # 'SVM': {
        #     'C': [0.1, 1, 10],
        #     'kernel': ['rbf', 'linear']
        # },
        'Decision Tree': {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== Tinh Chỉnh Tham Số Cho Mô Hình: {name} ===")
        
        # Sử dụng GridSearchCV để tìm tham số tối ưu
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='recall',  # Tối ưu hóa recall (Class 1)
            cv=5,  # 5-fold cross-validation
            n_jobs=-1,  # Sử dụng tất cả CPU
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train_res, y_train_res)
        training_time = time.time() - start_time

        # Lấy mô hình tốt nhất sau khi tinh chỉnh
        best_model = grid_search.best_estimator_
        print(f"Tham số tối ưu cho {name}: {grid_search.best_params_}")
        print(f"Recall (Class 1) tốt nhất trên tập train (cross-validation): {grid_search.best_score_:.3f}")

        # Lưu mô hình tốt nhất
        results[name] = {'model': best_model, 'training_time': training_time}
        
        # Lưu mô hình
        joblib.dump(best_model, f'models/{name.lower().replace(" ", "_")}_best_model.pkl')
        print(f"Đã lưu mô hình {name} tại models/{name.lower().replace(' ', '_')}_best_model.pkl")

    return results