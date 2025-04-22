import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import time
import os  # Thêm import này

def train_models(X_train_res, y_train_res, y_train):
    # Tạo thư mục models nếu chưa tồn tại
    if not os.path.exists('models'):
        os.makedirs('models')

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
    }

    # Tính scale_pos_weight cho XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    models['XGBoost'] = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss'
    )

    results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_res, y_train_res)
        training_time = time.time() - start_time
        results[name] = {'model': model, 'training_time': training_time}

        # Lưu mô hình
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.pkl')
        print(f"Đã lưu mô hình {name} tại models/{name.lower().replace(' ', '_')}_model.pkl")

    return results