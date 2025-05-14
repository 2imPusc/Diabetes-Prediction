import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN

def load_and_preprocess_data(file_path):
    # Nạp dữ liệu
    data = pd.read_csv(file_path)

    # Mã hóa biến phân loại
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    data['smoking_history'] = le.fit_transform(data['smoking_history'])

    # Loại bỏ giá trị bất thường
    data = data[data['bmi'] <= 60]

    return data

def split_and_balance_data(data, test_size=0.2, random_state=42, resampling_method='smote'):
    # Chia dữ liệu thành tập train và test trước khi chuẩn hóa
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Chuẩn hóa dữ liệu sau khi chia
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
    
    # Chuẩn hóa trên tập train
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Áp dụng chuẩn hóa cho tập test
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Chọn phương pháp resampling
    if resampling_method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif resampling_method == 'smote-enn':
        sampler = SMOTEENN(random_state=random_state)
    elif resampling_method == 'borderline-smote':
        sampler = BorderlineSMOTE(random_state=random_state)
    else:
        raise ValueError("Phương pháp resampling không hợp lệ. Chọn 'smote', 'smote-enn', hoặc 'borderline-smote'.")

    # Cân bằng dữ liệu bằng phương pháp đã chọn
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler