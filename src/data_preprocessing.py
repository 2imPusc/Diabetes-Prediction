import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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

def split_and_balance_data(data, test_size=0.2, random_state=42):
    # Chia dữ liệu thành tập train và test trước khi chuẩn hóa
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Chuẩn hóa dữ liệu sau khi chia
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
    
    # Chuẩn hóa trên tập train
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Áp dụng chuẩn hóa cho tập test (chỉ dùng transform, không fit lại)
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Cân bằng dữ liệu bằng SMOTE (chỉ áp dụng trên tập train)
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res, scaler