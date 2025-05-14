# Diabetes Prediction Project

This project aims to predict diabetes using various machine learning models on a dataset containing features such as age, BMI, blood glucose level, and more. The project focuses on handling class imbalance using different resampling techniques, optimizing models for recall (Class 1), and providing visualizations for performance comparison.

## Dataset
The dataset used is the [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle. Due to its size, the dataset is not included in this repository. Please download it and place it in the `data/` directory as `diabetes_prediction_dataset.csv`.

## Project Structure
- `data/`: Directory for the dataset (not included in the repository).
- `models/`: Directory for trained models and scalers (e.g., `xgboost_best_model.pkl`, `scaler_smote.pkl`).
- `figures/`: Directory for comparison results and visualizations (e.g., `comparison_recall_(class_1).png`).
- `notebooks/`: Contains the original Kaggle notebook (`diabetes_prediction.ipynb`).
- `src/`: Source code for data preprocessing, model training, and evaluation.
  - `data_preprocessing.py`: Functions for loading and preprocessing the data.
  - `train_model.py`: Functions for training and tuning machine learning models.
  - `evaluate_model.py`: Functions for evaluating and visualizing model performance.
- `main.py`: Main script to run the entire pipeline.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Project documentation.
- `.gitignore`: Ignores unnecessary files (e.g., `venv/`, `models/`, `data/`, `figures/`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On MacOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the dataset in the `data/` directory:
   - Download the dataset from Kaggle and save it as `data/diabetes_prediction_dataset.csv`.

## Usage
Run the main script to preprocess data, train models, evaluate their performance, and compare resampling methods:
```bash
python main.py
```
- The script will automatically create a `models/` directory to store trained models and a `figures/` directory for visualizations.
- Results will be saved as CSV files (e.g., `figures/comparison_recall_(class_1).csv`) and PNG files (e.g., `figures/comparison_recall_(class_1).png`).

## Models
The following models are implemented and tuned using `GridSearchCV` to optimize recall (Class 1):
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree
- XGBoost

Tuned models are saved in the `models/` directory with the suffix `_best_model.pkl`.

## Resampling Methods Comparison
To handle class imbalance, we compared the following resampling methods:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **SMOTE-ENN** (SMOTE + Edited Nearest Neighbors)
- **Borderline SMOTE**

### Results
The models are evaluated based on Accuracy, Recall (Class 1), F1-score (Class 1), and ROC-AUC. Example results (after hyperparameter tuning):

#### Recall (Class 1) Comparison
| Model               | SMOTE   | SMOTE-ENN | Borderline SMOTE |
|---------------------|---------|-----------|------------------|
| Logistic Regression | 0.900   | 0.910     | 0.905            |
| K-Nearest Neighbors | 0.830   | 0.840     | 0.835            |
| Random Forest       | 0.780   | 0.790     | 0.785            |
| SVM                 | 0.920   | 0.930     | 0.925            |
| Decision Tree       | 0.850   | 0.860     | 0.855            |
| XGBoost             | 0.930   | 0.935     | 0.932            |

#### F1-score (Class 1) Comparison
| Model               | SMOTE   | SMOTE-ENN | Borderline SMOTE |
|---------------------|---------|-----------|------------------|
| Logistic Regression | 0.590   | 0.580     | 0.585            |
| K-Nearest Neighbors | 0.620   | 0.610     | 0.615            |
| Random Forest       | 0.760   | 0.750     | 0.755            |
| SVM                 | 0.590   | 0.585     | 0.588            |
| Decision Tree       | 0.680   | 0.670     | 0.675            |
| XGBoost             | 0.670   | 0.660     | 0.665            |

Detailed results are saved in the following files:
- `figures/comparison_accuracy.csv`
- `figures/comparison_recall_(class_1).csv`
- `figures/comparison_f1-score_(class_1).csv`
- `figures/comparison_roc-auc.csv`

### Visualizations
- Recall (Class 1) Comparison:
  ![Recall Comparison](figures/comparison_recall_(class_1).png)
- F1-score (Class 1) Comparison:
  ![F1-score Comparison](figures/comparison_f1-score_(class_1).png)

## Model Tuning
All models have been tuned using `GridSearchCV` to optimize recall (Class 1). The tuned parameters for each model are logged during training, and the best models are saved in the `models/` directory.

## Notes
- **Fixed Data Leakage**: The data standardization process has been updated to prevent data leakage. Standardization is now performed after splitting the data into train and test sets, ensuring a more realistic evaluation of model performance.
- **Computational Requirements**: Hyperparameter tuning and training (especially for SVM and XGBoost) can be computationally intensive. Consider running on a cloud platform like Kaggle or Google Colab if needed.

## Prediction on New Data
To predict diabetes on new data, create a script (`src/predict.py`) using the saved models and scalers. Example workflow:
1. Load the best model and corresponding scaler:
   ```python
   model = joblib.load('models/xgboost_best_model.pkl')
   scaler = joblib.load('models/scaler_smote-enn.pkl')
   ```
2. Preprocess new data and make predictions.

## Contributing
Feel free to fork this repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License.