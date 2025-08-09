# Loan Approval Prediction

This project predicts loan approval status based on applicant data using **Logistic Regression** and **Decision Tree Classifier**. It includes data preprocessing, handling missing values, encoding categorical variables, addressing class imbalance with **SMOTE**, training models, and evaluating them with performance metrics and confusion matrices.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

Install dependencies with:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

## Dataset
The dataset file should be named `loan.csv` and contain both numerical and categorical applicant details such as income, credit history, education, property area, etc., along with a target column `Loan_Status` indicating approval (1) or rejection (0). Update the file path in the script if needed:
df = pd.read_csv(r'C:\Users\LENOVO\Documents\loan.csv')

## Workflow
1. **Data Loading**: Reads the dataset and standardizes column names to lowercase without spaces.
2. **Missing Value Handling**: Fills missing values using forward fill (`ffill`).
3. **Encoding**: Converts categorical features to numeric using Label Encoding.
4. **Feature/Target Split**: Splits data into predictors (X) and target variable (`loan_status`).
5. **Class Imbalance Handling**: Uses SMOTE to oversample the minority class in the training data.
6. **Model Training**:
   - Logistic Regression
   - Decision Tree Classifier
7. **Model Evaluation**:
   - Classification report (Precision, Recall, F1-score, Accuracy)
   - Confusion Matrix visualization using matplotlib

## Running the Project
1. Place your dataset in the specified location or update the path in the script.
2. Install dependencies.
3. Run the script:
   python loan_approval_prediction.py

## Example Output
**Class Distribution Before SMOTE**:
Approved: 332, Rejected: 110  
**After SMOTE**:
Approved: 332, Rejected: 332  

**Performance Metrics**:
