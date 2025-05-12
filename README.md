Credit Score Classification
This project builds a machine learning pipeline to classify individuals' credit scores as Poor, Standard, or Good based on various financial and demographic features. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tuning, and prediction deployment.

📂 Project Structure
credit_score.py: Main script implementing the full ML pipeline from data loading to model saving.

train.csv: Training dataset (not included in the repo).

test.csv: Test dataset (not included in the repo).

credit_score_model.pkl: Trained model (generated after execution).

scaler.pkl: Scaler used for input feature normalization.

🔍 Features Used
Age, Income, Credit Card Usage, Number of Loans, Payment History

Engineered features like:

Debt-to-Income Ratio

Payment Delay Ratio

Credit Utilization Category

Age Group

🧪 ML Models Evaluated
Logistic Regression

Random Forest

Decision Tree

K-Nearest Neighbors

XGBoost

The best model is chosen based on validation accuracy, and optionally tuned using RandomizedSearchCV.

🧰 Tech Stack
Python

Pandas, NumPy

Matplotlib, Seaborn (for EDA)

Scikit-learn, XGBoost

Joblib (for model serialization)

⚙️ How to Run
Install requirements

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
Place the train.csv and test.csv files in the project directory.

Run the script

bash
Copy
Edit
python credit_score.py
Output

Trained model saved as credit_score_model.pkl

Scaler saved as scaler.pkl

Accuracy scores and classification reports printed for all models.

🧾 Output Labels
0: Poor Credit Score

1: Standard Credit Score

2: Good Credit Score

🔮 Example Prediction
A sample prediction on custom user input is included at the end of the script with proper preprocessing and scaling.

📌 Notes
The script automatically handles missing values, outliers, and categorical encoding.

Some columns such as ID, SSN, and Name are excluded from training due to irrelevance or uniqueness.

Ensure all feature engineering steps are applied to both training and new input data.

📧 Contact
For questions or improvements, feel free to open an issue or submit a pull request.
