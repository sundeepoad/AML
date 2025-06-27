# 💰 Money Laundering Detection with XGBoost

This project focuses on detecting potential money laundering activities using machine learning. The dataset contains transactional data, and I used feature engineering, balancing techniques, and XGBoost to classify suspicious trasactions.

## 🚀 Project Highlights

- 📊 **Imbalanced Dataset Handling**: The dataset had large number of trasanctions. Downsampling was used to balance classes.
- 🧠 **Model**: Trained an XGBoost classifier for binary classification.
- 🧼 **Feature Engineering**:
  - Extracted temporal features (Year, Month, Day, Hour) from timestamps.
  - Label-encoded categorical fields.
    
- 📈 **Performance Evaluation**:
  - Classification report and accuracy score 85%.
  - Feature importance analysis using bar charts.

## 🛠️ Tech Stack

- Python (Pandas, NumPy)
- Visualization: Seaborn, Matplotlib
- Scikit-learn
- XGBoost

## 📂 Dataset
Source: "IBM Transactions for Anti Money Laundering (AML)" Kaggle
- File: `LI-Small_Trans.csv`
- Key Columns:
  - `Timestamp`: Date and time of transaction
  - `Amount Paid`
  - `Payment Currency`, `Receiving Currency`, `Payment Format`
  - `Is Laundering`: Target label


## 📊 Model Results

- 📌 **Accuracy**: ~Accuracy printed from `accuracy_score`
- 📌 **Top Features**:
  - `Payment Format`
  - `From Bank`


