# Assignment 2: End-to-End Machine Learning Pipeline (Titanic Dataset)

## 📌 Overview
This project demonstrates a **complete machine learning pipeline** applied to the Titanic dataset from Kaggle.  
The pipeline covers all major steps from raw data to model evaluation and conclusions.

---

## 📂 Contents
- `Assignment_2_Notebook.ipynb` → Colab notebook with full code, outputs, and explanations.  
- `Assignment_2_Report.pdf` → 2–3 page summary report with insights, visualizations, and conclusions.  
- `README.md` → This file.  

---

## 🛠️ Steps Performed

### 1. Dataset Handling
- Loaded Titanic dataset (`train.csv`) from Kaggle.
- Checked for missing values and duplicates.
- Handled missing data:
  - `Age` → filled with median.
  - `Embarked` → filled with mode.
  - `Cabin` → dropped (too many missing).
- Encoded categorical variables (`Sex`, `Embarked`) into numeric form.

### 2. Exploratory Data Analysis (EDA)
- Basic statistics with Pandas & NumPy.
- Visualizations:
  - **Histogram (Age distribution)**.
  - **Barplot (Survival by Sex)**.
  - **Correlation heatmap (numeric features)**.
  - **Plotly scatter (Age vs Fare, colored by Survival)**.

### 3. Feature Engineering
- Split into features (X) and target (y).
- Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`).
- Standardized features with `StandardScaler`.
- Train-test split (80-20).

### 4. Model Training (Baseline)
- Trained and evaluated:
  - **KNN Classifier**
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
- Compared baseline accuracies.

### 5. Feature Importance
- Extracted feature importance from Random Forest.
- Found top features: **Sex, Fare, Pclass, Age**.

### 6. Hyperparameter Tuning
- Used **RandomizedSearchCV** for:
  - **KNN** → n_neighbors, weights, metric.
  - **Decision Tree** → max_depth, min_samples_split, criterion.
  - **Random Forest** → n_estimators, max_depth, min_samples_split.
- Compared tuned vs baseline results.

### 7. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score.
- Confusion matrices for tuned models.
- ROC Curve for best model.

### 8. Conclusion
- **Best Model:** Tuned Random Forest (~82% accuracy, AUC ~0.86).
- **Most Important Features:** Sex, Fare, Pclass, Age.
- **Impact of Tuning:** Improved accuracy and F1-scores for all models.

---

## 📊 Example Results

| Model                  | Accuracy | Precision | Recall | F1-score |
|-------------------------|----------|-----------|--------|----------|
| KNN (baseline)         | 0.68     | 0.67      | 0.66   | 0.66     |
| Decision Tree (baseline)| 0.70     | 0.69      | 0.70   | 0.69     |
| Random Forest (baseline)| 0.78     | 0.77      | 0.78   | 0.77     |
| KNN (tuned)            | 0.73     | 0.72      | 0.72   | 0.72     |
| Decision Tree (tuned)  | 0.77     | 0.76      | 0.77   | 0.76     |
| Random Forest (tuned)  | **0.82** | **0.81**  | **0.82** | **0.81** |

---

## 🚀 How to Run
1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload `Assignment_2_Notebook.ipynb`.  
3. Upload `train.csv` (Titanic dataset from Kaggle).  
4. Run cells sequentially.  
5. View results and plots directly in Colab.  

---

## 📌 Requirements
- Python 3.x  
- Libraries:
  - numpy  
  - pandas  
  - matplotlib  
  - seaborn  
  - plotly  
  - scikit-learn  

---

## 🙌 Acknowledgment
- Dataset: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)  
