# 📈 Linear Regression Using Python

## 📌 Overview
This repository contains my learning and practical implementation of **Linear Regression**, a fundamental Supervised Machine Learning algorithm.

The purpose of this project is to understand how Linear Regression works, how to train a machine learning model, make predictions, and evaluate its performance using Python and Scikit-learn.

## 🎯 Objectives
- Understand the fundamentals of Linear Regression.
- Learn the relationship between independent and dependent variables.
- Perform data preprocessing.
- Split datasets into training and testing sets.
- Train a Linear Regression model.
- Make predictions using the trained model.
- Evaluate model performance using different metrics.
- Visualize actual and predicted values.

## 🧠 What is Linear Regression?

Linear Regression is a Supervised Machine Learning algorithm used to predict continuous numerical values by identifying the relationship between input features and an output variable.

**Mathematical Equation:**

\[
y = mx + c
\]

Where:
- `y` = Predicted output
- `m` = Slope (coefficient)
- `x` = Input feature
- `c` = Intercept

## 🛠️ Technologies & Libraries

The following Python libraries are used:

| Library | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical calculations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning implementation |

## ⚙️ Implementation Steps

1. Import the required Python libraries.
2. Load the dataset using Pandas.
3. Explore and understand the dataset.
4. Perform data preprocessing.
5. Select independent (X) and dependent (y) variables.
6. Split the dataset into training and testing sets.
7. Create a Linear Regression model.
8. Train the model using `fit()`.
9. Generate predictions using `predict()`.
10. Evaluate the model performance.

## 💻 Basic Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data.csv")

# Select features and target
X = df[['Feature']]
y = df['Target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## 📊 Model Evaluation Metrics

The following evaluation metrics help measure the performance of the Linear Regression model:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Measures the average squared difference between actual and predicted values.
- **R² Score:** Measures how well the model explains the variation in the target variable.

## 📦 Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/Mubir786/Linear-Regression.git
```

**2. Navigate to the project directory**

```bash
cd Linear-Regression
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Run the Jupyter Notebook**

```bash
jupyter notebook
```

## 📚 Learning Outcomes

Through this project, I am learning:

- Fundamentals of Supervised Machine Learning.
- Data preprocessing and feature selection.
- Training and testing machine learning models.
- Understanding regression algorithms.
- Evaluating prediction accuracy.
- Implementing machine learning using Scikit-learn.

## 🚀 Future Improvements

- Implement Multiple Linear Regression.
- Experiment with different datasets.
- Perform feature selection.
- Compare Linear Regression with other regression algorithms.
- Improve model performance through experimentation.

## 👨‍💻 Author

**Mubir Hussain**

- GitHub: [@Mubir786](https://github.com/Mubir786)
- LinkedIn: [Mubir Hussain](https://www.linkedin.com/in/mubir-hussain-669788250/)
- Portfolio: [My Portfolio](https://mubir-portfolio.netlify.app/)

---

⭐ If you find this repository helpful, consider giving it a star!

**Learning Machine Learning, One Algorithm at a Time.** 🚀
