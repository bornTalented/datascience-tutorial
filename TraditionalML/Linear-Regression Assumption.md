Understanding the assumptions of **linear regression** is crucial to interpret results reliably. Let's break down each assumption with **simple, intuitive examples** and small visualizable data cases.

---

### **1. Linearity**

**Definition**: There must be a linear relationship between the independent variable(s) and the dependent variable.

#### ✅ Example (Linearity Present):

```python
# Hours studied vs. Exam score
Hours_Studied = [1, 2, 3, 4, 5]
Exam_Score = [50, 60, 70, 80, 90]  # Linear increase
```

Plotting this shows a straight line: `Exam_Score = 10 * Hours_Studied + 40`

#### ❌ Violation:

```python
Hours_Studied = [1, 2, 3, 4, 5]
Exam_Score = [30, 55, 75, 80, 82]  # Plateauing curve
```

Here, the increase slows down—indicating a **nonlinear** relationship (e.g., logarithmic or exponential).

---

### **2. No Multicollinearity**

**Definition**: Independent variables should not be highly correlated with each other in **Multiple Linear Regression**.

#### ✅ Example (No Multicollinearity):

```python
# Predicting house price
Square_Footage = [1000, 1500, 2000, 2500, 3000]
Age_of_House   = [10,   8,    15,   5,    20]
```

These features are **not strongly correlated**, so each provides unique information.

#### ❌ Violation:

```python
Square_Footage = [1000, 1500, 2000, 2500, 3000]
# Total_Rooms is just square footage divided by 500
Total_Rooms    = [2,     3,    4,     5,    6]
```

Here, `Square_Footage` and `Total_Rooms` are **almost perfectly correlated**, leading to multicollinearity.

---

### **3. Homoscedasticity**

**Definition**: The variance of residuals (errors) should be constant across all values of independent variables.

#### ✅ Example (Homoscedastic):

```python
# Residuals show no pattern
Predicted = [50, 60, 70, 80, 90]
Actual    = [52, 59, 72, 78, 89]
Residuals = [ 2, -1,  2, -2, -1]  # Constant variance
```

#### ❌ Violation (Heteroscedasticity):

```python
Predicted = [50, 60, 70, 80, 90]
Actual    = [55, 65, 85, 105, 130]
Residuals = [5, 5, 15, 25, 40]  # Variance increases with prediction
```

This often shows up as a **cone shape** in residual plots.

---

### **4. Normality of Residuals**

**Definition**: The residuals (differences between actual and predicted values) should be **normally distributed**.

#### ✅ Example:

```python
Residuals = [-2, -1, 0, 1, 2]  # Symmetric around zero
```

Histogram or Q-Q plot of these will show a **bell curve**.

#### ❌ Violation:

```python
Residuals = [-1, -1, 0, 1, 8]  # Skewed distribution
```

Outliers (like `8`) can distort the normality.

---

### **5. Independence of Residuals**

**Definition**: Residuals should not be correlated with each other, especially important in **time series**.

#### ✅ Example (Independent Residuals):

```python
Residuals = [1, -1, 2, -2, 0]  # No pattern
```

#### ❌ Violation (Autocorrelation):

```python
Residuals = [1, 2, 3, 4, 5]  # Clear trend or sequence
```

This is common in time series data (e.g., sales by month), where residuals show **systematic structure**.

---

### 📌 Summary Table:

| Assumption             | Violated When...                                          | Example Violation                          |
| ---------------------- | --------------------------------------------------------- | ------------------------------------------ |
| Linearity              | Relationship is curved or step-like                       | Exam scores plateau with hours studied     |
| No Multicollinearity   | Independent variables are highly correlated               | Square footage and total rooms             |
| Homoscedasticity       | Residuals increase/decrease as predictions increase       | Errors grow with larger predicted values   |
| Normality of Residuals | Residuals are skewed or have outliers                     | One large residual throws off distribution |
| Independence of Errors | Residuals follow a trend or pattern (esp. in time series) | Residuals increase linearly with time      |

---

Let me know if you want **code snippets or visual plots** for any of these!

---

Great — here's a **complete working Python example** using **scikit-learn** and **statsmodels** to train a linear regression model and **validate all five assumptions** using appropriate **diagnostic tools**.

---

## ✅ **Goal**

Detect violations of:

1. **Linearity**
2. **Multicollinearity**
3. **Homoscedasticity**
4. **Normality of Residuals**
5. **Independence of Residuals**

---

## ✅ **1. Setup: Load Data & Train Model**

We'll use the Boston Housing dataset (you can replace it with any dataset).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
```

```python
# Load dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Residuals
residuals = y_test - y_pred
```

---

## ✅ **2. Assumption Checks**

---

### 🧠 **A. Linearity**

**How to Detect:**

* Residual plot (should be randomly scattered)
* Partial regression plots

```python
# Residual plot
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Linearity Check: Residual Plot")
plt.show()
```

✅ **Random scatter** → linearity holds
❌ **Curve or trend** → violation

---

### 🧠 **B. Multicollinearity**

**How to Detect:**

* **Variance Inflation Factor (VIF)**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add constant for statsmodels
X_vif = sm.add_constant(X_train)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i+1) for i in range(len(X.columns))]
print(vif_data)
```

✅ **VIF < 5** → no multicollinearity
❌ **VIF > 5 or 10** → possible collinearity

---

### 🧠 **C. Homoscedasticity**

**How to Detect:**

* Residuals vs. predicted plot
* Breusch–Pagan test

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Breusch–Pagan test
_, pval, __, f_pval = het_breuschpagan(residuals, sm.add_constant(X_test))
print(f"Breusch–Pagan p-value: {pval}")
```

✅ **p-value > 0.05** → homoscedastic
❌ **p-value < 0.05** → heteroscedastic

---

### 🧠 **D. Normality of Residuals**

**How to Detect:**

* Histogram + KDE
* Q-Q plot
* Shapiro-Wilk or Jarque-Bera test

```python
# Histogram + KDE
sns.histplot(residuals, kde=True)
plt.title("Normality Check: Histogram of Residuals")
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Normality test
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print(f'Shapiro-Wilk p-value: {p}')
```

✅ **p-value > 0.05** → residuals are normal
❌ **p-value < 0.05** → non-normal

---

### 🧠 **E. Independence of Residuals**

**How to Detect:**

* **Durbin-Watson test** (for time-based data)
* **Autocorrelation plots**

```python
# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f'Durbin-Watson: {dw}')
```

✅ **DW ≈ 2** → no autocorrelation
❌ **DW << 2 or >> 2** → positive/negative autocorrelation

---

## ✅ Summary of Methods to Detect Assumptions

| Assumption             | Detection Methods                                                  |
| ---------------------- | ------------------------------------------------------------------ |
| Linearity              | Residual plot, Partial regression plot                             |
| No Multicollinearity   | VIF (Variance Inflation Factor), Correlation matrix                |
| Homoscedasticity       | Residual plot, **Breusch–Pagan test**, Goldfeld–Quandt test        |
| Normality of Residuals | Histogram + KDE, Q-Q Plot, **Shapiro-Wilk test**, Jarque-Bera test |
| Independence           | **Durbin-Watson statistic**, ACF/PACF plots for time series        |

---

Let me know if you’d like a **notebook version**, or want to run this on a **custom dataset** like CSV input.

---

Absolutely! Below is a detailed guide for **each assumption check** in **linear regression**, listing:

* ✅ All **available methods** (visual + statistical),
* 🔍 **When to use** them,
* ⚙️ **How to implement** each in Python.

---

# ✅ 1. **Linearity Assumption**

### 🎯 Goal:

Check if the relationship between predictors and target is **linear**.

---

### 🔎 **Available Methods**

| Method                              | Type        | How to Interpret                         |
| ----------------------------------- | ----------- | ---------------------------------------- |
| Residual vs. Predicted Plot         | Visual      | Random scatter ⇒ linearity holds         |
| Partial Regression Plots            | Visual      | Linearity per feature                    |
| Component plus residual plot        | Visual      | Useful for detecting nonlinearity        |
| Polynomial Terms / Box-Tidwell Test | Statistical | Detect nonlinear patterns quantitatively |

---

### ⚙️ **How to Perform**

#### ✅ 1. Residual vs. Fitted Plot

```python
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual vs. Fitted (Linearity)")
plt.show()
```

#### ✅ 2. Partial Regression Plot (statsmodels)

```python
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(sm.OLS(y_train, sm.add_constant(X_train)).fit())
plt.show()
```

#### ✅ 3. Component + Residual Plot (partial residual plot)

```python
from statsmodels.graphics.regressionplots import plot_ccpr_grid
model_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
plot_ccpr_grid(model_sm)
plt.show()
```

---

# ✅ 2. **Multicollinearity Assumption**

### 🎯 Goal:

Ensure independent variables are not **highly correlated**.

---

### 🔎 **Available Methods**

| Method                          | Type        | How to Interpret             |
| ------------------------------- | ----------- | ---------------------------- |
| Correlation Matrix              | Visual      | Corr > 0.8 → red flag        |
| Variance Inflation Factor (VIF) | Statistical | VIF > 5 or 10 → collinearity |
| Condition Number                | Statistical | Large values → instability   |

---

### ⚙️ **How to Perform**

#### ✅ 1. Correlation Matrix

```python
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

#### ✅ 2. Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i + 1) for i in range(X_train.shape[1])]
print(vif)
```

#### ✅ 3. Condition Number

```python
model_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print("Condition number:", np.linalg.cond(model_sm.model.exog))
```

---

# ✅ 3. **Homoscedasticity Assumption**

### 🎯 Goal:

Ensure the **variance of residuals is constant** across predictions.

---

### 🔎 **Available Methods**

| Method                    | Type        | How to Interpret                     |
| ------------------------- | ----------- | ------------------------------------ |
| Residuals vs. Fitted Plot | Visual      | Constant spread ⇒ good               |
| Breusch–Pagan Test        | Statistical | p > 0.05 ⇒ homoscedastic             |
| White’s Test              | Statistical | More general than Breusch–Pagan      |
| Goldfeld–Quandt Test      | Statistical | Tests increasing/decreasing variance |

---

### ⚙️ **How to Perform**

#### ✅ 1. Residuals vs. Fitted

(Already shown above)

#### ✅ 2. Breusch–Pagan Test

```python
from statsmodels.stats.diagnostic import het_breuschpagan
_, pval, __, _ = het_breuschpagan(residuals, sm.add_constant(X_test))
print("Breusch–Pagan p-value:", pval)
```

#### ✅ 3. White’s Test

```python
from statsmodels.stats.diagnostic import het_white
_, pval, __, _ = het_white(residuals, sm.add_constant(X_test))
print("White’s test p-value:", pval)
```

#### ✅ 4. Goldfeld–Quandt Test

```python
from statsmodels.stats.diagnostic import het_goldfeldquandt
_, pval, _ = het_goldfeldquandt(residuals, sm.add_constant(X_test))
print("Goldfeld–Quandt p-value:", pval)
```

---

# ✅ 4. **Normality of Residuals**

### 🎯 Goal:

Residuals should be **normally distributed**.

---

### 🔎 **Available Methods**

| Method                  | Type        | How to Interpret           |
| ----------------------- | ----------- | -------------------------- |
| Histogram + KDE         | Visual      | Bell-shaped curve ⇒ normal |
| Q-Q Plot                | Visual      | Points on line ⇒ normal    |
| Shapiro-Wilk Test       | Statistical | p > 0.05 ⇒ normal          |
| Kolmogorov–Smirnov Test | Statistical | p > 0.05 ⇒ normal          |
| Jarque-Bera Test        | Statistical | Common in econometrics     |

---

### ⚙️ **How to Perform**

#### ✅ 1. Histogram

```python
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()
```

#### ✅ 2. Q-Q Plot

```python
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot")
plt.show()
```

#### ✅ 3. Shapiro-Wilk Test

```python
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print("Shapiro-Wilk p-value:", p)
```

#### ✅ 4. Jarque-Bera Test

```python
from statsmodels.stats.stattools import jarque_bera
_, pval, _, _ = jarque_bera(residuals)
print("Jarque-Bera p-value:", pval)
```

---

# ✅ 5. **Independence of Residuals**

### 🎯 Goal:

Ensure residuals are **not autocorrelated**, especially in **time series**.

---

### 🔎 **Available Methods**

| Method                         | Type        | How to Interpret        |
| ------------------------------ | ----------- | ----------------------- |
| Durbin–Watson Test             | Statistical | DW ≈ 2 ⇒ independent    |
| Autocorrelation Function (ACF) | Visual      | No spikes ⇒ independent |
| Runs Test                      | Statistical | Detects randomness      |
| Breusch-Godfrey Test           | Statistical | Lagged autocorrelation  |

---

### ⚙️ **How to Perform**

#### ✅ 1. Durbin–Watson Test

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print("Durbin–Watson:", dw)
```

#### ✅ 2. ACF Plot

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals)
plt.title("ACF of Residuals")
plt.show()
```

#### ✅ 3. Breusch–Godfrey Test

```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
model_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
_, pval, _, _ = acorr_breusch_godfrey(model_sm)
print("Breusch-Godfrey p-value:", pval)
```

---

## 🧾 Summary Table

| Assumption            | Methods                                              | Recommended Tools           |
| --------------------- | ---------------------------------------------------- | --------------------------- |
| **Linearity**         | Residual Plot, Partial Regression, CCPR, Box-Tidwell | `matplotlib`, `statsmodels` |
| **Multicollinearity** | Correlation Matrix, VIF, Condition Number            | `pandas`, `statsmodels`     |
| **Homoscedasticity**  | Residual Plot, Breusch–Pagan, White’s Test           | `statsmodels`               |
| **Normality**         | Histogram, Q-Q Plot, Shapiro, JB, KS                 | `scipy`, `statsmodels`      |
| **Independence**      | Durbin–Watson, ACF, Breusch–Godfrey                  | `statsmodels`               |

---

Would you like this in a ready-to-run **Jupyter Notebook template** or packaged into a function/class that you can reuse on any dataset?

