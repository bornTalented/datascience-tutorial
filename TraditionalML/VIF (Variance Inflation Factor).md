**VIF (Variance Inflation Factor)** — an important concept for detecting **multicollinearity** in regression models.

---

## 📘 What is VIF?

**VIF (Variance Inflation Factor)** quantifies how much the **variance of a regression coefficient** is inflated due to **multicollinearity** with other predictors.

* High VIF → the variable is highly correlated with other variables → unstable estimates → overfitting risk.
* It is used **only for continuous independent variables**, mostly in **linear regression**.

---

## 🔸 Why Is VIF Important?

* Multicollinearity inflates coefficient variances → less reliable estimates.
* High multicollinearity makes it hard to distinguish the individual effect of variables.
* VIF helps detect and remove redundant or strongly correlated features.

---

## 🔸 How Is VIF Calculated?

For each predictor $X_j$, you **regress it against all the other features**, and compute:

$\text{VIF}_j = \frac{1}{1 - R_j^2}$

Where:

* $R_j^2$ : the **R-squared** value obtained by regressing feature ( $X_j$ ) on all the other features.

---

## 🔹 Step-by-Step Example

1. Pick feature ( $X_1$ ).
2. Regress $X_1$  on all other features: ( $X_2, X_3, ..., X_n$ )
3. Compute $R_1^2$
4. Compute $\text{VIF}_1 = \frac{1}{1 - R_1^2}$

Repeat for each feature.

---

## 🔸 Interpretation of VIF Values

| VIF Value | Interpretation                    |
| --------- | --------------------------------- |
| 1         | No multicollinearity              |
| 1–5       | Moderate correlation (acceptable) |
| >5        | High correlation (caution)        |
| >10       | Very high (problematic)           |

These thresholds are flexible (some domains use >5, some use >10 as the warning zone).

---

## 🔸 How to Use VIF for Feature Selection

### ✅ Strategy:

1. Compute VIF for all predictors.
2. **Iteratively remove** the variable with the **highest VIF** (if above threshold).
3. Recompute VIFs after each removal.
4. Stop when all VIFs are below the desired threshold.

---

## 🔹 Python Example

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def compute_vif(df):
    X = add_constant(df)  # Add intercept
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data.drop(index=0)  # Drop the constant term
```

### Example Usage:

```python
from sklearn.datasets import load_boston
import pandas as pd

# Load dataset
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Compute VIF
compute_vif(df)
```

---

## 🔸 Difference: IV vs VIF

| Metric | Purpose                        | Applied On                | Helps Detect         |
| ------ | ------------------------------ | ------------------------- | -------------------- |
| IV     | Predictive power               | Feature vs Target         | Unimportant features |
| VIF    | Multicollinearity (redundancy) | Feature vs Other Features | Redundant features   |

* **Use IV** to **select important features**
* **Use VIF** to **eliminate redundant features**

---

## ✅ Summary

| Concept       | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| **VIF**       | Measures how much a feature is linearly explained by other features |
| **Formula**   | $\text{VIF}_j = \frac{1}{1 - R_j^2}$                                |
| **Usage**     | Detect multicollinearity in regression                              |
| **Ideal VIF** | Close to 1                                                          |
| **Action**    | Drop or combine variables with high VIF                             |

Would you like a notebook to calculate both IV and VIF for a dataset to demonstrate **combined feature selection** strategy?
