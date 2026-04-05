### 📘 What is **Information Value (IV)**?

**Information Value (IV)** is a metric used primarily in **credit scoring and risk modeling**, but it’s also very useful for **feature selection** in classification tasks.

It measures the **predictive power of a variable** in separating the binary target classes (typically “good” vs “bad”, or 0 vs 1).

---

## 🔸 Why is IV Useful?

* It helps identify **which variables are most informative** for distinguishing between classes.
* Variables with higher IV values are typically **better predictors**.
* Especially useful for **binary classification** problems.

---

## 🔸 How is IV Calculated?

It is based on **Weight of Evidence (WoE)**, which transforms raw values of a categorical or binned continuous variable into a scale showing its predictive power.

### ✅ Step-by-Step:

Let’s assume you have:

* A binary target variable $( y \in {0, 1} )$
* A predictor feature $( X )$, which is either categorical or binned into discrete intervals.

---

### **1. Bin the Feature**

For continuous variables, bin them into intervals (e.g., using quantiles). For categorical, use levels.

---

### **2. Calculate WoE (Weight of Evidence)** for each bin:

$\text{WoE}_i = \ln \left( \frac{\text{Distribution of Good}_i}{\text{Distribution of Bad}_i} \right)$

Where:

*  $\text{Distribution of Good}_i = \frac{\text{Number of good (y=0) in bin } i}{\text{Total good}}$ 
*  $\text{Distribution of Bad}_i = \frac{\text{Number of bad (y=1) in bin } i}{\text{Total bad}}$ 

---

### **3. Calculate IV**

$\text{IV} = \sum_i \left( \left( \text{Distribution of Good}_i - \text{Distribution of Bad}_i \right) \cdot \text{WoE}_i \right)$

This is done over all bins ( i ) of the variable.

---

## 🔸 IV Score Interpretation

| Information Value (IV) | Predictive Power         |
| ---------------------- | ------------------------ |
| IV < 0.02              | Useless predictor        |
| 0.02 ≤ IV < 0.1        | Weak predictor           |
| 0.1 ≤ IV < 0.3         | Medium predictor         |
| 0.3 ≤ IV < 0.5         | Strong predictor         |
| IV ≥ 0.5               | Suspicious (overfitting) |

---

## 🔸 Example Calculation

Suppose you have a feature "Age" binned into three bins:

| Age Bin | Good (y=0) | Bad (y=1) | Total |
| ------- | ---------- | --------- | ----- |
| <30     | 40         | 10        | 50    |
| 30-60   | 30         | 30        | 60    |
| >60     | 10         | 40        | 50    |

### Step-by-Step:

* Total Good = 40 + 30 + 10 = 80
* Total Bad = 10 + 30 + 40 = 80

Now compute **distribution** and **WoE**:

| Bin   | Dist. Good    | Dist. Bad     | WoE                      | IV Contribution                |
| ----- | ------------- | ------------- | ------------------------ | ------------------------------ |
| <30   | 40/80 = 0.5   | 10/80 = 0.125 | ln(0.5 / 0.125) = 1.386  | (0.5 - 0.125)*1.386 = 0.519    |
| 30-60 | 30/80 = 0.375 | 30/80 = 0.375 | ln(1) = 0                | 0                              |
| >60   | 10/80 = 0.125 | 40/80 = 0.5   | ln(0.125 / 0.5) = -1.386 | (0.125 - 0.5)*(-1.386) = 0.519 |

**Final IV = 0.519 + 0 + 0.519 = 1.038** → very strong predictor.

---

## 🔸 How to Use IV for Feature Selection

1. **Compute IV** for each feature (after binning if needed).
2. **Rank features by IV** (descending).
3. **Set a threshold**:
	* Keep features with IV > 0.02 or > 0.1 depending on your needs.
	* Drop features with IV < 0.02 (useless predictor).
4. Optionally combine with other methods (e.g., mutual information, recursive feature elimination).

---

## 🔸 Python Code Example

```python
import pandas as pd
import numpy as np

def calc_woe_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()
    df[feature] = pd.qcut(df[feature], bins, duplicates='drop') if pd.api.types.is_numeric_dtype(df[feature]) else df[feature]
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad']).replace({np.inf: 0, -np.inf: 0})
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
    return grouped[['woe', 'iv']], grouped['iv'].sum()
```

---

## ✅ Summary

| Concept     | Description                                           |
| ----------- | ----------------------------------------------------- |
| **IV**      | Measures feature's ability to separate target classes |
| **Use**     | Feature selection in binary classification            |
| **Steps**   | Bin → Compute Good/Bad distribution → WoE → IV        |
| **High IV** | Strong feature                                        |
| **Low IV**  | Weak or no predictive power                           |

Would you like a full notebook with visualization of IV distributions across features?
