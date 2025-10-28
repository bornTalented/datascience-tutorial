Supervised learning algorithms are trained on labeled data, meaning each training example includes an input and a corresponding output. Here’s a list of the most commonly used **supervised learning algorithms** in machine learning, grouped by type:

---

### **1. Linear Models**

* **Linear Regression** – for regression tasks
	* ❌ Classification  ✅ Regression
* **Logistic Regression** – for binary and multi-class classification
	* ✅ Classification ❌ Regression

---

### **2. Support Vector Machines**

* **Support Vector Classifier (SVC)**
	* ✅ Classification ❌ Regression
* **Support Vector Regression (SVR)** – variant of SVM for regression
	* ❌ Classification ✅ Regression

---

### **3. Tree-Based Methods** 
(have variants for classification and regression tasks)

* **Decision Tree**
* **Random Forest**
* **Gradient Boosting Machines (GBM)** – includes:
  * XGBoost
  * LightGBM
  * CatBoost
* **Extra Trees (Extremely Randomized Trees)**

---

### **4. Instance-Based Learning**

* **K-Nearest Neighbors (KNN)**
	* ✅ Classification ❌ Regression
* **K-Nearest Neighbors Regressor**
	* ❌ Classification ✅ Regression
---

### **5. Neural Networks**

* **Multi-Layer Perceptrons (MLP)**
	* ✅ Classification ❌ Regression
* **MLP Regressor**
	* ❌ Classification ✅ Regression
* **Convolutional Neural Networks (CNNs)** – mainly for image classification
	* ✅ Classification ❌ Regression
* **Recurrent Neural Networks (RNNs)** – for sequence data (can also be used in supervised settings)
	* ✅ Classification ❌ Regression

---

### **6. Bayesian Methods**

* **Naive Bayes (Gaussian, Multinomial, Bernoulli variants)**
	* ✅ Classification ❌ Regression
* **Bayesian Linear Regression**
	* ❌ Classification ✅ Regression
* **Bayesian Networks** (when used with labeled data)
	* ✅ Classification ❌ Regression

---

### **7. Ensemble Methods**
(have variants for classification and regression tasks)
* **Bagging**
* **Boosting**
* **Stacking**
* **Voting Classifier/Regressor**

---

### **8. Discriminant Analysis**

* **Linear Discriminant Analysis (LDA)**
	* ✅ Classification ❌ Regression
* **Quadratic Discriminant Analysis (QDA)**
	* ✅ Classification ❌ Regression

---

### **9. Regularized Models**

* **Ridge Regression (L2 regularization)**
	* ❌ Classification  ✅ Regression
* **Lasso Regression (L1 regularization)**
	* ❌ Classification  ✅ Regression
* **Elastic Net (combination of L1 and L2)**
	* ❌ Classification  ✅ Regression

---

Here is a **comprehensive list of supervised learning algorithms**, categorized by the **type of task** they are typically used for: **classification**, **regression**, or **both**.

---

## 🟡 **Algorithms That Can Be Used for Both Classification and Regression**

Some algorithms are flexible and can be configured for either task.

| Algorithm                                           | Tasks Supported                        |
| --------------------------------------------------- | -------------------------------------- |
| **Decision Trees**                                  | Classification, Regression             |
| **Random Forest**                                   | Classification, Regression             |
| **Gradient Boosting (XGBoost, LightGBM, CatBoost)** | Classification, Regression             |
| **K-Nearest Neighbors (KNN)**                       | Classification, Regression             |
| **Support Vector Machines**                         | Classification (SVC), Regression (SVR) |
| **Neural Networks (MLP)**                           | Classification, Regression             |
| **Ensemble Methods**                                | Classification, Regression             |
