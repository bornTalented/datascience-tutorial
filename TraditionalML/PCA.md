Principal Component Analysis (PCA) is a fundamental statistical technique used for **dimensionality reduction** in data preprocessing and exploratory data analysis. Its goal is to **reduce the number of features** (dimensions) in the dataset while **preserving as much variability (information)** as possible.

**Simple Analogy:** 
Think of a 3D object (say a sphere). If you look at its shadow on a wall (a 2D projection), you're trying to preserve the **shape and structure** of the object in a lower dimension. That’s PCA!

In simple terms, it's a way to simplify complex datasets by transforming a large number of potentially correlated variables into a smaller set of uncorrelated variables, called **principal components**, while retaining as much of the original information (variance) as possible.

Imagine you have a dataset with many features, like height, weight, age, blood pressure, cholesterol levels, etc., for a group of people. Some of these features might be correlated (e.g., height and weight are often related). PCA helps you find the most important "directions" or "patterns" in this data that capture the most variation, allowing you to represent the data with fewer dimensions without losing much valuable insight.

Here's a breakdown of the key concepts:

**A. The "Curse of Dimensionality"**
As the number of features (dimensions) in a dataset increases, it becomes harder to visualize, analyze, and process the data. This is known as the "curse of dimensionality," which can lead to:
* **Increased computational cost:** More features mean more calculations.
* **Overfitting:** Models might perform well on training data but poorly on new, unseen data due to too many features capturing noise.
* **Difficulty in visualization:** It's impossible to visualize data in more than three dimensions.

**B. The Goal of PCA**
PCA aims to address the curse of dimensionality by:
* **Reducing the number of variables:** Creating a smaller set of new variables.
* **Preserving maximum variance:** Ensuring that the new variables (principal components) capture as much of the original data's variability as possible.
* **Creating uncorrelated components:** The principal components are orthogonal (perpendicular) to each other, meaning they are independent and don't contain redundant information.

**C. How PCA Works (Intuitive Steps):**

**Prerequisites:**

* **Data Matrix:** You have a dataset represented as a matrix, let's call it $X$. Each row represents an observation (or sample), and each column represents a variable (or feature).
* **Linear Algebra Basics:** Familiarity with concepts like mean, variance, covariance, matrices, matrix multiplication, eigenvectors, and eigenvalues is helpful.

**Steps to Compute PCA:**

**1. Data Standardization (Centering and Scaling)**

This is a critical first step because PCA is sensitive to the scale of your variables. If one variable has a much larger range than others ( will inherently have larger variances), it might dominate the analysis (principal components), regardless of their actual importance. Therefore, it's crucial to standardize the data (e.g., by subtracting the mean and dividing by the standard deviation) so that all variables contribute equally.

* **Centering:** For each variable (column) in your dataset, subtract its mean from every data point in that column. This shifts the data so that the mean of each variable is 0.
    
    Let $X_{ij}$ be the value of the $j$-th variable for the $i$-th observation.
    Let $\mu_j$ be the mean of the $j$-th variable.
    
    The centered value $X'_{ij}$ is:
    $X'_{ij} = X_{ij} - \mu_j$
    
* **Scaling (Optional but Recommended):** After centering, divide each variable's values by its standard deviation. This ensures that all variables have a unit standard deviation (typically 1).
    
    Let $\sigma_j$ be the standard deviation of the $j$-th variable.
    
    The standardized value $Z_{ij}$ (often called a Z-score) is:
    $Z_{ij} = \frac{X'_{ij}}{\sigma_j} = \frac{X_{ij} - \mu_j}{\sigma_j}$
    
    The data matrix after standardization is often denoted as $Z$.

**2. Compute the Covariance Matrix**

This matrix shows how each variable in your dataset varies with every other variable. It helps identify relationships and redundancies between them. It's a square, symmetric matrix where the diagonal elements are the variances of individual variables, and the off-diagonal elements are the covariances between pairs of variables.

For a standardized data matrix $Z$ with $n$ observations and $p$ variables, the covariance matrix $C$ (or $\Sigma$) is calculated as:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$
$C = \frac{1}{n-1} Z^T Z$

Where:
* $Z^T$ is the transpose of the standardized data matrix $Z$.
* $n$ is the number of observations.
* The division by $(n-1)$ is for the sample covariance (Bessel's correction).

**Interpretation of Covariance:**
* A positive covariance indicates that two variables tend to increase or decrease together.
* A negative covariance indicates that one variable tends to increase when the other decreases.
* A covariance close to zero suggests little to no linear relationship.

**3. Compute Eigenvalues and Eigenvectors of the Covariance Matrix**

This is the mathematical core of PCA.

* **Eigenvectors:** These represent the "principal components" themselves. These are the directions (axes) in the data that capture the most variance. Each eigenvector is a unique direction.
* **Eigenvalues:** Each eigenvector has a corresponding eigenvalue, which indicates the amount of variance explained by that principal component. A larger eigenvalue means its corresponding eigenvector captures more of the data's variability.

To find the eigenvalues ($\lambda$) and eigenvectors ($v$) of the covariance matrix $C$, you solve the characteristic equation:

$C v = \lambda v$

Which can be rewritten as:

$(C - \lambda I) v = 0$

Where:
* $I$ is the identity matrix of the same dimensions as $C$.
* To find non-trivial solutions for $v$, the determinant of $(C - \lambda I)$ must be zero:
    $\det(C - \lambda I) = 0$

Solving this polynomial equation gives you the eigenvalues ($\lambda$). For each eigenvalue, you then substitute it back into the equation $(C - \lambda I) v = 0$ to find its corresponding eigenvector ($v$).

**4. Sort Eigenvalues and Corresponding Eigenvectors**

Once you have all the eigenvalue-eigenvector pairs, sort them in descending order based on their eigenvalues. The eigenvector with the largest eigenvalue is the first principal component (PC1), explaining the most variance. The eigenvector with the second largest eigenvalue is the second principal component (PC2), and so on.

**5. Select a Subset of Principal Components**

You don't typically use all principal components. The goal of PCA is dimensionality reduction, so you choose a smaller number of components that still capture most of the relevant information (variance).

* **Scree Plot:** A common method is to plot the eigenvalues in descending order. You look for an "elbow" point where the eigenvalues start to level off. Components before the elbow are usually retained.
* **Explained Variance Ratio:** Another approach is to select enough principal components to account for a certain percentage of the total variance (e.g., 90% or 95%). This is calculated as the sum of the selected eigenvalues divided by the sum of all eigenvalues.
 $$
\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}
$$

Let $k$ be the number of principal components you choose to keep. You form a **feature vector** (or projection matrix) $W$ by taking the first $k$ eigenvectors (the ones with the largest eigenvalues) and arranging them as columns.

**6. Transform the Data**

Finally, project your standardized data onto the selected principal components. This creates a new, lower-dimensional dataset. This new dataset is a linear combination of the original variables, represented in the new coordinate system defined by the principal components.

Let $Z$ be your standardized data matrix (from Step 1) and $W$ be your feature vector (from Step 5). The new transformed data matrix $Y$ is:

$Y = Z W$

Where:
* $Y$ is the new dataset with reduced dimensions. Each column in $Y$ represents a principal component.
* The rows of $Y$ are the new coordinates of your original data points in the principal component space.

**PCA Algorithm (Step-by-Step)**

1. **Standardize the data** (mean = 0, variance = 1).
2. Compute the **covariance matrix**.
3. Compute **eigenvectors** and **eigenvalues**.
4. **Sort eigenvalues** and their eigenvectors.
5. Choose top $k$ eigenvectors (principal components).
6. Project original data onto new feature space:

**Example (Conceptual):**

Imagine you have data on "height" and "weight."

1.  **Standardize:** Transform height and weight to Z-scores.
2.  **Covariance Matrix:** Calculate the covariance between height and weight, and their individual variances.
3.  **Eigenvalues/Eigenvectors:** Find the two principal components. PC1 might represent a general "size" factor (since height and weight are correlated), explaining most of the variance. PC2 would be orthogonal to PC1, explaining the remaining variance (perhaps related to body proportionality).
4.  **Sort and Select:** You'd likely keep PC1 as it explains the most variance.
5.  **Transform:** Project the original height and weight data onto the PC1 axis, effectively representing each person's "size" with a single number instead of two.

**4. Applications of PCA:**

* **Dimensionality Reduction:** The most common use, simplifying datasets for easier analysis and faster model training.
* **Data Visualization:** Reducing data to 2 or 3 dimensions allows for easier plotting and visual identification of patterns, clusters, and outliers.
* **Noise Reduction:** By focusing on the components that explain the most variance, PCA can effectively filter out noise in the data.
* **Feature Extraction:** PCA creates new features (principal components) that are combinations of the original ones, which can be more informative for machine learning models.
* **Data Compression:** Storing data in a reduced dimension saves space.
* **Handling Multicollinearity:** Since principal components are uncorrelated, PCA can address issues arising from highly correlated features in regression models.

**5. ⚠️ Limitations of PCA**

| Limitation                        | Description                                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Linear Assumption**             | PCA assumes linear relationships. It doesn't capture non-linear patterns.                       |
| **Loss of Interpretability**      | Principal components are **combinations** of original features, so interpreting them is harder. |
| **Sensitive to Scaling**          | PCA is sensitive to feature scaling. Always **standardize** the data.                           |
| **Not Good for Categorical Data** | PCA works best on continuous variables.                                                         |
| **Affected by Outliers**          | Outliers can dominate the variance and skew results.                                            |
| **Variance ≠ Importance**         | PCA assumes higher variance means more importance, which isn't always true.                     |

---

In essence, PCA helps you find the most meaningful and independent ways to look at your data, making it more manageable and insightful for various analytical and machine learning tasks
**Implementing PCA (Practically):**

While understanding the manual computation is valuable, in practice, you'll almost always use libraries for PCA due to the complexity of eigenvalue decomposition for larger datasets.

**Using Python with scikit-learn:**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# 1. Create some sample data
data = np.array([
    [170, 65],  # Height, Weight
    [175, 70],
    [160, 55],
    [180, 75],
    [165, 60],
    [190, 85]
])

df = pd.DataFrame(data, columns=['Height', 'Weight'])
print("Original Data:\n", df)

# 2. Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("\nScaled Data (mean=0, std=1):\n", pd.DataFrame(scaled_data, columns=['Height_scaled', 'Weight_scaled']))

# 3. Perform PCA
# You can specify the number of components or a variance threshold
pca = PCA(n_components=2) # Keep all 2 components for illustration
# pca = PCA(n_components=0.95) # Keep components that explain 95% of variance

principal_components = pca.fit_transform(scaled_data)

# Print explained variance ratio for each component
print("\nExplained Variance Ratio for each Principal Component:")
print(pca.explained_variance_ratio_)
print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.2f}")

# Print the Principal Components (Eigenvectors)
print("\nPrincipal Components (Eigenvectors - directions of maximum variance):\n", pca.components_)

# Print the transformed data
# Each column is a principal component
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print("\nTransformed Data (Principal Components):\n", pc_df)

# Visualize the data in the new PCA space (if 2 components)
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data Projected onto Principal Components')
plt.grid(True)
plt.show()

# You can also access the eigenvalues (explained variance)
print("\nEigenvalues (amount of variance explained by each component):")
print(pca.explained_variance_)
```

This code snippet demonstrates the practical application of PCA, performing the steps described above efficiently.

> [!IMPORTANT]
> 
> ### PCA Interview Questions and Answers
> 
> **Q1: What is Principal Component Analysis (PCA) and what is its primary goal?**
> 
> **A1:** PCA is an unsupervised linear dimensionality reduction technique. Its primary goal is to transform a high-dimensional dataset into a lower-dimensional one while retaining as much of the original variance (information) as possible. It achieves this by identifying new orthogonal (uncorrelated) variables called "principal components" that are linear combinations of the original variables.
> 
> **Q2: Why is dimensionality reduction important, and when would you use PCA?**
> 
> **A2:** Dimensionality reduction is important for several reasons:
> * **Mitigating the Curse of Dimensionality:** High-dimensional data can lead to increased computational cost, difficulty in visualization, and overfitting in machine learning models (more features can capture noise).
> * **Improved Model Performance:** Reducing noise and redundancy can sometimes lead to simpler, more robust models that generalize better.
> * **Faster Training:** Fewer features mean faster training times for many algorithms.
> * **Better Visualization:** Reducing data to 2 or 3 dimensions allows for easy plotting and identification of patterns, clusters, and outliers.
> * **Data Compression:** Storing data in a reduced dimension saves space.
> 
> You would use PCA when:
> * You have a dataset with a large number of features.
> * Some of your features are highly correlated (multicollinearity).
> * You need to visualize high-dimensional data.
> * You want to reduce noise in your dataset.
> * You want to improve the performance or training time of a machine learning model.
> 
> **Q3: Explain the core steps involved in computing PCA.**
> 
> **A3:** The core steps to compute PCA are:
> 
> 1.  **Data Standardization (Centering and Scaling):** Subtract the mean from each feature and divide by its standard deviation. This is crucial because PCA is sensitive to the scale of variables.
> 2.  **Compute the Covariance Matrix:** Calculate the covariance matrix of the standardized data. This matrix shows how each pair of variables varies together.
> 3.  **Compute Eigenvalues and Eigenvectors:** Find the eigenvalues and corresponding eigenvectors of the covariance matrix. Eigenvectors represent the principal components (directions of maximum variance), and eigenvalues indicate the amount of variance explained by each component.
> 4.  **Sort Eigenvalues and Eigenvectors:** Order the eigenvalue-eigenvector pairs in descending order based on the eigenvalues. The eigenvector with the largest eigenvalue is the first principal component, and so on.
> 5.  **Select a Subset of Principal Components:** Choose the top 'k' principal components (eigenvectors) that explain a significant cumulative proportion of the total variance (e.g., 90-95%). This can be determined by a scree plot or by examining the explained variance ratio.
> 6.  **Transform the Data:** Project the original (standardized) data onto the chosen 'k' principal components. This results in a new, lower-dimensional dataset.
> 
> **Q4: What is the significance of eigenvalues and eigenvectors in PCA?**
> 
> **A4:**
> * **Eigenvectors:** These are the principal components themselves. They represent the directions (axes) in the feature space along which the data varies the most. They are orthogonal to each other, meaning they are uncorrelated.
> * **Eigenvalues:** Each eigenvalue corresponds to an eigenvector and quantifies the amount of variance in the data that is captured along that eigenvector's direction. A larger eigenvalue means its corresponding eigenvector (principal component) explains more variance in the data. By sorting eigenvalues in descending order, we can identify the most important principal components.
> 
> **Q5: How do you determine the number of principal components to keep?**
> 
> **A5:** There are several common methods:
> 
> * **Scree Plot:** Plot the eigenvalues in descending order. Look for an "elbow" point where the explained variance starts to level off. Components before the elbow are typically retained. (PrincipleComponent vs EigenValue)
> * **Cumulative Explained Variance:** Calculate the cumulative sum of the explained variance ratio (each eigenvalue divided by the sum of all eigenvalues). Select enough components to capture a desired percentage of the total variance (e.g., 80%, 90%, 95%).
> * **Rule of Thumb (Less Common):** Sometimes, 'k' is chosen such that $k \le \text{number of features}$.
> * **Domain Knowledge:** In some cases, domain expertise might guide the selection, especially if certain components have clear interpretations.
> * **Cross-validation:** For machine learning tasks, you can use cross-validation to evaluate model performance with different numbers of principal components and choose the 'k' that optimizes performance.
> 
> **Q6: Is PCA a supervised or unsupervised learning technique? Explain why.**
> 
> **A6:** PCA is an **unsupervised** learning technique.
> 
> It is unsupervised because it does not use any information about the target variable or labels during its computation. PCA only analyzes the relationships (variance and covariance) within the input features ($X$) themselves to find the principal components. It aims to find the optimal projection that captures the most variance in the input data, irrespective of any output labels.
> 
> **Q7: When should you NOT use PCA? What are its limitations?**
> 
> **A7:** You should be cautious or avoid using PCA when:
> 
> * **Interpretability is paramount:** While PCA reduces dimensions, the new principal components are linear combinations of the original features and may not have clear, intuitive meanings. If you need to understand the direct impact of individual original features, PCA might obscure that.
> * **Non-linear relationships:** PCA is a linear transformation. If the intrinsic structure of your data is non-linear (e.g., data points lie on a manifold), PCA might not effectively capture the true underlying dimensions. Kernel PCA can sometimes address this.
> * **Loss of Information (if not careful):** While PCA aims to retain most variance, it *does* involve discarding information. If the "discarded" variance (the components with smaller eigenvalues) contains crucial information for your specific task (e.g., separating classes), then PCA might hurt performance.
> * **Data is already low-dimensional:** If your dataset already has a manageable number of features, PCA might not offer significant benefits and could even add unnecessary complexity.
> * **Scaling matters:** If features are not standardized properly, features with larger scales can unduly influence the principal components.
> 
> **Q8: Explain the difference between PCA and Factor Analysis.**
> 
> **A8:** Both PCA and Factor Analysis are dimensionality reduction techniques, but they have different underlying assumptions and goals:
> 
> * **PCA:**
>     * **Goal:** To explain the *total variance* in the observed variables.
>     * **Assumption:** Principal components are linear combinations of the observed variables.
>     * **Nature:** A data reduction technique. It transforms the observed variables into a new set of orthogonal variables (PCs) without assuming any latent structure.
>     * **Purpose:** Primarily for data compression, visualization, and noise reduction.
> 
> * **Factor Analysis:**
>     * **Goal:** To explain the *covariance* among observed variables by identifying underlying, unobserved (latent) "factors."
>     * **Assumption:** Observed variables are linear combinations of a smaller set of common factors plus unique error terms. It assumes there's an underlying causal structure.
>     * **Nature:** A statistical model that infers latent constructs.
>     * **Purpose:** To discover latent constructs or relationships that explain the observed correlations among variables. Often used in psychology, social sciences, and market research.
> 
> In summary, PCA seeks to summarize the data into fewer dimensions, while Factor Analysis seeks to explain the relationships between observed variables by attributing them to common underlying factors.
> 
> **Q9: Why is it important to standardize the data before applying PCA?**
> 
> **A9:** It's critical to standardize data before PCA because PCA is sensitive to the scale of the variables.
> 
> * **Impact of Scale:** Variables with larger values or wider ranges will naturally have larger variances. If not standardized, these variables would disproportionately influence the principal components, potentially dominating the first few components even if they are not truly more "important" or contain more meaningful information.
> * **Equal Contribution:** Standardization (e.g., Z-score normalization) ensures that all variables contribute equally to the calculation of the covariance matrix and, consequently, to the principal components. It places all features on a common scale, typically with a mean of 0 and a standard deviation of 1.
> 
> **Q10: Can PCA be used for anomaly detection? If so, how?**
> 
> **A10:** Yes, PCA can be used for anomaly detection, particularly for **reconstruction error-based anomaly detection**.
> 
> Here's how:
> 1.  **Train PCA:** Apply PCA to your *normal* (non-anomalous) training data. Select a number of principal components that capture most of the variance (e.g., 95%).
> 2.  **Reconstruct Data:** For any new data point, project it onto the selected principal components and then "reconstruct" it back into the original feature space using the inverse transformation.
> 3.  **Calculate Reconstruction Error:** Compute the difference (e.g., Euclidean distance) between the original data point and its reconstructed version.
> 4.  **Identify Anomalies:** Normal data points, which conform to the underlying structure captured by the principal components, will have a low reconstruction error. Anomalous data points, which deviate significantly from this structure, will have a high reconstruction error.
> 5.  **Thresholding:** A threshold is set on the reconstruction error. Data points exceeding this threshold are flagged as anomalies.
> 
> This method works well when anomalies represent deviations from the dominant linear relationships captured by PCA.
> 
> **Q11: What is a "loading" in PCA, and what does it tell you?**
> 
> **A11:** A "loading" in PCA refers to the coefficients of the original variables when forming a principal component. In simpler terms, if a principal component is a linear combination of original features (e.g., $PC1 = a \cdot \text{Feature1} + b \cdot \text{Feature2} + c \cdot \text{Feature3}$), then $a$, $b$, and $c$ are the loadings for Feature1, Feature2, and Feature3 on PC1, respectively.
> 
> What they tell you:
> * **Strength and Direction of Relationship:** Loadings indicate how strongly each original variable influences a principal component and in what direction (positive or negative). A large absolute loading value means that the original variable contributes significantly to that principal component.
> * **Interpretation of Components:** By examining the loadings of a principal component, you can often interpret what that component represents. For example, if PC1 has high positive loadings for "height," "weight," and "shoe size," you might interpret PC1 as a "general size" component.
> * **Variable Contribution:** Squaring the loadings (and dividing by the sum of squared loadings for that component) can give you the proportion of variance of the original variable explained by that principal component.
> 
> **Q12: How does PCA handle categorical variables?**
> 
> **A12:** PCA, in its traditional form, is designed for **numerical (continuous)** data. It relies on calculating means, variances, and covariances, which are well-defined for numerical features.
> 
> When dealing with categorical variables:
> 
> * **One-Hot Encoding:** The most common approach is to convert categorical variables into numerical ones using techniques like one-hot encoding. However, this can significantly increase dimensionality and introduce sparsity, which might not be ideal for PCA.
> * **Correspondence Analysis (CA) or Multiple Correspondence Analysis (MCA):** These are specialized techniques designed for analyzing categorical data and are conceptually similar to PCA but adapted for nominal scales.
> * **Categorical PCA (CATPCA):** Some statistical software packages offer extensions like CATPCA, which can handle mixed data types or specifically designed for categorical variables.
> 
> It's generally recommended to use PCA on numerical data or to preprocess categorical features thoughtfully before applying PCA.
> 
> **Q13: Can PCA cause information loss? If so, why and how do you mitigate it?**
> 
> **A13:** Yes, PCA can cause **information loss**.
> 
> * **Why:** When you reduce the dimensionality by selecting only a subset of principal components (e.g., keeping only the top 'k' components), you are explicitly discarding the components with the smallest eigenvalues. These discarded components, by definition, explain the least variance in the data. While they contain the "least" variance, they still contain *some* information.
> * **How to Mitigate:**
>     * **Careful Component Selection:** Use methods like scree plots or cumulative explained variance plots to ensure you retain enough components to capture a significant proportion (e.g., 90-95%) of the total variance.
>     * **Evaluate Downstream Task Performance:** Don't just rely on explained variance. If you're using PCA as a preprocessing step for a machine learning model, evaluate the model's performance (e.g., accuracy, F1-score) after PCA. If performance drops significantly, you might need to retain more components or reconsider if PCA is appropriate.
>     * **Consider Alternatives:** If information loss is critical and PCA is not suitable, explore other dimensionality reduction techniques (e.g., t-SNE for visualization, autoencoders for non-linear reduction) or feature selection methods.
> 
> **Q14: Describe a scenario where PCA would be beneficial in a real-world application.**
> 
> **A14:**
> **Scenario: Image Compression and Recognition**
> 
> Imagine you're working with a dataset of facial images, each represented by thousands of pixels (features).
> 
> 1.  **The Problem:**
>     * **High Dimensionality:** A 100x100 pixel grayscale image has 10,000 features. Training a face recognition model directly on this high-dimensional data can be computationally expensive and prone to overfitting.
>     * **Redundancy:** Adjacent pixels in an image are often highly correlated.
> 
> 2.  **PCA Application:**
>     * **Dimensionality Reduction:** Apply PCA to the image dataset. PCA can find the "eigenfaces" (principal components) that capture the most significant variations in facial features across the dataset.
>     * **Feature Extraction:** Instead of using 10,000 pixel values, each image can be represented by a much smaller number of principal components (e.g., 50-200 components), which are essentially compact representations of the facial structure.
>     * **Noise Reduction:** Minor variations or noise in individual pixels might be relegated to lower-variance components, effectively reducing noise.
>     * **Improved Recognition:** A face recognition algorithm can then be trained on these lower-dimensional principal components, leading to faster training times, reduced memory usage, and potentially better generalization due to less noise and redundancy.
>     * **Image Compression:** Storing images as their principal component scores is a form of data compression, saving storage space.
> 
> This is a classic example of PCA's power in feature extraction and dimensionality reduction for complex data like images.
> 
> ---