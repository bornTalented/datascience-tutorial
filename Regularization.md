### The Problem: Overfitting

Before we talk about regularization, let's understand the problem it solves: **Overfitting**.

Imagine you're studying for an exam.
* **Underfitting:** You don't study enough. You only learn the very basic concepts and don't pick up on any details. When you get to the exam, you perform poorly because your knowledge is too general.
* **Just Right:** You study diligently, understand the core concepts, and also grasp the nuances and exceptions. You do well on the exam because you've learned to generalize well.
* **Overfitting:** You *memorize* every single practice question, every example, every footnote in the textbook. You can ace those specific practice questions, but if the exam asks a question phrased slightly differently, or presents a new problem based on the same concepts, you get stuck. You've learned the *training data* (practice questions) perfectly, but you haven't learned to *generalize* to new, unseen data (the actual exam).

In machine learning, overfitting happens when our model learns the training data *too well*, including the noise and specific patterns that aren't representative of the underlying relationship. When this overfitted model encounters new, unseen data, its performance drops significantly because it can't generalize.

**Signs of Overfitting:**
* Very high accuracy on the training data.
* Significantly lower accuracy on the validation or test data.
* Complex models with many features and large coefficients.

### The Solution: Regularization

Regularization is a technique that modifies the learning algorithm to **prevent overfitting** by **penalizing overly complex models**. It essentially encourages the model to be simpler and more generalizable.

How does it do this? By adding a **penalty term** to the model's cost function (also known as the loss function). The cost function is what the model tries to minimize during training. By adding this penalty, we're telling the model, "Hey, while you're trying to minimize errors, also try to keep your parameters (weights) small."

Let's look at the general form of a cost function:

$$\text{Cost Function} = \text{Loss} + \text{Regularization Term}$$

Where:
* **Loss:** Measures how well your model is performing on the training data (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
* **Regularization Term:** This is the penalty. It's usually a function of the model's coefficients (weights).

Regularization helps keep the model **simple** by penalizing large weights (parameters).
### Types of Regularization

The two most common types of regularization are L1 and L2 regularization.
Suppose our cost function (for regression) is:
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$
#### 1. L2 Regularization (Ridge Regression)

**Concept:** L2 regularization adds a penalty proportional to the **square of the magnitude of the coefficients** to the loss function.

**Mathematical Form:**

For a linear regression model, the cost function with L2 regularization looks like this:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

Where:
* $J(\theta)$ is the total cost function.
* $\lambda$ (lambda) is the **regularization hyperparameter**. This is a value you tune.
    * If $\lambda = 0$, no regularization is applied.
    * As $\lambda$ increases, the penalty for large weights increases, forcing the weights to be smaller.
* $\sum_{j=1}^{n} \theta_j^2$ is the sum of the squares of all the weights (excluding the bias term, $b$, as it doesn't typically contribute to complexity).

**How it works:**
L2 regularization pushes the coefficients towards zero, but **doesn't force them to be exactly zero**. It effectively shrinks the coefficients, making the model simpler and less sensitive to individual data points. Imagine a ball rolling down a hill (our loss function) and also being pulled towards the origin (due to the L2 penalty).

**When to use it:**
* When you have many features and want to shrink the coefficients of less important features.
* When you suspect multicollinearity (features are highly correlated).

#### 2. L1 Regularization (Lasso Regression)

**Concept:** L1 regularization adds a penalty proportional to the **absolute value of the magnitude of the coefficients** to the loss function.

**Mathematical Form:**

For a linear regression model, the cost function with L1 regularization looks like this:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

Where:
* The terms are similar to L2, but the penalty is $\lambda \sum_{j=1}^{n} |\theta_j|$.

**How it works:**
L1 regularization has a unique property: it can **force some coefficients to become exactly zero**. This means it performs **feature selection**, effectively eliminating features that are not very important to the model.

**When to use it:**
* When you have a very large number of features and want to select a smaller subset of important features.
* When you want a more interpretable model, as it simplifies the model by setting some coefficients to zero.

### The Regularization Hyperparameter ($\lambda$)

This is a critical part of regularization. $\lambda$ controls the strength of the regularization.
* **Small $\lambda$**: Less regularization, model can still overfit.
* **Large $\lambda$**: More regularization, coefficients are pushed strongly towards zero. If $\lambda$ is too large, it can lead to **underfitting** (the model becomes too simple and doesn't learn enough from the data).

**How to choose $\lambda$:** You typically tune $\lambda$ using techniques like **cross-validation**. You train your model with different values of $\lambda$ and select the $\lambda$ that gives the best performance on a separate validation set.

### Why does Regularization Work? (Intuition)

Regularization asks the model:

> “Not only fit the data well, but **use the smallest possible weights** to do so.”

Why? Smaller weights → simpler models → less chance of overfitting.

Consider a simple linear model: $y = w_1 x_1 + w_2 x_2 + b$.

If $w_1$ is very large, a small change in $x_1$ will lead to a very large change in $y$. This makes the model very sensitive to small fluctuations in the input data, which is characteristic of overfitting (it's trying to fit every tiny wiggle in the training data).

Regularization penalizes these large weights. By forcing weights to be smaller, it makes the model:
* **Smoother:** Less prone to sharp changes.
* **Less sensitive to individual data points:** It generalizes better because it focuses on the overall trends rather than noise.
* **Less complex:** A model with smaller weights is inherently simpler.

### Beyond L1 and L2

While L1 and L2 are the most common, there are other regularization techniques:

* **Elastic Net Regularization:** Combines both L1 and L2 penalties. It's particularly useful when you have groups of correlated features.
* **Dropout (for Neural Networks):** During training, randomly sets a fraction of neurons to zero at each update. This prevents neurons from co-adapting too much and forces the network to learn more robust features.
* **Early Stopping:** Instead of letting your model train until convergence, you monitor its performance on a validation set and stop training when the validation error starts to increase (indicating overfitting).
* **Data Augmentation:** Creating more training data by transforming existing data (e.g., rotating images, adding noise). This increases the dataset's diversity and helps the model generalize.

### In Summary

Regularization is a cornerstone technique in machine learning for building robust and generalizable models. By adding a penalty to the cost function, it discourages overly complex models and helps prevent overfitting. Understanding L1 and L2 regularization, and how to choose the appropriate regularization strength ($\lambda$), are essential skills for any machine learning practitioner.


| Method     | Regularization Term | Effect                         |
| ---------- | ------------------- | ------------------------------ |
| Ridge      | $\sum \theta_j^2$   | Shrinks weights                |
| Lasso      | $\sum \|\theta_j\|$ | Shrinks and eliminates weights |
| ElasticNet | Both L1 and L2      | Hybrid approach                |


### 1. The Penalty Terms and Their Derivatives

Recall the penalty terms:
* **L1 Regularization:** $\lambda \sum_{j=1}^{n} |w_j|$
* **L2 Regularization:** $\lambda \sum_{j=1}^{n} w_j^2$

Now, let's consider how these penalties affect the optimization process, particularly through their derivatives (gradients). When an optimization algorithm like Gradient Descent is trying to minimize the total cost function, it moves the weights in the direction opposite to the gradient.

#### L2 Regularization (Ridge)

The derivative of the L2 penalty term with respect to a single weight $w_j$ is:
$$\frac{\partial}{\partial w_j} (\lambda w_j^2) = 2 \lambda w_j$$
$$
\text{Penalty: } \theta^2 \quad \Rightarrow \quad \frac{d}{d\theta} = 2\theta
$$
* **Behavior near zero:** As $w_j$ approaches zero, the derivative $2 \lambda w_j$ also approaches zero. This means that as a weight gets very small, the **penalty for not being exactly zero becomes proportionally smaller**.
* **Shrinkage:** The L2 penalty applies a force that is proportional to the current magnitude of the weight. It continuously pulls the weights towards zero, but the "pull" diminishes as the weight gets closer to zero. It's like applying friction that gets weaker as an object slows down.
* **Result:** Weights are shrunk towards zero, but they are rarely forced to be *exactly* zero unless $\lambda$ is extremely large (which would likely lead to underfitting). All features will retain some small, non-zero coefficient, meaning they all contribute to the model, albeit perhaps minimally.

#### L1 Regularization (Lasso)

The derivative of the L1 penalty term with respect to a single weight $w_j$ is:
$$\frac{\partial}{\partial w_j} (\lambda |w_j|) = \lambda \cdot \text{sgn}(w_j)$$
where $\text{sgn}(w_j)$ is the sign function (1 if $w_j > 0$, -1 if $w_j < 0$).

$$
\text{Penalty: } |\theta| \quad \Rightarrow \quad \frac{d}{d\theta} = 
\begin{cases}
+1, & \theta > 0 \\
-1, & \theta < 0 \\
\text{undefined}, & \theta = 0
\end{cases}
$$
* Not differentiable at $\theta = 0$, but **subgradients** can be used.

* **Behavior near zero:** This is the key difference! The derivative of the absolute value function is a **constant** ($\lambda$ or $-\lambda$), *regardless* of how close $w_j$ is to zero (as long as it's not exactly zero, where the derivative is undefined).
* **Constant Force:** This means L1 regularization applies a **constant "force"** to push the weights towards zero. Imagine a ball rolling down a hill (loss function) but also experiencing a constant friction force.
* **Sparsity:** If the weight is, say, $0.1$ and the gradient from the loss function is small, the constant $\lambda$ penalty might be strong enough to overcome that small gradient and push the weight all the way to zero. Because this "constant force" persists even for very small weights, L1 regularization is much more likely to drive less important features' coefficients to exactly zero.

This "constant force" property of L1 regularization is what enables its feature selection capability. It effectively prunes features that contribute little to the model's performance by setting their corresponding weights to zero.

### 2. Geometric Interpretation (Visualizing the Constraint)

This is a powerful way to understand the difference. We can think of regularization as solving an optimization problem with a constraint:

$$\text{minimize Loss}(\mathbf{w}, b) \quad \text{subject to} \quad \text{Penalty}(\mathbf{w}) \le C$$

Where $C$ is a constant related to $\lambda$ (a smaller $C$ means stronger regularization, equivalent to a larger $\lambda$).

Let's consider a simple case with only two weights, $w_1$ and $w_2$.

#### L2 Regularization's Constraint

The L2 constraint is: $|w_1|^2 + |w_2|^2 \le C$. This defines a **circle** in the $w_1$-$w_2$ plane centered at the origin.

The unregularized loss function (e.g., Mean Squared Error) typically has elliptical contour lines, with the center of the ellipses representing the unregularized minimum.

When we combine the loss function and the L2 penalty, we are looking for the point where the elliptical contours of the loss function first touch the circular constraint region.

* Because the circle is smooth and round, the point of tangency between the elliptical contours and the circular constraint boundary is **very unlikely** to occur exactly on one of the axes (where $w_1=0$ or $w_2=0$). It's much more likely to be a point where both $w_1$ and $w_2$ are non-zero.

#### L1 Regularization's Constraint

The L1 constraint is: $|w_1| + |w_2| \le C$. This defines a **diamond shape** (a square rotated by 45 degrees) in the $w_1$-$w_2$ plane centered at the origin.

Now, when we combine the loss function and the L1 penalty, we are looking for the point where the elliptical contours of the loss function first touch the diamond-shaped constraint region.

* The diamond shape has **sharp corners** that lie exactly on the axes (where one of the weights is zero).
* Due to these sharp corners, the elliptical contours of the loss function are **much more likely to touch the constraint boundary at one of these corners**. If an ellipse touches a corner, it means the corresponding weight (or weights) will be exactly zero.

**Visualizing it:**
Imagine blowing up a balloon (the loss function contours) inside a room shaped like either a circle (L2) or a diamond (L1).
* In the circular room (L2), the balloon will likely touch the wall somewhere in the middle, not necessarily at the exact "ends" of the axes.
* In the diamond-shaped room (L1), the balloon is much more likely to first touch one of the sharp corners of the diamond. These corners correspond to weight values of zero for some features.

### In essence:

* **L2 regularization** shrinks weights proportionally to their magnitude, resulting in smaller, non-zero weights for most features. It prevents any single feature from dominating the model but doesn't eliminate features.
* **L1 regularization** applies a constant shrinkage force, which is powerful enough to drive less important feature weights to exactly zero, thus performing automatic feature selection and leading to sparse models.

This ability of L1 regularization to set coefficients to zero is incredibly valuable in situations with high-dimensional data, as it can help simplify the model, reduce noise, and improve interpretability by identifying the most influential features.

---
🔍 What Happens When We Minimize with Regularization?

We are solving this kind of optimization problem:

* **With L2 (Ridge)**:

  $$
  \min_\theta \left[ \text{Loss}(\theta) + \lambda \sum_j \theta_j^2 \right]
  $$

* **With L1 (Lasso)**:

  $$
  \min_\theta \left[ \text{Loss}(\theta) + \lambda \sum_j |\theta_j| \right]
  $$

So what's the **mechanical difference** that leads L1 to set some $\theta_j = 0$, but not L2?

---

## ⚙️ Let's Understand the Gradients

### 📌 L2 Regularization:

$$
\frac{d}{d\theta_j} (\theta_j^2) = 2\theta_j
$$

* The gradient is **linear**.
* It becomes **exactly 0 only when $\theta_j = 0$**.
* During optimization, L2 pulls weights toward 0, but **never sharply enough to "snap" to zero** unless the data forces it exactly.
* It behaves like a **rubber band** gently pulling each weight toward the origin.

### 📌 L1 Regularization:

$$
\frac{d}{d\theta_j} |\theta_j| =
\begin{cases}
+1 & \theta_j > 0 \\
-1 & \theta_j < 0 \\
\text{undefined (use subgradient)} & \theta_j = 0
\end{cases}
$$

* The gradient is **constant** (±1) away from zero.
* There's a **sharp "kink" at 0**, which gives the optimizer a **clear incentive to stop at zero**.
* Imagine walking downhill and hitting a **flat floor** — you just stop.

---

## 🧭 Geometric Intuition (Critical Insight)

Consider you're minimizing the loss under a constraint:

* **L1** constraint region: a diamond
* **L2** constraint region: a circle

### 📉 Loss contours are ellipses. You minimize the loss **while staying inside the constraint**.

**In L1:**

* The diamond has **sharp corners** along the axes.
* The optimizer often lands **exactly on those corners**, which means some $\theta_j = 0$.

**In L2:**

* The circle has no corners — it's smooth.
* The optimizer will almost always **slide to a point inside the circle**, where **all $\theta_j$ are non-zero**.

This difference is **purely geometric** — L1 has "kinks" (points of nondifferentiability), and L2 doesn’t.

---

## 🧠 Tiny Example:

We have:

* Features: $x_1 = 1, x_2 = 1$
* Target: $y = 2$
* Initial weights: $w_1 = 1, w_2 = 1$
* Learning rate: $\eta = 0.1$
* Regularization strength: $\lambda = 0.1$

We'll now run **5 steps** of gradient descent for both **L1 (Lasso)** and **L2 (Ridge)**.

---

### 📘 L2 Regularization

Loss:
$$
\mathcal{L}_{\text{L2}} = \frac{1}{2}(y - w_1 x_1 - w_2 x_2)^2 + \lambda (w_1^2 + w_2^2)
$$
$$
\mathcal{L}_{\text{L2}} = \frac{1}{2}(2 - w_1 - w_2)^2 + \lambda(w_1^2 + w_2^2)
$$
To update the weights using gradient descent:

$$
w_i \leftarrow w_i - \eta \left[ -x_i (y - \hat{y}) + 2\lambda w_i \right]
$$

Gradient (since $\hat{y} = w_1 + w_2$):

$$
\frac{d\mathcal{L}}{dw_i} = -(2 - (w_1 + w_2)) + 2\lambda w_i
$$

We’ll track the values over steps:

### L2 Gradient Descent Steps

| Step | $w_1$  | Prediction $\hat{y}$ | Error ($y - \hat{y}$) | Gradient | Update  |
| ---- | ------ | -------------------- | --------------------- | -------- | ------- |
| 0    | 1.0000 | 2.0000               | 0.0000                | 0.2000   | -0.0200 |
| 1    | 0.9800 | 1.9600               | 0.0400                | 0.1960   | -0.0196 |
| 2    | 0.9604 | 1.9208               | 0.0792                | 0.1921   | -0.0192 |
| 3    | 0.9412 | 1.8824               | 0.1176                | 0.1882   | -0.0188 |
| 4    | 0.9224 | 1.8447               | 0.1553                | 0.1845   | -0.0184 |
| 5    | 0.9040 | 1.8081               | 0.1919                | 0.1808   | -0.0181 |
| 6    | 0.8860 | 1.7721               | 0.2279                | 0.1772   | -0.0177 |
| 7    | 0.8683 | 1.7366               | 0.2634                | 0.1737   | -0.0174 |
| 8    | 0.8509 | 1.7018               | 0.2982                | 0.1702   | -0.0170 |
| 9    | 0.8339 | 1.6677               | 0.3323                | 0.1668   | -0.0167 |
| 10   | 0.8172 | 1.6344               | 0.3656                | 0.1634   | -0.0163 |
| 11   | 0.8009 | 1.6017               | 0.3983                | 0.1602   | -0.0160 |
| 12   | 0.7849 | 1.5698               | 0.4302                | 0.1570   | -0.0157 |
| 13   | 0.7692 | 1.5385               | 0.4615                | 0.1539   | -0.0154 |
| 14   | 0.7538 | 1.5076               | 0.4924                | 0.1508   | -0.1508 |
| 15   | 0.7387 | 1.4774               | 0.5226                | 0.1477   | -0.0148 |
| 16   | 0.7239 | 1.4478               | 0.5522                | 0.1448   | -0.0145 |
| 17   | 0.7094 | 1.4188               | 0.5812                | 0.1419   | -0.0142 |
| 18   | 0.6952 | 1.3903               | 0.6097                | 0.1390   | -0.0139 |
| 19   | 0.6813 | 1.3626               | 0.6374                | 0.1363   | -0.0136 |
| 20   | 0.6677 | 1.3354               | 0.6646                | 0.1335   | -0.0133 |
| 21   | 0.6543 | 1.3087               | 0.6913                | 0.1309   | -0.0131 |
| 22   | 0.6412 | 1.2824               | 0.7176                | 0.1282   | -0.0128 |
| 23   | 0.6284 | 1.2568               | 0.7432                | 0.1257   | -0.0126 |
| 24   | 0.6158 | 1.2316               | 0.7684                | 0.1232   | -0.0123 |
| 25   | 0.6035 | 1.2070               | 0.7930                | 0.1207   | -0.1207 |
| 26   | 0.5914 | 1.1827               | 0.8173                | 0.1183   | -0.0118 |
| 27   | 0.5796 | 1.1592               | 0.8408                | 0.1160   | -0.0116 |
| 28   | 0.5680 | 1.1360               | 0.8640                | 0.1136   | -0.0114 |
| 29   | 0.5566 | 1.1132               | 0.8868                | 0.1113   | -0.0111 |
| 30   | 0.5455 | 1.0909               | 0.9091                | 0.1091   | -0.0109 |
| 31   | 0.5346 | 1.0692               | 0.9308                | 0.1070   | -0.0107 |
| 32   | 0.5239 | 1.0479               | 0.9521                | 0.1050   | -0.0105 |
| 33   | 0.5135 | 1.0270               | 0.9730                | 0.1030   | -0.0103 |
| 34   | 0.5032 | 1.0064               | 0.9936                | 0.1013   | -0.0101 |
| 35   | 0.4931 | 0.9862               | 1.0138                | 0.0996   | -0.0100 |
| 36   | 0.4831 | 0.9662               | 1.0338                | 0.0979   | -0.0098 |
| 37   | 0.4733 | 0.9466               | 1.0534                | 0.0963   | -0.0096 |

→ The weights decrease **smoothly and symmetrically**, but **never become zero**.

---

### 📙 L1 Regularization

Loss:
$$
\mathcal{L}_{\text{L1}} = \frac{1}{2}(y - w_1 x_1 - w_2 x_2)^2 + \lambda (|w_1| + |w_2|)
$$

$$
\mathcal{L}_{\text{L1}} = \frac{1}{2}(2 - w_1 - w_2)^2 + \lambda(|w_1| + |w_2|)
$$

The gradient is **not smooth** at $w_i = 0$, so we use **subgradients**:

$$
\frac{d}{dw_i} \left[ \lambda |w_i| \right] = 
\begin{cases}
+\lambda & \text{if } w_i > 0 \\\\
-\lambda & \text{if } w_i < 0 \\\\
\text{undefined, pick in } [-\lambda, +\lambda] & \text{if } w_i = 0
\end{cases}
$$

To update the weights using gradient descent:

$$
w_i \leftarrow w_i - \eta \left[ -x_i (y - \hat{y}) + \lambda \cdot \text{sign}(w_i)\right]
$$


Gradient (since $\hat{y} = w_1 + w_2$):

$$
\frac{d\mathcal{L}}{dw_i} = -(2 - (w_1 + w_2)) + \lambda \cdot \text{sign}(w_i)
$$

We’ll assume sign remains positive for now.

### L1 Gradient Descent Steps (until zero weight)

| Step | $w_1$   | Prediction $\hat{y}$ | Error ($y - \hat{y}$) | Gradient | Update  |
| ---- | ------- | -------------------- | --------------------- | -------- | ------- |
| 0    | 1.0000  | 2.0000               | 0.0000                | 0.1      | -0.01   |
| 1    | 0.9900  | 1.9800               | 0.0200                | 0.108    | -0.0108 |
| 2    | 0.9792  | 1.9584               | 0.0416                | 0.1162   | -0.0116 |
| 3    | 0.9676  | 1.9352               | 0.0648                | 0.1245   | -0.0124 |
| 4    | 0.9551  | 1.9102               | 0.0898                | 0.1329   | -0.0133 |
| 5    | 0.9418  | 1.8836               | 0.1164                | 0.1416   | -0.0142 |
| 6    | 0.9277  | 1.8554               | 0.1446                | 0.1505   | -0.0151 |
| 7    | 0.9126  | 1.8252               | 0.1748                | 0.1595   | -0.0160 |
| 8    | 0.8966  | 1.7932               | 0.2068                | 0.1687   | -0.0169 |
| 9    | 0.8797  | 1.7593               | 0.2407                | 0.1781   | -0.0178 |
| 10   | 0.8619  | 1.7239               | 0.2761                | 0.1876   | -0.0188 |
| 11   | 0.8431  | 1.6862               | 0.3138                | 0.1974   | -0.0197 |
| 12   | 0.8234  | 1.6468               | 0.3532                | 0.2073   | -0.0207 |
| 13   | 0.8027  | 1.6054               | 0.3946                | 0.2173   | -0.0217 |
| 14   | 0.7810  | 1.5620               | 0.4380                | 0.2274   | -0.0227 |
| 15   | 0.7583  | 1.5166               | 0.4834                | 0.2377   | -0.0238 |
| 16   | 0.7345  | 1.4690               | 0.5310                | 0.2481   | -0.0248 |
| 17   | 0.7097  | 1.4194               | 0.5806                | 0.2586   | -0.0259 |
| 18   | 0.6839  | 1.3678               | 0.6322                | 0.2692   | -0.0269 |
| 19   | 0.6570  | 1.3140               | 0.6860                | 0.2800   | -0.0280 |
| 20   | 0.6290  | 1.2580               | 0.7420                | 0.2908   | -0.0291 |
| 21   | 0.5999  | 1.1998               | 0.8002                | 0.3018   | -0.0302 |
| 22   | 0.5697  | 1.1394               | 0.8606                | 0.3129   | -0.0313 |
| 23   | 0.5384  | 1.0768               | 0.9232                | 0.3241   | -0.0324 |
| 24   | 0.5060  | 1.0120               | 0.9880                | 0.3354   | -0.0335 |
| 25   | 0.4725  | 0.9450               | 1.0550                | 0.3469   | -0.0347 |
| 26   | 0.4378  | 0.8756               | 1.1244                | 0.3584   | -0.0358 |
| 27   | 0.4019  | 0.8038               | 1.1962                | 0.3700   | -0.0370 |
| 28   | 0.3649  | 0.7298               | 1.2702                | 0.3817   | -0.0382 |
| 29   | 0.3267  | 0.6534               | 1.3466                | 0.3935   | -0.0394 |
| 30   | 0.2874  | 0.5748               | 1.4252                | 0.4054   | -0.0405 |
| 31   | 0.2470  | 0.4940               | 1.5060                | 0.4174   | -0.0417 |
| 32   | 0.2053  | 0.4106               | 1.5894                | 0.4295   | -0.0429 |
| 33   | 0.1624  | 0.3248               | 1.6752                | 0.4417   | -0.0442 |
| 34   | 0.1182  | 0.2364               | 1.7636                | 0.4540   | -0.0454 |
| 35   | 0.0728  | 0.1456               | 1.8544                | 0.4664   | -0.0466 |
| 36   | 0.0262  | 0.0524               | 1.9476                | 0.4789   | -0.0479 |
| 37   | -0.0217 | -0.0434              | 2.0434                | 0.4915   | -0.0491 |

🔚 At **step 36 → 37**, both weights cross zero — L1 regularization sets them to zero with continued gradient steps.

Notice:

* L1 updates stay **nonlinear** due to the constant $\lambda$ term.
* If weights become very small (say near 0.01), the subgradient can dominate and set them **exactly to zero**.

If we continue:

* **L1** could zero-out one or both weights.
* **L2** continues gently reducing them.

---

## ✨ Key Takeaway

* **L2 (Ridge)** behaves like soft shrinkage: all weights are reduced proportionally but never reach zero.
* **L1 (Lasso)** acts like hard thresholding: weights can be driven exactly to zero due to the constant pull of the absolute value penalty.

---
## 🟩 1. **L1 Regularization (Lasso)**

### 📉 **Limitations**

| Limitation                                   | Why it Happens                                                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ❌ **Instability with correlated features**   | When multiple features are correlated, Lasso **randomly picks one** and sets others to zero — which leads to unstable and unreliable feature selection.                        |
| ❌ **Bias introduced in large coefficients**  | L1 tends to **over-shrink** all coefficients equally, including the important ones.                                                                                            |
| ❌ **At most n non-zero features (if n < p)** | If the number of training examples $n$ is less than the number of features $p$, Lasso will select **at most n** features. This may be too aggressive in high-dimensional data. |

---

### ✅ **Solutions for L1**

#### 1. **Elastic Net Regularization**

* Combines L1 and L2:

  $$
  \text{Loss} + \lambda_1 \sum |\theta_j| + \lambda_2 \sum \theta_j^2
  $$
* Solves the correlated feature problem: it **selects groups** of correlated features together.
* Useful when you want **sparsity** but also **stability**.

#### 2. **Stability Selection**

* Run Lasso multiple times on subsamples of data.
* Select features that consistently show up.
* This **reduces randomness** in variable selection.

---

## 🟦 2. **L2 Regularization (Ridge)**

### 📉 **Limitations**

| Limitation                                            | Why it Happens                                                                                                      |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ❌ **Doesn’t produce sparse models**                   | L2 **never zeroes out** any weight. So, it doesn’t help with feature selection.                                     |
| ❌ **Poor interpretability**                           | Since all weights remain non-zero, you end up with a model that's hard to interpret, especially in high dimensions. |
| ❌ **Can underperform when irrelevant features exist** | If many features are noisy/irrelevant, Ridge spreads small weights to them rather than cutting them out.            |

---

### ✅ **Solutions for L2**

#### 1. **Preprocessing with Feature Selection**

* Use techniques like:
	* Variance thresholding
	* Mutual information
	* Recursive Feature Elimination (RFE)
* Then apply Ridge to the reduced set.

#### 2. **Use Elastic Net**

* Again, Elastic Net solves this by:
	* Shrinking all weights (L2)
	* Removing irrelevant ones (L1)

---

## 🧪 Elastic Net: Best of Both Worlds

### Formula:

$$
\text{Loss} + \alpha \left[ \lambda_1 \sum |\theta_j| + (1 - \lambda_1) \sum \theta_j^2 \right]
$$

* $\lambda_1 \in [0, 1]$: controls L1 vs. L2 mix
* $\alpha$: overall regularization strength

### 👍 Benefits:

* **Sparse** like Lasso
* **Stable** like Ridge
* **Group selection**: keeps groups of correlated features

---

## 🧠 Summary Table

| Regularizer     | Pros                             | Cons                                                 | Solution                               |
| --------------- | -------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| **L1 (Lasso)**  | Sparse models, feature selection | Instability with correlated features, over-shrinkage | Use Elastic Net or Stability Selection |
| **L2 (Ridge)**  | Stable, avoids overfitting       | No sparsity, poor interpretability                   | Use Feature Selection or Elastic Net   |
| **Elastic Net** | Balanced trade-off               | Needs two hyperparameters                            | Tune via cross-validation              |


---
## 🔁 Gradient Descent Without Regularization

Recall, in simple **linear regression**, the update rule for a parameter $\theta_j$ using gradient descent is:

$$
\theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}
$$

For **Mean Squared Error (MSE)**, the gradient of the loss is:

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

So the update rule becomes:

$$
\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

---

## 🔐 Gradient Descent **With L2 Regularization**

We modify the cost function to include the **L2 penalty**:

$$
J_{\text{L2}}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

Now, compute the gradient:

$$
\frac{\partial J_{\text{L2}}}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j
$$

The update rule becomes:

$$
\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]
$$

---

## 🧠 Interpreting the Update Rule

Let’s isolate the parts:

* **Data fit gradient** (original):

  $$
  \Delta_{\text{data}} = \frac{1}{m} \sum \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
  $$

* **Regularization term**:

  $$
  \Delta_{\text{reg}} = \frac{\lambda}{m} \theta_j
  $$

### Final update:

$$
\theta_j := \theta_j \cdot \left(1 - \alpha \cdot \frac{\lambda}{m} \right) - \alpha \cdot \Delta_{\text{data}}
$$

---

## 💡 Key Insight

### The term:

$$
\theta_j \cdot \left(1 - \alpha \cdot \frac{\lambda}{m} \right)
$$

acts like a **shrinkage factor**.

* Every update **pulls $\theta_j$ closer to zero**.
* This prevents weights from growing too large.
* The higher $\lambda$, the more aggressively it shrinks.

---
Great question!

L2 regularization (also called **Ridge Regularization**) can be **generalized to other supervised learning models** beyond linear regression, with only slight modifications to the cost function depending on the base model. Below is a breakdown for common supervised learning methods:

---

## 🔹 1. **Logistic Regression (with L2 Regularization)**

### **Cost Function:**

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

* First term: standard binary cross-entropy loss.
* Second term: L2 regularization penalty.
* Note: Bias term $\theta_0$ is usually **excluded** from regularization.

---

## 🔹 2. **Linear Regression (with L2 Regularization)**

As you already have:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

---

## 🔹 3. **Support Vector Machines (SVM with L2 Regularization)**

For **linear SVMs**, the hinge loss is used:

$$
J(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \max(0, 1 - y^{(i)}(w^T x^{(i)} + b))
$$

* The first term $\frac{1}{2} \|w\|^2$ is the L2 regularization.
* $C$ is the inverse of $\lambda$ (i.e., $C = \frac{1}{\lambda}$), controlling regularization strength.
* This formulation is **primal soft-margin SVM**.

---

## 🔹 4. **Ridge Regression (Linear Regression with L2 Regularization)**

This is just another name for linear regression with L2 regularization:

$$
\min_{\theta} \|X\theta - y\|^2 + \lambda \|\theta\|^2
$$

---

## 🔹 5. **Regularized Loss for General ML Models**

For any generic model with base loss function $\mathcal{L}(\theta)$, the L2 regularized version looks like:

$$
J(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

Where:

* $\mathcal{L}(\theta)$ could be MSE, cross-entropy, hinge loss, etc.
* The regularization term always adds a penalty for large values of $\theta_j$.

---

## 🔹 6. **Tree-Based Methods (e.g., XGBoost, LightGBM, etc.)**

Although tree-based models do **not** have weights $\theta$ in the same way, regularization is applied via:

* Penalizing **tree complexity**, e.g.:

  * Number of leaves.
  * Leaf weights (e.g., $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2$)
* So XGBoost regularization formula looks like:

$$
\text{Obj} = \sum_{i=1}^{m} l(\hat{y}_i, y_i) + \sum_{k=1}^K \Omega(f_k)
$$

$$
\text{where } \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2
$$

---

## 🔹 7. **Neural Networks**

Regularization is added to the **loss function**:

$$
J(\theta) = \text{Original Loss} + \frac{\lambda}{2m} \sum_{l=1}^{L} \sum_{j=1}^{n_l} \sum_{k=1}^{n_{l+1}} (W_{jk}^{(l)})^2
$$

* $W_{jk}^{(l)}$: weight from neuron $j$ in layer $l$ to neuron $k$ in layer $l+1$.
* Regularization typically excludes bias terms.

---

## 🔸 Summary Table

| Model               | Base Loss                | L2 Regularization Term                                  |
| ------------------- | ------------------------ | ------------------------------------------------------- |
| Linear Regression   | MSE                      | $\frac{\lambda}{2m} \sum \theta_j^2$                    |
| Logistic Regression | Cross-Entropy            | $\frac{\lambda}{2m} \sum \theta_j^2$                    |
| SVM                 | Hinge Loss               | $\frac{1}{2} \|w\|^2$                                   |
| Neural Networks     | Cross-Entropy or MSE     | $\frac{\lambda}{2m} \sum W_{jk}^{(l)^2}$                |
| XGBoost/GBM         | Custom loss + Complexity | $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum w_j^2$ |
| Ridge Regression    | MSE                      | $\lambda \|\theta\|^2$                                  |

---

If you'd like, I can also walk you through **L1 regularization** (Lasso), **Elastic Net**, or **when to prefer L1 over L2**.

---

Sure! Let’s now walk through **L1 regularization** across various supervised models in the same style.

---

## 🔹 What is L1 Regularization?

L1 regularization (also known as **Lasso** when used with linear models) adds a **penalty proportional to the absolute value** of the weights:

$$
\text{L1 Penalty} = \lambda \sum_{j=1}^{n} |\theta_j|
$$

* **Encourages sparsity**: It tends to **zero out** irrelevant features (feature selection).
* Bias term $\theta_0$ is typically **not regularized**.

---

## 🔸 General Cost Function (with L1)

$$
J(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

Where:

* $\mathcal{L}(\theta)$ is the base loss (e.g., MSE, cross-entropy).
* $\lambda$: regularization strength.
* Regularization term adds penalty for large **absolute** values of weights.

---

## 🔹 1. **Linear Regression (Lasso Regression)**

### Cost Function:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

* Similar to Ridge but uses absolute value.
* Encourages **sparse** solutions (many weights exactly 0).

---

## 🔹 2. **Logistic Regression (with L1 Regularization)**

### Cost Function:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right] + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

* Promotes **sparse models**, useful for feature selection in high-dimensional data (e.g., text, genomics).

---

## 🔹 3. **Support Vector Machines (L1-Regularized SVM / L1-SVM)**

$$
J(w, b) = \|w\|_1 + C \sum_{i=1}^m \max(0, 1 - y^{(i)}(w^T x^{(i)} + b))
$$

* L1 norm $\|w\|_1$ replaces $\|w\|^2$ of standard SVM.
* Leads to sparse $w$ (i.e., some features ignored entirely).
* Less common than L2-SVM, but useful in high-dimensional feature settings.

---

## 🔹 4. **Neural Networks (with L1 Regularization)**

$$
J(\theta) = \text{Original Loss} + \frac{\lambda}{m} \sum_{l=1}^{L} \sum_{j=1}^{n_l} \sum_{k=1}^{n_{l+1}} |W_{jk}^{(l)}|
$$

* Can zero out unimportant weights in the network.
* Sometimes used in conjunction with **L2 (Elastic Net)**.

---

## 🔹 5. **XGBoost / Tree-Based Models**

Tree models don’t use $\theta$, but XGBoost allows L1 via:

$$
\Omega(f) = \gamma T + \frac{\lambda_2}{2} \sum_j w_j^2 + \lambda_1 \sum_j |w_j|
$$

* Both L1 and L2 can be used.
* $\lambda_1$: L1 regularization on leaf weights.
* Helps control overfitting and encourages sparse trees.

---

## 🔸 Summary Table

| Model               | Base Loss                | L1 Regularization Term   | Notes         |   |                    |
| ------------------- | ------------------------ | ------------------------ | ------------- | - | ------------------ |
| Linear Regression   | MSE                      | ( \frac{\lambda}{m} \sum | \theta\_j     | ) | Lasso              |
| Logistic Regression | Cross-Entropy            | ( \frac{\lambda}{m} \sum | \theta\_j     | ) | Sparse classifier  |
| SVM                 | Hinge Loss               | $\|w\|_1$                | L1-SVM        |   |                    |
| Neural Networks     | Cross-Entropy or MSE     | ( \frac{\lambda}{m} \sum | W\_{jk}^{(l)} | ) | Sparse weights     |
| XGBoost             | Custom loss + Complexity | ( \lambda\_1 \sum        | w\_j          | ) | L1 on leaf weights |

---

## 🔹 When to Use L1?

| L1 (Lasso)                  | L2 (Ridge)                      |
| --------------------------- | ------------------------------- |
| Feature selection needed    | Multicollinearity in features   |
| Sparse model preferred      | All features are informative    |
| Works well in high-dim data | Works well in low/moderate dim  |
| Can zero out weights        | Shrinks weights but not to zero |

---

## 🔹 Elastic Net (Combination of L1 + L2)

$$
J(\theta) = \mathcal{L}(\theta) + \frac{\lambda_1}{m} \sum |\theta_j| + \frac{\lambda_2}{2m} \sum \theta_j^2
$$

* Balances sparsity and shrinkage.
* Often works better than pure L1 or L2 when features are correlated.

---

Would you like a **visual intuition** or **Python examples** comparing L1 vs L2 on real datasets (like `sklearn` or `XGBoost`)?
