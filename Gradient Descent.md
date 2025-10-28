### 🔍 **What Is a Gradient?**

At its core, a **gradient** is a **vector of partial derivatives** that points in the direction of the **steepest increase** of a function.

---

### 📐 **Mathematical Definition**

Suppose we have a function:

$$
f(\mathbf{x}) = f(x_1, x_2, \dots, x_n)
$$

The **gradient** of $f$ is a vector of all its partial derivatives:

$$
\nabla f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]
$$

This vector tells you how the function $f$ changes **locally** if you change each of the variables $x_i$.

---

### 🌄 **Geometric Interpretation**

* Think of a hill (the loss surface).
* The gradient points **uphill** — in the direction where the function (e.g., loss) increases most quickly.
* In deep learning, we want to **go downhill** (minimize loss), so we move in the **negative gradient** direction.

---

### 💡 **In Deep Learning**

In deep learning, models are trained to minimize a **loss function**, say $\mathcal{L}(\theta)$, where $\theta$ represents the model parameters (weights and biases).

1. **Compute the gradient**:

   $$
   \nabla_{\theta} \mathcal{L}(\theta)
   $$

   This tells us how each parameter affects the loss.

2. **Update the parameters** (via gradient descent):

   $$
   \theta := \theta - \eta \nabla_{\theta} \mathcal{L}(\theta)
   $$

   where $\eta$ is the **learning rate**.

---

### ⚙️ **Why Is It Important?**

* Without gradients, we wouldn't know how to adjust model parameters.
* The **backpropagation** algorithm computes gradients of the loss with respect to all parameters efficiently, using the **chain rule**.

---

### 🧠 Summary

| Concept          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| Gradient         | Vector of partial derivatives of a function              |
| Tells us         | Direction and rate of fastest increase in the function   |
| In deep learning | Used to adjust parameters to minimize loss function      |
| Optimizer        | Uses gradients to update model weights (e.g., SGD, Adam) |

---

### Gradient Descent (GD)
**Gradient Descent** is an optimization algorithm used to minimize a function (typically a loss function in machine learning). **Mini-batch gradient descent (MBGD)** is a variant that balances **efficiency** and **stability** by updating the model using small subsets (mini-batches) of the training data.

## 1. Gradient Descent: The Big Picture
### **Goal**:  
We want to find **optimal model parameters** (e.g., weights $\theta$) that **minimize** the loss function $J(\theta)$. 

$$
\theta = \theta - \alpha \cdot \nabla J(\theta)
$$

where:  
- $\theta$ = model parameters (e.g., weights in a neural network).  
- $\alpha$ = learning rate (step size).  
- $\nabla J(\theta)$ = gradient of the loss function.  

## 2. Types of Gradient Descent
### (a) Batch Gradient Descent (BGD)
- Uses **all training examples** to compute the gradient.  
- **Formula**:  
$$
\theta = \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla J_i(\theta)
$$
- **Pros**: Stable convergence.
- **Cons**: **Slow** for large datasets.

### (b) Stochastic Gradient Descent (SGD)
- Uses **one** random example at a time.  
- **Formula**:  
$$
\theta = \theta - \alpha \cdot \nabla J_i(\theta)
$$
- **Pros**: Faster updates.
- **Cons**: Noisy updates, unstable convergence.

### (c) Mini-Batch Gradient Descent (MBGD)
- Uses a **small subset (mini-batch) of size $b$** to compute the gradient.  
- **Formula**:  
$$
\theta = \theta - \alpha \cdot \frac{1}{b} \sum_{i=1}^{b} \nabla J_i(\theta)
$$
- **Pros**:  
	  - Faster than BGD.  
	  - More stable than SGD.  
	  - Can leverage parallel computing (e.g., GPUs).  

## 3. Why Use Mini-Batches?

| **Metric**                | **Batch GD** | **Mini-Batch GD** | **SGD** |
| ------------------------- | ------------ | ----------------- | ------- |
| **Computation Cost**      | High         | Medium            | Low     |
| **Convergence Stability** | High         | Medium            | Low     |
| **Speed**                 | Slow         | Fast              | Fastest |
| **Memory Usage**          | High         | Medium            | Low     |

Mini-batches provide a **trade-off**: 
- Small batches lead to **faster training** (like SGD). 
- Large batches ensure **stable convergence** (like BGD).  

## 4. Choosing the Mini-Batch Size
- **Small batch size (e.g., 32, 64):** Good for convergence, introduces slight randomness.
- **Large batch size (e.g., 512, 1024):** More stable, but requires more memory.
- **Rule of thumb:** Start with **32 or 64**, and tune based on model performance.

---

## 5. Python Implementation of Mini-Batch Gradient Descent
Let's implement **MBGD for linear regression**.

```python
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# Hyperparameters
learning_rate = 0.1
epochs = 1000
batch_size = 10
m = len(X)

# Initialize weights and bias
theta = np.random.randn(2, 1)  # [bias, weight]
X_b = np.c_[np.ones((m, 1)), X]  # Add bias term

# Mini-Batch Gradient Descent
for epoch in range(epochs):
    shuffled_indices = np.random.permutation(m)
    X_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        gradients = 2/batch_size * X_batch.T.dot(X_batch.dot(theta) - y_batch)
        theta -= learning_rate * gradients

print("Optimized Parameters:", theta)
```

---

## 6. Mini-Batch GD in Deep Learning (TensorFlow/Keras)
In deep learning, mini-batch training is **default**.

```python
import tensorflow as tf
from tensorflow import keras

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.boston_housing.load_data()

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss="mse")

# Train with Mini-Batches (batch_size=32)
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## Summary
- Mini-batch gradient descent balances **speed and accuracy**.  
- It is widely used in **deep learning (e.g., CNNs, RNNs, Transformers)**.  
- Choose **batch size wisely** (start with 32/64).  
- In **large-scale ML**, MBGD is the **standard optimization method**.

---

# 📘 From Taylor Series to Gradient Descent

## 🧠 Objective

To understand how **Taylor Series** helps determine the correct update direction and step size when optimizing parameters in machine learning using **Gradient Descent**.

---

## 📌 Setup: Parameter Update in Optimization

We begin with a vector of parameters:

$$
\theta = [w, b]
$$

Suppose we change these parameters by a small amount:

$$
\Delta \theta = [\Delta w, \Delta b]
$$

We want to update the parameters conservatively, i.e., by a small step:

$$
\theta_{\text{new}} = \theta + \eta \cdot \Delta\theta
$$

> 🔍 **Question:** What should the direction $\Delta \theta$ be?

---

## 📐 The Taylor Series Comes to the Rescue

### What is Taylor Series?

The **Taylor series** approximates any differentiable function $\mathcal{L}(w)$ using a polynomial expansion.

#### General Taylor Series Expansion:

$$
\mathcal{L}(w) = \mathcal{L}(w_0) + \frac{\mathcal{L}'(w_0)}{1!}(w - w_0) + \frac{\mathcal{L}''(w_0)}{2!}(w - w_0)^2 + \cdots
$$

---
We typically ignore higher-order terms:
#### ✅ Linear Approximation (First Order):

$$
\mathcal{L}(w) \approx \mathcal{L}(w_0) + \frac{\mathcal{L}'(w_0)}{1!}(w - w_0)
$$

#### ✅ Quadratic Approximation (Second Order):

$$
\mathcal{L}(w) \approx \mathcal{L}(w_0) + \frac{\mathcal{L}'(w_0)}{1!}(w - w_0) + \frac{\mathcal{L}''(w_0)}{2!}(w - w_0)^2
$$

---

## Applying Taylor Series in Optimization

Let:

$$
\Delta \theta = u
$$

Then,

$$
\mathcal{L}(\theta + \eta u) = \mathcal{L}(\theta) + \eta u^T \nabla_\theta \mathcal{L}(\theta) + \frac{\eta^2}{2!} u^T \nabla_\theta^2 \mathcal{L}(\theta) u + \cdots
$$

Ignoring higher-order terms:

$$
\mathcal{L}(\theta + \eta u) \approx \mathcal{L}(\theta) + \eta u^T \nabla_\theta \mathcal{L}(\theta)
$$

> **Goal:** We want the loss to decrease, i.e.,
> $$
\mathcal{L}(\theta + \eta u) - \mathcal{L}(\theta) < 0
$$
>Which means:
>$$
u^T \nabla_\theta \mathcal{L}(\theta) < 0
$$

---

## 📏 Geometric Insight: Angle Between Directions

Let $\beta$ be the angle between $u^T$ and $\nabla_\theta \mathcal{L}(\theta)$:

$$
\cos(\beta) = \frac{u^T \cdot \nabla_\theta \mathcal{L}(\theta)}{\|u^T\| \cdot \|\nabla_\theta \mathcal{L}(\theta)\|}
$$
So:

$$
u^T \nabla_\theta \mathcal{L}(\theta) = \|u^T\| \cdot \|\nabla_\theta \mathcal{L}(\theta)\| \cdot \cos(\beta)
$$
Since, 
$$
-1 \leq \cos(\beta) \leq 1
$$
🧠 To **decrease the loss the most**, we want:

* $\cos(\beta) = -1$
* So $\beta = 180^\circ$, i.e., **opposite direction** of gradient

---

## 🔁 Gradient Descent Rule

Choose:

$$
u = -\nabla_\theta \mathcal{L}(\theta)
$$
Thus, the update rule becomes:

$$
\theta_{\text{new}} = \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

> 📌 This is the **core principle of Gradient Descent**:
> Move in the opposite direction of the gradient to reduce loss.

---

### 🔄 Parameter Update Rule

For parameters $w$ and $b$, the update rule is:

$$
\begin{aligned}
w_{t+1} &= w_t - \eta \nabla w_t \\
b_{t+1} &= b_t - \eta \nabla b_t
\end{aligned}
$$

Where:

$$
\nabla w_t = \frac{\partial \mathcal{L}(w_t, b_t)}{\partial w}, \quad \nabla b_t = \frac{\partial \mathcal{L}(w_t, b_t)}{\partial b}
$$

---

## ✅ Conclusion

We now have a **principled approach** for parameter updates in the $(w, b)$ plane using **Taylor series**:

* It justifies **gradient descent**.
* Shows why we move in the **opposite direction of the gradient**.
* Explains why **small step sizes** $\eta$ are necessary.

> 📘 Instead of guesswork, we now use **calculus-backed optimization** to minimize our loss!

---


# 📘  Momentum-Based Gradient Descent

---
When using vanilla (basic) gradient descent:

* It **takes a lot of time** to navigate regions with a **gentle slope**.
* This happens because the **gradient** (i.e., slope of the loss function) in these regions is **very small**, resulting in **very small updates** to parameters.
* ❓ *Can we do better?*

Yes! Let’s explore **Momentum-Based Gradient Descent** — a technique that helps us move faster through these flat regions.

---

### 🧠 A Helpful Reminder

Before any optimization algorithm begins, it always starts by computing the **gradient of the loss with respect to the weights**:

$$
\nabla w_0
$$

---

## 💡 Intuition Behind Momentum

Imagine you're repeatedly being told to move in the **same direction**. Naturally, you’d gain **confidence** and take **larger steps**.

This is the idea behind **momentum** in optimization. Just like a **ball gains momentum** when rolling downhill, an optimizer can use past gradients to gain speed in a consistent direction.

> Momentum helps overcome small gradients by accumulating velocity.

---

## 🔁 Update Rule for Momentum-Based Gradient Descent

Let’s define the update rule with momentum:

$$
\begin{align*}
u_t &= \beta u_{t-1} + \nabla w_t \\
w_{t+1} &= w_t - \eta u_t
\end{align*}
$$

Where:

* $\beta \in [0, 1)$: Momentum factor.
* $\eta$: Learning rate.
* $u_t$: Velocity (momentum update, that accumulates gradients over time).
* $\nabla w_t$: Gradient at time $t$.
* $u_{-1} = 0$, $w_0$ is initialized randomly.

💡 **Key Idea**: Momentum considers **both the current gradient and the history of previous gradients.**

---

## 🔁 Working Through the First Few Steps

Let’s compute a few initial steps of updates:

$$
\begin{align*}
u_0 &= \nabla w_0 \quad \text{(as } u_{-1} = 0\text{)} \\
u_1 &= \beta u_0 + \nabla w_1 = \beta (\nabla w_0) + \nabla w_1 \\
u_2 &= \beta u_1 + \nabla w_2 = \beta^2 \nabla w_0 + \beta \nabla w_1 + \nabla w_2
\end{align*}
$$

This builds up an **exponentially weighted average** of past gradients:

$$
u_t = \sum_{\tau=0}^t \beta^{t - \tau} \nabla w_\tau
$$

📊 As shown in the bar graph, earlier gradients contribute less over time due to the exponential decay.

---

## 🤔 Some Questions to Ponder

Momentum helps take **larger steps** even in **flat regions**, enabling faster convergence.

But...

> Is moving fast always good?

Could momentum cause us to **overshoot** the minimum? Could it lead us to **miss** our goal entirely in some cases?

---

## 🌊 Momentum in Action

* **Momentum-based gradient descent** often **oscillates** in and out of the **minima valley**, carried by the accumulated velocity.
* It may require **multiple u-turns** before it finally converges.
* Still, it typically **converges faster** than standard (vanilla) gradient descent.

The animation shows how momentum-based descent and vanilla descent behave differently in navigating the same loss landscape.

Parameters used:

$$
w_0 = -2,\quad b_0 = -4,\quad \eta = 1
$$
---

## ✅ Summary

| Feature        | Vanilla Gradient Descent | Momentum-Based Gradient Descent |
| -------------- | ------------------------ | ------------------------------- |
| Slope Handling | Slow in gentle slopes    | Faster due to momentum          |
| Convergence    | May take longer          | Often converges faster          |
| Movement       | Direct but slow          | Oscillatory but accelerated     |

---

# 📘 Nesterov Accelerated Gradient (NAG)

Gradient descent with momentum helps speed up training and smoothen oscillations. But can we do better?
### ❓ Questions

**Can we do something to reduce these oscillations?**
Yes! Let's look at **Nesterov Accelerated Gradient (NAG)** — an improvement over traditional momentum-based gradient descent.

---

## 💡 Intuition Behind NAG

> **“Look before you leap.”**

In regular momentum-based methods, we update the velocity as:

$$
u_t = \beta u_{t-1} + \nabla w_t
$$

This means we move in the direction of the momentum ($βu_{t-1}$) and the current gradient ($∇w_t$).
So we already know we’re going to move by **at least** $βu_{t-1}$.

### 🔍 What if we "look ahead"?

Instead of evaluating the gradient at the current position, why not **peek ahead** to where momentum is taking us, and evaluate the gradient there?

This "look-ahead" idea is at the heart of NAG.

---

## 🔁 Update Rule for NAG

$$
u_t = \beta u_{t-1} + \nabla (w_t - \beta u_{t-1})
$$

$$
w_{t+1} = w_t - \eta u_t
$$

Where:

* `β` is the momentum term (`0 ≤ β < 1`)
* `η` is the learning rate
* $u_{-1}$ = 0 (initial velocity)

---
> [!NOTE]
> 
> #### Predicting the next position:
> 
>    * The formula $w_t - \beta u_{t-1}$ represents **"subtracting"** the predicted momentum step.
>    * Why subtract? Because $\beta u_{t-1}$ tells you where the momentum will push the parameters in the **next step**. To **predict the position before applying the momentum**, you move backward by $\beta u_{t-1}$, which gives you the **look-ahead position**.
> 
>    **In other words**:
> 
>    * $w_t$ is where the parameters are now.
>    * $\beta u_{t-1}$ tells you where the momentum is pushing you.
>    * By subtracting $\beta u_{t-1}$, you're effectively **predicting where you'll be** after you apply the momentum.
> 
> 
> #### Why Not $w_t + \beta u_{t-1}$?
> 
> You might ask: **Why don’t we just move forward with $w_t + \beta u_{t-1}$?**
> 
> If you were to use $w_t + \beta u_{t-1}$ instead, you'd be computing the gradient **at a position beyond the predicted momentum update**.
> 
> * This would mean you're computing the gradient at a point **after** you've already moved forward with the momentum, and you would be updating the momentum and parameters **too late** (i.e., after the actual update), rather than making an informed decision **before the update**.
> * The core idea of **look-ahead** is to anticipate where you will be and compute the gradient at that point, which improves the accuracy of the update.
> 
> In essence, **subtracting** $\beta u_{t-1}$ allows you to "look ahead" to the future position without prematurely applying the momentum. By looking ahead, you get a better sense of **where the function is heading**, and that leads to more informed updates.
## 👁️‍🗨️ Observations About NAG

* **Looking ahead** enables NAG to correct its course more quickly than standard momentum.
* Oscillations during training are **significantly reduced**.
* The risk of **overshooting** or **escaping shallow minima** is reduced — ideal for fine-tuning near the minima.

---

## 📊 Visual Gradient Derivation Example

For a neural network with sigmoid activation:

### Definitions:

* Input vector:

  $$
  x = \{x_1, x_2, x_3, x_4\}
  $$
* Weights:

  $$
  w = \{w_1, w_2, w_3, w_4\}
  $$
  
* Output:

  $$
  y = f(x) = \frac{1}{1 + e^{-(w^T x + b)}}
  $$

Given a single input-label pair $(x, y)$:

$$
\nabla w_1 = (f(x) - y) \cdot f(x) \cdot (1 - f(x)) \cdot x_1
$$

$$
\nabla w_2 = (f(x) - y) \cdot f(x) \cdot (1 - f(x)) \cdot x_2
$$


... and so on.

For multiple data points, sum gradients across all $n$ points.

$$
\nabla w_j = \sum_{i=1}^n (f(x^{(i)}) - y^{(i)}) \cdot f(x^{(i)}) \cdot (1 - f(x^{(i)})) \cdot x_j^{(i)}
$$
---

## ⚠️ Sparse Features: A Challenge

What if feature $x_2$ is **very sparse** (i.e., zero for most inputs)?

* Then, $∇w_2$ will be **zero** for most examples.
* This means **$w_2$ won’t be updated frequently**, which can hinder learning.

If $x_2$ is **sparse and important**, we should be more deliberate with how we update $w_2$.

---

## 🔧 Adaptive Learning Rates?

Yes! Consider using a **different learning rate per parameter**, based on how frequently the corresponding feature appears. This is one of the motivations behind **adaptive optimizers** like AdaGrad, RMSprop, and Adam.

---

## ✅ Summary

| Concept                  | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| **NAG**                  | Looks ahead by evaluating gradient at anticipated position.            |
| **Advantage**            | Reduces oscillations, faster convergence, better handling near minima. |
| **Update rule**          | $u_t = \beta u_{t-1} + \nabla(w_t - \beta u_{t-1})$                    |
| **Sparse feature issue** | Sparse inputs lead to under-updated weights.                           |
| **Solution**             | Use feature-aware learning rates (e.g., AdaGrad).                      |

---

# 📘 AdaGrad – Adaptive Gradient Algorithm

Gradient descent optimizers like SGD and Momentum struggle with **sparse data**. AdaGrad adapts the learning rate **per parameter** based on update history, making it well-suited for such scenarios.

---
## 💡 Intuition

> **Decay the learning rate for each parameter in proportion to how frequently it gets updated.**

* Parameters that receive **more frequent updates** get a **smaller learning rate** over time.
* Parameters updated **less often** retain a **larger learning rate**, allowing them to still learn effectively.

---
## 🔁 Update Rule for AdaGrad

For a parameter $w$:

$$
v_t = v_{t-1} + (\nabla w_t)^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot \nabla w_t
$$

Where:

* $v_t$: Accumulated squared gradients
* $\epsilon$: Small constant to avoid division by zero
* $\eta$: Global learning rate

> Similar updates apply for bias $b$ or any other parameters.

**The effect**: Parameters with large gradients will have their learning rates decreased more rapidly, while parameters with small gradients will have their learning rates increased.

> [!NOTE]
> 
> #### Gradient Descent with AdaGrad: Why Squared Gradients?
> 
> AdaGrad's key innovation is to **adapt the learning rate** based on the history of gradients. Specifically, it uses the **squared gradient** to scale the learning rate over time. Here’s why this is beneficial:
> 
> ##### a. Handling Large and Small Gradients Differently
> 
> * When we use the **squared gradient**, we’re **accumulating the magnitude** of the gradient over time, which allows us to adjust the learning rate dynamically.
> * **For parameters with large gradients**, the accumulation of squared gradients will result in a **larger denominator** in the update rule, reducing the learning rate for that parameter over time. This helps prevent overshooting and makes the optimizer more stable when a parameter is changing rapidly.
> * **For parameters with small gradients**, the accumulated squared gradient will be smaller, meaning the learning rate will remain larger, allowing faster movement in directions where the gradients are smaller.
> 
> ##### b. Emphasis on Larger Gradients
> 
> By squaring the gradients, you give **more weight to larger gradients**. This means that the optimizer will make **larger adjustments to weights** that have had consistently large gradients. Over time, this helps **converge faster** for weights that are "sensitive" to changes (i.e., where the gradient is large).

---

## 🧪 Experiment: Sparse Feature in a Toy Network

To see AdaGrad in action, we need to **simulate sparsity** in one of the features.

**How?**

* Our toy model has two parameters: **weight $w$** and **bias $b$**.
* Bias feature is always **active (1)**, so we **cannot** make it sparse.
* The only option: make **input feature $x$** (which connects to $w$) sparse.

### ✅ Solution:

* Generate **500** random $(x, y)$ pairs.
* For **80%** of these pairs, set $x = 0$.
* This results in **sparse updates for $w$** and **dense updates for $b$**.

---

## 🧠 Code Snippet: AdaGrad Implementation

```python
def do_adagrad(max_epochs):
    # Initialization
    w, b, eta = -2, -2, 0.1
    v_w, v_b, eps = 0, 0, 1e-8

    for i in range(max_epochs):
        # Zero grad
        dw, db = 0, 0
        for x, y in zip(X, Y):
            # Compute gradients
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        # Accumulate squared gradients
        v_w = v_w + dw ** 2
        v_b = v_b + db ** 2

        # Update parameters
        w = w - eta * dw / (np.sqrt(v_w) + eps)
        b = b - eta * db / (np.sqrt(v_b) + eps)
```

---

## 🔍 What’s Happening?

In our setup:

* All algorithms (SGD, Momentum, NAG) show **initial movement along the vertical axis** (i.e., update $b$).
* There's **minimal horizontal movement** (i.e., update $w$), because:
  * Feature corresponding to $w$ (i.e., $x$) is **sparse**.
  * Hence, $w$ gets **fewer updates**.

This is visible in the **optimization path visualization**, where curves move more in the direction of $b$ than $w$.

---

## 📉 Why This Matters

Sparsity is very **common in large neural networks** — especially those with:

* **Thousands of input features**
* **High-dimensional sparse data** (e.g., text, recommender systems)

If not addressed, important but rare features may **never get trained properly**.

---

## 📊 AdaGrad Performance Summary

| Parameter          | Value                         |
| ------------------ | ----------------------------- |
| Dataset size       | 500 samples                   |
| Sparsity           | 80% of $x = 0$                |
| Learning rate      | $\eta = 0.1$ (all algorithms) |
| Momentum (if used) | $\beta = 0.9$                 |

* **AdaGrad slows down near the minimum** because the learning rate **decays** over time due to accumulated squared gradients.

---

# 🔍 Deep Dive into AdaGrad: Sparse vs Dense Features

Let’s now **examine AdaGrad more closely** by analyzing how it handles different types of features — specifically, **sparse** vs **dense** inputs.

---

## 🔢 Accumulated Gradient Analysis for Weight $w$

$$
v_t^{(w)} = v_{t-1}^{(w)} + (\nabla w_t)^2
$$

Expanding from initial steps:

$$
v_0^{(w)} = (\nabla w_0)^2
$$

$$
v_1^{(w)} = (\nabla w_0)^2 + (\nabla w_1)^2
$$

$$
v_2^{(w)} = (\nabla w_0)^2 + (\nabla w_1)^2 + (\nabla w_2)^2
$$

$$
\Rightarrow v_t^{(w)} = \sum_{i=0}^{t} (\nabla w_i)^2
$$

Where:

$$
\nabla w = (f(x) - y) \cdot f(x) \cdot (1 - f(x)) \cdot x
$$

Since **$x$** is sparse, $\nabla w$ is often **zero**.

🧠 **Conclusion:**

* $v_t^{(w)}$ increases slowly
* The learning rate $\frac{\eta}{\sqrt{v_t^{(w)} + \epsilon}}$ **decays slowly**
  → allowing **larger updates** to $w$ over time

---

## 🔢 Accumulated Gradient Analysis for Bias $b$

$$
v_t^{(b)} = v_{t-1}^{(b)} + (\nabla b_t)^2
$$

Similarly,

$$
v_0^{(b)} = (\nabla b_0)^2
$$

$$
v_1^{(b)} = (\nabla b_0)^2 + (\nabla b_1)^2
$$

$$
\Rightarrow v_t^{(b)} = \sum_{i=0}^{t} (\nabla b_i)^2
$$

Where:

$$
\nabla b = (f(x) - y) \cdot f(x) \cdot (1 - f(x))
$$

Since this gradient is **independent of $x$** (or weakly dependent), it's **rarely zero**.

🧠 **Conclusion:**

* $v_t^{(b)}$ grows **rapidly**
* The learning rate $\frac{\eta}{\sqrt{v_t^{(b)} + \epsilon}}$ **decays rapidly**
  → resulting in **very small updates** to $b$ over time

---

## 📊 Visualization

### 📈 Weight $w$

$$
v_t^{(w)} = v_{t-1}^{(w)} + (\nabla w_t)^2
$$

* Blue: $v_t^{(w)}$ increasing slowly
* Red: $\eta_t$ (effective learning rate) remains relatively high
* Dashed: Gradient $\nabla w$ small most of the time

### 📉 Bias $b$

$$
v_t^{(b)} = v_{t-1}^{(b)} + (\nabla b_t)^2
$$

* Blue: $v_t^{(b)}$ increases rapidly
* Red: $\eta_t$ drops quickly
* Dashed: Gradient initially non-zero, then tapers off

> Observe: In AdaGrad, $v_t^{(w)}$ and $v_t^{(b)}$ **never become zero**, even after gradients diminish.

---

## 🧠 What This Means in Practice

* By using **parameter-specific learning rates**, AdaGrad:

  * Gives **higher learning rate to sparse features** (like $w$)
  * Applies **strong decay to frequently updated features** (like $b$)

✅ **Benefits:**

* Sparse features don’t get ignored
* Dense features stop over-updating

⚠️ **Caveat:**

* Over time, learning rate for frequently updated parameters like $b$ may decay **too much** → no further updates

---

## 🔍 Insight

> If you **remove the square root** from the denominator in AdaGrad, the learning rate decay becomes too aggressive and often causes **training to halt prematurely**.

> This highlights the subtle balance AdaGrad strikes using the square root — it's **not just cosmetic**, it's crucial to training dynamics.

---

> [!NOTE]
> ### **Why Take the Square Root of the Accumulated Gradient?**
> 
> #### **Key Reason: To Keep the Learning Rate in a Reasonable Range**
> 
> Without the square root, the accumulated sum of squared gradients ($v_t$) would grow **unbounded** over time, especially for parameters with large gradients. This would result in an **extremely small learning rate** for those parameters because the denominator in the update rule grows without bound.
> 
> The formula for the weight update is:
> 
> $$
> w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot \nabla w_t
> $$
> 
> * **$\eta$** is the base learning rate.
> * **$\nabla w_t$** is the gradient of the loss function at time step $t$.
> * **$\sqrt{v_t + \epsilon}$** is the term that controls the effective learning rate for each parameter. The **square root** ensures that as the accumulated gradient grows, the learning rate decreases at a **slower rate** and doesn’t become too small too quickly.
> 
> #### **Why the Square Root Works:**
> 
> 1. **Scaling Behavior**:
>    The square root of the accumulated gradients ensures that the updates become **progressively smaller** over time, but not **too quickly**. If we didn’t take the square root, the learning rate would decrease too aggressively and too fast, making the model converge too slowly or stop learning altogether.
> 
> 2. **Normalization**:
>    The square root essentially **normalizes** the accumulated sum, ensuring that the learning rate remains at a **reasonable scale**. This helps prevent the optimization process from stagnating, as the learning rate won’t shrink too quickly, even for parameters with consistently large gradients.
> 
> 3. **Preserving the Gradient Magnitude**:
>    The square root also **moderates the impact** of large gradients without completely discarding them. By squaring the gradient initially, we give larger gradients more influence during the accumulation, but then we moderate this influence by taking the square root. This helps balance the contributions of larger and smaller gradients in a more controlled way.


---
# 📘 RMSProp – Root Mean Squared Propagation

### 🛠️ Solving AdaGrad’s Decay Problem with RMSProp

#### ❓ Can We Avoid the Rapid Decay in AdaGrad?

### 🔍 Intuition

> AdaGrad decays the learning rate **very aggressively** due to the **accumulating squared gradients** in the denominator.

* This causes the **learning rate to shrink** too much over time.
* Frequently updated parameters (like bias $b$) get **negligibly small updates** after a while.

🧠 **Idea**:
Why not **decay the accumulated denominator itself**?
This would prevent unbounded growth and **retain an effective learning rate**.

---

## 🔄 RMSProp Update Rule

Instead of summing squared gradients like AdaGrad, **RMSProp uses an exponentially weighted moving average**:

### Update Equations:

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla w_t)^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla w_t
$$

... and a similar set of equations apply for $b$

Where:

* $\beta \in [0, 1]$ controls the decay rate of the moving average.
* Common default: $\beta = 0.9$

---

## 📊 Comparing AdaGrad vs RMSProp (with respect to bias $b$)

### 🧮 AdaGrad:

$$
v_t = v_{t-1} + (\nabla b_t)^2 = \sum_{i=0}^{t} (\nabla b_i)^2
$$

$$
\nabla b = (f(x) - y) \cdot f(x) \cdot (1 - f(x))
$$

* Since $\nabla b$ is **frequently non-zero**, the accumulated gradient grows rapidly.
* Therefore:

$$
\frac{\eta}{\sqrt{v_t + \epsilon}} \text{ decays rapidly}
$$

---

### 🧮 RMSProp:

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla b_t)^2
$$

Expanding:

$$\begin{align*}
v_0 &= 0.1(\nabla b_0)^2 \\
v_1 &= 0.09(\nabla b_0)^2 + 0.1(\nabla b_1)^2 \\
v_2 &= 0.08(\nabla b_0)^2 + 0.09(\nabla b_1)^2 + 0.1(\nabla b_2)^2
\end{align*}
$$

Or, in general:

$$
v_t = (1 - \beta)\sum_{\tau=0}^{t} \beta^{t - \tau}(\nabla b_\tau)^2
$$

* This exponential decay gives **more weight to recent gradients**.
* As a result:

$$
\frac{\eta}{\sqrt{v_t + \epsilon}} \text{ decays slowly (compared to AdaGrad)}
$$

---

## 📉 Convergence Comparison

Two visualizations show:

1. **Contour Plot** (left):

   * AdaGrad updates become horizontal due to over-decay.
   * RMSProp maintains diagonal progress.

2. **Learning Rate Plot** (middle):

   * With $\eta = 0.1, \beta = 0.5$, RMSProp retains a usable learning rate longer than AdaGrad.

✅ **RMSProp converges more quickly** by being **less aggressive on decay**.

---

## 💻 Python Code: RMSProp Implementation

```python
def do_rmsprop(max_epochs):
    # Initialization
    w, b, eta = -4.4, 0.1
    beta = 0.5
    v_w, v_b, eps = 0.0, 0.0, 1e-4

    for i in range(max_epochs):
        zero_grad()
        dw, db = 0, 0

        for x, y in zip(X, Y):
            # Compute gradients
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        # Update intermediate values
        v_w = beta * v_w + (1 - beta) * dw ** 2
        v_b = beta * v_b + (1 - beta) * db ** 2

        # Update parameters
        w -= eta * dw / (np.sqrt(v_w) + eps)
        b -= eta * db / (np.sqrt(v_b) + eps)
```

---

## 🤔 But Why Are There Oscillations?

> RMSProp sometimes shows oscillations due to the **delayed response** in the smoothed moving average.
> This can be addressed by:

* Tuning $\beta$
* Using **momentum** in combination (as in **Adam**)

---

# 🔄 RMSProp: Oscillation and Learning Rate Sensitivity

---

## ❓ Can RMSProp Cause Oscillations?

> After many iterations, **could the learning rate become constant**, causing the algorithm to **oscillate indefinitely around the minimum**?

Let’s explore how AdaGrad and RMSProp differ in behavior over long iterations.

---

## 🔁 Behavior of the Denominator $v_t$

### 📌 AdaGrad:

$$
v_t = v_{t-1} + (\nabla w_t)^2
$$

* Keeps **growing** even if gradients shrink (or become zero).
* Learning rate keeps **decreasing** due to the ever-growing denominator.
* Hence, gradient steps **shrink monotonically** over time.

---

### 📌 RMSProp:

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla w_t)^2
$$

* **Moving average** keeps $v_t$ more stable.
* Hence, the learning rate:

  * May **increase**
  * May **decrease**
  * Or may even **remain constant**

➡️ This can result in **oscillation** around the minima if the learning rate doesn’t shrink when needed.

---

## 📉 Visualization: Gradients Across 500 Iterations

The graphs show the behavior of gradients for:

* **AdaGrad**: Gradients approach zero as denominator grows.
* **RMSProp**: Gradients show **oscillatory behavior**, especially around minima.

👉 This happens **when the learning rate remains constant** and the optimizer "bounces" near the optimum.

---

## ✅ How Do We Fix This?

> 📌 **Solution**: Set the **initial learning rate** appropriately to avoid oscillations.

In this example, choosing:

$$
\eta = 0.05,\quad \epsilon = 0.0001
$$

Gives a learning rate:

$$
\eta = \frac{0.05}{\sqrt{10^{-4}}}
$$

This choice helps **smooth out the updates**, reducing oscillations and allowing convergence.

---

## ⚠️ Drawbacks of RMSProp (and AdaGrad)

Both optimizers are:

🔻 **Sensitive to**:

* Initial learning rate
* Initial conditions of parameters
* Magnitude of early gradients

---

### 📉 AdaGrad Specific Issue:

* If **initial gradients are large**, the accumulated denominator grows rapidly.
* This leads to **tiny learning rates** in the remaining training.
* Later, if the curve flattens, **there’s no way to recover** a larger step size.

📌 That’s why AdaGrad **struggles with long training**, while RMSProp improves upon it using a decaying moving average.

---

# 📘 AdaDelta Optimizer: Adaptive Learning Without Manual Tuning

---

## 📌 What is AdaDelta?

**AdaDelta** is an extension of RMSProp that avoids **manually setting an initial learning rate $\eta_0$**.

> It is called **Adaptive Delta** because it uses the change in parameters $\Delta w_t$ (delta) to adapt the learning rate automatically.

---

## ⚙️ AdaDelta Algorithm Steps

For each iteration $t \in \text{range}(1, N)$:

1. Compute gradient:

   $$
   \nabla w_t
   $$

2. Accumulate squared gradient using EMA:

   $$
   v_t = \beta v_{t-1} + (1 - \beta)(\nabla w_t)^2
   $$

3. Compute parameter update using historical updates:

   $$
   \Delta w_t = - \frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}} \nabla w_t
   $$

4. Update weights:

   $$
   w_{t+1} = w_t + \Delta w_t
   $$

5. Accumulate squared updates for use in next iteration:

   $$
   u_t = \beta u_{t-1} + (1 - \beta)(\Delta w_t)^2
   $$

> $\Delta w_t$ is used not just for weight update, but also as a historical input for computing the **next update**.

---

## 🧠 Why Is This Adaptive?

* In **RMSProp**, the numerator in effective learning rate is a **constant**.
* In **AdaDelta**, the numerator $\sqrt{u_{t-1} + \epsilon}$ is based on **past gradients and updates**, which **adapts the learning rate dynamically**.

---

## 🔍 How the Updates Work (Step-by-Step)

### Iteration $t = 0$:

* Initial update:

  $$
  v_0 = 0.1(\nabla w_0)^2,\quad \Delta w_0 = -\frac{\sqrt{\epsilon}}{\sqrt{v_0 + \epsilon}} \nabla w_0
  $$
* Weight update:

  $$
  w_1 = w_0 + \Delta w_0
  $$
* Accumulate update history:

  $$
  u_0 = 0.1(\frac{\sqrt{\epsilon}}{\sqrt{v_0 + \epsilon}} \nabla w_0)^2
  $$
store a fraction of a history for next iteration
---

### Iteration $t = 1$:

* Gradient squared average:

  $$
  v_1 = 0.9v_0 + 0.1(\nabla w_1)^2
  $$
  $$
  v_1 = 0.9(0.1(\nabla w_0)^2) + 0.1(\nabla w_1)^2
  $$
  $$
  v_1 = 0.09(\nabla w_0)^2 + 0.1(\nabla w_1)^2
  $$
  
* Update:

  $$
  \Delta w_1 = -\frac{\sqrt{u_0}}{\sqrt{v_1}} \nabla w_1
  $$
  ignoring $\epsilon$

$$
w_2 = w_1 + \Delta w_1
$$
* Accumulate update history:
  $$
  u_1 = 0.09(\frac{\sqrt{\epsilon}}{\sqrt{v_0 + \epsilon}} \Delta w_0)^2 + 0.1(\frac{\sqrt{u_0}}{\sqrt{v_1}} \nabla w_1)^2
  $$

---

### Iteration $t = 2$:

* Similarly compute $v_2, \Delta w_2, u_2$

> 🔁 For each iteration:
> Both $v_t$ and $u_t$ increase,
> But $u_t$'s magnitude < $v_t$,
> because we take only a **fraction** of $(\Delta w_t)^2$

---

## 🔬 Effective Learning Rate

### AdaDelta:

$$
\text{Effective LR} = -\frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}} \Delta w_t
$$

### RMSProp:

$$
\text{Effective LR} = -\frac{1}{\sqrt{v_t + \epsilon}} \Delta w_t
$$

➡️ **AdaDelta** adapts to historical gradient magnitudes, whereas RMSProp keeps a static numerator.

---

## 📈 Visual Understanding

* At **high curvature regions** (e.g., steep slopes):

  * AdaDelta adjusts the learning rate based on **past updates**
  * This avoids aggressive learning rate drops like in RMSProp
  * Hence, convergence is smoother and **more resilient to sharp gradients**

---

## ✅ Key Takeaways

* AdaDelta **eliminates the need for manual learning rate tuning**
* It uses **past updates and gradients** to adjust learning dynamically
* More stable than RMSProp in highly curved or noisy landscapes

---

# 🔄 AdaDelta: Adaptive Learning Rate – Continued

---

## 📉 Behavior in Low Curvature Regions

After several iterations $i$, the accumulated squared gradient $v_i$ **starts decreasing**, especially when gradients stay small. This causes the **ratio of numerator to denominator** in the update rule to **increase**.

### Key Equations:

* Gradient history:

  $$
  v_i = 0.034 \nabla w_0^2 + 0.038 \nabla w_1^2 + \cdots + 0.1 \nabla w_i^2
  $$

* Update rule:

  $$
  \Delta w_i = -\frac{\sqrt{u_{i-1} + \epsilon}}{\sqrt{v_i + \epsilon}} \nabla w_i
  $$

  $$
  w_{i+1} = w_i + \Delta w_i
  $$

* Update accumulation:

  $$
  u_i = 0.034 \Delta w_0^2 + 0.038 \Delta w_1^2 + \cdots + 0.1 \Delta w_i^2
  $$

> 🔁 If gradients remain **consistently low**, AdaDelta **increases** the effective learning rate accordingly.

---

## 🧠 Why This Matters?

* Unlike RMSProp, where the learning rate might continuously shrink in flat regions (due to cumulative gradient decay),
* **AdaDelta allows the numerator to "catch up"**, making the algorithm more responsive and balanced.

---

## 🔍 Illustration – Low Curvature Region

Just like in high curvature cases, the same update steps apply:

```text
1. → ∇wₜ
2. → vₜ = βvₜ₋₁ + (1 − β)(∇wₜ)²
3. → Δwₜ = − √(uₜ₋₁+ε) / √(vₜ+ε) ∇wₜ
4. → wₜ₊₁ = wₜ + Δwₜ
5. → uₜ = βuₜ₋₁ + (1 − β)(Δwₜ)²
```

---

# 📘 ⚡ Adam Optimizer – Combining the Best of All Worlds

## 🚨 Intuition

> Do everything RMSProp and AdaDelta do to handle gradient decay,
> **PLUS**: incorporate **momentum** and **bias correction**
> **AND**: use cumulative history of gradients.

---

## 🔬 Adam Algorithm Equations

* **Momentum**:

  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla w_t
  $$

* **Bias-corrected momentum**:

  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
  $$

* **Adaptive gradient accumulation**:

  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2
  $$

* **Bias-corrected accumulation**:

  $$
  \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$

* **Parameter update**:

  $$
  w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
  $$

> Typical values:
> $\beta_1 = 0.9$,
> $\beta_2 = 0.999$,
> $\epsilon = 10^{-8}$

---

# ❓ Million Dollar Question: Which Optimizer Should You Use?

### ✅ Adam – The Practical Default

* Default choice for most applications.
* Very robust to learning rate settings.
* For **sequence generation tasks**, lower learning rates like $\eta = 0.001$ or $0.0001$ work better.

### ⚖️ SGD with Momentum

* Works well with **annealing schedules**.
* Sometimes **outperforms Adam** in generalization, especially in simpler or well-structured tasks.

---

## ⚠️ Warnings with Adam:

* Some works report that **Adam may fail to converge** in certain edge cases.
* Models trained with Adam often **do not generalize well**.
* This is a known issue and is actively studied in the research community.

---

# 📚 Summary

| Optimizer          | Pros                                                                   | Cons                                           |
| ------------------ | ---------------------------------------------------------------------- | ---------------------------------------------- |
| **AdaGrad**        | Simple, good for sparse data                                           | Learning rate decays too fast                  |
| **RMSProp**        | Better than AdaGrad, adapts to curvature                               | Can shrink learning rate aggressively          |
| **AdaDelta**       | No manual learning rate, handles both curvature and low-gradient areas | Slower convergence in some cases               |
| **Adam**           | Combines RMSProp + Momentum + Bias Correction                          | May overfit or generalize poorly in some tasks |
| **SGD + Momentum** | Strong generalization, simple to tune                                  | Slower convergence without schedule            |


---

# 🎯 Why Do We Need Bias Correction in Adam?

Adam is an optimizer that combines the benefits of **momentum** and **adaptive learning rates** (like in RMSProp). However, it introduces a crucial step called **bias correction**, especially in the early stages of training.

---

## 🔁 Update Rule for Adam (Recap)

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla w_t
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

---

## 🧠 Why This Bias Correction is Needed?

When calculating $m_t$ and $v_t$, we're using **exponential moving averages (EMA)**, which tend to **bias the estimates toward zero initially**, especially when $t$ is small.

We use:

* $m_t$ to represent a **smoothed estimate of the gradient**
* $v_t$ for a **smoothed estimate of squared gradients**

However, initially, $m_t$ and $v_t$ are **biased toward zero** due to starting from zero vectors.

---

## 📈 Running Average and Bias

We want our smoothed estimates to reflect the **true mean gradient** over time:

* Instead of just using $\nabla w_t$, we use a running average to capture **overall behavior**
* But since the average starts at zero, it underestimates the true value in early iterations

---

## 📘 Deriving the Bias

Let’s recall the recursive momentum equation:

$$
m_t = \beta m_{t-1} + (1 - \beta) \nabla w_t
$$

If expanded out, it becomes:

$$
m_t = (1 - \beta) \sum_{\tau = 1}^{t} \beta^{t - \tau} \nabla w_\tau
$$

Taking expectation on both sides:

$$
E[m_t] = (1 - \beta) \sum_{\tau = 1}^{t} \beta^{t - \tau} E[\nabla w_\tau]
$$

Assuming all gradients come from the same distribution:

$$
E[\nabla w_\tau] = E[\nabla w] \quad \forall \tau
$$

Then:

$$
E[m_t] = E[\nabla w] (1 - \beta) \sum_{\tau = 1}^{t} \beta^{t - \tau}
$$

This is a **geometric progression**:

$$
\sum_{\tau = 1}^{t} \beta^{t - \tau} = \frac{1 - \beta^t}{1 - \beta}
$$

Thus:

$$
E[m_t] = E[\nabla w] (1 - \beta^t)
$$

So the bias is:

$$
\text{Bias in } m_t = 1 - \beta^t
$$

---

## ✅ Correcting the Bias

To get an unbiased estimate:

$$
\hat{m}_t = \frac{m_t}{1 - \beta^t} \quad \Rightarrow \quad E[\hat{m}_t] = E[\nabla w]
$$

This is why we need bias correction! Without it, especially in early steps, $m_t$ underestimates the true gradient.

The same logic applies to $v_t$ and $\hat{v}_t$.

---

## 🔍 Summary

| Term  | Purpose                                   | Issue                     | Fix                                     |
| ----- | ----------------------------------------- | ------------------------- | --------------------------------------- |
| $m_t$ | EMA of gradients (momentum)               | Biased towards 0 at start | $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ |
| $v_t$ | EMA of squared gradients (adaptive scale) | Biased towards 0 at start | $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ |


---

# 🎯 Bias Correction in Adam Optimizer – Illustrated Tutorial

## 🔍 What is Bias Correction?

Bias correction addresses the **initialization bias** that occurs in **exponentially weighted averages**, especially during early iterations. In optimizers like Adam, this helps stabilize the learning rate by preventing the averages from being skewed toward zero.

---

## 📉 Why Does Bias Occur?

Assume we only have **noisy observations** of a true function $f(t)$. We estimate the true function using **exponentially weighted averages (EWMA)**.

### 🔹 Without Bias Correction

* The smoothed estimate starts from 0 and is pulled toward zero in early steps.
* Result: **Poor approximation** during initial iterations.

<div align="center">
Estimated Function vs True Function:
- Noisy: Blue Dots  
- True Function \( f(t) \): Black Line  
- EWMA Estimate: Red Line (biased low at start)
</div>

---

## ✅ With Bias Correction

* Applying bias correction yields a better estimate of the expected value.
* It **attenuates the early bias**, giving more accurate updates.

---

## 🧠 What Happens Without Bias Correction?

Adam’s second moment estimate (variance) update:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2, \quad \beta_2 = 0.999
$$

### ➕ Example Without Bias Correction

Let $\nabla w_0 = 0.1$:

$$
v_0 = 0.999 \cdot 0 + 0.001 \cdot (0.1)^2 = 0.00001
$$

$$
\eta_t = \frac{1}{\sqrt{v_0}} = 316.22
$$

Next step:

$$
v_2 = 0.999 \cdot v_0 + 0.001 \cdot 0^2 = 0.0000099
$$

$$
\eta_t = \frac{1}{\sqrt{v_2}} = 316.38
$$

🔹 **Insight**: No bias correction results in **huge learning rates** initially due to small denominator values.

---

### ➕ Example With Bias Correction

Same gradient $\nabla w_0 = 0.1$:

$$
v_0 = 0.00001, \quad \hat{v}_0 = \frac{v_0}{1 - \beta_2} = \frac{0.00001}{0.001} = 0.01
$$

$$
\eta_t = \frac{1}{\sqrt{0.01}} = 10
$$

Next:

$$
v_2 = 0.0000099, \quad \hat{v}_2 = \frac{v_2}{1 - \beta_2^2} = \frac{0.0000099}{0.0019} \approx 0.0052
$$

$$
\eta_t = \frac{1}{\sqrt{0.0052}} \approx 13.8
$$

✅ **Conclusion**: Bias correction **stabilizes** the learning rate in early iterations and avoids erratic updates.

---

## 🧮 Revisit of $L^p$ Norm (Extra Conceptual Insight)

The $L^p$ norm is defined as:

$$
L^p = \left( |x_1|^p + |x_2|^p + \cdots + |x_n|^p \right)^{\frac{1}{p}}
$$

### 🔎 Visualization:

Let’s fix $L^p = 1$, and vary $p$:

$$
1 = \left( |x_1|^p + |x_2|^p \right)^{\frac{1}{p}} \Rightarrow 1^p = |x_1|^p + |x_2|^p
$$

We can choose any $p \geq 1$. As $p \to \infty$, this converges to:

$$
\max(|x_1|, |x_2|, \ldots, |x_n|)
$$

⚠️ When $p$ is very high, $|x|^p$ becomes too small to represent → **Numerical instability** can occur.

---

## 📌 Final Takeaway

Bias correction in Adam:

* Compensates for initialization bias in EMA.
* Prevents **oversized steps** in early updates.
* Is especially important when using **high decay rates** like $\beta_1, \beta_2 \approx 0.9 - 0.999$.

---

# 🔄 Alternative to Bias Correction: Using Max Norm in Adam Optimizer

## 📌 Objective

So, what’s the key takeaway?

Bias correction in Adam deals with the bias from zero initialization in exponentially weighted averages. But what if we **replace the L² norm with the max norm (L∞)**?

Let’s break it down.

---

## 🧾 Recall Adam’s Variance Update

The original Adam update for second moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Since $\sqrt{v_t}$ resembles an $L^2$ norm, why not explore an **$L^\infty$ (max norm)** alternative?

---

## ✅ Replacing $\sqrt{v_t}$ with Max Norm

We propose the following update rule:

$$
v_t = \max(\beta_2^{-1} |\nabla w_1|, \beta_2^{-2} |\nabla w_2|, \ldots, |\nabla w_t|)
$$

Or recursively,

$$
v_t = \max(\beta_2 v_{t-1}, |\nabla w_t|)
$$

$$
w_{t+1} = w_t - \frac{\eta_0}{v_t} \hat{m}_t
$$

**Note**: No bias correction needed!
Max norm is **not susceptible to initialization bias** (since it's not an average), hence no correction is required.

---

## 🧪 Illustrative Example

### Without Max Norm

* Estimating the true function using exponential averaging is **biased toward zero** early on.

### With Max Norm

* Estimating using $\max(\beta_2 v_{t-1}, |\nabla w_t|)$ yields:

  * Faster convergence
  * No bias toward zero at the beginning

📊 **Result**:

* Max norm successfully avoids the "cold start" problem.
* More accurate in early iterations.

---

## 📈 Sparse Gradient Scenarios

Suppose:

* Initial gradient $\nabla w_0$ is **high**
* Next gradients are **zero** (e.g., due to sparsity in input)

We **don’t want the learning rate to increase** just because current gradient is zero.

### Update:

$$
v_t = \max(\beta_2 v_{t-1}, |\nabla w_t|), \quad \beta_2 = 0.999
$$

---

### ⛳ Case Study: 50% of Input is Zero

Let:

* $\nabla w_0 = 1$
* Later gradients $\nabla w_t = 0$

Then:

$$
v_0 = \max(0, |\nabla w_0|) = 1, \quad \eta_1 = \frac{1}{1} = 1
$$

$$
v_1 = \max(0.999 \cdot 1, 0) = 0.999, \quad \eta_2 = \frac{1}{0.999} \approx 1.001
$$

$$
v_2 = \max(0.999 \cdot 0.999, 0) \approx 1, \quad \eta_3 = 1
$$

📉 **Observation**:

* Learning rate doesn’t oscillate drastically.
* Gradual, smooth decay (if any) in max value.
* Stable training even in **sparse data regimes**.

---

## 📊 Visual Summary

* **Top-Right Graph**: Spike at initialization, then low gradient
* **Middle Graph**: Maintains stable max norm
* **Bottom Graph**: Corresponding learning rate remains stable, oscillating slightly but not diverging

---

## 🧠 Final Insight

Using max norm:

* Avoids bias correction
* Improves performance in sparse data
* Reduces numerical instability
* Retains adaptive learning rate behavior

---

Let me know if you'd like this merged with the previous tutorial into a single document (PDF, DOCX, etc.), or if you want code-based walkthroughs or animations for better understanding.

Here is **Part 3** of the tutorial, incorporating and expanding on the content in the latest image. This section wraps up the discussion with the motivation, comparison, and final insights behind using **Max Norm** in adaptive optimizers.

---

# 📘 Part 3: From RMSProp to AdaMax — Embracing Max Norm

In this part, we further compare traditional exponential moving average–based optimizers with their **max norm–based** counterparts. We show how these formulations can lead to more stable and robust updates, especially in sparse settings or noisy gradients.

---

## 🧮 Exponential Averaging Revisited

Let’s consider the update rule using standard exponential averaging:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2, \quad \beta_2 = 0.999
$$

Assume:

$$
v_0 = 0.999 \cdot 0 + 0.001 (\nabla w_0)^2 = 0.001
$$

$$
\hat{v}_0 = \frac{0.001}{1 - 0.999} = 1, \quad \eta_t = \frac{1}{\sqrt{1}} = 1
$$

Now suppose next gradients vanish:

$$
v_1 = 0.999 \cdot 0.001 + 0.001 \cdot 0 = 0.000999
$$

$$
\hat{v}_1 = \frac{0.000999}{1 - 0.999^2} = 0.499, \quad \eta_t = \frac{1}{\sqrt{0.499}} \approx 1.41
$$

> **Problem**: Even with bias correction, exponential averaging causes the learning rate to **increase** when $\nabla w_t = 0$ — this is **undesirable**.

---

## ⚙️ Enter MaxProp

We now explore an alternative to RMSProp by introducing **MaxProp**, which replaces the L² norm with an L∞ norm.

### 📌 Update Rule for RMSProp

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla w_t)^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \nabla w_t
$$

### 🚀 Update Rule for MaxProp

$$
v_t = \max(\beta v_{t-1}, |\nabla w_t|)
$$

$$
w_{t+1} = w_t - \frac{\eta}{v_t + \epsilon} \nabla w_t
$$

📢 **Key Benefit**:
MaxProp maintains a stable learning rate in sparse gradients and avoids unexpected increases.

---

## 🔁 From Adam to AdaMax

Building on MaxProp, we extend this idea to the Adam optimizer, resulting in **AdaMax** — a variant proposed in the *original Adam paper*.

---

### 🧮 Update Rule for Adam

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla w_t
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

---

### 🚀 Update Rule for AdaMax

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla w_t
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
v_t = \max(\beta_2 v_{t-1}, |\nabla w_t|)
$$

$$
w_{t+1} = w_t - \frac{\eta}{v_t + \epsilon} \hat{m}_t
$$

✅ **Why AdaMax is Beneficial**:

* No need for bias correction of $v_t$
* Not **susceptible to initial bias towards zero**
* Stable in high-sparsity or high-variance settings

---

## 🧠 Final Thoughts

* Replacing $L^2$ norm with **max norm** results in optimizers that:

  * Are simpler (no bias correction needed)
  * Avoid pitfalls of exponential decay
  * Are robust in sparse/noisy environments
* AdaMax is not just a theoretical tweak—it **originated from the same paper** that introduced Adam.

---

# 📘 Part 4: NAdam — Nesterov Meets Adam

---

## 🧠 Motivation

> We know that **Nesterov Accelerated Gradient (NAG)** outperforms classical Momentum in gradient descent.
>
> 💡 **Question**: Why not combine NAG with Adam?
>
> ✅ **Answer**: Modify the momentum term $m_t$ in Adam to incorporate the **look-ahead** behavior of NAG.

---

## 🔁 Recap: Adam Update Rule

Adam uses both first and second moments of gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla w_t
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla w_t)^2
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

---

## 🔄 Momentum vs. NAG

### 🔸 Standard Momentum

$$
u_t = \beta u_{t-1} + \nabla w_t
$$

$$
w_{t+1} = w_t - \eta u_t
$$

### 🔹 NAG (Look-Ahead Gradient)

$$
u_t = \beta u_{t-1} + \nabla(w_t - \beta \eta u_{t-1})
$$

$$
w_{t+1} = w_t - \eta u_t
$$

This allows looking ahead before calculating the gradient — a subtle but effective tweak.

---

## 👀 Look-Ahead in Action (Illustration)

Using NAG:

$$
g_t = \nabla(w_t - \eta \beta m_{t-1}) \quad \text{(look-ahead)}
$$

$$
m_t = \beta m_{t-1} + g_t
$$

$$
w_{t+1} = w_t - \eta m_t
$$

### Observation:

* The same momentum vector $m_{t-1}$ is used **twice**:

  1. In computing the gradient
  2. In updating the parameter

📉 **Drawback**: This leads to **two weight updates**, which can be computationally expensive — especially for large neural networks.

---

## 🧪 Simulation: NAG Trajectory

Illustration from the image explains:

* $w_0 \to w_1 \to w_2$ steps
* Shows how gradients are calculated at **look-ahead points**
* You compute gradient at $w_1 - \beta m_0$, then move again

This double-update behavior is costly.

---

## ❓ Can We Fix This?

Yes! That's what **NAdam** aims to solve.

🔄 The idea is to **embed the look-ahead behavior directly into Adam**, such that:

* Gradients are computed at **future positions**
* No redundant computation or double-updates
* Still leverage **adaptive learning rates**

The detailed NAdam update rules (coming next) fix the double-update issue by carefully modifying $m_t$ without introducing inefficiencies.

---

# 📘 Part 5: Finalizing NAdam — Look Ahead + Adaptive Gradient

---

## 🔍 Why Not Look Ahead Earlier?

Recall from NAG:

$$
g_1 = \nabla(w_1 - \beta m_0)
$$

We ask:

> ❓ Why not compute this in the previous step?

➡️ Because $\beta m_0$ is not available during gradient computation at $w_0$ — it's based on prior time step only. So, we **can’t look ahead** until after the first momentum term is computed.

---

## 🔁 Rewritten NAG

To simplify computations and reduce redundant updates, we redefine NAG like so:

$$
g_{t+1} = \nabla w_t
$$

$$
m_{t+1} = \beta m_t + g_{t+1}
$$

$$
w_{t+1} = w_t - \eta(\beta m_{t+1} + g_{t+1})
$$

🔍 Now, look-ahead is embedded directly into the **momentum step**.
✅ This prevents double computation of $m_t$ and avoids two updates per step.

---

## 📉 Visualization of Rewritten NAG

For $\eta = 1$, $m_0 = 0$:

* Compute gradient: $g_1 = \nabla w_0$
* Update momentum: $m_1 = \beta m_0 + g_1$
* Update position: $w_1 = w_0 - \beta g_1 - g_1$

Then repeat for next step:

* $g_2 = \nabla w_1$
* $m_2 = \beta m_1 + g_2$
* $w_2 = w_1 - (\beta g_1 + g_2)$

---

## 🔁 NAdam Update Rule

To combine this **efficient look-ahead** with **adaptive learning rates**, we modify Adam as follows:

---

### 🔺 Step-by-step NAdam

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1)\nabla w_t
$$

$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}
$$

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2)(\nabla w_t)^2
$$

$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \left( \beta_1 \hat{m}_{t+1} + \frac{(1 - \beta_1)\nabla w_t}{1 - \beta_1^{t+1}} \right)
$$

This clever blend uses the **look-ahead direction** with **adaptive scaling** — the core of NAdam!

---

## 🧑‍💻 NAdam Python Implementation (SGD Variant)

```python
def do_adamax_sgd(max_epochs):
    # Initialization
    w, b, eta = -4, 0.1, 1
    beta1, beta2 = 0.9, 0.99
    m_w, m_hat, n_b, v_hat, v_b_hat = 0, 0, 0, 0, 0

    for i in range(max_epochs):
        dw, db = 0.0, 0.0
        eps = 1e-10
        for x, y in zip(X, Y):
            # Compute the gradients
            dw = grad_w_sgd(w, b, x, y)
            db = grad_b_sgd(w, b, x, y)

            # Compute intermediate values
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * dw**2
            v_b = beta2 * v_b + (1 - beta2) * db**2

            m_hat = m_w / (1 - beta1 ** (i + 1))
            v_hat = v_w / (1 - beta2 ** (i + 1))

            # Update parameters
            w = w - (eta / (np.sqrt(v_hat) + eps)) * \
                (beta1 * m_hat + (1 - beta1) * dw / (1 - beta1 ** (i + 1)))

            b = b - (eta / (np.sqrt(v_b_hat) + eps)) * \
                (beta1 * m_hat + (1 - beta1) * db / (1 - beta1 ** (i + 1)))
```

---

## ✅ Summary

| Optimizer    | Momentum | Adaptive | Look-ahead |
| ------------ | -------- | -------- | ---------- |
| **SGD**      | ❌        | ❌        | ❌          |
| **Momentum** | ✅        | ❌        | ❌          |
| **NAG**      | ✅        | ❌        | ✅          |
| **Adam**     | ✅        | ✅        | ❌          |
| **NAdam**    | ✅        | ✅        | ✅          |

---




