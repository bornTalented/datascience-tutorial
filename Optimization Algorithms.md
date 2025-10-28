## **Journey of Optimization Algorithms in Deep Learning** 🚀

### **Overview**

Optimization algorithms are used to update model parameters ($\theta$) to minimize the loss function. The evolution of these algorithms started from **Batch Gradient Descent (BGD)** and led to more advanced techniques like **Momentum, RMSProp, and Adam**.

---

## **1️⃣ Batch Gradient Descent (BGD)**

### **Concept**

* Computes the gradient using **all training examples**.
* **Update rule:**

  $$
  \theta = \theta - \alpha \cdot \nabla J(\theta)
  $$

  where $\alpha$ is the **learning rate**.

### **Pros & Cons**

✅ Stable convergence.
❌ **Slow** for large datasets.
❌ Cannot be used in **online learning** (real-time updates).

### **Why Move Beyond BGD?**

* **High memory requirement**: Stores gradients for the entire dataset.
* **Computationally expensive**: Updates only after processing all data.

---

## **2️⃣ Stochastic Gradient Descent (SGD)**

### **Concept**

* Updates weights **after each training example** instead of the full dataset.
* **Update rule:**

  $$
  \theta = \theta - \alpha \cdot \nabla J_i(\theta)
  $$

  where $\nabla J_i(\theta)$ is the gradient from **one data point**.

### **Pros & Cons**

✅ **Faster updates**, better for large datasets.
✅ Allows **online learning** (real-time updates).
❌ **Noisy convergence**: Fluctuates around the minimum.
❌ Hard to choose an **optimal learning rate**.

### **Why Move Beyond SGD?**

* **Convergence is unstable** due to high variance in updates.
* Learning rate tuning is **challenging**.

---

## **3️⃣ Mini-Batch Gradient Descent (MBGD)**

### **Concept**

* Uses a **small batch of training examples** (e.g., 32, 64) to compute updates.
* **Update rule:**

  $$
  \theta = \theta - \alpha \cdot \frac{1}{b} \sum_{i=1}^{b} \nabla J_i(\theta)
  $$

### **Pros & Cons**

✅ **Balances stability and speed**.
✅ Leverages **parallel processing (GPUs)**.
✅ **Smoother convergence** than SGD.
❌ Learning rate tuning is still difficult.

---

## **4️⃣ Gradient Descent with Momentum**

### **Concept**

* **Solves SGD’s fluctuation problem** by adding momentum to updates.
* Uses an **exponential moving average of past gradients** to accelerate training.
* **Update rule:**

  $$
  v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta)
  $$

  $$
  \theta = \theta - \alpha v_t
  $$

  where:

  * $v_t$ is the momentum term.
  * $\beta$ (e.g., 0.9) controls how much of the past gradient is remembered.

### **Intuition:**

* Think of **rolling a ball down a hill**. Momentum helps it **move faster** and avoid oscillations.

### **Pros & Cons**

✅ **Faster convergence** than SGD.
✅ **Reduces oscillations** in gradients.
✅ Helps in **ravines (steep valleys)** where gradients zig-zag.
❌ May overshoot if **momentum is too high**.

---

## **5️⃣ RMSProp (Root Mean Square Propagation)**

### **Concept**

* **Solves the problem of oscillating gradients** by using an **adaptive learning rate**.
* Instead of a fixed learning rate, RMSProp **adjusts** the step size based on recent gradient magnitudes.
* **Update rule:**

  $$
  s_t = \beta s_{t-1} + (1 - \beta) (\nabla J(\theta))^2
  $$

  $$
  \theta = \theta - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla J(\theta)
  $$

  where:

  * $s_t$ is a moving average of squared gradients.
  * $\epsilon$ (e.g., $10^{-8}$) prevents division by zero.

### **Intuition:**

* **If gradients are large**, RMSProp **reduces learning rate**.
* **If gradients are small**, RMSProp **increases learning rate**.

### **Pros & Cons**

✅ **Adapts learning rate dynamically**.
✅ Works well for **non-stationary problems** (e.g., RL).
✅ **Solves the vanishing learning rate issue in SGD**.
❌ **Does not use momentum**, which can slow convergence.

---

## **6️⃣ Adam (Adaptive Moment Estimation)**

### **Concept**

* **Combines Momentum and RMSProp** for **better convergence**.
* Maintains:

  1. **Momentum** (moving average of past gradients).
  2. **Adaptive learning rate** (like RMSProp).
* **Update rule:**

  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla J(\theta)
  $$

  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla J(\theta))^2
  $$

  $$
  \theta = \theta - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t
  $$

  where:

  * $m_t$ is momentum.
  * $v_t$ is adaptive learning rate.
  * $\beta_1 \approx 0.9, \beta_2 \approx 0.999$.

### **Pros & Cons**

✅ **Combines benefits of Momentum + RMSProp**.
✅ **Stable and fast convergence**.
✅ Works well **for deep networks**.
❌ Can sometimes **converge too fast** and miss an optimal solution.

---

## **7️⃣ Summary: Choosing the Best Algorithm**

| Algorithm     | Handles Large Datasets? | Adaptive Learning Rate? | Fast Convergence? | Avoids Oscillations? |
| ------------- | ----------------------- | ----------------------- | ----------------- | -------------------- |
| Batch GD      | ❌                       | ❌                       | ❌                 | ✅                    |
| SGD           | ✅                       | ❌                       | ✅                 | ❌                    |
| Mini-Batch GD | ✅                       | ❌                       | ✅                 | ✅                    |
| Momentum      | ✅                       | ❌                       | ✅                 | ✅                    |
| RMSProp       | ✅                       | ✅                       | ✅                 | ✅                    |
| **Adam**      | ✅                       | ✅                       | ✅                 | ✅                    |

- **Adam is the most widely used optimizer** for deep learning.
- **RMSProp is better for reinforcement learning.**
- **Momentum speeds up SGD without adaptivity.**

---

## **8️⃣ Python Code for Adam Optimization**

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

# Compile with Adam optimizer
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# Train with mini-batches (batch_size=32)
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## **Final Takeaway**

* **Started from Batch GD** → Too slow for large datasets.
* **SGD improved speed** → But unstable.
* **Mini-Batch GD balanced both**.
* **Momentum accelerated convergence**.
* **RMSProp made learning rates adaptive**.
* **Adam combined all the best techniques** → Best for deep learning!