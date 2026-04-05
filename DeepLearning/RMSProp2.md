Here is an improved, clean tutorial based on the uploaded image, continuing from the AdaGrad explanation into RMSProp. The structure adds clarity while retaining all your original content.

---

# 🚀 From AdaGrad to RMSProp: Can We Avoid Aggressive Decay?

---

## ❓ Problem with AdaGrad

### 💡 **Intuition**

AdaGrad **decays the learning rate very aggressively** because the denominator $\sqrt{v_t + \epsilon}$ grows rapidly over time.

🔁 As a result:

* Frequently updated parameters receive **very small updates**
* Training may **stall** prematurely

> **Can we fix this?**
> Yes! Instead of letting the denominator grow unbounded, we can **control its growth**.

---

## 🔄 Introducing RMSProp

RMSProp modifies AdaGrad by using an **exponentially decaying average** of past squared gradients.

### 📘 **Update Rule for RMSProp**

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla w_t)^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla w_t
$$

📝 A **similar set of equations** applies for the bias term $b$.

---

## ⚖️ Comparing AdaGrad vs RMSProp

### 🔁 AdaGrad (for bias $b$):

$$
v_t = v_{t-1} + (\nabla b_t)^2
\Rightarrow v_t = \sum_{i=0}^{t} (\nabla b_i)^2
$$

Recall:

$$
\nabla b = (f(x) - y) \cdot f(x) \cdot (1 - f(x))
$$

➡️ Since this is **frequently non-zero**, $v_t$ grows rapidly.
Thus, $\frac{\eta}{\sqrt{v_t + \epsilon}}$ **decays rapidly**.

---

### 🔁 RMSProp (for bias $b$):

$$
v_t = \beta v_{t-1} + (1 - \beta)(\nabla b_t)^2,\quad \beta \in [0, 1]
$$

Example:

* $\beta = 0.9$
* $v_0 = 0.1 (\nabla b_0)^2$
* $v_1 = 0.09 (\nabla b_0)^2 + 0.1 (\nabla b_1)^2$
* $v_2 = 0.08 (\nabla b_0)^2 + 0.09 (\nabla b_1)^2 + 0.1 (\nabla b_2)^2$

Generalized:

$$
v_t = (1 - \beta) \sum_{\tau=0}^{t} \beta^{t - \tau} (\nabla b_\tau)^2
$$

✅ As a result, $v_t$ grows **more gradually**, and:

$$
\frac{\eta}{\sqrt{v_t + \epsilon}} \quad \text{decays slowly}
$$

---

## 📉 Visual Insight

### 📊 Comparison of Optimization Paths

* **Left graph:** Shows how RMSProp (red path) stabilizes updates compared to AdaGrad (blue path)
* **Middle graph:** Demonstrates the learning rate ($\eta = 0.1$, $\beta = 0.5$)

  * AdaGrad decay is steep
  * RMSProp maintains learning rate better

📌 **Conclusion:**
RMSProp **converges more quickly** than AdaGrad by being **less aggressive** in learning rate decay.

---

## 🧑‍💻 RMSProp Code Snippet

```python
def do_rmsprop(max_epochs):
    # Initialization
    w, b, eta = -4.4, 0.1
    beta = 0.5
    v_w, v_b, eps = 0.0, 0.0, 1e-4

    for i in range(max_epochs):
        # Zero grads
        dw, db = 0.0, 0.0

        for x, y in zip(X, Y):
            # Compute gradients
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        # Compute intermediate values
        v_w = beta * v_w + (1 - beta) * dw**2
        v_b = beta * v_b + (1 - beta) * db**2

        # Update parameters
        w -= eta * dw / (np.sqrt(v_w) + eps)
        b -= eta * db / (np.sqrt(v_b) + eps)
```

---

## ⚠️ Oscillations?

RMSProp converges faster, but:

> ❓ **Why are there oscillations?**

* It’s due to the imbalance in decay rate or a poor choice of $\beta$ or $\eta$.
* Tuning these hyperparameters is **crucial** to get stable convergence.

---

Would you like a final merged tutorial that includes all parts from AdaGrad through RMSProp in a clean markdown or document file?

---

Here's a polished and structured **tutorial extension** for your previous AdaGrad discussion, now introducing **RMSProp**, with improved clarity and teaching flow. All original content is preserved and augmented for better understanding.

---

# 🛠️ Solving AdaGrad’s Decay Problem with RMSProp

## ❓ Can We Avoid the Rapid Decay in AdaGrad?

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

$$
v_0 = 0.1(\nabla b_0)^2 \\
v_1 = 0.09(\nabla b_0)^2 + 0.1(\nabla b_1)^2 \\
v_2 = 0.08(\nabla b_0)^2 + 0.09(\nabla b_1)^2 + 0.1(\nabla b_2)^2
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

Would you like me to compile this with your previous AdaGrad notes into a single cohesive markdown/tutorial file?
