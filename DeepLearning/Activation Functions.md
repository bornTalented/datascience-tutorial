Different types of **activation functions** are used in neural networks because they introduce **non-linearity** into the model, which allows the network to **learn complex patterns and relationships** in the data. The choice of activation function affects the performance, training stability, convergence speed, and representational power of the network. Here's why various types are used:
### 1. Introduce Non-Linearity

Without activation functions, a neural network composed only of linear operations (dot products and additions) would be equivalent to a single-layer linear model.

> **Why it matters**: Non-linear activation functions allow the network to approximate **non-linear functions**, enabling it to solve complex tasks like image recognition, natural language processing, etc.
* Without activation:
	$y = W_2(W_1x) = W'x$ → still linear
* With activation:
	$y = W_2 \cdot \sigma(W_1x)$ → now non-linear

### 2. Different Tasks, Different Needs

Different activation functions have different properties. Choosing the right one depends on:

* Task type (classification, regression)
* Layer type (hidden layer, output layer)
* Convergence behavior
* Gradient flow

Let’s break down commonly used ones:

---

## Common Activation Functions and Why They’re Used

### 1. Sigmoid

**$f(x) = 1 / (1 + e^{-x})$**

* ✅ Smooth, differentiable
* ✅ Used in output layer for binary classification
* ❌ Vanishing gradient for large positive/negative inputs
* ❌ Outputs not zero-centered

### 2. Tanh

**$f(x) = (e^x - e^{-x}) / (e^x + e^{-x})$**

* ✅ Output between -1 and 1 → zero-centered
* ❌ Still suffers from vanishing gradients

### 3. ReLU (Rectified Linear Unit)

**$f(x) = max(0, x)$**

* ✅ Fast convergence (used in deep CNNs)
* ✅ Sparsity (activates only some neurons)
* ❌ Dying ReLU problem: neurons can become inactive permanently

### 4. Leaky ReLU / Parametric ReLU

$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$

* ✅ Solves dying ReLU by allowing small gradients when x < 0
* ✅ Tunable or learnable slope

### 5. **Softmax**

**$f_i(x) = e^{x_i} / Σ e^{x_j}$**

* ✅ Used in output layer for multi-class classification
* ✅ Outputs probabilities that sum to 1

---

Some activation functions like **Swish**, **GELU**, or **Mish** (used in transformers) have been introduced to improve training dynamics and final accuracy in specific architectures.

### Summary Table

| Activation  | Range          | Non-linearity | Issues / Notes                                                     |
| ----------- | -------------- | ------------- | ------------------------------------------------------------------ |
| Sigmoid     | (0, 1)         | Yes           | Vanishing gradients                                                |
| Tanh        | (-1, 1)        | Yes           | Better than sigmoid, still vanishing.<br>Saturation, slow training |
| ReLU        | \[0, ∞)        | Yes           | Dying ReLU for negative inputs                                     |
| Leaky ReLU  | (-∞, ∞)        | Yes           | Fixes ReLU's dying issue                                           |
| GELU, Swish | (-∞, ∞)        | Yes           | Newer, smoother, better for transformers                           |
| Softmax     | \[0, 1], sum=1 | Yes           | For multi-class classification                                     |
| Linear      | (-∞, ∞)        | No            | Used in output layer for regression                                |

### Layer-wise Usage

| Layer Type           | Common Activations     | Reason                                    |
| -------------------- | ---------------------- | ----------------------------------------- |
| Hidden layers        | ReLU, Leaky ReLU, GELU | Fast training, prevent vanishing gradient |
| Output (Binary)      | Sigmoid                | Gives probability \[0, 1]                 |
| Output (Multi-class) | Softmax                | Converts to probability distribution      |
| Output (Regression)  | None / Linear          | Continuous output                         |

### Activation Functions Summary Table

| **Activation Function**              | **Formula**                                                                                   | **Derivative**                                                                               | **Advantages**                                                           | **Limitations**                                                 |
| ------------------------------------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| **Sigmoid**                          | $\sigma(x) = \frac{1}{1 + e^{-x}}$                                                            | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$                                                      | Smooth and bounded; maps input to (0,1); good for binary classification  | Vanishing gradient; not zero-centered; slow convergence         |
| **Tanh**                             | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$                                                | $\tanh'(x) = 1 - \tanh^2(x)$                                                                 | Zero-centered; stronger gradients than sigmoid                           | Vanishing gradient for large inputs                             |
| **ReLU**<br>(Rectified Linear Unit)  | $f(x) = \max(0, x)$                                                                           | $f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$             | Sparse activation; computationally efficient; avoids vanishing gradients | Dying ReLU problem (neurons output zero and stop learning)      |
| **Leaky ReLU**                       | $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$        | $f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{otherwise} \end{cases}$        | Fixes dying ReLU; allows small gradient for $x < 0$                      | α needs to be manually set; may still suffer from dying neurons |
| **ELU**<br>(Exponential Linear Unit) | $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}$ | $f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ f(x) + \alpha & \text{otherwise} \end{cases}$ | Smooth; avoids dying ReLU; better learning                               | Slightly more computationally expensive                         |
| **Swish**                            | $f(x) = x \cdot \sigma(x)$                                                                    | $f'(x) = \sigma(x) + x \cdot \sigma'(x)$                                                     | Smooth and non-monotonic; often outperforms ReLU                         | Higher computation cost                                         |
| **Softplus**                         | $f(x) = \ln(1 + e^x)$                                                                         | $f'(x) = \frac{1}{1 + e^{-x}} = \sigma(x)$                                                   | Smooth approximation to ReLU                                             | Computationally more expensive; not sparse                      |
| **Softmax**                          | $f_i(x) = \frac{e^{x_i}}{\sum_j e^{x_j}}$                                                     | $f_i'(x) = f_i(x)(1 - f_i(x))$ (diag),<br> $-f_i(x)f_j(x)$ (off-diag)                        | Used for multiclass classification; probabilistic interpretation         | Only used in output layer; not for hidden layers                |

### 📝 Notes:

* **ReLU family (ReLU, Leaky ReLU, ELU, Swish)** is generally preferred in deep networks due to their **non-saturating gradients**.
* **Softmax** is not used as a hidden layer activation—only for the output layer in classification tasks.
* **Swish** (developed by Google) is trainable and often gives better performance in deep networks.
* **Vanishing Gradient Problem** affects functions like **Sigmoid** and **Tanh**, causing slow learning or stagnation.
