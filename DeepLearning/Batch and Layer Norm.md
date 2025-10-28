## Batch Normalization
Batch Normalization (Batch Norm) is a technique used in deep neural networks to improve the training process, leading to faster convergence, better stability, and often improved performance.

Here's a breakdown of the concept:

### The Problem it Addresses: Internal Covariate Shift

Imagine a deep neural network with many layers. During training, as the parameters (weights and biases) of the earlier layers are updated, the distribution of activations (outputs) from those layers changes. This means that the input distribution to the subsequent layers is constantly shifting. This phenomenon is called **internal covariate shift**.

Why is this a problem?
* **Slower Training:** Each layer has to constantly adapt to new input distributions from the previous layer, making the training process slower and requiring lower learning rates.
* **Vanishing/Exploding Gradients:** In deep networks, this shifting distribution can push activations into saturation regions of activation functions (like sigmoid or tanh), where gradients become very small (vanishing gradients), or cause gradients to become extremely large (exploding gradients), both of which hinder learning.
* **Sensitivity to Initialization:** Networks without batch normalization are often very sensitive to the initial values of their weights, requiring careful initialization.

### How Batch Normalization Works

Batch normalization addresses internal covariate shift by normalizing the activations of each layer within a mini-batch during training. It typically operates on the output of a linear transformation (e.g., matrix multiplication in a fully connected layer or convolution in a convolutional layer) and *before* the non-linear activation function.

**During Training:**

Here's the process for a given layer and a mini-batch:

1.  **Calculate Mini-Batch Statistics:** For each feature (or dimension) in the mini-batch, the mean ($\mu_B$) and variance ($\sigma_B^2$) are calculated.
    * Mean: $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$
    * Variance: $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_B)^2$
    where $m$ is the mini-batch size and $x_i$ is the activation of the $i^{th}$ example in the mini-batch for a given feature.

2.  **Normalize Activations:** Each activation ($x^{(i)}$) in the mini-batch is then normalized using the calculated mean and variance:
    * $x_{norm}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
    where $\epsilon$ is a small constant added for numerical stability (to prevent division by zero). This step ensures that the normalized activations have a mean of approximately zero and a standard deviation of approximately one.

3.  **Scale and Shift (Learnable Parameters):** To allow the network to retain its expressive power and potentially "undo" the normalization if it finds it optimal, two learnable parameters are introduced:
    * **Gamma ($\gamma$):** A scaling factor.
    * **Beta ($\beta$):** A shifting factor (offset).
    The final output of the batch normalization layer for each activation is:
    * $\tilde{x}^{(i)} = \gamma x_{norm}^{(i)} + \beta$

    These $\gamma$ and $\beta$ parameters are learned during training through backpropagation, just like other network weights. This allows the model to learn the optimal scale and shift for the activations, even if it deviates from a mean of zero and unit variance.

4. Crucially, during training, the Batch Normalization layer also keeps track of a **running average** (also known as a moving average) of the mean and variance across all mini-batches seen so far. These are often called "moving mean" and "moving variance."
   Update **running statistics** using exponential moving average:
   $$
   \mu_{\text{running}} \leftarrow (1 - \alpha)\mu_{\text{running}} + \alpha \mu_B
   $$

   $$
   \sigma^2_{\text{running}} \leftarrow (1 - \alpha)\sigma^2_{\text{running}} + \alpha \sigma_B^2
   $$
The momentum hyperparameter $\alpha$ controls how much the new mini-batch statistics influence the running averages. A common value is 0.9 or 0.99.

**During Inference (Testing):**
The core problem during inference is that you might be processing a single example at a time, or a batch that is much smaller than your training batch, or a batch with statistics that are not representative of the overall data distribution.
During training, the mean and variance are calculated per mini-batch. However, during inference, we don't have mini-batches in the same way, and we need a deterministic output. So, instead of mini-batch statistics, the **population mean and variance** (or moving averages of the means and variances collected during training) are used to normalize the activations.

Here's how it works:

1.  **Saved Global Statistics:** Throughout the training process, the Batch Normalization layer maintains and updates a "moving average" of the means and variances of each feature. This is typically an exponentially weighted moving average, where more recent mini-batch statistics have a slightly higher influence.

2.  **Normalization with Global Statistics:** When the model is in inference (evaluation/prediction) mode, the Batch Normalization layer uses these saved `running_mean` and `running_variance` values (which represent an approximation of the population statistics of the entire training dataset) to normalize the incoming data.
    * For an input $x_i$ during inference, the normalization becomes:
        $$
        \hat{x}_i = \frac{x_i - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}
        $$
        $$
        y_i = \gamma \hat{x}_i + \beta
        $$

The learnable parameters $\gamma$ and $\beta$ are also "frozen" after training and used as they were learned.

**Why this approach?**

* **Consistency:** Using the fixed global statistics ensures that the output of the Batch Normalization layer is deterministic and consistent regardless of the batch size or the specific examples in the current inference batch.
* **Generalization:** The moving averages provide a more robust estimate of the overall data distribution than any single mini-batch, leading to better generalization on unseen data.
* **Practicality:** It allows you to run inference on single samples or small batches without issues, which is common in real-world deployment.

In essence, during training, Batch Norm is dynamic and adapts to each mini-batch. During inference, it becomes static, relying on the learned global statistics to apply a consistent normalization transformation.
### Benefits of Batch Normalization:

* **Accelerated Training:** By stabilizing the input distribution to each layer, batch normalization allows for much higher learning rates, significantly speeding up the training process and requiring fewer epochs to converge.
* **Improved Stability:** It reduces internal covariate shift, making the training process more stable and less sensitive to the choice of initial weights.
* **Acts as a Regularizer:** Batch normalization introduces a slight amount of noise to the activations due to the mini-batch statistics. This acts as a form of regularization, similar to dropout, which can help prevent overfitting and improve generalization performance. In some cases, it can even reduce the need for other regularization techniques like dropout.
* **Allows Deeper Networks:** By alleviating vanishing/exploding gradients, batch normalization makes it feasible to train much deeper neural networks than would otherwise be possible.
* **More Viable Activation Functions:** It helps keep activations in a healthy range, making a wider variety of activation functions, including those prone to vanishing gradients (like sigmoid), more effective in deep networks.
* **Smoother Loss Landscape:** By normalizing activations, batch normalization can smooth the optimization landscape, making it easier for gradient descent to find the optimal solution.

| Challenge                       | How BatchNorm Helps                         |
| ------------------------------- | ------------------------------------------- |
| Internal covariate shift        | Normalizes layer inputs                     |
| Slow convergence                | Stabilizes distribution, speeds up training |
| Vanishing/exploding gradients   | Ensures better gradient flow                |
| Overfitting                     | Acts as regularizer                         |
| Need for careful initialization | Makes training more forgiving               |
In essence, Batch Normalization acts as an adaptive pre-processing step at each layer, continuously adjusting the input distributions to keep them stable and within a good range for effective learning. This simple yet powerful technique has become a staple in modern deep learning architectures.

---

# Layer Normalization

**Layer Normalization (LayerNorm)** was introduced to solve some limitations of **Batch Normalization (BatchNorm)**—especially in architectures like **RNNs**, **Transformers**, and cases where **batch sizes are small or variable**.

Layer Normalization (LayerNorm) is another powerful technique used in deep neural networks to normalize activations and stabilize training, much like Batch Normalization. However, it takes a different approach to calculating the normalization statistics.
### The Core Idea: Normalizing Across Features for Each Sample

Unlike Batch Normalization, which computes mean and variance across the batch dimension for each feature, **Layer Normalization computes the mean and variance across all features for a single training example within a layer.**

Imagine a tensor representing the output of a layer. If the dimensions are (Batch Size, Sequence Length, Features) for an NLP task, or (Batch Size, Channels, Height, Width) for an image task, here's how LayerNorm operates:

* **Batch Normalization:** Normalizes *each feature channel* independently across the *entire batch*. So, for a given feature (e.g., the first feature in a fully connected layer, or the first channel in a convolutional layer), it calculates the mean and variance over all examples in the mini-batch.
* **Layer Normalization:** Normalizes *each individual sample* independently across *all its features within that layer*. For a single example, it calculates the mean and variance of all the feature activations in that layer.

### How Layer Normalization Works (for a single training example)

Let's say the output of a layer for a single input example is a vector $x = [x_1, x_2, \ldots, x_H]$, where $H$ is the number of features (or hidden units) in that layer.

1.  **Calculate Mean ($\mu$) and Variance ($\sigma^2$) for the current example:**
    * Mean: $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$
    * Variance: $\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$
    Here, the summation is over all features $H$ *for that specific input example*.

2.  **Normalize Activations:**
    * $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
    Again, $\epsilon$ is a small constant for numerical stability. This step ensures that the activations for *this single example* have a mean of approximately zero and a standard deviation of approximately one across its features.

3.  **Scale and Shift (Learnable Parameters):**
    * $y_i = \gamma \hat{x}_i + \beta$
    Similar to Batch Norm, $\gamma$ (gain/scaling factor) and $\beta$ (bias/shifting factor) are learnable parameters unique to each layer. These allow the network to learn the optimal scale and shift for the normalized activations, preserving the network's expressive power.

**Key Difference from Batch Normalization:**
The crucial distinction is the **scope of normalization**. LayerNorm's statistics ($\mu$ and $\sigma^2$) are computed *per sample*, independent of other samples in the mini-batch. This means:

* **Training and Inference are Identical:** Since the normalization depends only on the current input sample, the computation is exactly the same during training and inference. There's no need to store moving averages of statistics, unlike Batch Normalization.
* **Insensitivity to Batch Size:** This is LayerNorm's biggest advantage. Because it normalizes per sample, it works equally well with any batch size, including batch size 1. This is particularly beneficial for:
    * **Recurrent Neural Networks (RNNs):** Where sequence lengths can vary, making mini-batch statistics inconsistent.
    * **Generative Models:** Where training might involve very small batch sizes.
    * **Models with Variable Input Sizes:** Where batching strategies might be complex.

### Benefits of Layer Normalization:

* **Robust to Batch Size Variation:** This is the primary strength. It performs consistently regardless of whether you have large or small batches.
* **Suitable for RNNs and Transformers:** Due to its batch-size independence, LayerNorm is widely adopted in sequence models like RNNs (especially LSTMs and GRUs) and the Transformer architecture, where it plays a critical role in stabilizing training and improving performance.
* **Consistent Training and Inference:** No special handling or global statistics are needed for inference, simplifying deployment.
* **Stabilizes Hidden State Dynamics:** Particularly in RNNs, LayerNorm helps to prevent exploding or vanishing gradients by keeping hidden state activations within a stable range.
* **Faster Convergence:** Similar to Batch Norm, by stabilizing activations, LayerNorm can allow for higher learning rates and faster convergence during training.
* **Reduced Sensitivity to Initialization:** It makes the network less dependent on carefully chosen initial weights.

| Problem Solved                  | LayerNorm Benefit            |
| ------------------------------- | ---------------------------- |
| Variable/small batch sizes      | Independent of batch size    |
| Sequence modeling (RNNs)        | Preserves time structure     |
| Inconsistent training/inference | No dependency on batch stats |
| Transformers                    | Token-wise normalization     |
In summary, Layer Normalization provides a powerful and flexible way to normalize activations within neural networks, especially in architectures where batch-dependent normalization is problematic. Its per-sample normalization makes it robust to varying batch sizes and highly effective for sequential data and complex models like Transformers.