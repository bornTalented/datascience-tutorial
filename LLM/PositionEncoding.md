### Why Do We Need Positional Encoding?

Transformers **do not have recurrence** (like RNNs) or **convolutions** (like CNNs), which traditionally help capture the order of words in a sequence.

But **word order is critical** in natural language. For example:

* "The cat sat on the mat" ≠ "The mat sat on the cat"

To give the model a sense of the **position of each token** in a sequence, we inject **positional information** into the input embeddings.

---

### How Is Positional Encoding Implemented?

The original **Vaswani et al. (2017)** Transformer paper introduced **sinusoidal positional encodings**, which are **added to** the input embeddings before feeding them into the encoder.

Let’s denote:
* $\text{pos}$: position in the sequence (e.g., 0, 1, 2, …)
* $i$: dimension index of the embedding vector (e.g., 0 to $d_{\text{model}} - 1$)

#### Mathematical Formulas
$$
PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$
$$
PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

This means:
* Even dimensions use sine.
* Odd dimensions use cosine.
* Different frequencies (scales) are used for each dimension.

#### Example with $d_{\text{model}} = 4$

For position `pos = 1`:

* $PE(1, 0) = \sin(1 / 10000^{0/4}) = \sin(1)$
* $PE(1, 1) = \cos(1 / 10000^{0/4}) = \cos(1)$
* $PE(1, 2) = \sin(1 / 10000^{2/4}) = \sin(1 / 100)$
* $PE(1, 3) = \cos(1 / 10000^{2/4}) = \cos(1 / 100)$

So, each position gets a unique vector of values that can be added to the word embeddings.

---

### Properties of Sinusoidal Encoding

1. **Continuous and Differentiable** – helps with optimization.
2. **Fixed (non-learnable)** – no additional parameters.
3. **Relative Position Representability** – dot products between positional encodings for nearby positions encode relative distances.

---

### Visualization

Each dimension is a different wave with increasing wavelength:

```python
Position → 
Dim 0:   ~~~~~~~~ (high frequency)
Dim 1:   ~~~~~~~~ (high frequency)
Dim 2:   --------~~~~~~-------- (lower frequency)
Dim 3:   --------~~~~~~-------- (lower frequency)
```

Together, these patterns uniquely encode each position.

---

### Learnable vs Sinusoidal Encoding

While the original Transformer uses **fixed sinusoidal encoding**, many modern models (e.g., BERT, GPT) use **learnable positional embeddings**:

```python
self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
```

* These are trained along with the model.
* More flexible, but require more parameters.

---

### Summary

| Feature    | Sinusoidal Positional Encoding      |
| ---------- | ----------------------------------- |
| Type       | Fixed                               |
| Uses       | Original Transformer                |
| Purpose    | Inject position info                |
| Formula    | Uses sin/cos of varying frequencies |
| Learnable? | No                                  |
| Advantage  | Generalizes to longer sequences     |
| Limitation | Not task-adaptive                   |

---

Let’s **decode the positional encoding formula** more deeply and break down **why it uses 10000** and the particular **normalization scheme**. This will involve a bit of math intuition and signal processing insight.

---

### Recap of the Positional Encoding Formula

$$
PE_{\text{(pos, 2i)}} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

$$
PE_{\text{(pos, 2i+1)}} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

Where:

* $\text{pos}$ is the **position** in the sequence.
* $i$ is the **dimension** index.
* $d_{\text{model}}$ is the **embedding size** (like 512 in the original paper).
* The denominator is an **exponential scale** of 10000 raised to $\frac{2i}{d_{\text{model}}}$.

---

###  Why Use This Form?

#### 1. Different Frequencies Across Dimensions

Each dimension in the positional encoding corresponds to a **different wavelength** (or frequency). This lets the model **encode both fine-grained and coarse-grained** positional patterns.

* For small $i$, the exponent $\frac{2i}{d_{\text{model}}}$ is close to 0 → **higher frequency** sine/cosine.
* For large $i$, the exponent approaches 1 → **lower frequency**.

This allows the Transformer to capture:

* **Local word order** (short wavelengths).
* **Global structure** (long wavelengths).

---

#### 2. 10000 as a Base: A Design Choice

The base number `10000` is arbitrary but **chosen for a wide spread of frequencies**.

Let's make it intuitive:

* The **frequency** of the sine wave is controlled by the denominator.
* $10000^{\frac{2i}{d}}$ varies **logarithmically** across dimensions.
* For $d = 512$, this means:

  * At $i = 0$: $10000^{0} = 1$
  * At $i = 256$: $10000^1 = 10000$
  * So, frequencies vary from $\frac{1}{1}$ to $\frac{1}{10000}$

This gives the model access to a **wide spectrum of position sensitivity**.

---

#### 3. Normalized Position Inputs

Why divide by $10000^{\frac{2i}{d}}$ rather than multiply?

Because we want:

* **positional encoding = sin(angular frequency × position)**
* The denominator lets the frequency **decrease exponentially** across dimensions:

  $$
  \text{angular frequency} = \frac{1}{10000^{\frac{2i}{d}}}
  $$

This is inspired by **Fourier features** in signal processing — encoding information in **waveforms** of various frequencies.

---

#### 4. Combining Sin & Cos: Phase Shift

Using both sine and cosine for each position:

* Allows the model to distinguish not just **frequency** but also **phase** (start of wave).
* Helps it **encode relative distances** because:

  $$
  \sin(a - b) = \sin(a)\cos(b) - \cos(a)\sin(b)
  $$
* This means the **dot product of two position vectors** reflects their **relative positions**.

---

### Summary: Why This Specific Formula?

| Component                         | Purpose                                                      |
| --------------------------------- | ------------------------------------------------------------ |
| $\frac{\text{pos}}{10000^{2i/d}}$ | Scales positions by dimension-dependent frequencies          |
| $10000$                           | Gives wide frequency range from 1 to 1/10000                 |
| $\sin/\cos$                       | Makes encoding periodic and differentiable                   |
| Even/Odd split                    | Encodes phase shift and supports relative position reasoning |
| Fixed & continuous                | Generalizes to sequence lengths unseen during training       |

---

### Bonus: Learnable Positional Embeddings?

Later models like **BERT** and **GPT** replaced this with **learnable position embeddings**, which:

* Lose the nice frequency spread and generalization,
* But offer more **task-specific adaptability**.

Still, sinusoidal encoding remains **elegant, interpretable**, and **theoretically grounded**.

---

