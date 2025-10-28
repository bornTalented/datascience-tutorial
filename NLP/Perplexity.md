
## 📌 What is Perplexity?

**Perplexity**  measures how well a language model predicts a sequence of words. It is a standard metric used to evaluate the quality of language models. Intuitively, it measures **how "surprised" a model is by the test data**.

Mathematically, if a model assigns higher probabilities to the actual words in a sentence, it will have **lower perplexity**, meaning better performance.

---

### **Step 1: Normalizing Sentence Probability using the Geometric Mean**

#### **Step 1: Compute the Probability of the Sentence**

Given a sequence of $N$ words $W = (w_1, w_2, ..., w_N)$,  the probability of the entire sentence (joint probability) according to the language model is:

$$
P(W) = P(w_1, w_2, \dots, w_N) = \prod_{i=1}^{N} P(w_i | w_1, \dots, w_{i-1})
$$

However, this probability decreases exponentially as $N$ grows, making direct comparison difficult.

#### **Step 2: Normalize Using the Geometric Mean**

Since probabilities tend to be very small values (especially for long sentences), directly using this probability can be impractical. Instead, we **normalize it using the geometric mean**, which ensures that the probability is comparable across different sentence lengths:

$$
\text{Geometric Mean (Probability)} = P(W)^{\frac{1}{N}} = \left( \prod_{i=1}^{N} P(w_i | w_1, \dots, w_{i-1}) \right)^{\frac{1}{N}}
$$

Taking the logarithm (to avoid numerical underflow and make calculations easier):

$$
\log \text{Geometric Mean (Probability)} = \frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \dots, w_{i-1})
$$
$$
GM(\text{Probability}) = e^{\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})}
$$

The geometric mean ensures that we **normalize** the probability, preventing longer sentences from automatically having lower probabilities.

> [!NOTE]
> The formula for the geometric mean of a set of $n$ numbers is:
> 
> $$
> \text{Geometric Mean} = \left( \prod_{i=1}^{n} x_i \right)^{\frac{1}{n}}
> $$
> 
> Where:
> 
> * $x_1, x_2, \dots, x_n$ are the $n$ numbers in the set.
> * $\prod_{i=1}^{n} x_i$ represents the product of all the numbers in the set.
> * $n$ is the total number of values in the dataset.
> 
> In simpler terms, the geometric mean is the $n$th root of the product of all the values.
> 
> For example, if you wanted to find the geometric mean of 3, 6, and 9:
> 
> $$
> \text{Geometric Mean} = (3 \times 6 \times 9)^{\frac{1}{3}} = 162^{\frac{1}{3}} \approx 5.428
> $$
> 
> The geometric mean is often used when comparing things like growth rates or financial returns over time.


---

#### **Step 3: Compute Perplexity**

Perplexity is the **reciprocal** of the geometric mean probability:

$$
\text{Perplexity}(W) = PP(W) = \frac{1}{GM(W)}
$$

$$
\text{PP}(W) = \frac{1}{P(W)^{\frac{1}{N}}} = \left( P(W) \right)^{-\frac{1}{N}}
$$

or equivalently,

$$
\text{PP}(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, \dots, w_{i-1})}
$$

Using the natural logarithm:

$$
\text{PP}(W) = e^{-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \dots, w_{i-1})}
$$

This formula ensures that perplexity properly scales with sentence length and provides a comparable measure across different models.

---

### **Intuition Behind Perplexity**
- If a language model is perfect and always assigns a probability of 1 to the correct next word, the perplexity is 1.
- If the model is uncertain, it assigns lower probabilities to the correct words, increasing perplexity.
- A random model (uniform probability) has high perplexity.

For example:
- A perplexity of **10** means that on average, the model is as confused as if it had to choose among 10 equally probable words at each step.
- A lower perplexity (e.g., **5**) indicates the model is more confident in its predictions.
- A **higher** perplexity means the model struggles to predict the next word accurately.

---

### **Example Calculation**

Suppose we have a trigram language model, and it assigns the following probabilities to a 3-word sentence:

$$
P(w_1) = 0.2, \quad P(w_2 | w_1) = 0.3, \quad P(w_3 | w_1, w_2) = 0.4
$$

The joint probability of the sentence:

$$
P(W) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) = 0.2 \times 0.3 \times 0.4 = 0.024
$$

The geometric mean probability:

$$
GM(W) = P(W)^{\frac{1}{3}} = (0.024)^{\frac{1}{3}} \approx 0.29
$$

Perplexity:

$$
PP(W) = \frac{1}{0.29} \approx 3.45
$$

This means that, on average, the model is as uncertain as if it had to choose between **3.46 words** at each step.

---
## ⚠️ Important Observations

* If **any single probability is zero**, the **entire product becomes zero**, making perplexity **infinite**.
* To avoid this, we use **smoothing techniques**.

---

## 🧂 Add-One (Laplacian) Smoothing

One common method is **Add-One Smoothing** (also known as Laplacian Smoothing). It estimates conditional probabilities as:

$$
P(w_i | w_{i-(n-1)}, ..., w_{i-1}) = \frac{c(w_{i-(n-1)}, ..., w_i) + 1}{c(w_{i-(n-1)}, ..., w_{i-1}) + V}
$$

Where:

* $c(\cdot)$: count in training corpus
* $V$: number of unique unigrams in the training set **+1** (to include the `<end>` token)

---

## ❓ Why Add 1 to $V$?

Because:

* The `<end>` token is **explicitly predicted** by the model, so it counts as a vocabulary word.
* The `<start>` tokens are **not predicted** (they're context), so they're not included in $V$.

---

## 🔧 Example Task

We are given:

* **Train sentence**:
  *“This is the cat that killed the rat that ate the malt that lay in the house that Jack built.”*

* **Test sentence**:
  *“This is the house that Jack built.”*

### Step 1: Add Start and End Tokens

Since it’s a **trigram** model ($n = 3$), we add:

* $n-1 = 2$ start tokens: `<s1>`, `<s2>`
* 1 end token: `<end>`

So:

* **Train**:
  `<s1> <s2> This is the cat that killed the rat that ate the malt that lay in the house that Jack built <end>`

* **Test**:
  `<s1> <s2> This is the house that Jack built <end>`

---

### Step 2: Vocabulary Size

* Unique unigrams in the **train sentence** = 14
* So, $V = 14 + 1 = 15$

---

### Step 3: Test Sentence Length

* Number of words (including `<end>`) = 7
* So, $N = 7$
* This implies we need to compute **8** trigram probabilities

---

### Step 4: Compute Trigram Probabilities with Add-One Smoothing

Use the formula:

$$
P(w_i | w_{i-2}, w_{i-1}) = \frac{c(w_{i-2}, w_{i-1}, w_i) + 1}{c(w_{i-2}, w_{i-1}) + V}
$$

#### Observations:

* All bigrams in the test sentence exist in the training sentence → Denominator = $1 + 15 = 16$
* Only **one trigram** is unseen: `"is the house"` → Numerator = $0 + 1 = 1$, probability = $\frac{1}{16} = 0.0625$
* All other trigrams are seen once: Numerator = $1 + 1 = 2$, probability = $\frac{2}{16} = 0.125$

---

### Step 5: Compute Perplexity

There are 8 trigram probabilities:

$$
P = 0.0625 \times 0.125^7
$$

So, perplexity is:

$$
\text{Perplexity} = P^{-\frac{1}{N}} = (0.0625 \cdot 0.125^7)^{-1/7} \approx 11.89
$$

---

## 🧠 Key Takeaways

* Perplexity is an **evaluation metric**: lower = better.
* Smoothing is **critical** to avoid infinite perplexity.
* Add-one smoothing is **simple but effective**, though not always the best in practice.
* Efficient perplexity computation often relies on **reusing counts** and observing **shared contexts**.

---

### **Perplexity in Neural Language Models**
In deep learning models (e.g., Transformer-based models like GPT), perplexity is computed using the model’s predicted probability distribution over the vocabulary.

For a neural network that computes loss as the negative log-likelihood (NLL), the perplexity is:

$$
PP = e^{\text{Loss}}
$$

where **Loss** is the average negative log-likelihood.

---

### **Conclusion**
- Perplexity is a fundamental metric for evaluating language models.
- Lower perplexity means better predictions.
- It helps compare different models (e.g., bigram vs. neural models).
- However, perplexity alone is not always sufficient; human evaluation and downstream task performance are also essential.

