
# 📚 Language Modeling – A Tutorial

---

## 🔤 1. Language Modeling

### Goal:

To **predict probabilities of a sequence of words** $W = w_1, w_2, ..., w_k$

---

### 🔗 Chain Rule of Probability:

$$
P(w_1, w_2, ..., w_k) = P(w_1) P(w_2|w_1) P(w_3|w_1, w_2) \cdots P(w_k|w_1,...,w_{k-1})
$$

### ⚠ Problem:

Too many dependencies! This is computationally expensive and data-hungry.

---

### ✅ Markov Assumption:

We assume that the probability of a word depends only on the **previous n−1 words**:

$$
P(w_i|w_1, ..., w_{i-1}) \approx P(w_i | w_{i-(n-1)}, ..., w_{i-1})
$$

This leads to **n-gram models**.

---

## 🔢 2. N-Gram Language Models

### 🟦 Bi-gram Model (n=2):

$$
P(W) = \prod_{i=1}^{k+1} P(w_i | w_{i-1})
$$

### 🟨 Tri-gram Model (n=3):

$$
P(W) = \prod_{i=1}^{k+1} P(w_i | w_{i-2}, w_{i-1})
$$

### 🟩 General n-gram Model:

$$
P(W) = \prod_{i=1}^{k+1} P(w_i | w_{i-n+1}^{i-1})
$$

Note: This means we condition only on the **last (n−1) words**, not the full history.

---

### 🔧 Training an N-Gram Model:

Let $N$ be the total length (number of words) of the training corpus.

To **estimate parameters**, use **maximum likelihood estimation (MLE)**:

$$
P(w_i | w_{i-n+1}^{i-1}) = \frac{\text{count}(w_{i-n+1}^{i})}{\text{count}(w_{i-n+1}^{i-1})}
$$

---

### 🧪 Model Evaluation

#### 🔹 Likelihood:

$$
P(W_{\text{test}}) = \prod_{i=1}^{N+1} P(w_i | w_{i-n+1}^{i-1})
$$

#### 🔹 Log Likelihood:

$$
\log P(W) = \sum_{i=1}^{N+1} \log P(w_i | w_{i-n+1}^{i-1})
$$

---

### ❓ Which Language Model is Better?

Use **Perplexity**:

$$
\text{Perplexity}(W) = \left( P(W) \right)^{-1/N}
$$

A **lower perplexity** implies a better language model.

🔍 Intuition: A good LM assigns **high probability to real text** → low perplexity → less “perplexed.”

---

