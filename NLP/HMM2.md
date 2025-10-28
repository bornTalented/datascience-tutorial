# 📚 Hidden Markov Models (HMM) – A Tutorial

## 🧠 1. Markov Chains and POS Tagging

### Markov Chain Basics:

* A **state** represents a condition (like a POS tag).
* We move from one state to another with certain **transition probabilities**.

Let:

* $Q = \{q_1, q_2, ..., q_n\}$: Set of possible states (e.g., POS tags)
* $A$: Transition matrix with $a_{ij} = P(q_j | q_i)$

---

## 🤖 2. Hidden Markov Model (HMM)

### 🎯 Goal:

Given a sequence of **observed words**, find the most probable **sequence of hidden states (POS tags)**.

### Definition of HMM:

An HMM is a **5-tuple**:

$$
(S, V, \pi, A, B)
$$

* $S$: Hidden states (POS tags)
* $V$: Vocabulary (observed words)
* $\pi$: Initial state probabilities
* $A$: Transition matrix $a_{ij} = P(s_j | s_i)$
* $B$: Emission matrix $b_{ik} = P(v_k | s_i)$

### Assumptions:

1. **Markov Assumption**:

   $$
   P(y_t | y_{t-1}, ..., y_1) = P(y_t | y_{t-1})
   $$

2. **Output Independence**:

   $$
   P(x_t | y_t, ..., y_1, x_{t-1}, ..., x_1) = P(x_t | y_t)
   $$

---

### ⛓️ Joint Probability:

$$
P(X, Y) = P(Y) \cdot P(X | Y) = \prod_{t=1}^T P(y_t | y_{t-1}) \cdot P(x_t | y_t)
$$

Where:

* $X = x_1, ..., x_T$ (observed sequence)
* $Y = y_1, ..., y_T$ (hidden state sequence)

---

### 🔍 Prediction: Find Most Probable Tag Sequence

$$
\arg\max_Y P(Y|X) = \arg\max_Y P(X, Y)
$$

This is typically solved using the **Viterbi Algorithm**.

---

## 📊 3. Building Transition and Emission Matrices

### 1. Transition Probabilities:

$$
P(t_i | t_{i-1}) = \frac{\text{count}(t_{i-1}, t_i)}{\sum_{t'} \text{count}(t_{i-1}, t')}
$$

If you encounter zero counts, apply **smoothing** to avoid zero probability.

---

### 2. Emission Probabilities:

$$
P(w_i | t_i) = \frac{\text{count}(t_i, w_i) + \epsilon}{\sum_{w_j} \text{count}(t_i, w_j) + V \cdot \epsilon}
$$

Note: Don't smooth the **initial state** distribution.

---

## 🧩 4. Conditional vs Generative Models

### Conditional Model:

Estimate $P(Y|X)$ directly.

### Generative Model (e.g., HMM):

Estimate $P(X, Y) = P(Y) \cdot P(X|Y)$, then infer $P(Y|X)$.

This is often preferred in unsupervised or semi-supervised learning.

---