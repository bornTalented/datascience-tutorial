Inference parameters in Large Language Models (LLMs) control how the model generates text when given a prompt. These parameters help balance **creativity, coherence, determinism, and diversity** in the output. Below are the most common inference parameters:
### 1. Temperature

_Temperature (T)_ is a hyperparameter that controls the randomness of token selection by scaling the logits before they’re passed through the softmax function.

* **Purpose**: controls the randomness of a language model's output
* **Range**: `0.0` to `1.0+`
* **Working Principle**:
	* A lower temperature makes the model more **deterministic** (picks the most likely next token).
	* A higher temperature adds **creativity and diversity** by sampling more from lower-probability tokens.
* **Example**:
	  Prompt: *"The knight entered the castle and found..."*
	
	* `temperature=0.2`: "a hidden treasure guarded by soldiers."
	* `temperature=1.0`: "a portal to a parallel universe of talking dragons."

It affects how **creative** or **deterministic** the responses are.

- **Lower values (0.0 – 0.3)** → More **deterministic** and predictable.  
- **Higher values (0.7 – 1.5)** → More **random**, creative, and diverse.


| Use Case                                    | Recommended Temperature |
|---------------------------------------------|--------------------------|
| Factual answers (math, code, facts)         | 0.0 – 0.3               |
| Balanced response (general QA, explanations)| 0.5 – 0.7               |
| Creative writing, storytelling, jokes       | 0.9 – 1.2               |
| Maximum randomness (wild ideas, brainstorming) | 1.5+                  |

### 2. Top-k Sampling

_Top-K sampling_ restricts token selection to only the k most likely tokens from the vocabulary, effectively truncating the long tail of the probability distribution.

* **Purpose**: Limits choices to the top **k** most likely tokens.
* **Range**: Integer (`k ≥ 1`)
* **Working Principle**:
	* The model samples the next word from the **top k** tokens based on their probability distribution.
	* If `k=1`, it’s greedy decoding.
* **Example**:
	  Prompt: *"The wizard cast a spell and..."*
	
	* `top_k=5`: Selects from top 5 tokens (e.g., "vanished", "glowed", "appeared", "shouted", "smiled")

### 3. Top-p (Nucleus) Sampling

_Nucleus sampling_ dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold p. Hence, it’s also called top-P sampling.

* **Purpose**: Samples from the **smallest possible set** of tokens whose cumulative probability exceeds **p**.
* **Range**: Float between `0.0` and `1.0`
* **Working Principle**:
	* Dynamically includes tokens **until** their cumulative probability ≥ `p`.
	* Balances **quality** and **diversity** better than top-k.
* **Example**:
	  Prompt: *"The AI robot said..."*
	
	* `top_p=0.9`: Selects from a dynamic subset of tokens where cumulative probability is 90%.

Nucleus sampling adapts to the model’s confidence at each step. When the model is very confident, few tokens are considered (as with low top-K), and when the model is uncertain, more tokens are considered (as with high top- K). Therefore, it generally produces more natural text than fixed top-K sampling does.

---

### Max Tokens (or Max Length)

* **Purpose**: Limits the length of the generated output.
* **Range**: Integer
* **Working Principle**:
	* Caps the number of tokens the model generates in a response.
* **Example**:
	  Prompt: *"Write a short poem."*
	
	* `max_tokens=10`: "Roses bloom in springtime air, sun"

---

### Repetition Penalty

* **Purpose**: Penalizes repeating the same phrases or tokens.
* **Range**: Typically `1.0` to `2.0`
* **Working Principle**:
	* If a token is repeated, its probability is reduced by multiplying with `1 / repetition_penalty`.
* **Example**:
	  Prompt: *"He walked and walked and walked..."*
	
	* `repetition_penalty=1.2`: Reduces the chance of repeated "walked".


#### Presence Penalty vs. Frequency Penalty (used in OpenAI's models)

| Parameter             | Purpose                                 | Working Principle                                          |
| --------------------- | --------------------------------------- | ---------------------------------------------------------- |
| **Presence Penalty**  | Penalizes **whether** a token appears   | Reduces the chance of repeating any already-used tokens    |
| **Frequency Penalty** | Penalizes **how often** a token appears | Reduces likelihood of over-using frequently repeated words |

* **Range**: Float (commonly `0.0` to `2.0`)
* **Example**:
  Prompt: *"Tell me a joke about cats."*

  * With presence penalty: The model tries **not to reuse previous ideas**.
  * With frequency penalty: The model avoids **repeating the word "cat"** too many times.

---

### Stop Sequences

* **Purpose**: Instructs the model to stop generation upon encountering specific tokens.
* **Working Principle**:
	* If the generated output matches a stop sequence (like `"\n\n"`), generation halts immediately.
* **Example**:
	  Prompt: *"Answer the question below:\nQ: What is the capital of France?\nA:"*
	
	* `stop=["\n"]`: Ensures the model ends after the answer.

---

##  Putting It All Together (Example)

```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a fantasy story intro."}],
    temperature=0.8,
    top_p=0.9,
    max_tokens=100,
    frequency_penalty=0.5,
    presence_penalty=0.6,
    stop=["THE END"]
)
```

This setup will:

* Generate a creative and varied story intro,
* Limit length to 100 tokens,
* Avoid repeating words or ideas,
* And stop if "THE END" appears.

---
#### Let's walk through **step-by-step examples** of how **Top-k** and **Top-p (nucleus) sampling** work during inference in a Large Language Model.
---

### 🎯 Scenario:

Prompt: `"The cat sat on the"`

The model has computed the following **probabilities** (simplified) for the next word:

| Token    | Probability |
| -------- | ----------- |
| mat      | 0.30        |
| couch    | 0.20        |
| floor    | 0.15        |
| table    | 0.10        |
| rug      | 0.08        |
| bed      | 0.07        |
| chair    | 0.05        |
| laptop   | 0.03        |
| window   | 0.01        |
| keyboard | 0.01        |

These probabilities sum to **1.00**, as required.

---

### Step-by-Step: **Top-k Sampling (e.g., k = 3)**

#### Parameter: `top_k = 3`

1. **Sort tokens by probability (done above).**
2. **Select top 3 tokens:**
	* mat (0.30)
	* couch (0.20)
	* floor (0.15)

3. **Normalize the selected probabilities:**

| Token | Original P | Normalized P         |
| ----- | ---------- | -------------------- |
| mat   | 0.30       | 0.30 / 0.65 ≈ 0.4615 |
| couch | 0.20       | 0.20 / 0.65 ≈ 0.3077 |
| floor | 0.15       | 0.15 / 0.65 ≈ 0.2308 |

4. **Sample from this reduced distribution.**

*Outcome*: The next word is chosen randomly from {mat, couch, floor}, with higher preference for “mat.”

---

### Step-by-Step: **Top-p (Nucleus) Sampling (e.g., p = 0.8)**

#### Parameter: `top_p = 0.8`

1. **Sort tokens by probability (already sorted).**
2. **Cumulatively add probabilities until they exceed `p = 0.8`:**

| Token | P    | Cumulative P |
| ----- | ---- | ------------ |
| mat   | 0.30 | 0.30         |
| couch | 0.20 | 0.50         |
| floor | 0.15 | 0.65         |
| table | 0.10 | 0.75         |
| rug   | 0.08 | 0.83 ✅       |

➡ Stop at rug. The cumulative probability now exceeds 0.8.

3. **Selected tokens:**
	* {mat, couch, floor, table, rug}

4. **Normalize the probabilities:**

| Token | P    | Normalized P         |
| ----- | ---- | -------------------- |
| mat   | 0.30 | 0.30 / 0.83 ≈ 0.3614 |
| couch | 0.20 | 0.20 / 0.83 ≈ 0.2410 |
| floor | 0.15 | 0.15 / 0.83 ≈ 0.1807 |
| table | 0.10 | 0.10 / 0.83 ≈ 0.1205 |
| rug   | 0.08 | 0.08 / 0.83 ≈ 0.0964 |

5. **Sample from this dynamically chosen subset.**

*Outcome*: The next word is sampled from {mat, couch, floor, table, rug}.

---

### Key Differences

| Feature          | Top-k (k=3)                                     | Top-p (p=0.8)           |
| ---------------- | ----------------------------------------------- | ----------------------- |
| Fixed or Dynamic | Fixed `k` tokens                                | Dynamic token count     |
| Diversity        | May exclude plausible options if not in top `k` | Includes more variation |
| Control          | Simple and deterministic                        | Adaptive and nuanced    |

---

Great — let’s build a **deep, intuitive + technical understanding** of the *temperature parameter* step by step.

---

# 🔥 What Temperature *Really* Does

At its core, temperature controls **how confident vs. exploratory** the model is when choosing the next token.

The model always produces probabilities like:

```
"cat" → 0.6  
"dog" → 0.3  
"car" → 0.1
```

Temperature **reshapes this distribution before sampling**.

---

# 🧠 The Math (Core Insight)

Temperature modifies probabilities using this formula:

[
P_i' = \frac{P_i^{1/T}}{\sum_j P_j^{1/T}}
]

Where:

* (T) = temperature
* (P_i) = original probability
* (P_i') = adjusted probability

---

# 📊 What Happens at Different Temperatures

### ✅ 1. **T = 1 (Default)**

* No change
* Output follows the model’s natural probabilities

---

### ❄️ 2. **T < 1 (e.g., 0.2, 0.5)** → Sharper distribution

* High probabilities get **even higher**
* Low probabilities get **suppressed**

Example:

```
Before:   cat 0.6, dog 0.3, car 0.1
After:    cat ~0.85, dog ~0.14, car ~0.01
```

👉 Result:

* More **predictable**
* More **repetitive**
* Less creative

---

### 🔥 3. **T > 1 (e.g., 1.2, 1.5)** → Flatter distribution

* Differences between probabilities shrink

Example:

```
Before:   cat 0.6, dog 0.3, car 0.1
After:    cat ~0.45, dog ~0.35, car ~0.20
```

👉 Result:

* More **diverse**
* More **creative**
* Higher chance of **errors or nonsense**

---

### 🚨 4. **Extreme Cases**

#### T → 0

* Becomes **greedy decoding**
* Always picks the highest-probability token
* Fully deterministic

#### T → ∞

* Becomes **uniform random**
* All tokens equally likely (pure chaos)

---

# 🎯 Intuition (Very Important)

Think of temperature as:

👉 **Confidence vs Curiosity knob**

| Temperature      | Behavior                             |
| ---------------- | ------------------------------------ |
| Low (0–0.3)      | “I’ll stick to what I’m sure about.” |
| Medium (0.7–1.0) | “Balanced thinking.”                 |
| High (1.2+)      | “Let’s explore weird possibilities.” |

---

# 🔄 Temperature vs Other Sampling Methods

Temperature is often combined with:

### 🔹 Top-k sampling

* Only choose from top *k* tokens

### 🔹 Top-p (nucleus sampling)

* Choose smallest set of tokens whose probability sum ≥ p

👉 Key point:

* **Temperature reshapes probabilities**
* **Top-k / top-p restrict choices**

---

# 🧪 Real-World Effects

### Low temperature (0.2)

* Factual answers
* Code generation
* Math
* Repetitive phrasing

### Medium (0.7)

* General conversation
* Balanced responses

### High (1.0+)

* Storytelling
* Brainstorming
* Creative writing

---

# ⚠️ Common Misconceptions

### ❌ “Temperature increases randomness directly”

Not exactly — it **reshapes probabilities**, then sampling introduces randomness.

---

### ❌ “Higher temperature = better creativity”

Not always:

* Too high → incoherent output
* Sweet spot matters

---

### ❌ “Temperature works with beam search”

* Beam search is **deterministic**
* Temperature is mainly for **sampling-based decoding**

---

# 🧩 Visual Intuition

Imagine probabilities as a mountain:

* **Low temperature** → tall sharp peak 🏔️
* **High temperature** → flattened hill 🌄

---

# 🧠 One-Line Summary

👉 Temperature controls **how strongly the model prefers high-probability tokens over low-probability ones**.

---

If you want, I can:

* Show a **step-by-step numeric example with real logits**
* Explain how temperature is applied to **logits (before softmax)** (more precise than probabilities)
* Or compare **temperature vs top-p vs beam search in one unified framework**
