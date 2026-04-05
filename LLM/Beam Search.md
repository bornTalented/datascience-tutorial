## 🧠 Notes: Beam Search in Language Models

### 1. What is Beam Search?

* **Beam Search** is a **deterministic search algorithm** used by language models during text generation.
* Instead of selecting **one token at a time**, the model:
	* Explores **multiple possible token sequences simultaneously**.
	* Keeps track of the **most probable sequences**.
* This allows the model to evaluate the **probability of the entire sequence**, not just the next token.

### 2. Key Idea

* At each step:
	* The model generates several candidate continuations.
	* It keeps only the **top “k” best sequences** (called the **beam width**).
* This helps produce **more coherent and likely text outputs**.

---

# ⚙️ Beam Search Control Parameters

## 1. Repetition Penalties

These reduce the chances of generating the same words repeatedly.

### 🔹 Frequency Penalty

* Increases the penalty **each time a word or phrase appears** in generated text.
* The **more frequently a token appears**, the **less likely it will appear again**.

**Purpose:**
Prevents excessive repetition.

---

### 🔹 Presence Penalty

* Applied **whenever a token appears at least once**.
* Encourages the model to **introduce new words**.

**Purpose:**
Promotes **diversity in generated text**.

---

## 2. Length Penalties

These control how long the generated text should be.

### 🔹 Minimum Length Penalty

* Forces the model to generate **at least a specified number of tokens**.

**Purpose:**
Prevents responses from being **too short**.

---

### 🔹 Maximum Length Penalty

* Limits the generation to **a maximum number of tokens**.

**Purpose:**
Prevents **overly long outputs**.

---

### 🔹 Length Normalization Penalty

* Adjusts sequence scores by **dividing them by a function of sequence length**.

**Purpose:**
Prevents beam search from **favoring shorter sequences**.

## 3. **Beam Search Width**

* **Beam width** is the **number of parallel sequences considered during generation**.
* A **larger beam width** means:
	* More possible sequences are explored.
	* Potentially **better results but higher computation cost**.

---

✅ **Quick Summary**

* **Beam Search:** Finds the most likely sequence by exploring multiple possibilities.
* **Repetition Penalties:** Reduce repeated tokens.
	* Frequency Penalty
	* Presence Penalty
* **Length Penalties:** Control sequence length.
	* Minimum Length
	* Maximum Length
	* Length Normalization
* **Beam Width:** Number of parallel sequences explored.
