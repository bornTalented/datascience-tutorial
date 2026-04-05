## **LLM Evaluation – Metrics**

### **ROUGE**

* Used for **text summarization**
* Compares a generated summary to one or more **reference (human) summaries**
* Focuses on **recall** (how much of the reference is captured)

---

### **BLEU Score**

* Used for **machine translation**
* Compares generated text to **human translations**
* Focuses more on **precision** (how much generated text is correct)

---

## **Example**

**Reference (human):**

> It is cold outside.

**Generated output:**

> It is very cold outside.

---

## **ROUGE-1 (Unigram matches)**

**Unigrams in Reference:**

* `It`, `is`, `cold`, `outside` → **4 unigrams**

**Unigrams in Output:**

* `It`, `is`, `very`, `cold`, `outside` → **5 unigrams**

**Matching unigrams:**

* `It`, `is`, `cold`, `outside` → **4 matches**

**ROUGE-1 Precision**
= matches / unigrams in output
= 4 / 5 = **0.8**

**ROUGE-1 Recall**
= matches / unigrams in reference
= 4 / 4 = **1.0**

**ROUGE-1 F1**
= 2 × (Precision × Recall) / (Precision + Recall)
= 2 × (0.8 × 1) / (0.8 + 1)
= 1.6 / 1.8 = **0.89**

---

### ⚠️ Important Insight

If the generated output is:

> It is not cold outside.

**Unigrams:**

* `It`, `is`, `not`, `cold`, `outside`

Still **4 matching unigrams**, so **ROUGE-1 stays the same**, even though the meaning is opposite.

👉 This shows a **limitation of ROUGE-1**: it ignores **semantic meaning**.

---

## **ROUGE-2 (Bigram matches)**

**Bigrams in Reference:**

* `It is`
* `is cold`
* `cold outside` → **3 bigrams**

**Bigrams in Output:**

* `It is`
* `is very`
* `very cold`
* `cold outside` → **4 bigrams**

**Matching bigrams:**

* `It is`
* `cold outside `→ **2 matches**

**ROUGE-2 Precision**
= 2 / 4 = **0.5**

**ROUGE-2 Recall**
= 2 / 3 = **0.67**

**ROUGE-2 F1**
= 2 × (0.5 × 0.67) / (0.5 + 0.67)
= 0.67 / 1.17 ≈ **0.57**

---

## **ROUGE-L (Longest Common Subsequence)**

* Measures the **longest common subsequence (LCS)** between reference and output
* Unlike n-grams, it:
	* Preserves **word order**
	* Allows **gaps** (not necessarily contiguous)

---

### **LCS Example**

Reference:

> It is cold outside

Output:

> It is very cold outside

**Longest Common Subsequence:**

> It → is → cold → outside → **length = 4**

---

### **ROUGE-L Calculations**

**Precision**
= LCS / length of output
= 4 / 5 = **0.8**

**Recall**
= LCS / length of reference
= 4 / 4 = **1.0**

**F1 Score**
= 2 × (0.8 × 1) / (0.8 + 1)
= **0.89**

---

## **Summary of ROUGE Variants**

| Metric  | What it Measures            | Strength                    | Weakness                         |
| ------- | --------------------------- | --------------------------- | -------------------------------- |
| ROUGE-1 | Word overlap (unigrams)     | Simple, high recall         | Ignores meaning                  |
| ROUGE-2 | Phrase overlap (bigrams)    | Captures fluency            | Still surface-level              |
| ROUGE-L | Longest sequence similarity | Considers order + structure | Still not semantic understanding |

---

## **Key Takeaways**

* ROUGE focuses on **overlap, not meaning**
* Higher-order ROUGE (2, L) captures **structure better**
* Even perfect ROUGE **does not guarantee correctness**
* That’s why modern evaluation also uses:

  * **BERTScore**
  * **LLM-based evaluation**

---

ROUGE rewards **word overlap** with the reference text.  
So if a model learns this, it can “game the system” by:

- Copying phrases directly
- Repeating important words
- Adding irrelevant but overlapping content

👉 Even if the result is **awkward, redundant, or misleading**, the score can still be high.
### Simple Example

**Reference (human):**

> It is cold outside

**Generated output:**

> cold cold cold cold

**Reference unigrams:**  
It, is, cold, outside → 4 unigrams

**Output unigrams:**  
cold, cold, cold, cold → 4 unigrams

**Naive ROUGE-1 Matching:**

If we just count all occurrences, each "cold" would be considered a match → **4 matches**

- ROUGE-1 Precision (naive) = matches / output unigrams = 4 / 4 = 1.0
- ROUGE-1 Recall (naive) = matches / reference unigrams = 4 / 4 = 1.0

> ⚠️ This gives a perfect score **even though the generated output is clearly wrong!**

This is the essence of **ROUGE hacking**.

ROUGE hacking =

> **Optimizing for word overlap instead of true understanding**

A high ROUGE score **does not guarantee**:

* Correctness
* Clarity
* Usefulness

---

## ✂️ ROUGE Clipping

### 🔍 Problem

Models may repeat words:

> cold cold cold cold

### ✅ Solution: Clipping to Prevent Overcounting

$\text{match} = \min(\text{count in reference}, \text{count in output})$

👉 Prevents:

- Artificial score inflation
- Repetition-based cheating

**ROUGE-1 Calculation with Clipping:**

- **Precision** = matches / total output unigrams = 1 / 4 = 0.25
- **Recall** = matches / total reference unigrams = 1 / 4 = 0.25
- **F1 Score** = 2 × (0.25 × 0.25) / (0.25 + 0.25) = 0.125 / 0.5 = 0.25

> Much more realistic! The score reflects that the output **mostly misses the reference meaning**.

---


## **BLEU Score (Bilingual Evaluation Understudy)**

* Used for **machine translation**
* Compares a **generated translation** to one or more **human reference translations**
* Focuses on **precision**: what fraction of generated words/n-grams appear in the reference
* Includes **brevity penalty** to avoid very short translations getting artificially high scores

---

### **BLEU Intuition**

Unlike ROUGE, which emphasizes **recall**, BLEU emphasizes **precision**.

* High BLEU = most of the generated n-grams appear in the reference
* Low BLEU = many generated words are not in reference

---

## **Example**

**Reference (human):**

> It is cold outside

**Generated output:**

> It is very cold outside

---
_BLEU metric = Avg(precision across range of n-gram sizes)_
### **Step 1: Count n-gram matches**

#### **Unigrams (1-grams)**

* Reference unigrams: `It`, `is`, `cold`, `outside` → 4
* Output unigrams: `It`, `is`, `very`, `cold`, `outside` → 5
* Matching unigrams (with **clipping**):

  * `It` → 1 match
  * `is` → 1 match
  * `cold` → 1 match
  * `outside` → 1 match

**Total matches = 4**

---

### **Step 2: Precision for unigrams**

$\text{Precision}_1 = \frac{\text{matches}}{\text{total output unigrams}} = \frac{4}{5} = 0.8$

---

#### **Bigrams (2-grams)**

* Reference bigrams: `It is`, `is cold`, `cold outside` → 3
* Output bigrams: `It is`, `is very`, `very cold`, `cold outside` → 4
* Matching bigrams:

  * `It is` → 1 match
  * `cold outside` → 1 match

**Total matches = 2**

$\text{Precision}_2 = \frac{2}{4} = 0.5$

---

### **Step 3: Geometric mean of n-gram precisions**

For BLEU-N (up to n-grams), we combine precisions using **geometric mean**:

$P = (\text{Precision}_1 \times \text{Precision}_2 \times ... \times \text{Precision}_N)^{1/N}$

For BLEU-2:

$P = \sqrt{0.8 \times 0.5} = \sqrt{0.4} \approx 0.632$

---

### **Step 4: Apply brevity penalty (BP)**

BLEU penalizes outputs shorter than the reference:

$$
BP =
\begin{cases}
1 & \text{if } len(o) > len(r)  \\
\
e^{(1 - len(r)/len(o))} & \text{if } len(o) \le len(r)
\end{cases}
$$

* `o` = output (candidate) sentence
* `r` = reference sentence

Here, `len(o) i.e. 5 > len(r) i.e. 4` → BP = 1 (no penalty)

---

### **Step 5: Final BLEU Score**

$\text{BLEU} = BP \times P = 1 \times 0.632 \approx 0.63$

> ✅ BLEU = 0.63 → reasonably good overlap with reference

---

### ⚠️ Important Insight

BLEU **ignores meaning beyond matching n-grams**:

* If output is: `It is not cold outside`

  * Unigrams match: `It`, `is`, `cold`, `outside` → high BLEU
  * Meaning is **opposite**!

> Like ROUGE, BLEU can be “gamed” by word overlap without true semantic correctness.

---

### **BLEU vs ROUGE**

| Metric | Focus                           | Strength                                    | Weakness                             |
| ------ | ------------------------------- | ------------------------------------------- | ------------------------------------ |
| ROUGE  | Recall (overlap with reference) | Good for summarization, captures content    | Ignores precision & semantic meaning |
| BLEU   | Precision (matches in output)   | Good for translation, penalizes extra words | Sensitive to synonyms & word order   |

---

### **Key Takeaways**

* BLEU = precision-oriented, ROUGE = recall-oriented
* Both can be **misleading if used alone**
* Modern evaluations often combine with:
	* **BERTScore** (semantic similarity)
	* **Human evaluation** (fluency, adequacy, meaning)

---

## **BERTScore**

- Uses **contextual embeddings** from transformer models (like BERT, RoBERTa)
- Measures **semantic similarity** between generated text and reference text
- Goes **beyond exact word overlap**, capturing meaning even if wording differs

---

### **Why BERTScore?**

ROUGE and BLEU only count **word or n-gram overlap**:

- “It is freezing outside” vs. “It is very cold outside” → low ROUGE/BLEU
- Meaning is almost identical → BERTScore can detect this

> BERTScore is **semantic-aware**, making it better for evaluating paraphrases or nuanced language.

---

## **How BERTScore Works**

1. **Embed words using a pretrained model**
    - Each token in the candidate and reference gets a vector
2. **Compute pairwise cosine similarity** between candidate and reference embeddings
3. **Aggregate scores**:
    - **Precision**: fraction of candidate words that match reference
    - **Recall**: fraction of reference words matched by candidate
    - **F1 Score**: harmonic mean of precision and recall

---

### **Example**

**Reference:**

> It is cold outside

**Generated output:**

> It is freezing outside

---

### **Step 1: Token embeddings**

|Word (Reference)|Embedding vector|
|---|---|
|It|`[0.12, -0.05, ...]`|
|is|`[0.03, 0.11, ...]`|
|cold|`[0.44, -0.22, ...]`|
|outside|`[0.30, 0.08, ...]`|

|Word (Output)|Embedding vector|
|---|---|
|It|`[0.12, -0.05, ...]`|
|is|`[0.03, 0.11, ...]`|
|freezing|`[0.41, -0.21, ...]`|
|outside|`[0.30, 0.08, ...]`|

### **Step 2: Compute cosine similarity**

- Cosine similarity measures alignment of embedding vectors
- Example matches:

|Candidate → Reference|Similarity|
|---|---|
|It → It|1.0|
|is → is|1.0|
|freezing → cold|0.85|
|outside → outside|1.0|

### **Step 3: Precision, Recall, F1**

- **Precision** = avg max similarity for each candidate token  
    $P = \frac{1.0 + 1.0 + 0.85 + 1.0}{4} = 0.9625 \approx 0.96$  
    
- **Recall** = avg max similarity for each reference token  
    $R = \frac{1.0 + 1.0 + 0.85 + 1.0}{4} = 0.96$  
    
- **F1 Score**
    $F1 = 2 \times \frac{P \cdot R}{P + R} = 0.96$  
    

> ✅ BERTScore ≈ 0.96 → recognizes the output is **semantically very close**, even though “freezing” ≠ “cold”

---

### ⚠️ Important Insight

BERTScore handles:

- **Synonyms**: cold ≈ freezing
- **Rephrasing**: “It is chilly outside” → still high score
- **Paraphrases**: “Outside, it is cold” → order changes minimally affect score

Unlike ROUGE or BLEU, **BERTScore evaluates meaning, not just overlap**.

---
### **BERTScore vs ROUGE/BLEU**

|Metric|Focus|Strength|Weakness|
|---|---|---|---|
|ROUGE|Recall, word overlap|Simple, high recall for summarization|Ignores synonyms/semantics|
|BLEU|Precision, n-gram|Good for translation, penalizes extra words|Sensitive to word order & synonyms|
|BERTScore|Semantic similarity|Captures meaning & paraphrases|Requires heavy model & compute|

---

### **Key Takeaways**

- **ROUGE & BLEU** = surface-level evaluation (words & phrases)
- **BERTScore** = semantic evaluation (meaning-aware)
- Modern LLM evaluation often **combines all three**, plus human review, for **reliability**

---

### Other Evaluation Benchmarks
- GLUE
	- GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
- SuperGLUE
	- SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems“
- MMLU (Massive Multitask Language Understanding)
	- Measuring Massive Multitask Language Understanding
- BIG-bench
	- Challenging BIG-Bench tasks and whether chain-of-thought can solve them
- HELM
	- Holistic Evaluation of Language Models (HELM)
	- Metrics:
		1. Accuracy
		2. Calibration
		3. Robustness
		4. Fairness
		5. Bias
		6. Toxicity
		7. Efficiency

If you want, I can now create a **full comparison table with ROUGE, BLEU, and BERTScore**, using the same sample outputs, so you can see exactly how each metric scores the same candidate text. It’s very visual and intuitive.

Do you want me to do that?
