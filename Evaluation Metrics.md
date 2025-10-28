A **comprehensive list of evaluation metrics** across key ML tasks, along with their **mathematical formulas**.

---

## 🔵 1. Classification Metrics

### A. Binary Classification

Let:

* TP = True Positives
* TN = True Negatives
* FP = False Positives
* FN = False Negatives

1.  Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

2. Precision (Positive Predictive Value)

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

3. Recall (Sensitivity, True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

4. Specificity (True Negative Rate)

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

5. F1 Score (Harmonic mean of Precision and Recall)

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

6. $F_β$ Score

$$
F_{\beta} = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}
$$
	Where β > 0 determines the weight of recall relative to precision.

7. Matthews Correlation Coefficient (MCC)

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

* N = Total number of samples
* $y_i \in \{0, 1\}$ = true label, $\hat{y}_i \in [0, 1]$ = predicted probability

8. **Balanced Accuracy**

$$
\text{Balanced Accuracy} = = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{\text{TPR} + \text{TNR}}{2}
$$

9. **False Positive Rate (FPR)**

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

10. **False Negative Rate (FNR)**

$$
\text{FNR} = \frac{FN}{FN + TP}
$$

11. **False Discovery Rate (FDR)**

$$
\text{FDR} = \frac{FP}{FP + TP}
$$

12. **False Omission Rate (FOR)**

$$
\text{FOR} = \frac{FN}{FN + TN}
$$

13. **Prevalence**

$$
\text{Prevalence} = \frac{TP + FN}{TP + TN + FP + FN}
$$

14. **Threat Score (Critical Success Index)**

$$
\text{TS} = \frac{TP}{TP + FN + FP}
$$

15. **Cohen's Kappa**

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

	where $p_o =$ observed agreement, $p_e =$ expected agreement.

16. **Brier Score**

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

17. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

	No single formula, but:
	
	* Plot TPR vs FPR across thresholds.
	* Area under the ROC curve is calculated using trapezoidal integration.
	
$$
\text{AUC} = \int_{0}^{1} \text{TPR}(FPR) \, dFPR
$$
	*Alternatively, AUC can be interpreted as the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.*

18. **PR-AUC (Precision-Recall AUC)**
    No closed form: area under the Precision-Recall curve.

19. Log Loss (Cross-Entropy Loss)

$$
\text{LogLoss} = - \frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]
$$

---

### B. Multi-class Classification

For multi-class problems, metrics are often averaged:
* **Macro-Averaging**: Unweighted mean across classes.
* **Micro-Averaging**: Aggregates contributions of all classes to compute the average metric (e.g. global TP/FP/FN).
* **Weighted-Averaging**: Weighted mean by the number of instances per class.

Let:

* C = Number of classes
* $y_i \in \{1, 2, ..., C\}$, $\hat{y}_i \in \{1, 2, ..., C\}$

1. **Macro-Averaged Metrics**

$$
\text{Macro-Precision} = \frac{1}{C} \sum_{c=1}^C \text{Precision}_c
$$

(similarly for Recall, F1)

2. **Micro-Averaged Metrics**

$$
\text{Micro-Precision} = \frac{\sum_c TP_c}{\sum_c TP_c + \sum_c FP_c}
$$

(similarly for Recall, F1)

3. **Weighted Metrics**

$$
\text{Weighted-F1} = \sum_{c=1}^C \frac{n_c}{N} F1_c
$$

4. **Top-k Accuracy** (e.g., Top-1, Top-5)

$$
\text{Top-k Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(y_i \in \hat{y}_i^{(k)})
$$
	*Where $\hat{y}_i^{(k)}$ is the set of top-k predicted classes for instance $i$.

5. **Hamming Loss**

$$
\text{Hamming Loss} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(y_i \ne \hat{y}_i)
$$

6. **Jaccard Index (Intersection over Union)**

$$
\text{Jaccard} = \frac{TP}{TP + FP + FN}
$$

---

### C. Multi-label Classification


Let:

* L = total number of labels
* $y_i \in \{0,1\}^L$, $\hat{y}_i \in \{0,1\}^L$

1. **Subset Accuracy**

$$
\text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(y_i = \hat{y}_i)
$$

2. **Hamming Loss**

$$
\text{Hamming Loss} = \frac{1}{N \cdot L} \sum_{i=1}^N \sum_{j=1}^L \mathbb{1}(y_{ij} \ne \hat{y}_{ij})
$$
	Where $L$ is the number of labels.

3. **Jaccard Index** (Intersection over Union)

$$
\text{Jaccard} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|}
$$

4. **Label Ranking Average Precision (LRAP)**
   Custom metric based on label rankings.


$$
\text{LRAP} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{|Y_i|} \sum_{j \in Y_i} \frac{|\{k: r_{ik} \leq r_{ij}, k \in Y_i\}|}{r_{ij}}
$$

	Where $r_{ij}$ is the rank of label $j$ for instance $i$.

---

## 🟢 2. Regression Metrics

Let:

* $y_i$: True value
* $\hat{y}_i$: Predicted value
* $\bar{y}$: Mean of true values
* $N$: Number of instances

$y_i \in \mathbb{R}$, $\hat{y}_i \in \mathbb{R}$, $\bar{y} = \frac{1}{N} \sum y_i$

1. **Mean Absolute Error (MAE)**

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$

2. **Mean Squared Error (MSE)**

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

3. **Root Mean Squared Error (RMSE)**

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

4. **R-squared ($R^2$)** (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
$$

5. **Adjusted $R^2$**

$$
\text{Adjusted } R^2 =\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{N - 1}{N - p - 1}
$$

	Where $p$ is number of predictors.

6. **Mean Absolute Percentage Error (MAPE)**

$$
\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

7. **Symmetric MAPE (SMAPE)**

$$
\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
$$

8. **Huber Loss**

$$
L_\delta(a) = \begin{cases}
\frac{1}{2} a^2 & \text{for } |a| \leq \delta \\
\delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$
	Where $a = y_i - \hat{y}_i$ and $\delta$ is a threshold parameter.

	- **What it is**: Huber loss combines **MSE** and **MAE** by using squared error for smaller values and absolute error for larger values, depending on a threshold δ\deltaδ.
	    
	- **Why it’s useful with outliers**: For errors smaller than the threshold δ, it uses MSE (which is sensitive to small errors), and for errors larger than δ, it uses MAE (which is robust to large errors). This approach reduces the influence of outliers.
	
9. **Mean Bias Deviation (MBD)**

$$
\text{MBD} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)
$$

10. **Mean Squared Logarithmic Error (MSLE)**

$$
\text{MSLE} = \frac{1}{N} \sum_{i=1}^{N} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2
$$


---

## 🔶 3. Clustering Metrics

Let:

* $U$: Ground truth labels
* $V$: Predicted cluster labels


1. **Adjusted Rand Index (ARI)**

$$
ARI = \frac{RI - \mathbb{E}[RI]}{\max(RI) - \mathbb{E}[RI]}
$$
	Where RI is the Rand Index and $\mathbb{E}$ means Expected
	(Requires combinatorial computation)

2. **Normalized Mutual Information (NMI)**

$$
\text{NMI}(U, V) = \frac{2 \cdot I(U;V)}{H(U) + H(V)}
$$

3. **Silhouette Score**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
	Where:
	
	* $a(i)$ = avg distance to points in same cluster.
	* $b(i)$ = lowest avg distance to points in other clusters.

4. **Davies-Bouldin Index**: based on intra-/inter-cluster distances (no simple formula).

---

## 🔷 4. Ranking / Information Retrieval / Recommender Metrics

1. **Precision\@k**

$$
P@k = \frac{\text{Number of relevant items in top-}k}{k}
$$

2. **Recall\@k**

$$
R@k = \frac{\text{Number of relevant items in top-}k}{\text{Total number of relevant items}}
$$

3. **Mean Reciprocal Rank (MRR)**

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

4. **Normalized Discounted Cumulative Gain (NDCG@k)**

$$
DCG_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}, \quad NDCG_k = \frac{DCG_k}{IDCG_k}
$$
	Let $rel_i$ be relevance at rank $i$.

Other metrices:
* **Mean Average Precision (MAP)**
* **Normalized Discounted Cumulative Gain (NDCG)**
* **Hit Rate**
* **Coverage**
* **Diversity**
* **Novelty**
* **Mean Percentile Rank (MPR)**

---

## 🟣 5. Anomaly Detection Metrics

Same as binary classification: ROC-AUC, PR-AUC, F1, etc.

---

## 🔶 6. Survival Analysis (Time-to-event prediction)

1. **Concordance Index (C-index)**

$$
\text{C-index} = \frac{\text{Number of concordant pairs}}{\text{Number of comparable pairs}}
$$
Other:
* **Brier Score**
* **Log-rank Test**
* **Time-dependent AUC**
* **Integrated Brier Score**

---

## 🔵 7. Time Series Forecasting

Same as regression with:


1. **Mean Directional Accuracy (MDA)**

$$
\text{MDA} = \frac{1}{n-1} \sum_{t=2}^n \mathbb{1}[(y_t - y_{t-1})(\hat{y}_t - \hat{y}_{t-1}) > 0]
$$

Other:
* **Weighted RMSE**
* **Mean Scaled Error (MSE)**
* **Mean Interval Score**
* **Pinball Loss** (for probabilistic forecasts)

---

## 🔘 8. NLP Metrics


1. **BLEU** (for translation/summarization)

$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n \right)
$$
	Where $p_n$ is modified n-gram precision, $BP$ is brevity penalty.

2. **ROUGE-L** (Longest Common Subsequence)

$$
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\text{Recall} + \beta^2 \cdot \text{Precision}}
$$

3. **Perplexity** (for language modeling)

$$
\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log p(w_i)\right)
$$

---

## 🟠 9. Computer Vision


1. **Intersection over Union (IoU)**

$$
IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|}
$$

2. **Dice Coefficient**

$$
\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}
$$

---

## 🔴 Summary Table (Tasks → Metrics)

| ML Task                    | Common Metrics                                             |
| -------------------------- | ---------------------------------------------------------- |
| Binary Classification      | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss |
| Multi-class Classification | Accuracy, Macro/Micro F1, Top-k Accuracy, Log Loss         |
| Multi-label Classification | Hamming Loss, Subset Accuracy, Jaccard, LRAP               |
| Regression                 | MSE, MAE, RMSE, R², Adjusted R², MAPE, SMAPE               |
| Clustering                 | ARI, NMI, Silhouette, Davies-Bouldin, CH Index             |
| Ranking / Recommendation   | NDCG, MAP, MRR, Precision\@k, Recall\@k                    |
| Anomaly Detection          | ROC-AUC, PR-AUC, F1, Precision\@N                          |
| Survival Analysis          | Concordance Index, Brier Score                             |
| Time Series Forecasting    | MAE, RMSE, MAPE, SMAPE, MDA                                |
| Generative Models          | FID, IS, KID, BERTScore                                    |
| NLP Tasks                  | BLEU, ROUGE, METEOR, Perplexity, EM, F1, BERTScore         |
| CV Tasks                   | mAP, IoU, Dice Coeff, Pixel Accuracy                       |


---
Here’s a **comprehensive tutorial** on *Information Retrieval Evaluation* based on the [Amitness article](https://amitness.com/posts/information-retrieval-evaluation/), enriched with **clear examples** for each metric and made more **readable for learners**. All original content is preserved and expanded.

---

## 📘 Tutorial: Evaluating Information Retrieval Systems

---

### 1. 📌 Why Evaluate IR Systems?

Evaluation tells us **how good** our retrieval system is—how well it surfaces *relevant* documents for a query.

* **Offline evaluation** uses predefined datasets and static relevance labels.
* **Online evaluation** uses real-time user data like clicks or time spent.

---

## 2. Order‑Unaware Metrics (Don't Care About Ranking)

These metrics focus only on *which documents* are retrieved, **not** their *order*.

---

### 2.1 🎯 Precision\@K

**Definition**: Fraction of top‑K retrieved items that are relevant.

$$
\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Retrieved}_{@K}|}{K}
$$

#### 🔍 Example:

* Retrieved top-5 docs: `[D1, D2, D3, D4, D5]`
* Relevant docs: `{D1, D3, D6}`
* Precision\@5 = 2 relevant in top 5 →

  $$
  P@5 = \frac{2}{5} = 0.4
  $$

---

### 2.2 🧲 Recall\@K

**Definition**: Fraction of all relevant documents that were retrieved in top‑K.

$$
\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved}_{@K}|}{|\text{Relevant}|}
$$

#### 🔍 Example:

* Retrieved top-5: `[D1, D2, D3, D4, D5]`
* Relevant docs: `{D1, D3, D6}`
* Recall\@5 = 2 out of 3 →

  $$
  R@5 = \frac{2}{3} ≈ 0.667
  $$

---

### 2.3 ⚖️ F1-Score\@K

**Definition**: Harmonic mean of Precision\@K and Recall\@K.

$$
F1@K = 2 \cdot \frac{P@K \cdot R@K}{P@K + R@K}
$$

#### 🔍 Example:

* From above: $P@5 = 0.4$, $R@5 ≈ 0.667$

$$
F1@5 = 2 \cdot \frac{0.4 \cdot 0.667}{0.4 + 0.667} ≈ 0.5
$$

---

## 3. Order‑Aware Metrics (Ranking Matters)

These metrics **reward systems** that return relevant documents **higher** in the list.

---

### 3.1 🎯 Mean Reciprocal Rank (MRR)

**Definition**: Inverse of the rank of the first relevant document.

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

#### 🔍 Example:

| Query | Retrieved Docs | First Relevant   | Reciprocal Rank |
| ----- | -------------- | ---------------- | --------------- |
| Q1    | `[D2, D4, D1]` | D1 at position 3 | 1/3             |
| Q2    | `[D3, D5, D4]` | D3 at position 1 | 1               |

$$
MRR = \frac{1}{2} \cdot \left(\frac{1}{3} + 1\right) = 0.667
$$

---

### 3.2 📚 Average Precision (AP) & MAP

**AP**: Precision averaged at every rank where a relevant document is found.

$$
AP = \frac{1}{|\text{Relevant}|} \sum_{k=1}^n P@k \cdot rel(k)
$$

**MAP**: Mean AP over multiple queries.

#### 🔍 Example:

* Retrieved: `[D1, D2, D3, D4, D5]`
* Relevant: `{D1, D3, D5}`
* Relevance: `[1, 0, 1, 0, 1]`

$$
\begin{align*}
P@1 &= 1.0 \quad \text{(D1 is relevant)} \\
P@3 &= \frac{2}{3} \quad \text{(D3 is relevant)} \\
P@5 &= \frac{3}{5} \quad \text{(D5 is relevant)} \\
AP &= \frac{1.0 + 0.667 + 0.6}{3} = 0.755
\end{align*}
$$

---

### 3.3 📈 DCG & NDCG

**DCG**: Discounts the relevance of lower-ranked results.

$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
$$

**NDCG**: Normalizes DCG by the ideal DCG.

#### 🔍 Example:

* Relevance scores: `[3, 2, 3, 0, 1]` for positions 1 to 5

$$
DCG@5 = \frac{3}{\log_2(2)} + \frac{2}{\log_2(3)} + \frac{3}{\log_2(4)} + 0 + \frac{1}{\log_2(6)} \approx 6.15
$$

* Sort the list ideally: `[3, 3, 2, 1, 0]`

$$
IDCG@5 ≈ 3 + 3/\log_2(3) + 2/\log_2(4) + 1/\log_2(5) ≈ 6.89
$$

$$
NDCG@5 = \frac{6.15}{6.89} ≈ 0.892
$$

---

## 4. Summary Table of Metrics

| Metric       | Ranking Used | Relevance Used | Best Use Case                            |
| ------------ | ------------ | -------------- | ---------------------------------------- |
| Precision\@K | ❌            | Binary         | Accuracy in top‑K                        |
| Recall\@K    | ❌            | Binary         | Coverage of relevant items               |
| F1\@K        | ❌            | Binary         | Balance between precision and recall     |
| MRR          | ✅            | Binary         | Finding a single relevant item quickly   |
| MAP          | ✅            | Binary         | Ranking multiple relevant items properly |
| NDCG         | ✅            | Graded         | Graded relevance (e.g. 0–3 scale)        |

---

## 5. Online Evaluation Metrics (for Live Systems)

These metrics are used *after* deployment to evaluate system behavior with real users.

* **Click-through rate (CTR)**: How often results are clicked.
* **Abandonment rate**: Users leaving without clicking.
* **Dwell time**: Time spent on clicked documents.
* **Conversion rate**: Did the user complete a desired action?

---

## 6. 🧠 Choosing the Right Metric

| Situation                            | Suggested Metric  |
| ------------------------------------ | ----------------- |
| Need first result to be correct      | MRR, Precision\@1 |
| Multiple relevant documents          | MAP, NDCG         |
| Relevance has levels (graded)        | NDCG              |
| Only care about relevance, not order | Precision/Recall  |
| Real user behavior in production     | CTR, Dwell Time   |

---

## ✅ Conclusion

Evaluating an IR system requires both **offline rigor** (like Precision\@K, MAP, NDCG) and **online intuition** (like CTR). Always align your metric with your **use case**.
