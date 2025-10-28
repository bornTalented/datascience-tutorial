Fine-tuning a large language model (LLM) involves updating the model's weights using additional data to adapt it for a specific task or domain. There are **three main approaches** to fine-tuning LLMs, each varying in computational cost, flexibility, and performance:

---

### 1. Full Fine-Tuning (End-to-End Fine-Tuning)

**Approach:**
Update **all model weights** using backpropagation on task-specific data.

**Use Case:**
* High-resource settings
* When task performance is critical and domain-shift is significant

**Example:**
Fine-tuning GPT-2 on medical text to create a medical chatbot.

**Logic:**
* Input the task-specific data $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$
* Minimize task-specific loss, e.g., cross-entropy:

$$
\mathcal{L} = -\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_{\theta}(y_t^{(i)} | y_{<t}^{(i)}, x^{(i)})
$$

* Update **all parameters** $\theta$ using gradient descent:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}
$$

### 2. Parameter-Efficient Fine-Tuning (PEFT)

Instead of updating all parameters, update only a **small subset**, saving compute and memory.
#### Popular Methods:

**2.1. LoRA (Low-Rank Adaptation)**
* Freeze the model weights.
* Inject trainable **low-rank matrices** into transformer layers.

**Example:**
Fine-tuning LLaMA 7B on legal data using LoRA.

**Logic:**
Assume a weight matrix $W \in \mathbb{R}^{d \times k}$. Instead of updating $W$, add a low-rank perturbation:
$$
\tilde{W} = W + \Delta W,\quad \Delta W = A B,\quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}
$$
Train only $A$ and $B$, where $r \ll d, k$

**Benefits:**
* Drastically reduces trainable parameters
* Can be merged back into the original model post-training

### 3. Adapter Tuning

**Approach:**
Insert small trainable **adapter layers** between frozen transformer blocks.

**Example:**
Adapter-based fine-tuning of BERT for sentiment classification.

**Logic:**
* Given transformer hidden state $h \in \mathbb{R}^d$
* Adapter: a bottleneck MLP
$$
\text{Adapter}(h) = W_{\text{up}}(\text{ReLU}(W_{\text{down}} h))
$$

where:

* $W_{\text{down}} \in \mathbb{R}^{r \times d}$, $W_{\text{up}} \in \mathbb{R}^{d \times r}$
* Only $W_{\text{down}}, W_{\text{up}}$ are trained

---

### 4. Prefix / Prompt Tuning

Inject learnable prompts or tokens **without modifying model weights**.

**Types:**
* **Prefix Tuning:** Train a continuous embedding prefix to guide attention.
	- **Mathematical View:**
		Given transformer layers, for each attention layer:
	$$
\text{Attention}(Q, K, V) \Rightarrow \text{Attention}(Q, [P_K; K], [P_V; V])
	$$
		Where $P_K, P_V$ are **trainable prefix keys and values**
	
* **Prompt Tuning:** Train soft prompt embeddings prepended to inputs.
	- **Example:**
		Prompt-tuning GPT-3 for question answering in a finance domain.

---

### Comparison Table

| Approach             | Parameters Trained | Memory Cost | Performance    | Notes                                  |
| -------------------- | ------------------ | ----------- | -------------- | -------------------------------------- |
| Full Fine-tuning     | 100%               | High        | High           | Best if data and compute are available |
| LoRA                 | <1%                | Low         | High           | Preferred for large models             |
| Adapter Tuning       | \~1-5%             | Moderate    | Moderate       | Good for modular design                |
| Prefix/Prompt Tuning | Very Low           | Very Low    | Task-dependent | Efficient for few-shot or many tasks   |

---

### Final Thoughts

The choice depends on the **task complexity**, **compute budget**, and **desired generalization**:
* Use **LoRA** or **adapter tuning** when dealing with very large models (e.g., LLaMA, GPT).
* Use **prompt tuning** when few-shot learning is sufficient.
* Use **full fine-tuning** only when complete model adaptation is needed.