The terms **prompt engineering** and **prompt tuning** are related to how we steer the behavior of large language models (LLMs), but they differ fundamentally in **method**, **learning**, and **use cases**.

---

## 1. Prompt Engineering

**Definition:** *Manual* design of textual input prompts to guide the LLM's output behavior, using *natural language*.

**Goal:** Find the best phrasing or structure to elicit the desired response from a *frozen* model (no learning involved).

**Method:**
* Write instructions or examples (zero-shot, few-shot)
* Tune input wording manually
* Use trial-and-error

**No Learning:**
No parameters are updated; the model remains unchanged.

📝 **Example:**

```text
"Translate the following English sentence to French: 'I am hungry.'"
```

or

```text
"Write a professional email declining an invitation to a meeting."
```

📦 **Use Case:**
* API-based LLM usage (like OpenAI's GPT-4)
* Rapid prototyping
* Low/no compute resources

---

## 2. Prompt Tuning

**Definition:** *Automated*, trainable method where a small number of *continuous prompt embeddings* are learned and prepended to the model input.

**Goal:** Fine-tune the model's behavior on a specific task *without changing model weights*.

**Method:**
* Learn soft (continuous vector) prompts $P \in \mathbb{R}^{m \times d}$
* These embeddings are trained using gradient descent on task-specific data
* The base LLM remains frozen

**Learning Involved:**
Only the soft prompt vectors are trained, not the full model.

📝 **Example:**

Instead of training GPT-2, you learn a prompt embedding vector that, when prepended to the input, makes the model output summaries of medical texts.

📦 **Use Case:**
* Resource-efficient fine-tuning
* Use in limited compute environments
* When access to model weights is restricted (e.g., black-box models)

---

## Comparison Table

| Aspect               | Prompt Engineering              | Prompt Tuning                                  |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| **Learning**         | None (manual)                   | Yes (learnable embeddings)                     |
| **Prompt Type**      | Natural language (discrete)     | Continuous vector embeddings                   |
| **Model Access**     | Black-box (API or frozen model) | White-box or semi-white-box (embeddings added) |
| **Trainable Params** | 0                               | Small (e.g., a few thousand vectors)           |
| **Goal**             | Steer model with crafted input  | Task adaptation without full fine-tuning       |
| **Flexibility**      | Quick changes, easy to iterate  | Requires training, more precise adaptation     |

## Visual Analogy

* **Prompt Engineering** is like giving better instructions to a smart assistant.
* **Prompt Tuning** is like training the assistant to interpret your instructions better for a specific job, without changing its brain.

---

Example of how to implement **prompt tuning** using Hugging Face Transformers?
- Here's how you can implement **prompt tuning** using Hugging Face's `transformers` library with **T5** or **GPT-2** models. Hugging Face has built-in support via the [`PromptTuningConfig`](https://huggingface.co/docs/transformers/main_classes/prompt_tuning) and PEFT-compatible models.
### Example: Prompt Tuning for Text Summarization with T5

We’ll fine-tune a **frozen T5 model** on a small summarization task using **prompt tuning**.

#### Install Required Packages

```bash
pip install transformers datasets accelerate peft
```

#### Step-by-Step Code

```python
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    PromptTuningConfig,
    Seq2SeqTrainer
)
from datasets import load_dataset

# Load base model and tokenizer (frozen model)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a summarization dataset (e.g., CNN/DailyMail)
dataset = load_dataset("xsum")
train_data = dataset["train"].select(range(1000))  # Subset for demo
val_data = dataset["validation"].select(range(100))

# Tokenization
def preprocess(batch):
    inputs = tokenizer(
        ["summarize: " + x for x in batch["document"]],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    targets = tokenizer(
        batch["summary"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

train_data = train_data.map(preprocess, batched=True)
val_data = val_data.map(preprocess, batched=True)

# Prompt tuning config
prompt_config = PromptTuningConfig(
    task_type="seq2seq",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name
)

# Load model with prompt tuning
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    use_auth_token=True,
    device_map="auto",
    prompt_tuning_config=prompt_config
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./prompt-tuned-t5",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=5e-4,
    save_total_limit=2,
)

# Data collator
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=collator
)

# Train prompt embeddings
trainer.train()
```

#### What’s Happening?

* The model weights are **frozen**.
* A set of **learnable prompt embeddings** is prepended to the input embeddings.
* Only these prompt vectors are trained — no full model updates.

After training:
* The prompt vectors are saved in the model directory.
* You can use them to generate summaries with the frozen T5 model.

---

