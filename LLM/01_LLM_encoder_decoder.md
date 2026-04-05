The **three core transformer architectures**—**encoder-only**, **decoder-only**, and **encoder-decoder (sequence-to-sequence)**—are designed for different types of NLP tasks. Here's a breakdown of their differences and representative models:

---

### 1. Encoder-Only Architecture

Used For: **Understanding** tasks (e.g., classification, named entity recognition, sentence embeddings)
Masked Language Modeling (MLM)

* The encoder processes the **entire input sequence** and produces contextualized embeddings.
* **Self-attention** is **bidirectional**, meaning it attends to both left and right contexts.
#### Key Characteristics:

* Good at capturing **context** and **semantics** of the input.
* Not suitable for text generation.

#### Good use cases:
- Sentiment analysis
- Named entity recognition
- Word classification

#### Examples:

| Model          | Description                                             |
| -------------- | ------------------------------------------------------- |
| **BERT**       | Bidirectional Encoder Representations from Transformers |
| **RoBERTa**    | Robustly optimized BERT                                 |
| **DistilBERT** | Lighter version of BERT                                 |
| **ALBERT**     | Lite version of BERT                                    |

---

### 2. Decoder-Only Architecture

Used For: **Generation** tasks (e.g., text generation, code generation, chatbots)
Causal Language Modeling (CLM)

* Takes a **prompt** as input and generates the output sequence **token by token**.
* Self-attention is **causal (unidirectional)**—a token can only attend to previous tokens.
* Often trained as a **language model**.
#### Key Characteristics:

* Excellent at **auto-regressive generation**.
* No encoder block; it learns to predict the next token.

#### Good use cases:
- Text generation
- Other emergent behavior
	- Depends on model size
#### Examples:

| Model                        | Description                                |
| ---------------------------- | ------------------------------------------ |
| **GPT, GPT-2, GPT-3, GPT-4** | Generative Pre-trained Transformers        |
| **LLaMA**                    | Meta's language model                      |
| **Mistral**                  | Efficient open-source decoder-only LM      |
| **Falcon**                   | Decoder-only model optimized for inference |

---

###  3. Encoder-Decoder (Seq2Seq) Architecture

Used For: **Transformation** tasks (e.g., machine translation, summarization, question answering)

* The encoder processes the **input sequence**, while the decoder **generates an output sequence** based on the encoder's output.
* The decoder uses both **self-attention (causal)** and **cross-attention** to the encoder output.

#### Key Characteristics:

* Ideal for **input-output mappings** where the output depends heavily on the input.
* Uses **teacher forcing** during training (ground-truth fed into decoder).

#### Good use cases:
- Translation
- Text summarization
- Question answering

#### Examples:

| Model                                  | Description                               |
| -------------------------------------- | ----------------------------------------- |
| **T5**                                 | Text-to-Text Transfer Transformer         |
| **BART**                               | BERT + GPT hybrid (denoising autoencoder) |
| **mBART**                              | Multilingual version of BART              |
| **MarianMT**                           | Translation-focused seq2seq model         |
| **Transformer (Vaswani et al., 2017)** | The original architecture                 |

---

### Summary Table:

| Architecture    | Directionality         | Best For                   | Examples              |
| --------------- | ---------------------- | -------------------------- | --------------------- |
| Encoder-only    | Bidirectional          | Classification, NLU        | BERT, RoBERTa, ALBERT |
| Decoder-only    | Unidirectional         | Text generation            | GPT, LLaMA, Mistral   |
| Encoder-Decoder | Bidirectional + Causal | Translation, Summarization | T5, BART, MarianMT    |

