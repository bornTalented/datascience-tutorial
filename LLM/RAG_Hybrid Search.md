## 📘 **Tutorial: Hybrid Search in Retrieval-Augmented Generation (RAG)**

## Introduction to Hybrid Search in RAG

In a Retrieval-Augmented Generation (RAG) system, the goal is to fetch the most relevant documents for a user query and use them to generate a response via a language model (LLM). **Hybrid Search** combines multiple search strategies—**semantic** and **syntactic**—to maximize retrieval accuracy.

### What is Hybrid Search?
Combines Multiple Search Techniques:

1. **Semantic Search** → Dense Vector Similarity (meaning-based)
2. **Syntactic Search** → Exact Match / Keyword Search (token-based)

Steps:

1. Documents are encoded as:
	
	* **Dense Vectors** (for semantic search)
	* **Sparse Matrix** (for keyword search)

2. User query is also embedded in both forms:

	* **Dense** for semantic similarity
	* **Sparse** for keyword overlap

3. Perform two types of retrieval:
	
	* **Vector Search** → returns `Result_k1`
	* **Keyword Search** → returns `Result_k2`

4. Merge both result sets using:

	* **Rank Fusion** or
	* **Weighted Heuristics**

5. Pass merged documents to the LLM for final **response generation**.

---
- ### 1. Semantic Search (Dense Vector Search)

	### 📥 Document Embedding
	
	* Input text documents (D1, D2, ..., Dn) are converted into **dense embedding vectors** using models like:
		* OpenAI Embeddings (e.g. `text-embedding-3-small`, `text-embedding-3-large` and `text-embedding-ada-002`)
		* Hugging Face Transformers (e.g., `all-MiniLM-L6-v2`, `bge-base-en-v1.5`)
		* Ollama Embeddings (e.g. `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`)
	
	### 📤 Storage
	
	* These vectors are stored in a **vector database** like:
		* **FAISS**
		* **Chroma**
		* Other `VectorStoreDB` systems
	
	### 📈 Querying
	
	* User query is also embedded into a dense vector.
	* The system uses **cosine similarity** to retrieve semantically similar documents.
	
	### 🧠 Response Generation
	
	* Retrieved vectors go into a **prompt template**.
	* The LLM processes this context and generates a summarized **response**.

- ### 2. Syntactic Search (Keyword Search)

	### 📘 Sparse Matrix Creation
	
	* Text is converted into **sparse representations** using traditional techniques:
		* One-Hot Encoding (OHE)
		* Bag-of-Words (BoW)
		* TF-IDF
	
	### 🔎 Sparse Vector Search
	
	* A sparse matrix is generated for each document.
	* User queries are vectorized in the same sparse form.
	* Exact or near-exact matches are retrieved based on token overlap.

---

### Reciprocal Rank Fusion in Hybrid Search

**Reciprocal Rank Fusion** is a simple and effective method for **combining ranked lists** from multiple retrieval methods (e.g., semantic, keyword, graph search).

It works by assigning **higher scores to top-ranked results**, and **combining scores from multiple lists** in a reciprocal manner.

####  RRF Formula:

$$
\text{Final Score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

* $R$: Set of ranked lists (e.g., from semantic and keyword search)
* $\text{rank}_r(d)$: Rank of document `d` in retrieval list `r` (1 for top, 2 for second, etc.)
* $k$: A constant (commonly set to 60) to control the influence of rank
* If a document is not in a list, it contributes nothing from that list
#### Example:

Assume we perform **2 types of search**:

* **Semantic Search** (e.g., via cosine similarity)
* **Keyword Search** (exact matching)

Each search returns a **ranked list** of documents:

> **Semantic Search Result (Dense Vectors):**

| Rank | Document   | Score (1 / Rank) |
| ---- | ---------- | ---------------- |
| 1    | Document 1 | $1.00$           |
| 2    | Document 2 | $0.5$            |
| 3    | Document 3 | $0.33$           |
| 4    | Document 4 | $0.25$           |
| 5    | Document 5 | $0.20$           |

> **Keyword Search Result (Sparse Vectors):**

| Rank | Document   | Score (1 / Rank) |
| ---- | ---------- | ---------------- |
| 1    | Document 5 | $1.00$           |
| 2    | Document 3 | $0.5$            |
| 3    | Document 2 | $0.33$           |
| 4    | Document 4 | $0.25$           |
| 5    | Document 1 | $0.20$           |

**Step-by-Step RRF Score Calculation**

Let’s use the formula:

$$
\text{RRF Score} = \frac{1}{k + \text{rank}_\text{semantic}} + \frac{1}{k + \text{rank}_\text{keyword}}
$$

Assume $k = 0$ (for simplicity in this example).

| **Document** | **Semantic Rank** | **Semantic Score (1/rank)** | **Keyword Rank** | **Keyword Score (1/rank)** | **Total RRF Score** |
| ------------ | ----------------- | --------------------------- | ---------------- | -------------------------- | ------------------- |
| Document 1   | 1                 | 1.00                        | 5                | 0.20                       | **1.20**            |
| Document 2   | 2                 | 0.50                        | 3                | 0.33                       | **0.83**            |
| Document 3   | 3                 | 0.33                        | 2                | 0.50                       | **0.83**            |
| Document 4   | 4                 | 0.25                        | 4                | 0.25                       | **0.50**            |
| Document 5   | 5                 | 0.20                        | 1                | 1.00                       | **1.20**            |

🏆 **Final RRF Ranking**

| Document   | RRF Score | Final Rank  |
| ---------- | --------- | ----------- |
| Document 1 | 1.2       | ✅ 1st       |
| Document 5 | 1.2       | ✅ 1st (tie) |
| Document 2 | 0.83      | 3rd         |
| Document 3 | 0.83      | 3rd (tie)   |
| Document 4 | 0.5       | 5th         |

---

### Weighted Merging Option (Alternative to RRF)

* You can assign weights:
	* 50% Semantic
	* 50% Keyword
	* Or any other ratio like 70:30, depending on your use-case

This approach works well when you have varying confidence in your semantic vs keyword models.

### Summary

| Component         | Description                                       |
| ----------------- | ------------------------------------------------- |
| Dense Embeddings  | Capture semantics using OpenAI, Hugging Face etc. |
| Sparse Embeddings | Capture exact keyword match using TF-IDF/BoW      |
| Vector DB         | FAISS, Chroma, used for storing dense vectors     |
| Hybrid Search     | Combines semantic + keyword retrieval             |
| Rank Fusion       | Combines results using reciprocal rank formula    |
| Output            | Final merged context passed to LLM for response   |

### 🛠️ Tools & Frameworks for Implementation

* **LangChain** or **LlamaIndex** for RAG orchestration
* **OpenAI** / **HuggingFace Transformers** for embeddings
* **FAISS**, **Chroma** for vector storage
* **Scikit-learn** / **Scipy** for TF-IDF and sparse matrices
* **LLMs** like GPT-4, Mistral, LLaMA for generating final response

---
## 📘 **Extended Tutorial: Hybrid + Graph Knowledge Search in RAG**

**Graph Knowledge Search**, an advanced retrieval method that complements **Hybrid Search** (Semantic + Syntactic) by incorporating **structured relationships** from a **Knowledge Graph (KG)**.

---
## Motivation for Graph-Based Retrieval in RAG

While **semantic** and **keyword** searches retrieve documents based on vector similarity or lexical overlap, they often **miss explicit, structured relationships** like:

* Who is the CEO of a company?
* What symptoms are connected to a disease?
* Which regulations cite a particular legal clause?

This is where **Knowledge Graph Search** excels.

### **What is a Knowledge Graph (KG)?**

A **Knowledge Graph** is a collection of **triples** in the form:

$$
\text{(subject, predicate, object)}
$$

For example:

* ("Elon Musk", "CEO\_of", "Tesla")
* ("Fever", "is\_symptom\_of", "Flu")

Components:
* **Nodes**: Entities (e.g., people, companies, concepts)
* **Edges**: Relationships (e.g., works\_for, causes, located\_in)

---
### How Graph-Based Search Works in RAG

Step-by-step Workflow:

1. **Document Ingestion**

   * Text is parsed to extract **entities** and **relations** using:

     * Named Entity Recognition (NER)
     * Relation Extraction models

2. **Graph Construction**

   * Entities and relations are structured into a graph (Neo4j, RDF store, NetworkX, etc.)

3. **User Query Interpretation**

   * The query is converted into a **graph query**:

     * SPARQL (for RDF)
     * Cypher (for Neo4j)

4. **Graph Query Execution**

   * Search traverses the graph to find exact entity/relationship matches.

5. **Hybrid Integration**

   * Combine graph search results with:

     * **Semantic Search** → via dense embeddings
     * **Syntactic Search** → via TF-IDF/BoW

6. **Context Construction**

   * Results from **KG**, **vector**, and **keyword** search are merged using **Reciprocal Rank Fusion** or **weighting strategy**.

7. **LLM Response**

   * Enriched context passed to LLM to generate accurate and grounded response.

---

**Merging Graph Search with Hybrid Search**

Three-way Retrieval:

| Search Type     | Output                                |
| --------------- | ------------------------------------- |
| Semantic Search | Meaning-based similar docs            |
| Keyword Search  | Exact match docs                      |
| Graph Search    | Facts/triples based on entity linkage |

####  Unified Scoring (Extended Rank Fusion):

Use **Weighted Score Fusion**:

$$
\text{Final Score} = w_1 \cdot R_{\text{semantic}} + w_2 \cdot R_{\text{keyword}} + w_3 \cdot R_{\text{graph}}
$$

Where $w_1 + w_2 + w_3 = 1$

Example:

* $w_1 = 0.4$
* $w_2 = 0.3$
* $w_3 = 0.3$

You can use **Reciprocal Rank Fusion** or **learned weights** using validation data.

#### Graph Search: Tools & Frameworks

| Task                  | Tool / Framework                     |
| --------------------- | ------------------------------------ |
| KG Storage            | Neo4j, RDFStore, TigerGraph          |
| Triple Extraction     | spaCy + RE, OpenIE, Stanford CoreNLP |
| KG Querying           | Cypher (Neo4j), SPARQL               |
| Integration           | LangChain + Neo4jGraph, LlamaIndex   |
| LLM Context Generator | OpenAI GPT, LLaMA, Claude            |
#### Graph Search: Use Cases

| Domain     | Example Query                            | Graph Utility                |
| ---------- | ---------------------------------------- | ---------------------------- |
| Healthcare | “What diseases cause chest pain?”        | Disease → symptom relation   |
| Legal      | “Which laws cite section 420?”           | Clause → citation relation   |
| Finance    | “Who owns majority shares in Tesla?”     | Ownership relation           |
| Education  | “What are prerequisites for ML courses?” | Course → dependency relation |

#### Benefits of Adding Graph Search

* Enhances factual correctness.
* Enables logical reasoning and multi-hop queries.
* Extracts precise answers grounded in structured data.
* Reduces hallucinations by LLMs.

#### Visual Pipeline Overview

```python
                        ┌────────────────────┐
                        │     Documents      │
                        └────────────────────┘
                                 │
     ┌───────────────┬──────────┴───────────┬──────────────┐
     │ Semantic Emb  │ TF-IDF Vectorizer   │  Triples Extractor
     │ (Dense Vector)│ (Sparse Matrix)     │  (Graph Builder)
     └──────┬────────┴────────────┬────────┴──────┬────────┘
            ▼                     ▼                ▼
       Vector DB           Sparse Index        Graph DB
            │                     │                │
            ▼                     ▼                ▼
     Dense Vector Search   Keyword Search     Graph Traversal
            │                     │                │
            └────────────┬────────┴───────┬────────┘
                         ▼                ▼
                    Fusion Module (Rank Fusion / Weighted)
                                 │
                                 ▼
                          Context Construction
                                 │
                                 ▼
                              Prompted LLM
                                 │
                                 ▼
                             Final Answer
```

---

### Conclusion: Unified Hybrid + Graph Retrieval

* **Hybrid Search**: Combines **semantic** + **syntactic**
* **Graph Search**: Adds structured reasoning on top
* Together, they form a **powerful retrieval system** that enhances accuracy, interpretability, and response quality for LLMs.

---

Would you like me to provide a working Python example using FAISS + TF-IDF + Reciprocal Rank Fusion?
Would you like a code-based example integrating `LangChain` + `Neo4jGraph` + `FAISS` + `TF-IDF` into a unified RAG pipeline?

