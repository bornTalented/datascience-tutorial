Advance techniques aim to improve relevance, accuracy, personalization, and diversity of the retrieved content.

Retrieval-Augmented Generation (RAG) involves the following key stages:

1. **Query Preparation**
2. **Retrieval**
3. **Post-Retrieval Reranking**
4. **Result Aggregation & Augmentation**

---

### 1. Query Preparation

Techniques to enhance or modify the input query for better retrieval results:

#### 🔹 Query Expansion / Reformulation

* Enhances the query by adding synonyms, related terms, or rephrases the query.
* Techniques:
	* Pseudo Relevance Feedback (PRF)
		* Automatically expands queries using top-ranked documents to refine the search.
		* Particularly useful for short/ambiguous queries.
	* Neural Query Expansion (NQE)

---
### 2. Retrieval Stage

These techniques impact how documents are retrieved in response to a query.

#### 🔹 Hybrid Retrieval

* Combines multiple retrieval strategies:
	* **Sparse (keyword) retrieval** (e.g., BM25, TF-IDF)
	* **Dense (semantic) retrieval** (e.g. DPR \[Dense Passage Retrieval], ColBERT, Contriever)
		* **Similarity metric**: Cosine similarity, dot product, or Euclidean distance.
* **Fusion techniques**: Reciprocal Rank Fusion (RRF), Linear Interpolation, or Learning-to-Rank

- Notes:
	- **Dense (Semantic) Retrieval**
		* Uses vector embeddings to match semantics.
		* Models: DPR, Contriever, ColBERT
		* **Similarity metrics**: cosine, dot product, Euclidean.
		  
	-  **Sparse Neural Retrieval**
		  * Deep models that produce **sparse representations** of queries/documents.
		  * Example: **SPLADE**
		  * Bridges classical IR and neural models.

#### 🔹 Personalized Retrieval

* Adapts results based on user profile, preferences, or historical behavior.
* Uses **joint embeddings** of query and user context  for contextual retrieval.
* Useful in recommender systems.

#### 🔹 Context-aware / Session-based Retrieval

* Includes recent user interactions or session history in retrieval context.
* Useful in conversational agents.

#### 🔹 Temporal Retrieval

* Weights documents based on **recency** or temporal relevance.

#### 🔹 Graph-based Retrieval

* Constructs a knowledge graph or entity-relation graph from documents.
* Retrieval is performed through **graph traversals**, **embedding-based search**, or **subgraph matching**.
* Suitable for reasoning-based queries.
* Useful in biomedical or legal domains.
* Example: **Neural Graph Retrieval**, **GNN-enhanced IR**

#### 🔹 Knowledge-Augmented Retrieval

* Retrieves from or augments using structured knowledge like:
	* Knowledge graphs
	* Ontologies
	* External databases
* Can improve factual grounding and reasoning.

#### 🔹 Passage Retrieval

* Retrieves **fine-grained** short, relevant passages instead of whole documents.
* Improves precision for QA tasks.
* Example: **Dense Passage Retrieval (DPR)**

#### 🔹 Federated / Multi-source Retrieval

* Retrieves from multiple heterogeneous sources or indices or databases.
* Combines and ranks results across them.

#### 🔹 Multi-hop Retrieval

* Retrieves chains of documents where reasoning spans multiple sources.
* Used in complex QA or reasoning.

#### 🔹 Multi-vector Retrieval

* Represents documents with multiple embeddings (e.g., per token/chunk).
* Late interaction improves fine-grained relevance.
* Examples: **ColBERT**, **TAS-B**

#### 🔹 Multimodal Retrieval

* Retrieves documents based on queries from different modalities (text, image, audio).
* Example: Cross-modal search using text queries for image retrieval.
* Useful in e-commerce, medicine, or multimedia search.

---

### 3. Post-Retrieval Reranking

Methods to improve the **ranking** of retrieved documents.
#### 🔹 Cross-Encoder / Reranking

* After initial retrieval (sparse or dense), a cross-encoder (e.g., BERT-based model) reranks top-k results by computing **deep pairwise similarity** between the query and each document.
	* **Models**: `MonoT5`, `CrossEncoder-BERT`
	* **Input**: `[CLS] query [SEP] document` as input
* Improves precision after initial retrieval.

#### 🔹 Maximal Marginal Relevance (MMR)

* Promotes **diversity** and reduces **redundancy** in retrieved docs.
* Prevents redundancy by penalizing similar documents.
* Balances relevance vs novelty.

#### 🔹 Learning to Rank (LTR)

* Trains models using supervised relevance data to rank documents based on relevance features.
* Algorithms:
	* **RankNet**
	* **LambdaMART**
	* **XGBoost Ranker**

---

### 4. Result Aggregation & Filtering

Techniques to finalize or filter the set of retrieved results before generation.

#### 🔹 Metadata Filtering / Faceted Search

* Filters documents based on metadata (e.g., date, author, topic, tags).
* Efficient for narrowing down search space.
* Common in vector DBs like **Pinecone**, **FAISS**, **Weaviate**, **Qdrant**

---

### 5. Training & Embedding Optimization

Training approaches that improve **retriever model quality**:

#### 🔹 Contrastive Retrieval Learning

* Improves dense retrievers by distinguishing positive vs negative pairs.
* Frameworks: **SimCSE**, **DPR**, **CLIP** (for vision-language)

#### 🔹 Asymmetric Retrieval

* Uses **different encoders** for queries and documents.
* Found in **dual-encoder systems** like CLIP.

---

### 📌 Summary Table

| Stage                     | Techniques                                                                                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Query Preparation**     | Query Expansion, Neural PRF                                                                                                                                             |
| **Retrieval**             | Hybrid, Dense, Sparse Neural (SPLADE), Multi-hop, Passage, Context-aware, Personalized, Temporal, Graph-based, Federated, Multimodal, Knowledge-Augmented, Multi-vector |
| **Reranking**             | Cross-Encoder, MMR, Learning to Rank                                                                                                                                    |
| **Aggregation/Filtering** | Metadata Filtering                                                                                                                                                      |
| **Training Enhancements** | Contrastive Learning, Asymmetric Retrieval                                                                                                                              |

---

#### ⛓️ Retrieval with External Tools or Agents

* Uses tools like **search engines, APIs, or calculators** during retrieval.
* Common in **Agentic RAG** or **Tool-augmented LLMs**.

#### ⚙️ Efficient Retrieval Infrastructure

* ***Approximate Nearest Neighbor (ANN) Search**
  * Scalable vector search using tools like **FAISS**, **ScaNN**, **HNSW**, **Milvus**, etc.

---

