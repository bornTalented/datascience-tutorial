**MMR (Maximal Marginal Relevance)** is not a *distance metric* like cosine or dot product, but it **enhances vector search** by combining **relevance and diversity** during retrieval.

---

## 🧠 What is MMR (Maximal Marginal Relevance)?

MMR is a **re-ranking strategy** used after retrieving top-k vectors based on similarity.

It balances:

* **Relevance** to the query (e.g., cosine similarity)
* **Diversity** among the results (so you don’t get redundant answers)

---

### MMR Formula

Let:

* `Q` be the query embedding
* `D` be the set of candidate documents
* `S` be the selected documents so far

MMR chooses the next document `d ∈ D \ S` that maximizes:

$$
\text{MMR}(d) = \lambda \cdot \text{sim}(Q, d) - (1 - \lambda) \cdot \max_{s \in S} \text{sim}(d, s)
$$

Where:

* `sim(Q, d)` = similarity between query and document (e.g., cosine)
* `sim(d, s)` = similarity between this doc and already selected ones
* `λ` ∈ \[0, 1] = trade-off parameter

  * λ = 1 → only relevance
  * λ = 0 → only diversity

---

## ✅ Use Case

Let’s say you ask: **"How can I cancel my subscription?"**
A vanilla cosine similarity search might return:

* Doc 1: Cancel via mobile app
* Doc 2: Cancel via desktop
* Doc 3: Cancel via app again (duplicate)

MMR avoids showing duplicates by penalizing very similar results.

---

## 🔍 When to Use MMR

| Use Case            | Benefit                                             |
| ------------------- | --------------------------------------------------- |
| RAG-style pipelines | Ensures **varied** supporting contexts              |
| Semantic search     | Improves **user experience** by reducing redundancy |
| Chat memory search  | Prevents repetition in memory recall                |
| Question answering  | Surfaces **diverse perspectives** on a topic        |

---

## 🚀 MMR in LangChain

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10, # Final number of documents returned
        "fetch_k": 25,  # Number of initial candidates fetched before filtering
        "lambda_mult": 0.5  # balance between relevance(1) and diversity(0)
    }
)
```

---

## ⚖️ MMR vs Distance Metrics

| Feature                   | Distance Metric (e.g., cosine) | MMR                         |
| ------------------------- | ------------------------------ | --------------------------- |
| Purpose                   | Score relevance                | Rank top-k with diversity   |
| Works on                  | Pairs (query, doc)             | Entire result set           |
| Replaces distance metric? | ❌ No                           | ❌ No (works on top of it)   |
| Helps reduce redundancy?  | ❌ No                           | ✅ Yes                       |
| Needs tuning?             | Not much                       | λ (relevance vs. diversity) |

---

### ✅ TL;DR

* MMR is **not an alternative metric** but an **alternative retrieval strategy**.
* It improves search results by balancing **relevance and diversity**.
* You still need a **base similarity function** (like cosine or dot product).
* Great for **RAG**, **chatbots**, and **document QA** tasks.

---

Would you like a notebook that:

* Uses **LangChain + FAISS** with `search_type="mmr"`?
* Shows how MMR retrieval results differ from default cosine-based search?

Let me know, and I’ll prepare it.


Perfect! Here's a **LangChain + FAISS + OpenAI Embeddings notebook** that demonstrates:

> ✅ Retrieval with Cosine Similarity
> ✅ Retrieval with **MMR (Maximal Marginal Relevance)**
> ✅ Comparison of results to show how MMR improves diversity

---

## 📘 MMR Retrieval with LangChain + FAISS + OpenAI

### 🧪 Install dependencies (if not done already)

```bash
pip install openai langchain faiss-cpu tiktoken
```

---

### ✅ Python Notebook Code

```python
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Step 1: Sample documents (intentionally similar for MMR test)
docs = [
    Document(page_content="Cancel your subscription from mobile settings."),
    Document(page_content="Cancel your subscription from desktop settings."),
    Document(page_content="Refund requests are processed in 5-7 days."),
    Document(page_content="To cancel, go to your account > subscription."),
    Document(page_content="You can cancel your subscription anytime."),
    Document(page_content="Refunds are not available for yearly plans."),
    Document(page_content="Mobile app allows cancellation under profile section."),
]

# Step 2: Create embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 3: Index with FAISS
vector_store = FAISS.from_documents(docs, embedding_model)

# Step 4: Define retrievers
query = "How can I cancel my subscription?"

# a) Default (Cosine Similarity)
default_results = vector_store.similarity_search(query, k=4)

# b) MMR-based Retriever
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 8, "lambda_mult": 0.5}
)
mmr_results = mmr_retriever.get_relevant_documents(query)

# Step 5: Print and Compare Results
print("\n🔎 DEFAULT COSINE SIMILARITY RESULTS:")
for i, doc in enumerate(default_results, 1):
    print(f"{i}. {doc.page_content}")

print("\n🌈 MMR-ENHANCED DIVERSE RESULTS:")
for i, doc in enumerate(mmr_results, 1):
    print(f"{i}. {doc.page_content}")
```

---

### 🧠 What You Should Observe:

* The **default cosine-based results** often return **redundant answers** (e.g., multiple similar “cancel via settings” lines).
* The **MMR results** show a **more varied set** (cancel steps + refund info) because it penalizes similar documents.

---

### ⚙️ Optional Tuning

You can adjust `lambda_mult`:

* `λ = 1.0`: only relevance (same as cosine)
* `λ = 0.0`: only diversity
* `λ = 0.5`: balanced (default)

Try changing `fetch_k` to see how pulling more candidates affects diversity.

---
Maximal Marginal Relevance (MMR) is a widely used algorithm, particularly in information retrieval and summarization, designed to select items (e.g., documents, sentences, keywords) that are both **relevant to a query** and **diverse from previously selected items**.

The core problem MMR addresses is that traditional relevance-ranking systems often return a list of items that are highly relevant but also highly redundant. For example, if you search for "best vacation spots," a pure relevance-based search might give you ten articles all about beaches in Greece. While relevant, this doesn't provide a comprehensive overview of "best vacation spots." MMR aims to solve this by introducing diversity.

Here's a breakdown of the concept:

* **Balancing Relevance and Diversity:** MMR operates on the principle of maximizing a "marginal relevance" score for each candidate item. This score is a weighted combination of:
    * **Relevance to the query:** How similar the candidate item is to the user's initial query. This is typically measured using a similarity metric like cosine similarity between embeddings of the query and the item.
    * **Dissimilarity to already selected items:** How different the candidate item is from the items that have already been chosen and added to the result set. This is usually measured by finding the maximum similarity between the candidate item and any of the already selected items, and then penalizing for it.

* **The MMR Formula:** The general formula for MMR is:

    $$\text{MMR}(D_i) = \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in R} \text{Sim}_2(D_i, D_j)$$

    Where:
    * $D_i$: The candidate document/item being considered.
    * $Q$: The query.
    * $R$: The set of documents/items already selected.
    * $\text{Sim}_1(D_i, Q)$: The similarity between the candidate item $D_i$ and the query $Q$.
    * $\text{Sim}_2(D_i, D_j)$: The similarity between the candidate item $D_i$ and an already selected item $D_j$. (Often, $\text{Sim}_1$ and $\text{Sim}_2$ are the same similarity metric, e.g., cosine similarity).
    * $\lambda$ (lambda): A tunable parameter between 0 and 1.
        * If $\lambda = 1$: MMR focuses purely on relevance, acting like a standard relevance-ranked list.
        * If $\lambda = 0$: MMR focuses purely on diversity, selecting the most dissimilar items.
        * If $0 < \lambda < 1$: MMR balances relevance and diversity. A higher $\lambda$ emphasizes relevance, while a lower $\lambda$ emphasizes diversity.

* **Iterative Selection Process:** MMR works iteratively:
    1.  **Initial Selection:** The first item selected is typically the one with the highest pure relevance to the query.
    2.  **Subsequent Selection:** In each subsequent step, from the remaining unselected items, the algorithm calculates the MMR score for each. The item with the highest MMR score is then added to the result set.
    3.  **Repetition:** This process repeats until a desired number of items have been selected.

**Why is MMR useful?**

* **Reduces Redundancy:** Prevents the result set from being dominated by very similar pieces of information.
* **Improves Coverage/Comprehensiveness:** Ensures a broader range of relevant information is presented to the user.
* **Enhanced User Experience:** For many applications, users prefer a diverse set of results that cover different aspects of their query.
* **Applications:** MMR is widely used in:
    * **Search engines and recommender systems:** To provide more varied and useful results.
    * **Document summarization:** To select sentences or passages that are both relevant to the document's topic and non-redundant.
    * **Retrieval-Augmented Generation (RAG) in LLMs:** To select diverse and relevant context for large language models, preventing repetitive information from being fed into the limited context window, leading to more comprehensive and nuanced answers.
    * **Keyphrase extraction:** To identify a diverse set of representative keywords for a document.

By allowing a tunable balance between relevance and diversity, MMR offers a powerful approach to delivering more effective and informative retrieval and summarization results.

---

Okay, let's walk through a dummy example to illustrate how Maximal Marginal Relevance (MMR) works.

**Scenario:** Imagine you're building a simple search engine for blog posts, and a user searches for the query: **"Healthy Breakfast Ideas"**

Let's assume we have a pool of 5 blog posts that our initial relevance-ranking algorithm has identified as potentially relevant (ranked by initial relevance score):

- **Document D1:** "Quick & Easy Oatmeal Recipes for a Healthy Start" (Relevance Score: 0.9)
    
- **Document D2:** "Mediterranean Diet Breakfasts: Shakshuka and More" (Relevance Score: 0.8)
    
- **Document D3:** "Top 10 High-Protein Smoothies for Fitness Buffs" (Relevance Score: 0.75)
    
- **Document D4:** "Benefits of Oatmeal for Heart Health" (Relevance Score: 0.7)
    
- **Document D5:** "Traditional Indian Breakfasts: Poha and Upma" (Relevance Score: 0.6)
    

For simplicity, we'll use a very basic similarity metric (e.g., based on shared keywords or conceptual overlap) and a λ value of **0.7** (emphasizing relevance slightly more than diversity).

**Similarity Assumptions (Hypothetical Values):**

Let's define a similarity function Sim(A,B) where a higher number means more similar.

- Sim(Oatmeal Recipes,Oatmeal Benefits)=0.8 (very similar)
    
- Sim(Oatmeal Recipes,Smoothies)=0.3 (somewhat different)
    
- Sim(Oatmeal Recipes,Mediterranean Breakfasts)=0.2 (quite different)
    
- Sim(Oatmeal Recipes,Indian Breakfasts)=0.1 (very different)
    

We'll assume similar logic for other pairs. The critical point is that highly related documents will have high Sim2​ values.

MMR Formula:

$$\text{MMR}(D_i) = \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in R} \text{Sim}_2(D_i, D_j)$$

Given λ=0.7:

$$\text{MMR}(D_i) = (0.7) \cdot \text{Sim}_1(D_i, Q) - (0.3) \cdot \max_{D_j \in R} \text{Sim}_2(D_i, D_j)$$
---

**Step-by-Step Working:**

**Goal: Select 3 diverse and relevant breakfast ideas.**

**Iteration 1: Selecting the first document**

- When R (the set of selected documents) is empty, the maxDj​∈R​Sim2​(Di​,Dj​) term is effectively 0 or ignored.
    
- So, the first document is simply the one with the highest initial relevance score.
    
    - **D1:** "Quick & Easy Oatmeal Recipes for a Healthy Start" (Relevance: 0.9)
        
    - **D2:** "Mediterranean Diet Breakfasts: Shakshuka and More" (Relevance: 0.8)
        
    - **D3:** "Top 10 High-Protein Smoothies for Fitness Buffs" (Relevance: 0.75)
        
    - **D4:** "Benefits of Oatmeal for Heart Health" (Relevance: 0.7)
        
    - **D5:** "Traditional Indian Breakfasts: Poha and Upma" (Relevance: 0.6)
        
- **Result of Iteration 1: D1 is selected.**
    
- `Selected Set (R) = {D1}`
    
- `Remaining Candidates = {D2, D3, D4, D5}`
    

---

**Iteration 2: Selecting the second document**

Now we calculate the MMR score for each remaining candidate, considering its similarity to the query (Q) and its dissimilarity to the already selected `D1`.

- **For D2 ("Mediterranean Diet Breakfasts"):**
    
    - Sim1​(D2,Q)=0.8 (given initial relevance)
        
    - maxDj​∈{D1}​Sim2​(D2,Dj​)=Sim(D2,D1)=Sim(Mediterranean,Oatmeal)=0.2 (relatively low similarity, good for diversity)
        
    - MMR(D2)=(0.7⋅0.8)−(0.3⋅0.2)=0.56−0.06=0.50
        
- **For D3 ("Top 10 High-Protein Smoothies"):**
    
    - Sim1​(D3,Q)=0.75
        
    - maxDj​∈{D1}​Sim2​(D3,Dj​)=Sim(D3,D1)=Sim(Smoothies,Oatmeal)=0.3
        
    - MMR(D3)=(0.7⋅0.75)−(0.3⋅0.3)=0.525−0.09=0.435
        
- **For D4 ("Benefits of Oatmeal for Heart Health"):**
    
    - Sim1​(D4,Q)=0.7
        
    - maxDj​∈{D1}​Sim2​(D4,Dj​)=Sim(D4,D1)=Sim(Oatmeal Benefits,Oatmeal Recipes)=0.8 (HIGH similarity, this will be penalized heavily)
        
    - MMR(D4)=(0.7⋅0.7)−(0.3⋅0.8)=0.49−0.24=0.25
        
- **For D5 ("Traditional Indian Breakfasts"):**
    
    - Sim1​(D5,Q)=0.6
        
    - maxDj​∈{D1}​Sim2​(D5,Dj​)=Sim(D5,D1)=Sim(Indian,Oatmeal)=0.1 (very low similarity, good for diversity but low relevance)
        
    - MMR(D5)=(0.7⋅0.6)−(0.3⋅0.1)=0.42−0.03=0.39
        
- **Comparing MMR scores:**
    
    - D2: 0.50
        
    - D3: 0.435
        
    - D4: 0.25
        
    - D5: 0.39
        
- **Result of Iteration 2: D2 is selected.** It has a good balance of relevance (0.8) and is relatively diverse from D1 (0.2 similarity).
    
- `Selected Set (R) = {D1, D2}`
    
- `Remaining Candidates = {D3, D4, D5}`
    

---

**Iteration 3: Selecting the third document**

Now we calculate MMR scores, considering similarity to both D1 and D2. The maxDj​∈R​Sim2​(Di​,Dj​) term will take the highest similarity to _either_ D1 or D2.

- **For D3 ("Top 10 High-Protein Smoothies"):**
    
    - Sim1​(D3,Q)=0.75
        
    - maxDj​∈{D1,D2}​Sim2​(D3,Dj​)=max(Sim(Smoothies,Oatmeal),Sim(Smoothies,Mediterranean))
        
        - Let's assume Sim(Smoothies,Mediterranean)=0.4 (perhaps both are "healthy" but different types of meals).
            
        - So, max(0.3,0.4)=0.4
            
    - MMR(D3)=(0.7⋅0.75)−(0.3⋅0.4)=0.525−0.12=0.405
        
- **For D4 ("Benefits of Oatmeal for Heart Health"):**
    
    - Sim1​(D4,Q)=0.7
        
    - maxDj​∈{D1,D2}​Sim2​(D4,Dj​)=max(Sim(Oatmeal Benefits,Oatmeal Recipes),Sim(Oatmeal Benefits,Mediterranean))
        
        - Assume Sim(Oatmeal Benefits,Mediterranean)=0.1
            
        - So, max(0.8,0.1)=0.8 (still highly similar to D1)
            
    - MMR(D4)=(0.7⋅0.7)−(0.3⋅0.8)=0.49−0.24=0.25
        
- **For D5 ("Traditional Indian Breakfasts"):**
    
    - Sim1​(D5,Q)=0.6
        
    - maxDj​∈{D1,D2}​Sim2​(D5,Dj​)=max(Sim(Indian,Oatmeal),Sim(Indian,Mediterranean))
        
        - Assume Sim(Indian,Mediterranean)=0.15
            
        - So, max(0.1,0.15)=0.15
            
    - MMR(D5)=(0.7⋅0.6)−(0.3⋅0.15)=0.42−0.045=0.375
        
- **Comparing MMR scores:**
    
    - D3: 0.405
        
    - D4: 0.25
        
    - D5: 0.375
        
- **Result of Iteration 3: D3 is selected.** It's reasonably relevant and adds a new category (smoothies) that is somewhat distinct from oatmeal and Mediterranean dishes.
    
- `Selected Set (R) = {D1, D2, D3}`
    

---

**Final Result (Top 3):**

1. **D1:** "Quick & Easy Oatmeal Recipes for a Healthy Start" (Initial relevance: 0.9)
    
2. **D2:** "Mediterranean Diet Breakfasts: Shakshuka and More" (Initial relevance: 0.8)
    
3. **D3:** "Top 10 High-Protein Smoothies for Fitness Buffs" (Initial relevance: 0.75)
    

**Comparison with Pure Relevance Ranking:**

If we had just gone by pure relevance, the top 3 would be:

1. D1: "Quick & Easy Oatmeal Recipes for a Healthy Start" (0.9)
    
2. D2: "Mediterranean Diet Breakfasts: Shakshuka and More" (0.8)
    
3. D3: "Top 10 High-Protein Smoothies for Fitness Buffs" (0.75)
    
    (Wait, it's the same in this specific dummy example, why? Let's tweak D4's initial relevance slightly to make the difference clear.)
    

**Let's re-run with a slight change to D4's initial relevance to better demonstrate the "redundancy" problem:**

- **Document D1:** "Quick & Easy Oatmeal Recipes for a Healthy Start" (Relevance Score: **0.9**)
    
- **Document D4:** "Benefits of Oatmeal for Heart Health" (Relevance Score: **0.85**) - _Higher relevance now!_
    
- **Document D2:** "Mediterranean Diet Breakfasts: Shakshuka and More" (Relevance Score: 0.8)
    
- **Document D3:** "Top 10 High-Protein Smoothies for Fitness Buffs" (Relevance Score: 0.75)
    
- **Document D5:** "Traditional Indian Breakfasts: Poha and Upma" (Relevance Score: 0.6)
    

**Pure Relevance Ranking (original order based on scores):**

1. D1 (0.9)
    
2. D4 (0.85)
    
3. D2 (0.8)
    
    (Notice: D1 and D4 are both about oatmeal, leading to redundancy)
    

---

**MMR Re-run (with D4's higher initial relevance):**

**Iteration 1: Select D1.**

- `Selected Set (R) = {D1}`
    
- `Remaining Candidates = {D2, D3, D4, D5}`
    

**Iteration 2: Calculate MMR for remaining:**

- **For D2 ("Mediterranean Diet Breakfasts"):**
    
    - Sim1​(D2,Q)=0.8
        
    - maxDj​∈{D1}​Sim2​(D2,Dj​)=Sim(D2,D1)=0.2
        
    - MMR(D2)=(0.7⋅0.8)−(0.3⋅0.2)=0.56−0.06=0.50
        
- **For D3 ("Top 10 High-Protein Smoothies"):**
    
    - Sim1​(D3,Q)=0.75
        
    - maxDj​∈{D1}​Sim2​(D3,Dj​)=Sim(D3,D1)=0.3
        
    - MMR(D3)=(0.7⋅0.75)−(0.3⋅0.3)=0.525−0.09=0.435
        
- **For D4 ("Benefits of Oatmeal for Heart Health"):**
    
    - Sim1​(D4,Q)=0.85 (Higher now!)
        
    - maxDj​∈{D1}​Sim2​(D4,Dj​)=Sim(D4,D1)=0.8 (Still HIGH similarity to D1)
        
    - MMR(D4)=(0.7⋅0.85)−(0.3⋅0.8)=0.595−0.24=0.355
        
- **For D5 ("Traditional Indian Breakfasts"):**
    
    - Sim1​(D5,Q)=0.6
        
    - maxDj​∈{D1}​Sim2​(D5,Dj​)=Sim(D5,D1)=0.1
        
    - MMR(D5)=(0.7⋅0.6)−(0.3⋅0.1)=0.42−0.03=0.39
        
- **Comparing MMR scores:**
    
    - D2: 0.50
        
    - D3: 0.435
        
    - D4: 0.355
        
    - D5: 0.39
        
- **Result of Iteration 2: D2 is selected.** (Still the highest MMR)
    
- `Selected Set (R) = {D1, D2}`
    
- `Remaining Candidates = {D3, D4, D5}`
    

**Iteration 3: Calculate MMR for remaining (D3, D4, D5) based on D1 and D2 being in R:**

- **For D3 ("Top 10 High-Protein Smoothies"):**
    
    - Sim1​(D3,Q)=0.75
        
    - maxDj​∈{D1,D2}​Sim2​(D3,Dj​)=max(Sim(Smoothies,Oatmeal)=0.3,Sim(Smoothies,Mediterranean)=0.4)=0.4
        
    - MMR(D3)=(0.7⋅0.75)−(0.3⋅0.4)=0.525−0.12=0.405
        
- **For D4 ("Benefits of Oatmeal for Heart Health"):**
    
    - Sim1​(D4,Q)=0.85
        
    - maxDj​∈{D1,D2}​Sim2​(D4,Dj​)=max(Sim(Oatmeal Benefits,Oatmeal Recipes)=0.8,Sim(Oatmeal Benefits,Mediterranean)=0.1)=0.8
        
    - MMR(D4)=(0.7⋅0.85)−(0.3⋅0.8)=0.595−0.24=0.355
        
- **For D5 ("Traditional Indian Breakfasts"):**
    
    - Sim1​(D5,Q)=0.6
        
    - maxDj​∈{D1,D2}​Sim2​(D5,Dj​)=max(Sim(Indian,Oatmeal)=0.1,Sim(Indian,Mediterranean)=0.15)=0.15
        
    - MMR(D5)=(0.7⋅0.6)−(0.3⋅0.15)=0.42−0.045=0.375
        
- **Comparing MMR scores:**
    
    - D3: 0.405
        
    - D4: 0.355
        
    - D5: 0.375
        
- **Result of Iteration 3: D3 is selected.**
    

---

**Final Result (Top 3 with MMR, after D4's relevance change):**

1. **D1:** "Quick & Easy Oatmeal Recipes for a Healthy Start"
    
2. **D2:** "Mediterranean Diet Breakfasts: Shakshuka and More"
    
3. **D3:** "Top 10 High-Protein Smoothies for Fitness Buffs"
    

**The crucial difference with MMR:** Even though **D4 ("Benefits of Oatmeal for Heart Health")** had a very high initial relevance score (0.85) and would have been selected second in a purely relevance-based system, MMR significantly penalized it because it was too similar to D1, which was already selected. This allowed **D2 ("Mediterranean Diet Breakfasts")** and **D3 ("Top 10 High-Protein Smoothies")** to be chosen instead, providing a more diverse set of "Healthy Breakfast Ideas."

This example clearly shows how MMR actively works to reduce redundancy and increase the overall comprehensiveness of the retrieved results by balancing relevance with diversity.