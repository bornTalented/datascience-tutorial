## Naive Bayes Algorithm

Naive Bayes is a supervised machine learning algorithm used for classification tasks, particularly in natural language processing (NLP) and text classification. It's based on Bayes' theorem with a "naive" assumption of independence among predictors.

Here's a breakdown of the key concepts:

### 1. Bayes' Theorem

At its core, Naive Bayes relies on Bayes' theorem, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event. The formula is:

$P(A|B) = P(B|A) * P(A) / P(B)$

Where:
* $P(A|B)$ (Posterior Probability): The probability of hypothesis A given the evidence B. This is what we want to calculate.
* $P(B|A)$ (Likelihood): The probability of evidence B given that hypothesis A is true.
* $P(A)$ (Prior Probability): The probability of hypothesis A being true independently of the evidence.
* $P(B)$ (Evidence Probability): The probability of evidence B being true independently of the hypothesis.

### 2. The "Naive" Assumption

The "naive" part of Naive Bayes comes from its simplifying assumption: **it assumes that all features (predictors) are independent of each other given the class variable.**

In reality, this assumption is rarely perfectly true. For example, if you're classifying emails as spam or not spam, words like "Viagra" and "prescription" are probably not truly independent; they often appear together in spam emails. However, despite this strong assumption, Naive Bayes often performs surprisingly well, especially with large datasets.

### 3. How it Works for Classification

Let's say you want to classify a new data point (e.g., an email) into one of several classes (e.g., "spam" or "not spam"). Naive Bayes calculates the probability of that data point belonging to each class and then assigns it to the class with the highest probability.

To do this, it effectively calculates:

$P(\text{Class} | \text{Features}) = P(\text{Features} | \text{Class}) * P(\text{Class}) / P(\text{Features})$

Since $P(\text{Features})$ is constant for all classes, we can simplify this to:

$P(\text{Class} | \text{Features}) \propto P(\text{Features} | \text{Class}) * P(\text{Class})$

Due to the independence assumption, $P(\text{Features} | \text{Class})$ can be broken down into the product of individual feature probabilities:

$P(\text{Features} | \text{Class}) = P(\text{Feature}_1 | \text{Class}) * P(\text{Feature}_2 | \text{Class}) * ... * P(\text{Feature}_n | \text{Class})$

So, the full equation becomes:

$P(\text{Class} | \text{Feature}_1, ..., \text{Feature}_n) \propto P(\text{Class}) * \prod_{i=1}^{n} P(\text{Feature}_i | \text{Class})$

### 4. Training the Model

During the training phase, the algorithm learns these probabilities from the training data:

* **Prior Probabilities ($P(\text{Class})$):** The probability of each class occurring in the training data. This is simply the count of instances in that class divided by the total number of instances.
* **Likelihoods ($P(\text{Feature}_i | \text{Class})$):** The probability of each feature appearing given a particular class. For example, in text classification, this would be the probability of a specific word appearing in a "spam" email versus a "not spam" email. This is calculated by counting the occurrences of the feature within each class and normalizing.

### 5. Classification of New Data

When a new, unseen data point arrives, the algorithm:

1.  Calculates $P(\text{Class}) * \prod_{i=1}^{n} P(\text{Feature}_i | \text{Class})$ for each possible class.
2.  Assigns the new data point to the class that yields the highest probability.

### Types of Naive Bayes Classifiers

There are several variations of Naive Bayes, primarily differing in how they estimate $P(\text{Feature}_i | \text{Class})$:

* **Gaussian Naive Bayes:** Used when features have continuous values and are assumed to follow a Gaussian (normal) distribution.
* **Multinomial Naive Bayes:** Primarily used for discrete counts, like word counts in text classification. It's well-suited for document classification problems.
* **Bernoulli Naive Bayes:** Designed for binary/boolean features (features that are either present or absent). It counts how many times a feature is present or absent in a particular class.

### Advantages of Naive Bayes

* **Simple and Fast:** Easy to implement and computationally efficient, making it suitable for large datasets.
* **Good for High-Dimensional Data:** Performs well with a large number of features, especially in text classification.
* **Handles Missing Data (Implicitly):** Can sometimes handle missing data gracefully because it only considers the probabilities of features that are present.
* **Requires Less Training Data (relatively):** Can perform reasonably well even with a smaller training dataset compared to some other complex algorithms.

### Disadvantages of Naive Bayes

* **Strong Independence Assumption:** The "naive" assumption of feature independence can lead to inaccurate probability estimates if the features are highly correlated. However, it often still yields good classification results.
* **Zero-Frequency Problem:** If a particular feature-value combination doesn't appear in the training data for a certain class, the probability for that combination will be zero. This can cause the entire posterior probability to become zero, regardless of other features. (This is often addressed using smoothing techniques like Laplace smoothing.)
* **Poor Estimator of Probabilities:** While it's a good classifier, the actual probability outputs might not be very accurate or well-calibrated.

### Real-World Applications

* **Spam Filtering:** Classifying emails as spam or not spam.
* **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) of text.
* **Text Classification:** Categorizing documents into different topics.
* **Medical Diagnosis:** Predicting the likelihood of a disease based on symptoms.
* **Recommendation Systems:** Suggesting products or content to users.

In summary, Naive Bayes is a powerful and efficient classification algorithm, especially in areas like text analysis, due to its simplicity and ability to handle high-dimensional data, despite its strong independence assumption.

---

I understand you'd like a dummy example to illustrate the working of the Naive Bayes algorithm. Let's use a classic example: **classifying whether a fruit is an "Orange" or an "Apple" based on its features.**

### Dummy Example: Fruit Classification

**Our Goal:** To classify a new fruit as either an "Orange" or an "Apple" based on its color and shape.

**Features:**
* **Color:** Red, Green, Orange
* **Shape:** Round, Oval

**Classes:**
* **Apple**
* **Orange**

---

### Step 1: Training Data

First, we need some training data to learn the probabilities. Let's assume we have the following 10 fruits:

| Fruit ID | Color  | Shape | Class   |
| :------- | :----- | :---- | :------ |
| 1        | Red    | Round | Apple   |
| 2        | Red    | Round | Apple   |
| 3        | Red    | Oval  | Apple   |
| 4        | Green  | Round | Apple   |
| 5        | Orange | Round | Orange  |
| 6        | Orange | Round | Orange  |
| 7        | Orange | Oval  | Orange  |
| 8        | Red    | Oval  | Apple   |
| 9        | Green  | Oval  | Apple   |
| 10       | Orange | Round | Orange  |

---

### Step 2: Calculate Prior Probabilities ($P(\text{Class})$)

These are the probabilities of each class occurring in our training data.

* **Total Fruits:** 10
* **Apples:** 6
* **Oranges:** 4

Therefore:
* $P(\text{Apple}) = \text{Count of Apples} / \text{Total Fruits} = 6 / 10 = 0.6$
* $P(\text{Orange}) = \text{Count of Oranges} / \text{Total Fruits} = 4 / 10 = 0.4$

---

### Step 3: Calculate Likelihoods ($P(\text{Feature} | \text{Class})$)

This is where we look at the probability of each feature value given each class.

**For Class: Apple** (Total 6 Apples)

| Color  | Count (Apple) | $P(\text{Color}\|\text{Apple})$ |
| :----- | :------------ | :------------------------------ |
| Red    | 3             | $3/6 = 0.5$                     |
| Green  | 2             | $2/6 = 0.33$                    |
| Orange | 1             | $1/6 = 0.17$                    |

| Shape | Count (Apple) | $P(\text{Shape}\|\text{Apple})$ |
| :---- | :------------ | :------------------------------ |
| Round | 3             | $3/6 = 0.5$                     |
| Oval  | 3             | $3/6 = 0.5$                     |

**For Class: Orange** (Total 4 Oranges)

| Color  | Count (Orange) | $P(\text{Color} \| \text{Orange})$ |
| :----- | :------------- | :--------------------------------- |
| Red    | 0              | $0/4 = 0$                          |
| Green  | 0              | $0/4 = 0$                          |
| Orange | 4              | $4/4 = 1$                          |

| Shape | Count (Orange) | $P(\text{Shape} \| \text{Orange})$ |
| :---- | :------------- | :--------------------------------- |
| Round | 3              | $3/4 = 0.75$                       |
| Oval  | 1              | $1/4 = 0.25$                       |

---

### Step 4: Classify a New Fruit

Now, let's say we have a **new fruit** with the following features:

* **Color:** **Orange**
* **Shape:** **Oval**

We want to predict if it's an "Apple" or an "Orange". We use the formula:

$P(\text{Class} | \text{Features}) \propto P(\text{Class}) * P(\text{Color} | \text{Class}) * P(\text{Shape} | \text{Class})$

#### Calculation for **Apple**:

$P(\text{Apple} | \text{Color=Orange, Shape=Oval}) \propto P(\text{Apple}) * P(\text{Color=Orange} | \text{Apple}) * P(\text{Shape=Oval} | \text{Apple})$

$P(\text{Apple} | \text{new fruit}) \propto 0.6 * 0.17 * 0.5$
$P(\text{Apple} | \text{new fruit}) \propto 0.051$

#### Calculation for **Orange**:

$P(\text{Orange} | \text{Color=Orange, Shape=Oval}) \propto P(\text{Orange}) * P(\text{Color=Orange} | \text{Orange}) * P(\text{Shape=Oval} | \text{Orange})$

$P(\text{Orange} | \text{new fruit}) \propto 0.4 * 1 * 0.25$
$P(\text{Orange} | \text{new fruit}) \propto 0.1$

---

### Step 5: Make a Prediction

Compare the two calculated values:

* $P(\text{Apple} | \text{new fruit}) \propto 0.051$
* $P(\text{Orange} | \text{new fruit}) \propto 0.1$

Since $0.1 > 0.051$, the Naive Bayes algorithm would classify the new fruit as an **Orange**.

---

### Addressing the Zero-Frequency Problem (Important Note)

In our example, notice that $P(\text{Color=Red} | \text{Orange})$ and $P(\text{Color=Green} | \text{Orange})$ were both 0. If our new fruit had been "Red" and "Oval", then $P(\text{Orange} | \text{new fruit})$ would become 0, even if other factors strongly suggested it was an orange.

To avoid this, a common technique called **Laplace Smoothing** (or add-one smoothing) is used. It involves adding a small constant (usually 1) to all counts so that no probability ever becomes zero.

Let's quickly see how it would affect our calculations for $P(\text{Color=Orange} | \text{Apple})$:

* Without smoothing: $P(\text{Color=Orange} | \text{Apple}) = 1/6$
* With Laplace smoothing (k=1):
    * Count of "Orange" color in Apples: 1
    * Number of possible "Color" values: 3 (Red, Green, Orange)
    * Total Apples: 6
    * $P(\text{Color=Orange} | \text{Apple}) = (1 + 1) / (6 + 3) = 2 / 9 \approx 0.22$

This ensures that even unseen feature combinations have a small, non-zero probability, preventing the entire product from becoming zero. For simplicity, I omitted it in the main example, but it's crucial in real-world implementations.

This dummy example illustrates the core steps of how the Naive Bayes algorithm calculates probabilities and makes a classification decision based on those probabilities and the "naive" assumption of feature independence.

---

Sure, let's delve into the different variants of the Naive Bayes algorithm. The core difference between these variants lies in how they handle the **likelihood term ($P(\text{Feature}_i | \text{Class})$)**, which depends on the nature of your features.

### Naive Bayes Variants

Recall the main Naive Bayes formula:

$P(\text{Class} | \text{Features}) \propto P(\text{Class}) * \prod_{i=1}^{n} P(\text{Feature}_i | \text{Class})$

The variations come from the specific probability distribution assumed for $P(\text{Feature}_i | \text{Class})$.

---

### 1. Gaussian Naive Bayes

* **When to use:** This variant is used when your features are **continuous numerical values** and are assumed to follow a **Gaussian (normal) distribution**.

* **How it works:** Instead of counting frequencies, Gaussian Naive Bayes calculates the mean ($\mu$) and standard deviation ($\sigma$) of each continuous feature for each class during the training phase.

    For a new data point, it then uses the Probability Density Function (PDF) of the Gaussian distribution to calculate the likelihood of observing that feature value given the class. The formula for the Gaussian PDF is:

    $P(x | \text{Class}) = \frac{1}{\sqrt{2\pi\sigma^2}} * e^{-\frac{(x - \mu)^2}{2\sigma^2}}$

    Where:
    * $x$ is the value of the feature for the new data point.
    * $\mu$ is the mean of the feature values for that specific class (learned from training data).
    * $\sigma^2$ is the variance (square of standard deviation) of the feature values for that specific class (learned from training data).

* **Example:** Classifying whether a person has heart disease based on continuous features like "Blood Pressure" or "Cholesterol Level". For each class (e.g., "Heart Disease" or "No Heart Disease"), the model would learn the mean and standard deviation of blood pressure and cholesterol levels. When a new patient comes in, it calculates the probability of their blood pressure and cholesterol levels given each disease class using the Gaussian PDF.

---

### 2. Multinomial Naive Bayes

* **When to use:** This is arguably the most common variant for **text classification** problems. It's used when features represent **counts or frequencies** of events, such as word counts in a document. The features are typically discrete.

* **How it works:** It models the likelihood of observing word counts in a document given a class. It's based on the multinomial distribution.

    The likelihood $P(\text{word}_i | \text{Class})$ is calculated as:

    $P(\text{word}_i | \text{Class}) = \frac{\text{Count of word}_i \text{ in documents of Class} + \alpha}{\text{Total words in documents of Class} + \alpha * N}$

    Where:
    * $\text{Count of word}_i \text{ in documents of Class}$ is the number of times $\text{word}_i$ appears in all training documents belonging to that specific class.
    * $\text{Total words in documents of Class}$ is the total number of words (sum of all word counts) in all training documents belonging to that specific class.
    * $\alpha$ is the Laplace/Lidstone smoothing parameter (typically 1 for Laplace smoothing). This prevents zero probabilities for words not seen in a particular class.
    * $N$ is the total number of unique words in the vocabulary (across all classes).

* **Example:** Spam detection. Features are the individual words in an email.
    * To classify an email as "Spam" or "Not Spam".
    * The model learns the probability of each word appearing in "Spam" emails and "Not Spam" emails. For instance, $P(\text{"Viagra"} | \text{Spam})$ would likely be high, while $P(\text{"meeting"} | \text{Spam})$ might be low.
    * When a new email arrives, it calculates the likelihood of all its words appearing given each class (Spam or Not Spam) and combines them with the prior probabilities to make a prediction.

---

### 3. Bernoulli Naive Bayes

* **When to use:** This variant is suitable for **binary or boolean features**. It's used when features represent the presence or absence of a particular term, rather than their frequency.

* **How it works:** It assumes that features are binary-valued (e.g., 0 or 1, true or false). The likelihood $P(\text{Feature}_i | \text{Class})$ is calculated based on whether a feature is present or absent in a document belonging to a specific class.

    For each feature $i$ and class $C_k$, it calculates:
    * $P(X_i = 1 | C_k)$: The probability that feature $X_i$ is present (1) given class $C_k$.
    * $P(X_i = 0 | C_k)$: The probability that feature $X_i$ is absent (0) given class $C_k$.

    The formula for the likelihood calculation incorporates both the presence and absence of features:

    $P(\text{Features} | \text{Class}) = \prod_{i=1}^{n} (P(X_i = 1 | \text{Class}) * I(X_i) + P(X_i = 0 | \text{Class}) * (1 - I(X_i)))$

    Where $I(X_i)$ is an indicator function (1 if $X_i$ is present, 0 if $X_i$ is absent).

* **Example:** Document classification where you only care if a word exists in a document, not how many times it appears.
    * If you're classifying movie reviews as "positive" or "negative" and your features are simply the presence (1) or absence (0) of certain sentiment words ("great", "terrible", "enjoyable").
    * A review might be represented as `[great:1, terrible:0, enjoyable:1, ...]`
    * Bernoulli Naive Bayes would then learn the probability that "great" is present in a positive review vs. a negative review, and similarly for "terrible", etc.

---

### Summary of Differences:

| Variant           | Feature Type         | Likelihood Calculation                                                | Common Use Cases                                 |
| :---------------- | :------------------- | :-------------------------------------------------------------------- | :----------------------------------------------- |
| **Gaussian NB** | Continuous           | Assumes Gaussian (normal) distribution; uses mean & standard deviation. | Medical diagnosis, financial modeling.           |
| **Multinomial NB** | Discrete (Counts)    | Based on frequency counts; suitable for word counts.                  | Text classification (spam, sentiment, topic).    |
| **Bernoulli NB** | Binary (Presence/Absence) | Based on presence or absence of features.                             | Text classification (presence of words), feature selection. |

Choosing the right Naive Bayes variant is crucial and depends directly on the nature and distribution of your dataset's features. If you have a mix of feature types (e.g., continuous and categorical), you might need to combine approaches or preprocess your data accordingly.


----


Yes, absolutely! Let's walk through dummy examples for each of the Naive Bayes variants to make their workings clear.

---

### 1. Gaussian Naive Bayes Dummy Example

**Scenario:** We want to classify if a person will be **"Fit"** or **"Unfit"** based on their `Weight` (in kg) and `Height` (in cm). These are continuous numerical features.

**Training Data:**

| Person ID | Weight (kg) | Height (cm) | Class    |
| :-------- | :---------- | :---------- | :------- |
| 1         | 60          | 170         | Fit      |
| 2         | 65          | 175         | Fit      |
| 3         | 58          | 168         | Fit      |
| 4         | 80          | 160         | Unfit    |
| 5         | 85          | 165         | Unfit    |
| 6         | 90          | 170         | Unfit    |

**Step 1: Calculate Prior Probabilities ($P(\text{Class})$)**

* Total Persons: 6
* Fit Persons: 3
* Unfit Persons: 3

* $P(\text{Fit}) = 3/6 = 0.5$
* $P(\text{Unfit}) = 3/6 = 0.5$

**Step 2: Calculate Likelihoods ($P(\text{Feature} | \text{Class})$) using Gaussian PDF**

For each class, we need the mean ($\mu$) and standard deviation ($\sigma$) for each feature.

**For Class: Fit** (Weight: 60, 65, 58; Height: 170, 175, 168)
* **Weight (Fit):**
    * $\mu_{\text{Weight,Fit}} = (60+65+58)/3 = 61$
    * $\sigma_{\text{Weight,Fit}}$ (Standard Deviation): $\approx 3.78$
* **Height (Fit):**
    * $\mu_{\text{Height,Fit}} = (170+175+168)/3 = 171$
    * $\sigma_{\text{Height,Fit}}$ (Standard Deviation): $\approx 2.94$

**For Class: Unfit** (Weight: 80, 85, 90; Height: 160, 165, 170)
* **Weight (Unfit):**
    * $\mu_{\text{Weight,Unfit}} = (80+85+90)/3 = 85$
    * $\sigma_{\text{Weight,Unfit}}$ (Standard Deviation): $\approx 4.08$
* **Height (Unfit):**
    * $\mu_{\text{Height,Unfit}} = (160+165+170)/3 = 165$
    * $\sigma_{\text{Height,Unfit}}$ (Standard Deviation): $\approx 4.08$

**Step 3: Classify a New Person**

Let's classify a new person with:
* **Weight:** 70 kg
* **Height:** 172 cm

We use the Gaussian PDF formula: $P(x | \text{Class}) = \frac{1}{\sqrt{2\pi\sigma^2}} * e^{-\frac{(x - \mu)^2}{2\sigma^2}}$

#### Calculation for **Fit**:

$P(\text{Fit} | \text{Weight=70, Height=172}) \propto P(\text{Fit}) * P(\text{Weight=70} | \text{Fit}) * P(\text{Height=172} | \text{Fit})$

* $P(\text{Weight=70} | \text{Fit})$: Using Gaussian PDF with $\mu=61, \sigma=3.78, x=70$: $\approx 0.0076$
* $P(\text{Height=172} | \text{Fit})$: Using Gaussian PDF with $\mu=171, \sigma=2.94, x=172$: $\approx 0.132$

$P(\text{Fit} | \text{new person}) \propto 0.5 * 0.0076 * 0.132 \approx 0.0005016$

#### Calculation for **Unfit**:

$P(\text{Unfit} | \text{Weight=70, Height=172}) \propto P(\text{Unfit}) * P(\text{Weight=70} | \text{Unfit}) * P(\text{Height=172} | \text{Unfit})$

* $P(\text{Weight=70} | \text{Unfit})$: Using Gaussian PDF with $\mu=85, \sigma=4.08, x=70$: $\approx 0.0000004$
* $P(\text{Height=172} | \text{Unfit})$: Using Gaussian PDF with $\mu=165, \sigma=4.08, x=172$: $\approx 0.040$

$P(\text{Unfit} | \text{new person}) \propto 0.5 * 0.0000004 * 0.040 \approx 0.000000008$

**Step 4: Make a Prediction**

* $P(\text{Fit} | \text{new person}) \propto 0.0005016$
* $P(\text{Unfit} | \text{new person}) \propto 0.000000008$

Since $0.0005016 > 0.000000008$, the Gaussian Naive Bayes algorithm would classify the new person as **Fit**.

---

### 2. Multinomial Naive Bayes Dummy Example

**Scenario:** We want to classify movie reviews as **"Positive"** or **"Negative"** based on the words they contain.

**Training Data:**

| Review ID | Text (Tokenized)                | Class    |
| :-------- | :------------------------------ | :------- |
| 1         | `[great, movie, amazing]`       | Positive |
| 2         | `[loved, it, great, acting]`    | Positive |
| 3         | `[bad, acting, terrible]`       | Negative |
| 4         | `[boring, movie, awful]`        | Negative |
| 5         | `[good, movie]`                 | Positive |

**Step 1: Calculate Prior Probabilities ($P(\text{Class})$)**

* Total Reviews: 5
* Positive Reviews: 3
* Negative Reviews: 2

* $P(\text{Positive}) = 3/5 = 0.6$
* $P(\text{Negative}) = 2/5 = 0.4$

**Step 2: Create Vocabulary and Calculate Likelihoods ($P(\text{Word} | \text{Class})$) with Laplace Smoothing ($\alpha=1$)**

**Vocabulary (Unique words):** `[great, movie, amazing, loved, it, acting, bad, terrible, boring, awful, good]`
* Total unique words (N) = 11

**For Class: Positive** (Total 3 reviews, Total words in Positive reviews: 3+4+2 = 9)

| Word      | Count in Positive Reviews | $P(\text{Word} | \text{Positive})$ (with $\alpha=1$) = (Count + 1) / (Total Positive Words + N) |
| :-------- | :------------------------ | :-------------------------------------------------------------------------------------------------- |
| great     | 2                         | $(2+1) / (9+11) = 3/20 = 0.15$                                                                      |
| movie     | 2                         | $(2+1) / (9+11) = 3/20 = 0.15$                                                                      |
| amazing   | 1                         | $(1+1) / (9+11) = 2/20 = 0.1$                                                                       |
| loved     | 1                         | $(1+1) / (9+11) = 2/20 = 0.1$                                                                       |
| it        | 1                         | $(1+1) / (9+11) = 2/20 = 0.1$                                                                       |
| acting    | 1                         | $(1+1) / (9+11) = 2/20 = 0.1$                                                                       |
| good      | 1                         | $(1+1) / (9+11) = 2/20 = 0.1$                                                                       |
| *Others (bad, terrible, boring, awful)* | 0                         | $(0+1) / (9+11) = 1/20 = 0.05$                                                                      |

**For Class: Negative** (Total 2 reviews, Total words in Negative reviews: 3+3 = 6)

| Word      | Count in Negative Reviews | $P(\text{Word} | \text{Negative})$ (with $\alpha=1$) = (Count + 1) / (Total Negative Words + N) |
| :-------- | :------------------------ | :--------------------------------------------------------------------------------------------------- |
| bad       | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| acting    | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| terrible  | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| boring    | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| movie     | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| awful     | 1                         | $(1+1) / (6+11) = 2/17 \approx 0.117$                                                                |
| *Others (great, amazing, loved, it, good)* | 0                         | $(0+1) / (6+11) = 1/17 \approx 0.059$                                                                |

**Step 3: Classify a New Review**

New Review: `[great, movie, bad]`

#### Calculation for **Positive**:

$P(\text{Positive} | \text{new review}) \propto P(\text{Positive}) * P(\text{great} | \text{Positive}) * P(\text{movie} | \text{Positive}) * P(\text{bad} | \text{Positive})$

$P(\text{Positive} | \text{new review}) \propto 0.6 * 0.15 * 0.15 * 0.05 = 0.000675$

#### Calculation for **Negative**:

$P(\text{Negative} | \text{new review}) \propto P(\text{Negative}) * P(\text{great} | \text{Negative}) * P(\text{movie} | \text{Negative}) * P(\text{bad} | \text{Negative})$

$P(\text{Negative} | \text{new review}) \propto 0.4 * 0.059 * 0.117 * 0.117 \approx 0.000324$

**Step 4: Make a Prediction**

* $P(\text{Positive} | \text{new review}) \propto 0.000675$
* $P(\text{Negative} | \text{new review}) \propto 0.000324$

Since $0.000675 > 0.000324$, the Multinomial Naive Bayes algorithm would classify the new review as **Positive**. (Even though it contains "bad", the "great" and "movie" words pull it towards positive based on our small dataset).

---

### 3. Bernoulli Naive Bayes Dummy Example

**Scenario:** Classify customer feedback as **"Complaint"** or **"Praise"** based on the presence/absence of certain keywords.

**Features (Binary):**
* `Problem`: 1 if present, 0 if absent
* `Happy`: 1 if present, 0 if absent
* `Issue`: 1 if present, 0 if absent
* `Good`: 1 if present, 0 if absent

**Training Data:**

| Feedback ID | Problem | Happy | Issue | Good | Class     |
| :---------- | :------ | :---- | :---- | :--- | :-------- |
| 1           | 1       | 0     | 1     | 0    | Complaint |
| 2           | 1       | 0     | 0     | 0    | Complaint |
| 3           | 0       | 1     | 0     | 1    | Praise    |
| 4           | 0       | 1     | 0     | 0    | Praise    |
| 5           | 1       | 0     | 1     | 0    | Complaint |

**Step 1: Calculate Prior Probabilities ($P(\text{Class})$)**

* Total Feedback: 5
* Complaints: 3
* Praise: 2

* $P(\text{Complaint}) = 3/5 = 0.6$
* $P(\text{Praise}) = 2/5 = 0.4$

**Step 2: Calculate Likelihoods ($P(\text{Feature} | \text{Class})$) with Laplace Smoothing ($\alpha=1$)**

We need to calculate $P(X_i=1 | \text{Class})$ and $P(X_i=0 | \text{Class})$ for each feature.

**For Class: Complaint** (Total 3 Complaints)

| Feature | Count (Feature=1) | Count (Feature=0) | $P(\text{Feature=1} | \text{Complaint})$ = (Count+1)/(Total Complaints + 2) | $P(\text{Feature=0} | \text{Complaint})$ = (Count+1)/(Total Complaints + 2) |
| :------ | :---------------- | :---------------- | :-------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| Problem | 3                 | 0                 | $(3+1)/(3+2) = 4/5 = 0.8$                                                   | $(0+1)/(3+2) = 1/5 = 0.2$                                                   |
| Happy   | 0                 | 3                 | $(0+1)/(3+2) = 1/5 = 0.2$                                                   | $(3+1)/(3+2) = 4/5 = 0.8$                                                   |
| Issue   | 2                 | 1                 | $(2+1)/(3+2) = 3/5 = 0.6$                                                   | $(1+1)/(3+2) = 2/5 = 0.4$                                                   |
| Good    | 0                 | 3                 | $(0+1)/(3+2) = 1/5 = 0.2$                                                   | $(3+1)/(3+2) = 4/5 = 0.8$                                                   |

**For Class: Praise** (Total 2 Praise)

| Feature | Count (Feature=1) | Count (Feature=0) | $P(\text{Feature=1} | \text{Praise})$ = (Count+1)/(Total Praise + 2) | $P(\text{Feature=0} | \text{Praise})$ = (Count+1)/(Total Praise + 2) |
| :------ | :---------------- | :---------------- | :---------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| Problem | 0                 | 2                 | $(0+1)/(2+2) = 1/4 = 0.25$                                              | $(2+1)/(2+2) = 3/4 = 0.75$                                              |
| Happy   | 2                 | 0                 | $(2+1)/(2+2) = 3/4 = 0.75$                                              | $(0+1)/(2+2) = 1/4 = 0.25$                                              |
| Issue   | 0                 | 2                 | $(0+1)/(2+2) = 1/4 = 0.25$                                              | $(2+1)/(2+2) = 3/4 = 0.75$                                              |
| Good    | 1                 | 1                 | $(1+1)/(2+2) = 2/4 = 0.5$                                               | $(1+1)/(2+2) = 0.5$                                                     |

**Step 3: Classify New Feedback**

New Feedback: `[Problem: 0, Happy: 1, Issue: 0, Good: 1]` (i.e., No "Problem", "Happy" is present, No "Issue", "Good" is present)

The likelihood formula for Bernoulli NB is:
$P(\text{Features} | \text{Class}) = \prod_{i=1}^{n} (P(X_i = 1 | \text{Class}) * I(X_i) + P(X_i = 0 | \text{Class}) * (1 - I(X_i)))$
Where $I(X_i)$ is 1 if feature $X_i$ is present, 0 if absent.

#### Calculation for **Complaint**:

$P(\text{Complaint} | \text{new feedback}) \propto P(\text{Complaint}) *$
$P(\text{Problem=0} | \text{Complaint}) *$
$P(\text{Happy=1} | \text{Complaint}) *$
$P(\text{Issue=0} | \text{Complaint}) *$
$P(\text{Good=1} | \text{Complaint})$

$P(\text{Complaint} | \text{new feedback}) \propto 0.6 * 0.2 * 0.2 * 0.4 * 0.2 = 0.00384$

#### Calculation for **Praise**:

$P(\text{Praise} | \text{new feedback}) \propto P(\text{Praise}) *$
$P(\text{Problem=0} | \text{Praise}) *$
$P(\text{Happy=1} | \text{Praise}) *$
$P(\text{Issue=0} | \text{Praise}) *$
$P(\text{Good=1} | \text{Praise})$

$P(\text{Praise} | \text{new feedback}) \propto 0.4 * 0.75 * 0.75 * 0.75 * 0.5 = 0.084375$

**Step 4: Make a Prediction**

* $P(\text{Complaint} | \text{new feedback}) \propto 0.00384$
* $P(\text{Praise} | \text{new feedback}) \propto 0.084375$

Since $0.084375 > 0.00384$, the Bernoulli Naive Bayes algorithm would classify the new feedback as **Praise**.

---

These dummy examples demonstrate how each Naive Bayes variant adapts its probability calculation based on the nature of the input features (continuous, count-based, or binary).

---


