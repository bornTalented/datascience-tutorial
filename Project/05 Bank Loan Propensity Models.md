
Project Name: Lead Conversion Prediction + Sales Nudge Engine
Staff Role: Project Lead (DS)

Responsibilities
- Led the design and implementation of ML models for lead conversion prediction at multiple sales funnel stages (File-to-Sanction, Sanction-to-Disbursement), ensuring robustness through IV/VIF-based feature selection and hyperparameter tuning.
- Built stable, production-ready pipelines incorporating optimal binning, WoE encoding, and batch inference orchestration using Airflow DAGs.
- Developed monitoring systems using KS-statistic and PSI to track model performance across DEV, OOS, and OOT datasets.
- Designed a performance-based nudge engine integrating ML and rule-based logic to generate personalized nudges for sales officers, aligned with targets and behavioral patterns.
- Drove adoption of data-driven decision-making by enabling dynamic prioritization of high-potential leads and embedding the solution into frontline operations.
- Led cross-functional coordination between sales, analytics, and tech teams to ensure successful deployment and operational impact.
---
---

## **📌 Project Story: Driving Sales Efficiency at XYZ Bank Using ML & Nudges**

When I started working with XYZ Bank, the sales team was facing a classic but critical problem—**they had too many files to process, but no clear way to prioritize them**.

Every file required effort, but not every file had the same likelihood of conversion. As a result:

* High-potential cases were sometimes delayed
* Sales teams spent time on low-probability files
* Conversion rates and productivity were suboptimal

So the core question we tried to answer was:

👉 *“Can we intelligently predict which files are worth focusing on—and guide the sales team accordingly?”*

---

## **🔍 Step 1: Breaking Down the Problem**

We framed the problem into **two key prediction layers**:

1. **File → Sanction**
	* What is the probability that a file gets sanctioned quickly?

2. **Sanction → Disbursement**
	* Once sanctioned, how likely is it to convert into actual business?

But prediction alone wasn’t enough.

So we added a **third layer**:
👉 A **Nudge Engine** to *influence behavior*, not just measure it.

---

## **📊 Step 2: Building a Strong Data Foundation**

We pulled data from **RDS databases**, working closely with the data engineering team and business stakeholders.

Instead of jumping into modeling, we first:

* Created a **data funnel** to understand the journey of a file
* Validated it with the client to ensure business alignment
* Identified key feature groups like:
	* Customer demographics
	* Loan details
	* Credit history
	* Risk triggers

This step was crucial—it ensured we were solving the *right problem with the right data*.

---

## **📈 Step 3: Extracting Insights Through EDA**

We actually created **two separate EDAs**:

### 1. Business EDA

* Identified bottlenecks in the funnel
* Found patterns affecting conversions
* Suggested **Next Best Actions (NBAs)**

### 2. ML EDA

* Checked variable strength using **Information Value (IV)**
* Removed multicollinearity using **correlation & VIF**
* Ensured statistical significance using **p-values**
* Monitored stability using **PSI**

This dual approach helped us bridge **business intuition + statistical rigor**.

---

## **⚙️ Step 4: Feature Engineering & Preprocessing**

We defined the target in a very business-relevant way:

👉 A file is **“Good”** if it gets sanctioned within **4.75 working days**

Then we:

* Applied **WoE encoding** for categorical variables
* Used **optimal binning** for numeric features
* Handled missing values contextually

We started with **160 features** and reduced them to **12 highly impactful ones**.

---

## **🤖 Step 5: Model Building with a Business Constraint**

We experimented with multiple models:

* Random Forest
* XGBoost
* LightGBM
* Logistic Regression

While some complex models performed well, we had a key constraint:

👉 The model had to be **explainable**

So we chose **Logistic Regression**, because:

* It performed competitively
* It provided clear reasoning behind predictions

---

## **📏 Step 6: Measuring What Matters**

Our primary metric was **KS Statistic**.

👉 In simple terms:
It tells us how well the model separates **good vs bad cases**.

* A KS of **30–40** indicated strong performance
* We also ensured:
	* Stable performance across datasets
	* Minimal variation (<3 points difference)

This ensured the model wasn’t just accurate—but also **reliable over time**.

---

## **🚀 Step 7: From Model to Real Impact**

We didn’t stop at modeling—we built a **complete production pipeline**:

### Development

* Built on **SageMaker**

### Deployment

* Automated using **Airflow DAGs on EC2**

### Monitoring

* Continuous tracking using:
	* KS statistics
	* PSI reports

---

## **🔔 Step 8: Nudge Engine – Driving Behavior Change**

This was one of the most impactful parts.

We designed nudges at three levels:

* **Self** → “You’re below your usual performance”
* **Target** → “You need X more to hit your goal”
* **Peer** → “Your peers are performing better in this segment”

👉 This shifted the system from **passive prediction → active decision support**

---

## **📊 Final Impact**

By the end of the project:

* Sales teams could **prioritize high-probability files**
* Conversion rates improved across stages
* Decision-making became **data-driven**
* Sales officers received **actionable, personalized guidance**

---

## **🎯 One-Line Summary**

👉 *“I built an end-to-end explainable ML system that not only predicted sales conversions but also influenced sales behavior through targeted nudges—leading to better prioritization and improved business outcomes.”*

---

---

# **Sales Conversion Optimization & Nudge Engine for XYZ Bank**

---

## **1. Business Objective**

The primary objective of this project is to improve the efficiency and productivity of XYZ Bank’s sales process by:

* Prioritizing high-probability cases
* Reducing turnaround time (TAT)
* Driving higher conversion rates across funnel stages
* Enabling data-driven decision-making via intelligent nudges

### **Key Use Cases**

#### **1.1 File-to-Sanction Conversion**

* Predict the probability of a loan file being sanctioned within a defined SLA.
* Prioritize high-propensity files for faster processing.
* Improve operational efficiency and reduce delays.

#### **1.2 Sanction-to-Disbursement Conversion**

* Estimate likelihood of sanctioned cases converting to disbursement.
* Focus resources on high-conversion cases.
* Minimize drop-offs post-sanction.

#### **1.3 Nudge Engine**

* Generate personalized nudges for:
	* Sales Officers
	* Team Leads
* Based on:
	* Individual performance
	* Target achievement
	* Peer benchmarking
* Objective:
	* Improve productivity
	* Drive behavioral change
	* Enhance target achievement

---

## **2. Data Collection & Understanding**

### **2.1 Data Source**

* Data extracted from **AWS RDS (PostgreSQL)** using:
	* `pandas`
	* `psycopg2`

### **2.2 Stakeholder Alignment**

* Conducted multiple discussions with:
	* Business teams
	* Data Engineering (DE) team
* Understood:
	* Data schema
	* Business definitions
	* Feature relevance

### **2.3 Data Mapping & Funnel**

* Utilized DE-provided mapping sheet
* Built and validated:
	* End-to-end **data funnel**
	* Stage transitions (Login → Sanction → Disbursement)

### **2.4 Data Domains**

| Category     | Features                           |
| ------------ | ---------------------------------- |
| Bureau Data  | DPD, enquiries, credit utilization |
| Demographics | Tier classification, dependents    |
| Financial    | Income, obligations                |
| Employment   | Occupation, salary                 |
| Credit       | CIBIL score, history               |
| Loan Info    | Mortgage, loan type                |
| Others       | Co-applicants, risk flags          |

---

## **3. Exploratory Data Analysis (EDA)**

### **3.1 Business EDA**

* Identified patterns impacting conversions
* Derived **Next Best Actions (NBAs)** such as:
	* Prioritizing low FOIR cases
	* Fast-tracking high credit score customers
	* Reducing delays in document collection

### **3.2 ML-Focused EDA**

#### **Univariate Analysis**

* Missing value analysis
* Outlier detection
* Cardinality checks
* Skewness evaluation

#### **Bivariate Analysis**

* Statistical significance (p-values)
* Chi-square test (categorical)
* Information Value (IV)

#### **Multivariate Analysis**

* Correlation:
	* Pearson (numerical)
	* Cramér’s V (categorical)
* Multicollinearity:
	* Variance Inflation Factor (VIF)

#### **Stability Analysis**

* Population Stability Index (PSI)

---

## **4. Feature Engineering**

### **4.1 Target Definition**

* **Good (1):** File sanctioned within **4.75 working days**
* **Bad (0):** Otherwise

### **4.2 Derived Features**

* **FOIR (Fixed Obligation to Income Ratio)**
* **DPD Features** (delinquencies over last N months)
* **Enquiry Trends**
* **Credit Utilization**
* **Customer Profile Features**
* **Co-applicant Indicators**
* **Risk Flags**

### **4.3 Feature Bucketing**

* Logical grouping into:
	* Demographics
	* Financial
	* Credit behavior
	* Loan attributes

---

## **5. Data Preprocessing**

### **5.1 Data Cleaning**

* Standardization of formats
* Handling inconsistent values
* Feature-specific null treatment

### **5.2 Numerical Features**

* Outlier treatment (capping/winsorization)
* Scaling (if required)
* Optimal binning using **OptBinning**

### **5.3 Categorical Features**

* Weight of Evidence (WoE) encoding
* Handling rare categories

---

## **6. Variable Selection & Reduction**

### **6.1 Initial Features**

* Total features: **~160**

### **6.2 Reduction Techniques**

| Technique                | Threshold               |
| ------------------------ | ----------------------- |
| Correlation              | > 0.8 removed           |
| VIF                      | > 20 removed            |
| IV                       | < 0.02 or > 0.5 removed |
| Statistical Significance | p-value > 0.05 removed  |

### **6.3 Final Selection**

* Reduced to: **37 features**
* Final model input: **12 features**

---

## **7. Data Splitting Strategy**

| Dataset | Purpose              |
| ------- | -------------------- |
| DEV     | Model training (80%) |
| OOS     | Validation (20%)     |
| OOT     | Last 3 months        |
| PDV     | Last 6 months        |
| PSI     | Last 2 months        |

---

## **8. Model Development**

### **8.1 Constraints**

* **Trend:** Monotonic (strictly increasing/decreasing)
* **KS Statistic:** Ideal range of **30–40**
	* **Decile Distribution:** Equal-sized deciles
	* **Stability:** KS difference across datasets < **3 points**

### **8.2 Models Evaluated**

* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* XGBoost
* LightGBM

### **8.3 Final Model: Logistic Regression**

#### **Reason for Selection**

* High interpretability
* Regulatory friendliness
* Comparable performance to complex models
* Stable across datasets

---

## **9. Hyperparameter Tuning**

* Used **Bayesian Optimization**
* Advantages:
	* Faster than grid search
	* Efficient parameter exploration

---

## **10. Model Evaluation**

### **10.1 Metrics Used**

* Accuracy
* Precision
* Recall
* F1 Score
* KS Statistic

### **10.2 KS Statistic (Kolmogorov-Smirnov Statistic)**

* Measures separation between Good vs Bad distributions
* Defined as the **maximum difference between cumulative distributions** of the two classes. Higher KS indicates better discriminatory power.
* Ideal range for this project: **30–40**

### **10.3 Stability Validation**

* Evaluated on:
	* OOS
	* OOT datasets
* Ensured:
	* Minimal performance drift
	* Consistent KS values

---

## **11. Nudge Engine Design**

### **11.1 Inputs**

* Model predictions
* Sales performance metrics
* Peer benchmarks

### **11.2 Types of Nudges**

* Performance alerts
* Target reminders
* Opportunity prioritization
* Behavioral recommendations

### **11.3 Delivery Mechanism**

* Dashboard / CRM integration (assumed)
* Daily/weekly triggers

---

## **12. End-to-End Pipeline Architecture**

### **12.1 Development Pipeline**

* Built on:
	* AWS SageMaker
	* EC2 (testing)

### **12.2 Deployment Pipeline**

* Hosted on:
	* EC2
* Orchestrated using:
	* Airflow DAGs

### **12.3 Monitoring Pipeline**

* Runs periodic scoring
* Tracks:
	* KS statistic
	* PSI
	* Feature drift

---

## **13. Model Deployment**

### **13.1 Components**

* Training Script
* Inference Script
* Monitoring Script

### **13.2 Infrastructure**

* EC2 → Model hosting
* S3 → Data storage
* Airflow → Scheduling
* SNS → Alerts

### **13.3 Workflow**

* Incremental data ingestion
* Daily batch scoring
* Output pushed to downstream systems

---

## **14. Monitoring & Maintenance**

### **14.1 Monitoring Metrics**

* KS drift
* PSI
* Feature distribution changes
* Prediction stability

### **14.2 Tools Used**

* AWS EMR
* Athena
* Airflow UI
* SageMaker
* EC2
* S3

### **14.3 Activities**

* Debugging pipelines
* Model retraining (if needed)
* Nudge effectiveness tracking

We deployed the final model on an EC2 instance and built a data pipeline to load incremental data through multiple stages (from staging to mirror to consumption). 
A daily batch scoring job was scheduled using Airflow, which runs every morning. 

---

## **15. Assumptions & Risks (Added Section)**

### **Assumptions**

* Data quality remains consistent over time
* Business definitions (e.g., 4.75 days SLA) remain unchanged
* No major policy changes affecting loan approvals

### **Risks**

* Data drift impacting model performance
* Changing customer behavior
* External economic factors
* Model over-reliance without human validation

---

## **16. Governance & Compliance (Added Section)**

* Model explainability ensured via Logistic Regression
* Feature traceability maintained
* Periodic audits possible
* Compliance with internal risk policies

---

## **17. Key Outcomes**

* Improved file prioritization
* Increased conversion rates
* Reduced turnaround time
* Enhanced sales productivity
* Data-driven decision-making via nudges
* Robust, scalable, and interpretable ML system

---

If you want, I can next:

* Convert this into a **resume project description**
* Create a **PPT for interviews**
* Or make a **short storytelling version (very important for interviews)**
