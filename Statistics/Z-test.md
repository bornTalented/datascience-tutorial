The Z-test is a widely used statistical hypothesis test that helps determine if there's a significant difference between a sample mean and a population mean, or between the means of two populations. It's particularly useful when you have a large sample size and know the population standard deviation.

**Core Concept of Z-test**

At its heart, the Z-test calculates a "Z-score" (or Z-statistic). This Z-score measures how many standard deviations a data point (or a sample mean) is away from the population mean. In hypothesis testing, this Z-score is then compared to critical values from the standard normal distribution (or used to calculate a p-value) to decide whether to reject or fail to reject the null hypothesis.

**Key Assumptions for Z-test:**

For a Z-test to be valid, certain assumptions should be met:

1.  **Known Population Standard Deviation ($\sigma$):** This is a crucial assumption. If the population standard deviation is unknown, a t-test is generally preferred.
2.  **Large Sample Size ($n \ge 30$):** Due to the Central Limit Theorem, if the sample size is large enough (generally considered 30 or more), the sampling distribution of the sample mean will be approximately normally distributed, even if the original population distribution is not. This allows us to use the Z-distribution.
3.  **Independent Observations:** Each data point in the sample must be independent of the others.
4.  **Normally Distributed Population (or sufficiently large sample):** Ideally, the population from which the sample is drawn should be normally distributed. However, as mentioned, with a large sample size, this assumption becomes less critical due to the Central Limit Theorem.
5.  **Continuous Data:** The variable being tested should be continuous.

**General Steps to Perform a Z-test:**

1.  **State the Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$):**
    * $H_0$: This is the statement of no effect or no difference (e.g., the sample mean is equal to the population mean).
    * $H_1$: This is what you're trying to prove (e.g., the sample mean is different from, greater than, or less than the population mean).
2.  **Choose a Significance Level ($\alpha$):** This is the probability of rejecting the null hypothesis when it is actually true (Type I error). Common values are 0.05 (5%) or 0.01 (1%).
3.  **Calculate the Z-statistic:** Use the appropriate formula based on the type of Z-test.
4.  **Determine the Critical Value(s) or P-value:**
    * **Critical Value Approach:** Find the critical Z-value(s) from the standard normal table corresponding to your chosen $\alpha$ and the type of test (one-tailed or two-tailed).
    * **P-value Approach:** Calculate the p-value associated with your calculated Z-statistic.
5.  **Make a Decision:**
    * **Critical Value Approach:** If the calculated Z-statistic falls into the rejection region (beyond the critical value), reject $H_0$.
    * **P-value Approach:** If the p-value is less than or equal to $\alpha$, reject $H_0$.
6.  **Formulate a Conclusion:** State your conclusion in the context of the problem.

---

**Variants of the Z-test (with Examples):**

The Z-test comes in several variants, each designed for different scenarios.

1.  **One-Sample Z-test for Mean:**
    * **Purpose:** To compare the mean of a single sample to a known or hypothesized population mean.
    * **Formula:**
        $Z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$
        Where:
        * $\bar{x}$ = sample mean
        * $\mu$ = hypothesized population mean
        * $\sigma$ = population standard deviation
        * $n$ = sample size
    * **Example:** A company claims that the average weight of its "200g" potato chip bags is truly 200g. You take a random sample of 50 bags and find the average weight to be 198g. The population standard deviation of chip bag weights is known to be 5g. At a 0.05 significance level, can you say the average weight is different from 200g?
        * $H_0: \mu = 200$
        * $H_1: \mu \ne 200$ (Two-tailed test)
        * $\bar{x} = 198$, $\mu = 200$, $\sigma = 5$, $n = 50$
        * $Z = \frac{198 - 200}{5 / \sqrt{50}} = \frac{-2}{5 / 7.07} = \frac{-2}{0.707} \approx -2.83$
        * For a two-tailed test with $\alpha = 0.05$, the critical values are $\pm 1.96$. Since $-2.83 < -1.96$, we reject the null hypothesis. There is sufficient evidence to conclude that the average weight of the chip bags is significantly different from 200g.

2.  **Two-Sample Z-test for Means (Independent Samples):**
    * **Purpose:** To compare the means of two independent samples to determine if there's a significant difference between their respective population means. This assumes both population standard deviations are known.
    * **Formula:**
        $Z = \frac{(\bar{x}_1 - \bar{x}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}$
        Where:
        * $\bar{x}_1$, $\bar{x}_2$ = sample means of group 1 and group 2
        * $\mu_1 - \mu_2$ = hypothesized difference between population means (often 0, meaning no difference)
        * $\sigma_1^2$, $\sigma_2^2$ = population variances of group 1 and group 2
        * $n_1$, $n_2$ = sample sizes of group 1 and group 2
    * **Example:** A marketing team wants to compare the average spending of male and female customers at a department store. They randomly sample 60 male customers and find their average spending to be $120. They also randomly sample 70 female customers and find their average spending to be $135. The population standard deviation for male spending is known to be $20, and for female spending, it's $25. At $\alpha = 0.01$, is there a significant difference in average spending between male and female customers?
        * $H_0: \mu_1 = \mu_2$ (or $\mu_1 - \mu_2 = 0$)
        * $H_1: \mu_1 \ne \mu_2$ (Two-tailed test)
        * $\bar{x}_1 = 120$, $n_1 = 60$, $\sigma_1 = 20$
        * $\bar{x}_2 = 135$, $n_2 = 70$, $\sigma_2 = 25$
        * $Z = \frac{(120 - 135) - 0}{\sqrt{\frac{20^2}{60} + \frac{25^2}{70}}} = \frac{-15}{\sqrt{\frac{400}{60} + \frac{625}{70}}} = \frac{-15}{\sqrt{6.67 + 8.93}} = \frac{-15}{\sqrt{15.6}} \approx \frac{-15}{3.95} \approx -3.80$
        * For a two-tailed test with $\alpha = 0.01$, the critical values are $\pm 2.576$. Since $-3.80 < -2.576$, we reject the null hypothesis. There is significant evidence to suggest a difference in average spending between male and female customers.

3.  **Z-test for Proportions (One Sample):**
    * **Purpose:** To test whether a sample proportion differs significantly from a known or hypothesized population proportion.
    * **Formula:**
        $Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$
        Where:
        * $\hat{p}$ = sample proportion
        * $p_0$ = hypothesized population proportion
        * $n$ = sample size
    * **Example:** A company claims that 15% of its customers prefer a new product design. In a survey of 200 customers, 40 stated they preferred the new design. At $\alpha = 0.05$, is the company's claim accurate?
        * $H_0: p = 0.15$
        * $H_1: p \ne 0.15$ (Two-tailed test)
        * $\hat{p} = 40/200 = 0.20$
        * $p_0 = 0.15$
        * $n = 200$
        * $Z = \frac{0.20 - 0.15}{\sqrt{\frac{0.15(1-0.15)}{200}}} = \frac{0.05}{\sqrt{\frac{0.15 \times 0.85}{200}}} = \frac{0.05}{\sqrt{\frac{0.1275}{200}}} = \frac{0.05}{\sqrt{0.0006375}} = \frac{0.05}{0.0252} \approx 1.98$
        * For a two-tailed test with $\alpha = 0.05$, the critical values are $\pm 1.96$. Since $1.98 > 1.96$, we reject the null hypothesis. There is sufficient evidence to suggest that the proportion of customers preferring the new design is different from 15%.

4.  **Z-test for Proportions (Two Samples):**
    * **Purpose:** To compare the proportions of a certain characteristic in two independent samples.
    * **Formula:**
        $Z = \frac{(\hat{p}_1 - \hat{p}_2)}{\sqrt{\hat{p}_c(1-\hat{p}_c)(\frac{1}{n_1} + \frac{1}{n_2})}}$
        Where:
        * $\hat{p}_1$, $\hat{p}_2$ = sample proportions of group 1 and group 2
        * $n_1$, $n_2$ = sample sizes of group 1 and group 2
        * $\hat{p}_c$ = pooled sample proportion (calculated as $\frac{x_1 + x_2}{n_1 + n_2}$, where $x_1$ and $x_2$ are the number of successes in each sample)
    * **Example:** A vaccine manufacturer wants to know if their new vaccine is more effective than an older one. In a trial, 75 out of 100 people who received the old vaccine did not get sick, while 90 out of 110 people who received the new vaccine did not get sick. At $\alpha = 0.05$, is the new vaccine significantly more effective?
        * $H_0: p_1 = p_2$ (or $p_1 - p_2 = 0$)
        * $H_1: p_1 < p_2$ (New vaccine is more effective, so proportion not getting sick is higher; this is a left-tailed test as we'd expect $\hat{p}_1 - \hat{p}_2$ to be negative)
        * $\hat{p}_1 = 75/100 = 0.75$
        * $\hat{p}_2 = 90/110 \approx 0.818$
        * $n_1 = 100$, $n_2 = 110$
        * $x_1 = 75$, $x_2 = 90$
        * $\hat{p}_c = \frac{75 + 90}{100 + 110} = \frac{165}{210} \approx 0.786$
        * $Z = \frac{(0.75 - 0.818)}{\sqrt{0.786(1-0.786)(\frac{1}{100} + \frac{1}{110})}} = \frac{-0.068}{\sqrt{0.786 \times 0.214 \times (0.01 + 0.009)}} = \frac{-0.068}{\sqrt{0.168 \times 0.019}} = \frac{-0.068}{\sqrt{0.003192}} = \frac{-0.068}{0.0565} \approx -1.20$
        * For a left-tailed test with $\alpha = 0.05$, the critical value is $-1.645$. Since $-1.20 > -1.645$, we fail to reject the null hypothesis. There is not enough evidence to conclude that the new vaccine is significantly more effective.

**Z-test vs. T-test:**

It's important to differentiate between Z-tests and T-tests. The primary difference lies in the knowledge of the population standard deviation ($\sigma$) and sample size:

* **Z-test:** Used when the population standard deviation ($\sigma$) is **known** and the sample size ($n$) is **large** (typically $n \ge 30$).
* **T-test:** Used when the population standard deviation ($\sigma$) is **unknown** and the sample size ($n$) is **small** (typically $n < 30$). When $\sigma$ is unknown, the sample standard deviation (s) is used as an estimate, and the t-distribution is used, which accounts for the additional uncertainty. For larger sample sizes, the t-distribution approximates the Z-distribution.