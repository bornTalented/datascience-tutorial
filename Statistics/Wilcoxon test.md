The Wilcoxon test is a family of **non-parametric statistical tests** used when the assumptions for parametric tests (like the t-test) are not met. Primarily, this means that the data is not normally distributed, or the sample sizes are small, or the data is ordinal (ranked).

Instead of comparing means directly, Wilcoxon tests compare **medians** or the **distributions** of groups based on the ranks of the data. This makes them robust to outliers and skewed distributions.

**Core Concept of Wilcoxon Tests: Ranking Data**

The central idea behind all Wilcoxon tests is to convert the raw data values into their **ranks**.

1.  **Combine and Rank:** For independent samples, you combine all data points from both groups and rank them from smallest to largest. For paired samples, you calculate the differences and then rank the absolute differences.
2.  **Sum Ranks:** You then sum the ranks for specific groups or for positive/negative differences.
3.  **Compare Sums:** The test statistic is based on these rank sums. If the sums of ranks are significantly different, it suggests a difference between the groups (or conditions).

**Why use Wilcoxon instead of T-test?**

* **Non-normal Data:** If your data is not normally distributed and your sample size is small, the t-test's validity is compromised. Wilcoxon tests don't require normality.
* **Ordinal Data:** If your data is inherently ordinal (e.g., ratings on a Likert scale where the intervals aren't necessarily equal), rank-based tests are more appropriate.
* **Outliers:** Wilcoxon tests are less sensitive to outliers because they rely on ranks rather than the raw values themselves. An extreme outlier will still have a high rank, but its exact magnitude won't disproportionately influence the test statistic.

**Key Assumptions for Wilcoxon Tests:**

1.  **Independence:**
    * **Wilcoxon Rank-Sum Test (Mann-Whitney U):** Observations within each sample must be independent, and the two samples must be independent of each other.
    * **Wilcoxon Signed-Rank Test:** Pairs of observations must be independent of each other (though observations *within* a pair are dependent).
2.  **Measurement Scale:** The dependent variable should be at least ordinal (meaning you can rank the observations). For the Signed-Rank test, the differences should also be rankable.
3.  **Symmetry (for Signed-Rank Test):** The distribution of the *differences* between paired observations should be symmetric. This assumption is debated and sometimes considered less critical for simply detecting a shift in median.
4.  **Continuous Underlying Distribution (ideal, but ties handled):** While the tests operate on ranks, they ideally assume that the underlying variable is continuous, which would avoid tied ranks. However, methods for handling ties (assigning average ranks) are well-established.

**General Steps to Perform a Wilcoxon Test:**

1.  **State the Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$):**
    * These generally relate to whether the medians or distributions are the same.
2.  **Choose a Significance Level ($\alpha$):** (e.g., 0.05 or 0.01).
3.  **Perform Ranking and Summation:** This is the core calculation step and varies by test type.
4.  **Calculate the Test Statistic (W or U):** Based on the rank sums.
5.  **Determine Critical Value or P-value:**
    * For small samples, you use specific tables for the Wilcoxon statistic.
    * For larger samples (typically $n > 10$ or $n > 20$ depending on the specific test), the sampling distribution of the test statistic approximates a normal distribution, and a Z-score approximation can be used to find a p-value.
6.  **Make a Decision:** Compare the calculated statistic/p-value to the critical value/$\alpha$.
7.  **Formulate a Conclusion:** Interpret the results in the context of your research question.

---

## Variants of the Wilcoxon Test (with Examples):

There are two primary variants of the Wilcoxon test, each addressing a different type of data comparison:

1.  **Wilcoxon Signed-Rank Test:**
    * **Purpose:** The non-parametric alternative to the **paired-samples t-test**. It is used to compare two related samples (e.g., before-and-after measurements on the same individuals, or matched pairs) when the differences are not normally distributed or the sample size is small. It assesses whether the median of the differences is significantly different from zero.
    * **Hypotheses:**
        * $H_0$: The median difference between the paired observations is zero. (No change/no effect)
        * $H_1$: The median difference is not zero (two-tailed), or is greater than zero (right-tailed), or is less than zero (left-tailed).
    * **Process:**
        1.  Calculate the difference ($d_i$) for each pair of observations.
        2.  Exclude any pairs where the difference is zero.
        3.  Take the absolute value of each non-zero difference ($|d_i|$).
        4.  Rank these absolute differences from smallest to largest. If there are ties, assign the average rank to the tied values.
        5.  Reapply the original sign of the difference to each rank. This creates "signed ranks."
        6.  Sum the positive ranks ($W_+$) and sum the negative ranks ($W_-$).
        7.  The test statistic ($W$) is typically the *smaller* of $|W_+|$ and $|W_-|$ (though some software uses the sum of positive ranks as $W$).
    * **Degrees of Freedom (df):** Not applicable in the same way as parametric tests. The critical values depend on the sample size (number of non-zero differences).
    * **Example:** A researcher wants to test if a new meditation technique reduces stress levels. 8 participants' stress levels are measured (on a scale of 1-10) before and after practicing the technique for a month. The data is not normally distributed.
        * **Null Hypothesis ($H_0$):** The median stress level before and after meditation is the same.
        * **Alternative Hypothesis ($H_1$):** The median stress level after meditation is lower than before (one-tailed, so we expect positive differences if 'Before' > 'After').

        | Participant | Before | After | Difference (B - A) | Abs. Difference | Rank of Abs. Diff. | Signed Rank |
        | :---------- | :----- | :---- | :----------------- | :-------------- | :----------------- | :---------- |
        | 1           | 7      | 5     | 2                  | 2               | 2                  | +2          |
        | 2           | 8      | 6     | 2                  | 2               | 2                  | +2          |
        | 3           | 6      | 6     | 0                  | -               | -                  | -           |
        | 4           | 9      | 7     | 2                  | 2               | 2                  | +2          |
        | 5           | 5      | 4     | 1                  | 1               | 1                  | +1          |
        | 6           | 7      | 8     | -1                 | 1               | 1                  | -1          |
        | 7           | 6      | 4     | 2                  | 2               | 2                  | +2          |
        | 8           | 10     | 8     | 2                  | 2               | 2                  | +2          |

        * **Note on Ties:** Here, we have multiple differences of 2 and 1. We would rank all the non-zero absolute differences (1, 1, 2, 2, 2, 2, 2).
            * Absolute differences of '1' occur twice, and would occupy ranks 1 and 2. So, they both get the average rank of $(1+2)/2 = 1.5$.
            * Absolute differences of '2' occur five times, and would occupy ranks 3, 4, 5, 6, 7. So, they all get the average rank of $(3+4+5+6+7)/5 = 5$.

        * **Corrected Table with Tie Handling:**
            | Participant | Before | After | Difference (B - A) | Abs. Difference | Rank of Abs. Diff. | Signed Rank |
            | :---------- | :----- | :---- | :----------------- | :-------------- | :----------------- | :---------- |
            | 1           | 7      | 5     | 2                  | 2               | 5                  | +5          |
            | 2           | 8      | 6     | 2                  | 2               | 5                  | +5          |
            | 3           | 6      | 6     | 0                  | -               | -                  | -           |
            | 4           | 9      | 7     | 2                  | 2               | 5                  | +5          |
            | 5           | 5      | 4     | 1                  | 1               | 1.5                | +1.5        |
            | 6           | 7      | 8     | -1                 | 1               | 1.5                | -1.5        |
            | 7           | 6      | 4     | 2                  | 2               | 5                  | +5          |
            | 8           | 10     | 8     | 2                  | 2               | 5                  | +5          |

        * **Sum of positive ranks ($W_+$):** $5+5+5+1.5+5+5 = 26.5$
        * **Sum of negative ranks ($W_-$):** $-1.5$
        * **Test Statistic (W):** The smaller of the absolute sums: $|-1.5| = 1.5$.
        * **Number of non-zero differences ($n$):** 7 (Participant 3 had a 0 difference).
        * For $n=7$ and a one-tailed test with $\alpha = 0.05$, the critical value for $W$ from a Wilcoxon signed-rank table is 4.
        * **Decision:** Since our calculated $W = 1.5$ is less than or equal to the critical value of 4, we reject the null hypothesis.
        * **Conclusion:** There is statistically significant evidence that the meditation technique reduces stress levels.

2.  **Wilcoxon Rank-Sum Test (also known as Mann-Whitney U Test):**
    * **Purpose:** The non-parametric alternative to the **independent-samples t-test**. It is used to compare two independent groups when their data is not normally distributed or is ordinal. It assesses whether two independent samples come from the same distribution (often interpreted as comparing medians, assuming similar shapes).
    * **Hypotheses:**
        * $H_0$: The distributions of the two populations are the same (or, the median of Group 1 is equal to the median of Group 2, assuming similar shapes).
        * $H_1$: The distributions are different (two-tailed), or one is stochastically larger/smaller than the other (one-tailed).
    * **Process (two common methods):**
        * **Method 1 (Wilcoxon Rank-Sum based):**
            1.  Combine all data from both groups into one dataset.
            2.  Rank all observations from smallest to largest, assigning average ranks for ties.
            3.  Sum the ranks for *one* of the groups (e.g., the smaller sample size, or arbitrarily Group 1). This sum is often denoted as $W$ or $R_1$.
            4.  Compare this sum to a critical value or calculate a p-value.
        * **Method 2 (Mann-Whitney U statistic based, mathematically equivalent):**
            1.  For every observation in Group 1, count how many observations in Group 2 are smaller than it. Sum these counts to get $U_1$.
            2.  For every observation in Group 2, count how many observations in Group 1 are smaller than it. Sum these counts to get $U_2$.
            3.  The test statistic ($U$) is the *smaller* of $U_1$ and $U_2$.
            * There's a relationship between $W$ (rank sum) and $U$: $U_1 = R_1 - \frac{n_1(n_1+1)}{2}$ and $U_2 = R_2 - \frac{n_2(n_2+1)}{2}$. So, they are equivalent.
    * **Degrees of Freedom (df):** Not applicable in the traditional sense. Critical values depend on the sample sizes of both groups ($n_1$ and $n_2$).
    * **Example:** A teacher wants to compare the effectiveness of two different online learning platforms (Platform A and Platform B) on student engagement scores (on a scale of 0-20). They randomly assign 6 students to Platform A and 7 students to Platform B. The engagement scores are not normally distributed.
        * **Platform A Scores:** 12, 15, 10, 18, 13, 11
        * **Platform B Scores:** 14, 9, 16, 17, 13, 11, 20
        * **Null Hypothesis ($H_0$):** The distributions of engagement scores for Platform A and Platform B are the same.
        * **Alternative Hypothesis ($H_1$):** The distributions of engagement scores are different (two-tailed).

        * **Combine and Rank All Scores:**
            | Score | Group | Rank |
            | :---- | :---- | :--- |
            | 9     | B     | 1    |
            | 10    | A     | 2    |
            | 11    | A     | 3.5  | (Tied with another 11)
            | 11    | B     | 3.5  | (Tied with another 11)
            | 12    | A     | 5    |
            | 13    | A     | 6.5  | (Tied with another 13)
            | 13    | B     | 6.5  | (Tied with another 13)
            | 14    | B     | 8    |
            | 15    | A     | 9    |
            | 16    | B     | 10   |
            | 17    | B     | 11   |
            | 18    | A     | 12   |
            | 20    | B     | 13   |

        * **Sum of Ranks for Group A ($R_A$):** $2 + 3.5 + 5 + 6.5 + 9 + 12 = 38$
        * **Sum of Ranks for Group B ($R_B$):** $1 + 3.5 + 6.5 + 8 + 10 + 11 + 13 = 53$
        * (Check: Total ranks = $N(N+1)/2 = 13(14)/2 = 91$. $R_A + R_B = 38 + 53 = 91$. Correct.)

        * **Calculate Mann-Whitney U statistics:**
            $U_A = R_A - \frac{n_A(n_A+1)}{2} = 38 - \frac{6(6+1)}{2} = 38 - \frac{6 \times 7}{2} = 38 - 21 = 17$
            $U_B = R_B - \frac{n_B(n_B+1)}{2} = 53 - \frac{7(7+1)}{2} = 53 - \frac{7 \times 8}{2} = 53 - 28 = 25$
            The test statistic $U$ is the smaller of $U_A$ and $U_B$, so $U = 17$.

        * **For $\alpha = 0.05$ (two-tailed), $n_1=6$, $n_2=7$:** We'd consult a Mann-Whitney U critical value table. The critical value for U is often 8 (or 9, depending on the table and how it's structured).
        * **Decision:** Since our calculated $U = 17$ is *greater than* the critical value (e.g., 8 or 9), we *fail to reject* the null hypothesis. (Note: For Mann-Whitney, usually smaller U values lead to rejection in two-tailed tests, or values outside a critical range. This is why using a p-value from software is often easier.)
        * **Conclusion (if using p-value):** If the p-value was, say, 0.20, we would fail to reject $H_0$. There is not enough evidence to conclude a significant difference in engagement scores between Platform A and Platform B.

**When to Choose Which Test (Parametric vs. Non-parametric):**

* **Continuous Data, Normally Distributed, Sufficient Sample Size:** Use Z-test (if $\sigma$ known) or T-test (if $\sigma$ unknown).
* **Continuous Data, Not Normally Distributed OR Small Sample Size, Paired:** Use Wilcoxon Signed-Rank Test.
* **Continuous Data, Not Normally Distributed OR Small Sample Size, Independent:** Use Wilcoxon Rank-Sum Test (Mann-Whitney U).
* **Categorical Data:** Use Chi-Square Test.

Wilcoxon tests are invaluable tools when the stricter assumptions of parametric tests cannot be met, providing robust alternatives for analyzing differences based on ranks.