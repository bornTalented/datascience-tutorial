The Chi-Square ($\chi^2$) test is a non-parametric statistical test primarily used for **categorical data**. Unlike the Z-test and T-test which deal with means of continuous data, the Chi-Square test is concerned with **frequencies or counts** and examines whether observed frequencies differ significantly from expected frequencies.

**Core Concept of Chi-Square Test**

The fundamental idea behind the Chi-Square test is to compare the observed frequencies (the actual counts you collect from your sample) with the expected frequencies (the counts you would expect if the null hypothesis were true, i.e., if there were no effect or no relationship).

The Chi-Square statistic quantifies the difference between these observed and expected frequencies. A larger Chi-Square value indicates a greater discrepancy between what you observed and what you expected by chance.

**Formula (General Form):**

$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$

Where:
* $\chi^2$ = Chi-Square test statistic
* $\sum$ = Summation across all categories or cells
* $O_i$ = Observed frequency in category/cell $i$
* $E_i$ = Expected frequency in category/cell $i$

**Key Assumptions for Chi-Square Test:**

1.  **Categorical Data:** The variables being analyzed must be categorical (nominal or ordinal).
2.  **Independent Observations:** Each observation must be independent of all other observations. This means that each subject/case contributes data to only one category/cell.
3.  **Random Sampling:** The data should be obtained from a random sample.
4.  **Sufficient Expected Frequencies:** This is crucial.
    * Typically, all expected cell counts ($E_i$) should be at least 5.
    * Some sources suggest that no more than 20% of the cells should have expected counts less than 5, and no cell should have an expected count less than 1.
    * If this assumption is violated (especially with small expected counts), the Chi-Square test might not be reliable, and alternative tests (like Fisher's Exact Test) might be more appropriate.

**General Steps to Perform a Chi-Square Test:**

1.  **State the Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$):** These will vary based on the specific type of Chi-Square test.
2.  **Choose a Significance Level ($\alpha$):** Common values are 0.05 or 0.01.
3.  **Create a Contingency Table (if applicable):** Organize your observed frequencies into a table.
4.  **Calculate Expected Frequencies ($E_i$):** Determine what you would expect to see in each category/cell if the null hypothesis were true.
5.  **Calculate the Chi-Square Statistic ($\chi^2$):** Use the formula provided above.
6.  **Determine Degrees of Freedom (df):** The formula for df depends on the specific variant of the test.
7.  **Determine the Critical Value or P-value:**
    * **Critical Value Approach:** Look up the critical $\chi^2$ value in a Chi-Square distribution table using your chosen $\alpha$ and df.
    * **P-value Approach:** Calculate the p-value associated with your computed $\chi^2$ statistic and df using statistical software.
8.  **Make a Decision:**
    * **Critical Value Approach:** If the calculated $\chi^2$ statistic is greater than the critical $\chi^2$ value, reject $H_0$.
    * **P-value Approach:** If the p-value is less than or equal to $\alpha$, reject $H_0$.
9.  **Formulate a Conclusion:** State your conclusion in the context of the problem.

---

## Variants of the Chi-Square Test (with Examples):

The most common variants of the Pearson's Chi-Square test are for "Goodness-of-Fit" and "Independence." A third variant, the "Test for Homogeneity," is mathematically identical to the Test of Independence but differs in its experimental design and the way the data is collected.

1.  **Chi-Square Goodness-of-Fit Test:**
    * **Purpose:** To determine if an observed frequency distribution for a **single categorical variable** differs significantly from a hypothesized or expected distribution. In other words, does your sample data "fit" a theoretical distribution?
    * **Formula:** Same as the general Chi-Square formula: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$
        * **Degrees of Freedom (df):** $k - 1$, where $k$ is the number of categories.
    * **Example:** A geneticist theorizes that a certain genetic cross should produce offspring with flower colors in a specific ratio: 25% red, 50% pink, and 25% white. In an experiment, 120 offspring are observed, yielding: 28 red, 62 pink, and 30 white flowers. At $\alpha = 0.05$, do these observed frequencies fit the hypothesized genetic ratio?
        * **Categories:** Red, Pink, White
        * **Hypothesized Proportions ($P_i$):** Red = 0.25, Pink = 0.50, White = 0.25
        * **Total Observations ($N$):** 120
        * **Observed Frequencies ($O_i$):** $O_{red} = 28$, $O_{pink} = 62$, $O_{white} = 30$

        * **Calculate Expected Frequencies ($E_i$):**
            * $E_{red} = N \times P_{red} = 120 \times 0.25 = 30$
            * $E_{pink} = N \times P_{pink} = 120 \times 0.50 = 60$
            * $E_{white} = N \times P_{white} = 120 \times 0.25 = 30$

        * **Hypotheses:**
            * $H_0$: The observed distribution of flower colors fits the hypothesized 25%:50%:25% ratio.
            * $H_1$: The observed distribution does not fit the hypothesized ratio.

        * **Calculate $\chi^2$ Statistic:**
            $\chi^2 = \frac{(28-30)^2}{30} + \frac{(62-60)^2}{60} + \frac{(30-30)^2}{30}$
            $\chi^2 = \frac{(-2)^2}{30} + \frac{(2)^2}{60} + \frac{(0)^2}{30}$
            $\chi^2 = \frac{4}{30} + \frac{4}{60} + 0$
            $\chi^2 = 0.133 + 0.067 + 0 = 0.20$

        * **Degrees of Freedom:** $df = k - 1 = 3 - 1 = 2$
        * For $\alpha = 0.05$ and $df = 2$, the critical $\chi^2$ value is 5.991.
        * **Decision:** Since the calculated $\chi^2 = 0.20$ is less than the critical value of 5.991, we fail to reject the null hypothesis.
        * **Conclusion:** There is not enough evidence to suggest that the observed distribution of flower colors significantly differs from the hypothesized 25%:50%:25% ratio. The observed frequencies fit the genetic model.

2.  **Chi-Square Test of Independence:**
    * **Purpose:** To determine if there is a statistically significant association or relationship between **two categorical variables** from a single population. In other words, are the two variables independent of each other?
    * **Formula:** Same as the general Chi-Square formula: $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$ (where $i$ refers to rows and $j$ to columns in a contingency table).
        * **Degrees of Freedom (df):** $(r - 1)(c - 1)$, where $r$ is the number of rows and $c$ is the number of columns in the contingency table.
        * **Expected Frequencies Calculation:** For each cell, $E_{ij} = \frac{(\text{Row Total}_i \times \text{Column Total}_j)}{\text{Grand Total}}$
    * **Example:** A survey asks 300 randomly selected adults about their highest education level and their preference for a particular brand of coffee (Brand X vs. Other).
        * **Categorical Variable 1:** Education Level (High School, Bachelor's, Graduate)
        * **Categorical Variable 2:** Coffee Preference (Brand X, Other)

        **Observed Frequencies:**
        | Education Level | Brand X | Other | Row Total |
        | :-------------- | :------ | :---- | :-------- |
        | High School     | 50      | 80    | 130       |
        | Bachelor's      | 60      | 40    | 100       |
        | Graduate        | 20      | 50    | 70        |
        | **Column Total**| **130** | **170**| **300** |

        * **Hypotheses:**
            * $H_0$: Education level and coffee preference are independent (no association).
            * $H_1$: Education level and coffee preference are dependent (there is an association).

        * **Calculate Expected Frequencies for each cell (using $E_{ij} = \frac{(\text{Row Total}_i \times \text{Column Total}_j)}{\text{Grand Total}}$):**
            * $E_{\text{HS, Brand X}} = (130 \times 130) / 300 = 56.33$
            * $E_{\text{HS, Other}} = (130 \times 170) / 300 = 73.67$
            * $E_{\text{Bach, Brand X}} = (100 \times 130) / 300 = 43.33$
            * $E_{\text{Bach, Other}} = (100 \times 170) / 300 = 56.67$
            * $E_{\text{Grad, Brand X}} = (70 \times 130) / 300 = 30.33$
            * $E_{\text{Grad, Other}} = (70 \times 170) / 300 = 39.67$

        * **Calculate $\chi^2$ Statistic:**
            $\chi^2 = \frac{(50-56.33)^2}{56.33} + \frac{(80-73.67)^2}{73.67} + \frac{(60-43.33)^2}{43.33} + \frac{(40-56.67)^2}{56.67} + \frac{(20-30.33)^2}{30.33} + \frac{(50-39.67)^2}{39.67}$
            $\chi^2 \approx 0.71 + 0.54 + 6.30 + 4.67 + 3.55 + 2.63 \approx 18.4$

        * **Degrees of Freedom:** $df = (r - 1)(c - 1) = (3 - 1)(2 - 1) = 2 \times 1 = 2$
        * For $\alpha = 0.05$ and $df = 2$, the critical $\chi^2$ value is 5.991.
        * **Decision:** Since the calculated $\chi^2 = 18.4$ is greater than the critical value of 5.991, we reject the null hypothesis.
        * **Conclusion:** There is a significant association between education level and coffee preference.

3.  **Chi-Square Test for Homogeneity:**
    * **Purpose:** To determine if the distribution of a **single categorical variable** is the same across **two or more independent populations or groups**. It asks if the populations are homogeneous with respect to the characteristic.
    * **Formula:** Mathematically identical to the Test of Independence: $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$
        * **Degrees of Freedom (df):** $(r - 1)(c - 1)$
        * **Expected Frequencies Calculation:** For each cell, $E_{ij} = \frac{(\text{Row Total}_i \times \text{Column Total}_j)}{\text{Grand Total}}$
    * **Key Distinction from Test of Independence:** The difference lies in the sampling design.
        * **Independence:** One sample from one population, classified by two variables.
        * **Homogeneity:** Multiple samples (one from each population/group), and for each sample, subjects are classified according to one variable. The row/column totals for the independent variable are fixed by the researcher.
    * **Example:** A marketing firm wants to see if the TV viewing preferences (News, Drama, Comedy) are the same for adults in **City A** and **City B**. They randomly sample 150 adults from City A and 200 adults from City B.
        * **Independent Variable (Grouping Factor):** City (City A, City B) - these are the populations
        * **Dependent Variable (Characteristic):** TV Preference (News, Drama, Comedy)

        **Observed Frequencies:**
        | City  | News | Drama | Comedy | Row Total (Fixed by Sample Size) |
        | :---- | :--- | :---- | :----- | :------------------------------- |
        | City A| 40   | 60    | 50     | 150                              |
        | City B| 70   | 90    | 40     | 200                              |
        | **Column Total**| **110**| **150** | **90** | **350** |

        * **Hypotheses:**
            * $H_0$: The distribution of TV viewing preferences is the same for City A and City B.
            * $H_1$: The distribution of TV viewing preferences differs between City A and City B.

        * **Calculate Expected Frequencies:** (Same process as independence test)
            * $E_{\text{A, News}} = (150 \times 110) / 350 = 47.14$
            * $E_{\text{A, Drama}} = (150 \times 150) / 350 = 64.29$
            * $E_{\text{A, Comedy}} = (150 \times 90) / 350 = 38.57$
            * $E_{\text{B, News}} = (200 \times 110) / 350 = 62.86$
            * $E_{\text{B, Drama}} = (200 \times 150) / 350 = 85.71$
            * $E_{\text{B, Comedy}} = (200 \times 90) / 350 = 51.43$

        * **Calculate $\chi^2$ Statistic:** (Calculations are similar to the independence test, comparing observed to expected for each cell and summing)
            $\chi^2 = \frac{(40-47.14)^2}{47.14} + \frac{(60-64.29)^2}{64.29} + \frac{(50-38.57)^2}{38.57} + \frac{(70-62.86)^2}{62.86} + \frac{(90-85.71)^2}{85.71} + \frac{(40-51.43)^2}{51.43}$
            $\chi^2 \approx 1.08 + 0.29 + 3.32 + 0.81 + 0.21 + 2.54 \approx 8.25$

        * **Degrees of Freedom:** $df = (r - 1)(c - 1) = (2 - 1)(3 - 1) = 1 \times 2 = 2$
        * For $\alpha = 0.05$ and $df = 2$, the critical $\chi^2$ value is 5.991.
        * **Decision:** Since the calculated $\chi^2 = 8.25$ is greater than the critical value of 5.991, we reject the null hypothesis.
        * **Conclusion:** There is a significant difference in TV viewing preferences between City A and City B.

**Other "Variants" (Special Cases/Related Tests):**

* **Yates' Correction for Continuity:** Applied to Chi-Square tests (especially for 2x2 tables) when expected frequencies are small (e.g., between 5 and 10). It slightly reduces the calculated Chi-Square value to provide a more conservative result, better approximating the continuous chi-square distribution with discrete data.
* **Fisher's Exact Test:** Used for 2x2 contingency tables when expected cell counts are very small (typically less than 5), where the Chi-Square test assumptions are violated. It directly calculates the exact probability of observing the given cell frequencies, assuming the marginal totals are fixed.
* **McNemar's Test:** A specific Chi-Square based test used for **paired categorical data**, often in before-and-after designs or when comparing two related proportions (e.g., same subjects rated by two different observers). It tests the marginal homogeneity of a 2x2 table for dependent samples.

In summary, the Chi-Square test is your go-to tool when working with categorical data, allowing you to assess if observed counts deviate from expected counts (Goodness-of-Fit) or if two categorical variables are related (Independence/Homogeneity).