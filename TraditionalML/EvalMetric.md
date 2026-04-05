### F1 Score vs ROC AUC vs Accuracy vs PR AUC: Which Evaluation Metric Should You Choose?

PR AUC and F1 Score are very robust evaluation metrics that work great for many classification problems, but from my experience, the most commonly used metrics are accuracy and ROC AUC. Are they better? Not really. As with the famous “AUC vs. accuracy” discussion, there are real benefits to using both. The big question is when. 

There are many questions that you may have right now:

- When is accuracy a better evaluation metric than ROC AUC?
- What is the F1 score good for?
- What is the PR curve, and how do you actually use it?
- If my dataset is highly imbalanced, should I use ROC AUC or PR AUC?

As always, it depends, but understanding the trade-offs between different metrics is crucial when it comes to making the correct decision.

In this blog post, I will:

- Talk about some of the most common binary classification metrics, like F1 score, ROC AUC, PR AUC, and accuracy.
- Compare them using an example binary classification problem.
- Tell you what you should consider when deciding to choose one metric over the other (F1 score vs. ROC AUC).

Ok, let’s do this!

## Evaluation metrics recap

I will start by introducing each of those classification metrics. Specifically:

- What is the definition and intuition behind it?
- The non-technical explanation
- How to calculate or plot it
- When should you use it?

> [!TIP]
> If you have read my previous blog post, [“24 Evaluation Metrics for Binary Classification (And When to Use Them)”](https://neptune.ai/blog/evaluation-metrics-binary-classification/),  you may want to skip this section and scroll down to the [evaluation metrics comparison](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc#h-evaluation-metrics-comparison).

### Accuracy

It measures how many observations, both positive and negative, were correctly classified.

$accuracy = \frac{tp+tn}{tp+fp+tn+fn}$

You shouldn’t use accuracy on imbalanced problems. Then, it is easy to get a high accuracy score by simply classifying all observations as the majority class.

In Python, you can calculate it in the following way:
```python
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = (tp + tn) / (tp + fp + fn + tn)

# or simply

accuracy_score(y_true, y_pred_class)
```

Since the accuracy score is calculated on the predicted classes (not prediction scores), we need to apply a certain threshold before computing it. The obvious choice is the threshold of 0.5, but it can be suboptimal.

Let’s see an example of how accuracy depends on the threshold choice:
![[Pasted image 20250604123643.png]]
You can use charts like the one above to determine the optimal threshold. In this case, choosing something a bit over the standard 0.5 could bump the score by a tiny bit (0.9686–0.9688), but in other cases, the improvement can be more substantial.

So, when does it make sense to use it? 

- When your problem is balanced, using accuracy is usually a good start. An additional benefit is that it is really easy to explain it to non-technical stakeholders in your project.
- When every class is equally important to you.

### F1 score

Simply put, it combines precision and recall into one metric by calculating the harmonic mean between those two. It is actually a special case of the more general function F beta:

$f_{\beta} = (1+{\beta}^2) \frac{{precision}*{recall}}{{\beta^2}*{precision}+{recall}}$

When choosing beta in your F-beta score, the more you care about recall over precision, the higher beta you should choose. For example, with the F1 score, we care equally about recall and precision; with the F2 score, recall is twice as important to us.

![F beta by beta](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/f_by_beta.png?ssl=1)

_F beta threshold by beta_

With 0<beta<1, we care more about precision, and so the higher the threshold, the higher the F beta score. When beta > 1, our optimal threshold moves toward lower thresholds, and when beta = 1, it is somewhere in the middle.

It can be easily computed by running:

```python
from sklearn.metrics import f1_score

y_pred_class = y_pred_pos > threshold
f1_score(y_true, y_pred_class)
```

It is important to remember that the F1 score is calculated from precision and recall, which, in turn, are calculated from the predicted classes (not prediction scores).

How should we choose an optimal threshold? Let’s plot the F1 score over all possible thresholds:

![f1 score by threshold](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/f1_by_thres.png?ssl=1)

_F1 score by threshold_

We can adjust the threshold to optimize the F1 score. Notice that for both precision and recall, you could get perfect scores by increasing or decreasing the threshold. The good thing is that you can find a sweet spot for F1 scores. As you can see, getting the threshold just right can actually improve your score from 0.8077->0.8121.

When should you use it?

- Pretty much in every binary classification problem where you care more about the positive class. It is my go-to metric when working on those problems. 
- It can be easily explained to business stakeholders, which in many cases can be a deciding factor. Always remember that machine learning is just a tool to solve a business problem. 

### ROC AUC

AUC means “area under the curve.” So, to speak about the ROC AUC score, we need to define the ROC curve first. 

It is a chart that visualizes the trade-off between the true positive rate (TPR) and the false positive rate (FPR). Basically, for every threshold, we calculate TPR and FPR and plot them on one chart.

Of course, the higher the TPR and the lower the FPR for each threshold, the better, and so classifiers that have curves that are more top-left-side are better.

An extensive discussion of the ROC curve and the ROC AUC score can be found in this [article by Tom Fawcett](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.9777&rep=rep1&type=pdf).

![roc curve](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/roc_auc_curve.png?ssl=1)

_ROC curves_

We can see a healthy ROC curve pushed towards the top-left side for both positive and negative classes. It is not clear which one performs better across the board, as with FPR < ~0.15 the positive class is higher, and starting from FPR~0.15 the negative class is above.

In order to get one number that tells us how good our curve is, we can calculate the Area Under the ROC Curve or ROC AUC score. The more top-left your curve is, the higher the area, and hence, the higher the ROC AUC score.

Alternatively, [it can be shown](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Area-under-curve_\(AUC\)_statistic_for_ROC_curves) that the ROC AUC score is equivalent to calculating the rank correlation between predictions and targets. From an interpretation standpoint, it is more useful because it tells us that this metric shows how good your model is at ranking predictions. It tells you what the probability is that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

```python
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred_pos)
```
- You should use it when you ultimately care about ranking predictions and not necessarily about outputting well-calibrated probabilities (read this [article by Jason Brownlee](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/) if you want to learn about probability calibration).
- You should not use it when your data is heavily imbalanced. This was discussed extensively in this [article by Takaya Saito and Marc Rehmsmeier](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/). The intuition is the following: the false positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives.
- You should use it when you care equally about positive and negative classes. It naturally extends the imbalanced data discussion from the last section. If we care about true negatives as much as we care about true positives, then it totally makes sense to use ROC AUC.

### PR AUC | Average Precision

Similarly to ROC AUC, in order to define PR AUC, we need to define the precision-recall curve.

It is a curve that combines precision (PPV) and recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot them. The higher the y-axis on your curve, the better your model’s performance.

You can use this plot to make an educated decision when it comes to the classic precision/recall dilemma. Obviously, the higher the recall, the lower the precision. Knowing at which recall your precision starts to fall fast can help you choose the threshold and deliver a better model.

![precision recall curve](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/prec_rec_curve.png?ssl=1)

_Precision-Recall curve_

We can see that for the negative class, we maintain high precision and recall almost throughout the entire range of thresholds. For the positive class, precision starts to fall as soon as we recall 0.2 of true positives, and by the time we hit 0.8, it decreases to around 0.7.

Similarly to the ROC AUC score, you can calculate the area under the precision-recall curve (PR AUC) to get one number that describes model performance.

You can also think of PR AUC as the average of precision scores calculated for each recall threshold. You can also adjust this definition to suit your business needs by choosing or clipping recall thresholds if needed.

```python
from sklearn.metrics import average_precision_score

average_precision_score(y_true, y_pred_pos)
```

- when you want to communicate a precision or recall decision to other stakeholders
- when you want to choose the threshold that fits the business problem.
- when your data is heavily imbalanced. As mentioned before, it was discussed extensively in this [article by Takaya Saito and Marc Rehmsmeier](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/). The intuition is the following: since PR AUC focuses mainly on the positive class (PPV and TPR), it cares less about the frequent negative class.
- when you care more about positive than negative class. If you care more about the positive class, and hence PPV and TPR, you should go with the precision-recall curve and PR AUC (average precision).

## Evaluation metrics comparison

We will compare the metrics we discussed so far with a use case that’s close to what you might typically see day-to-day as data scientists.

Based on a [Kaggle competiton](https://www.kaggle.com/c/ieee-fraud-detection/overview) I created an example fraud detection problem:

- I selected only 43 features. 
- I sampled 66000 observations from the original dataset.
- I adjusted the fraction of the positive class to 0.09.

We’ll train a bunch of [LightGBM classifiers](https://lightgbm.readthedocs.io/en/stable/) with different hyperparameters and will use the metrics to get an intuition as to which models are “truly” better. Specifically, I suspect that the model with only 10 trees is worse than a model with 100 trees. Of course, with more trees and smaller learning rates, it gets tricky, but I think it is a decent proxy. 

To generate the results you will see below, run the following snippets of code in unison by changing the [hyperparameters of LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

### [Reference](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc#:~:text=F1%20score%20vs%20Accuracy&text=That%20makes%20a%20big%20difference,the%20metric%20you%20should%20choose.)
