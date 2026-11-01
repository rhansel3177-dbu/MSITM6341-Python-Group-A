# 4) Model Building & Evaluation

We trained two classifiers on the Task 2.1 training set (5,634 customers, 28 features): **Logistic Regression** and **Random Forest**. Only 26.5% of customers churn, so both models use `class_weight="balanced"`. Without it, a model can do well by mostly predicting "stays". Hyperparameters were tuned with 5-fold `GridSearchCV` on the training set only, scoring on F1. For Logistic Regression we tuned C, and the best value was 0.01. For Random Forest we tuned n_estimators and max_depth, and the best was 100 trees with max_depth = 10. All results below come from the held-out test set (1,409 customers, 374 churners).

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Predict everyone stays | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Baseline Logistic Regression | 0.733 | 0.498 | 0.791 | 0.612 | 0.842 |
| Baseline Random Forest | 0.785 | 0.624 | 0.479 | 0.542 | 0.826 |
| Tuned Logistic Regression | 0.742 | 0.510 | 0.773 | 0.614 | 0.839 |
| Tuned Random Forest | 0.767 | 0.546 | 0.717 | 0.620 | 0.841 |

**Why accuracy is misleading.** A model that predicts "everyone stays" is 73.5% accurate but finds no churners at all. The untuned Random Forest has the highest accuracy (78.5%) but misses more than half of the churners.

**Effect of tuning.** Tuning mattered most for Random Forest. Limiting tree depth stopped it from memorizing the training data, and recall rose from 48% to 72%. Logistic Regression barely changed, which suggests its default settings were already close to optimal.

**Best model: tuned Logistic Regression.** It found 77% of churners (about 289 of 374), and 51% of the customers it flagged really did leave. Tuned Random Forest found 72% (268 of 374) with 55% precision. Their F1 and ROC AUC are almost the same (about 0.84). We chose Logistic Regression on recall and F1 rather than accuracy, because losing a customer costs more than offering a discount to someone who would have stayed. It is also easier to explain to the business.

**Threshold trade-off.** At the default threshold of 0.5 the model catches 77% of churners. Lowering the threshold to 0.3 catches 93%, but precision falls to 41%, meaning more false alarms. The company can choose the threshold based on its retention budget.

**Key drivers.** Both models rank tenure, contract type, monthly charges, fiber-optic internet, and electronic-check payment near the top. Customers who are new, on month-to-month contracts, using fiber optic, or paying by electronic check churn most. Long tenure, one- or two-year contracts, and online security lower the risk.
