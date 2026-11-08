# Student Dropout & Academic Success – Multi-Model Classification

**ML Assignment 2 · M.Tech (AIML), BITS Pilani WILP**
**Name:** Chaitanya Rai  **BITS ID:** 2026ak05673

- **GitHub repository:** https://github.com/Chaitanyarai899/ml-assignment2-student-dropout
- **Live Streamlit app:** STREAMLIT_APP_URL

---

## a. Problem statement

Universities lose a significant share of every intake to dropout, and the earlier an at-risk student is identified, the more useful an intervention (mentoring, financial aid, academic support) can be. The goal of this project is to predict, from information available at enrolment and at the end of the first two semesters, whether a student will **drop out**, **remain enrolled** beyond the normal course duration, or **graduate** on time.

This is a **3-class classification problem**. Five classical classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared with six evaluation metrics. The trained models are served through an interactive Streamlit app that evaluates them on uploaded test data.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | *Predict Students' Dropout and Academic Success* |
| Source | UCI Machine Learning Repository, dataset id 697 ([link](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)) (Realinho et al., 2021) |
| Instances | 4,424 students (one row per student) |
| Features | 36 input features (all numeric or integer-coded categorical) + 1 target |
| Target | `Target` ∈ {Dropout, Enrolled, Graduate} |
| Class balance | Graduate 2,209 (49.9%) · Dropout 1,421 (32.1%) · Enrolled 794 (17.9%) |
| Missing values | None |

The data comes from a higher-education institution in Portugal and covers undergraduate degrees such as agronomy, design, education, nursing, journalism, management, social service and technologies. The features fall into five groups:

- **Demographic:** marital status, nationality, gender, age at enrolment, displaced, international.
- **Socio-economic:** mother's/father's qualification and occupation, scholarship holder, debtor, tuition fees up to date, educational special needs.
- **Academic path at enrolment:** application mode and order, course, day/evening attendance, previous qualification and its grade, admission grade.
- **Academic performance:** curricular units credited/enrolled/evaluated/approved, grade and units without evaluation for the **1st and 2nd semesters**.
- **Macro-economic:** unemployment rate, inflation rate, GDP at the time of enrolment.

**Preprocessing.** The raw UCI file is `;`-separated with a UTF-8 BOM and a stray tab in one header, so the header is cleaned on load (`Nacionality` is also renamed to `Nationality`). The target is label-encoded. The data is split **80/20 with stratification** (`random_state=5673`), giving 3,539 training and 885 test rows. The test split is saved as `test_data.csv`. Each scale-sensitive model (LR, KNN, NB, SVM) is wrapped in a `Pipeline` with a `StandardScaler` fitted on the training data only, so the saved `.joblib` file contains its own preprocessing and there is no train/test leakage.

## c. GitHub repository link

https://github.com/Chaitanyarai899/ml-assignment2-student-dropout

```
ml-assignment2-student-dropout/
|-- app.py                     # Streamlit app (inference only, no retraining)
|-- train_models.py            # trains the 5 models and writes metrics + artifacts
|-- requirements.txt
|-- README.md
|-- test_data.csv              # held-out 20% split (885 rows, 36 features + Target)
|-- data/
|   `-- student_dropout_full.csv   # original UCI file
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- label_encoder.joblib       # preprocessing artifact (target encoding)
    |-- feature_columns.json       # preprocessing artifact (expected column order)
    `-- metrics.json               # metrics reported below
```

**Reproduce:** `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler → multinomial LogisticRegression (C = 0.5, max_iter = 3000) |
| Decision Tree | DecisionTreeClassifier (max_depth = 7, min_samples_leaf = 12) |
| K-Nearest Neighbour | StandardScaler → KNN (k = 15, distance-weighted) |
| Naive Bayes | StandardScaler → GaussianNB (features are continuous/integer-coded, so Gaussian was chosen over Multinomial) |
| Support Vector Machine | StandardScaler → RBF-kernel SVC (C = 1.5, gamma = "scale"), probability-calibrated with `CalibratedClassifierCV` so AUC can be computed |

**Metric definitions for 3 classes:** Precision, Recall and F1 are **support-weighted averages** over the three classes. AUC is the **one-vs-rest ROC-AUC, weighted by support**. MCC is the multi-class Matthews correlation coefficient. All values below are measured on the 885-row held-out test set.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7831** | **0.9104** | **0.7649** | **0.7831** | **0.7669** | **0.6401** |
| Decision Tree | 0.7356 | 0.8614 | 0.7262 | 0.7356 | 0.7270 | 0.5621 |
| KNN | 0.7153 | 0.8585 | 0.6928 | 0.7153 | 0.6748 | 0.5282 |
| Naive Bayes | 0.6915 | 0.8123 | 0.6729 | 0.6915 | 0.6753 | 0.4830 |
| Support Vector Machine | 0.7695 | 0.8807 | 0.7544 | 0.7695 | 0.7510 | 0.6180 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on all six metrics (Acc 0.783, AUC 0.910, MCC 0.640). Most of the signal sits in the 1st/2nd-semester "units approved" and "grade" features, whose effect on the log-odds is close to monotonic, so a linear decision boundary on standardised features fits well and generalises better than the more flexible models. Its weak spot, shared by every model, is the *Enrolled* class (recall 0.34): those students look academically like a mix of graduates and dropouts. |
| Decision Tree | Mid-table (Acc 0.736, MCC 0.562). Depth 7 with at least 12 samples per leaf keeps it from overfitting, and its splits on approved units and tuition-fee status are easy to read. Axis-aligned piecewise boundaries are coarser than the smooth ones of LR/SVM, which costs about 5 points of accuracy and gives the lowest AUC after NB, because leaf probabilities are coarse. It does have the best *Enrolled* recall (0.37), as its hard splits carve out a dedicated region for that class. |
| KNN | Accuracy (0.715) is reasonable, but weighted F1 (0.675) and MCC (0.528) show the problem: it almost never predicts *Enrolled* (recall 0.13) and pushes uncertain points to the majority *Graduate* class (recall 0.97). With 36 dimensions, many of them integer-coded categoricals that lose meaning under Euclidean distance, neighbourhoods are noisy and dominated by the majority class. |
| Naive Bayes | Weakest model (Acc 0.692, AUC 0.812, MCC 0.483). The independence assumption is badly violated: the 1st- and 2nd-semester features are strongly correlated (enrolled, evaluated and approved units, and grades), so their evidence is effectively counted several times and the probabilities become over-confident. Many features are coded categoricals that are not Gaussian. It is still fast and gives a useful baseline. |
| Support Vector Machine | A close second (Acc 0.770, MCC 0.618). The RBF kernel captures mild non-linearities, but here it gains nothing over the linear model, which suggests the classes are almost linearly separable in the standardised feature space. It has the highest *Dropout* precision among the strong models (0.837). AUC (0.881) is a little lower than LR's because its probabilities come from post-hoc calibration rather than a native probabilistic model. |
| **Overall winner for this dataset** | **Logistic Regression.** It has the highest accuracy, AUC, F1 and MCC, is the cheapest to train and serve, and its coefficients are interpretable, which helps an institution explain *why* a student was flagged. SVM is a close runner-up. For a dropout early-warning system, LR's high Dropout precision/recall (0.82/0.77) and best AUC make it the most practical choice. The main remaining gap is the minority *Enrolled* class; class weighting or resampling would be the next step. |

## Streamlit app features

- **CSV upload** of test data (the bundled `test_data.csv` is used by default and can be downloaded from the sidebar).
- **Model selection dropdown** for the five saved classifiers.
- **Evaluation metrics** for the chosen model: Accuracy, AUC, Precision, Recall, F1 and MCC.
- **Confusion matrix** heatmap and **classification report**, plus predicted vs actual class distribution.
- **Row-level predictions** with confidence, a "misclassified only" filter and a CSV download.
- **Compare all models** tab, which scores all five on the uploaded file, highlights the best value per metric and plots any chosen metric.
- The app checks the uploaded file for missing columns and unknown labels. It **never retrains**; it only loads the `.joblib` pipelines from `model/`.

## Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
