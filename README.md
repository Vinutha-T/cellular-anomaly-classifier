# LTE Cell Anomaly Classifier

## a. Problem statement
Mobile operators monitor thousands of LTE cells through KPIs collected every 15 minutes. Unusual
cell behaviour (abnormal load, throughput or user counts) can indicate faults, misconfiguration or
unexpected traffic events, and should be flagged to the network operations team automatically.
The goal is to build and compare five classification models that predict whether a 15-minute
cell sample is **normal (0)** or **anomalous (1)**, and to expose them through an interactive
Streamlit app that evaluates the saved models on uploaded test data.

## b. Dataset description
- **Source:** Kaggle, *Anomaly Detection in 4G Cellular Networks* (ML-MATT competition),
  file `ML-MATT-CompetitionQT2021_train.csv`
- **Instances:** 36,904 (36,801 after removing 103 duplicate rows)
- **Input features (13):**

| Feature | Description |
|---|---|
| `Time` | Time of day of the 15-minute sample (HH:MM) |
| `CellName` | LTE cell identifier (33 cells, e.g. `6ALTE` = site 6, sector A) |
| `PRBUsageUL`, `PRBUsageDL` | Physical Resource Block utilisation, uplink / downlink |
| `meanThr_DL`, `meanThr_UL` | Mean throughput, downlink / uplink |
| `maxThr_DL`, `maxThr_UL` | Maximum throughput, downlink / uplink |
| `meanUE_DL`, `meanUE_UL` | Mean number of active UEs, downlink / uplink |
| `maxUE_DL`, `maxUE_UL` | Maximum number of active UEs, downlink / uplink |
| `maxUE_UL+DL` | Maximum active UEs, uplink + downlink combined |

- **Target:** `Unusual`, where 0 = normal (72.4%) and 1 = anomaly (27.6%). This is an imbalanced binary classification problem.
- **Preprocessing:** duplicates removed; `Time` converted to cyclic hour-of-day (sin/cos);
  `CellName` one-hot encoded; KPIs log1p-transformed and standardised. All preprocessing lives
  inside each saved scikit-learn pipeline.
- **Split:** stratified 80/20 (seed 2026), giving 29,440 training and 7,361 test samples (`test_data.csv`).

## c. GitHub repository
https://github.com/Vinutha-T/cellular-anomaly-classifier

**Live Streamlit app:** https://lte-cell-anomaly-vinutha.streamlit.app/

```
cellular-anomaly-classifier/
├── app.py              Streamlit app (loads saved models, no retraining)
├── train_models.py     Preprocessing, grid search, training, evaluation, model export
├── features.py         Shared column definitions, time encoding and metric helpers
├── requirements.txt
├── test_data.csv       Hold-out test set (raw features + Unusual)
├── data/               Original Kaggle training file
└── model/              Five trained pipelines (.joblib) + metrics.json
```

Run locally:
```bash
pip install -r requirements.txt
python train_models.py   # optional, the models are already in model/
streamlit run app.py
```

## d. Models used
All five models were trained on the same training split. Hyperparameters were tuned with 3-fold
stratified grid search optimising MCC, and the models were evaluated on the hold-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7720 | 0.7701 | 0.7042 | 0.3039 | 0.4246 | 0.3519 |
| Decision Tree | 0.9626 | 0.9736 | 0.9603 | 0.9023 | 0.9304 | 0.9057 |
| KNN | 0.7919 | 0.8009 | 0.7124 | 0.4158 | 0.5251 | 0.4274 |
| Naive Bayes | 0.6183 | 0.5947 | 0.3466 | 0.4286 | 0.3832 | 0.1126 |
| Support Vector Machine | 0.8548 | 0.8995 | 0.8168 | 0.6127 | 0.7001 | 0.6178 |

Tuned hyperparameters: LR `C=10`, Decision Tree `max_depth=None, min_samples_leaf=10`,
KNN `n_neighbors=5, weights=distance`, Gaussian NB `var_smoothing=1e-3`, SVM (RBF) `C=10, gamma=scale`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy (0.772) is slightly above the 72% obtained by always predicting "normal", and recall is just 0.30, so it misses about 70% of anomalies. A linear boundary cannot represent the narrow, non-monotonic bands of `meanUE_UL` and `PRBUsageUL` in which anomalies occur, even after log scaling. Precision (0.70) is reasonable, so its few alarms are mostly genuine, but MCC of 0.35 shows weak overall agreement. |
| Decision Tree | Clearly the best model on every metric (accuracy 0.963, F1 0.930, MCC 0.906). Its axis-aligned threshold splits fit the anomaly structure: `meanUE_UL` alone provides about 83% of the feature importance and `PRBUsageUL` about 11%, while cell ID and hour of day contribute almost nothing. The tree is deep (41 levels, 428 leaves), but `min_samples_leaf=10` limits overfitting, and test MCC (0.906) closely matches cross-validation MCC (0.911). |
| KNN | Moderate performance (F1 0.525, MCC 0.427) with low recall (0.42). Distances are computed over 46 dimensions after one-hot encoding, most of which (33 cell columns, hour, and correlated throughput/UE KPIs) carry little signal, so the informative uplink features are diluted and neighbours are often not meaningful. It is also slower at prediction time and has the largest saved model (about 1.9 MB), because it stores the whole training set. |
| Naive Bayes | The weakest model (accuracy 0.618, below the 72% majority baseline; MCC 0.113, close to random). Its assumptions are badly violated: features are strongly dependent (`maxUE_UL+DL` is exactly `maxUE_DL + maxUE_UL`, and `maxUE_UL` correlates 0.94 with it), `meanUE_UL` takes two clusters of values rather than a Gaussian shape, and the one-hot cell columns are binary. It does reach a recall of 0.43, but at the cost of many false alarms (precision 0.35). |
| Support Vector Machine | Second best (accuracy 0.855, AUC 0.900, MCC 0.618). The RBF kernel captures non-linear boundaries that Logistic Regression cannot, improving recall from 0.30 to 0.61, but a smooth kernel boundary still approximates the sharp threshold bands less well than a tree. It was also the most expensive model to train (about 110 s for the grid search, compared with under 10 s for the others). |
| **Overall winner for this dataset** | **Decision Tree.** It leads on all six metrics, with MCC 0.906 against 0.618 for the runner-up SVM, and catches 90% of anomalies with 96% precision. Anomalies in this data are defined by sharp thresholds on a few uplink KPIs, which rule-based splits model naturally. It is also fast to train, small (22 KB) and interpretable, which suits network operations, where engineers need to understand why a cell was flagged. Because the classes are imbalanced, MCC and F1 were used to rank the models instead of accuracy. |

