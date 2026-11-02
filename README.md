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

**Live Streamlit app:** <!-- add your streamlit.app link after deployment -->

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

Tuned hyperparameters: LR `C=10`*, Decision Tree `max_depth=None, min_samples_leaf=10`,
KNN `n_neighbors=5, weights=distance`, Gaussian NB `var_smoothing=1e-3`, SVM (RBF) `C=10, gamma=scale`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <!-- your observation --> |
| Decision Tree | <!-- your observation --> |
| KNN | <!-- your observation --> |
| Naive Bayes | <!-- your observation --> |
| Support Vector Machine | <!-- your observation --> |
| **Overall winner for this dataset** | <!-- your conclusion --> |
