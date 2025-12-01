# Predicting CPU Burst Times with Machine Learning
Comparative analysis and implementation of ML/AI algorithms and a neural network model to predict CPU burst times of processes, with application to operating systems scheduling (SJF/SRTF).

This project reproduces and extends the idea from academic work that more accurate CPU burst predictions can enhance the performance of Shortest-Job-First (SJF) and Shortest-Remaining-Time-First (SRTF) schedulers by providing better runtime estimates.

Contents
- How to run the project
- Dataset description
- Algorithms / Models
- Implementation steps
- Hardware / Software environment
- Comparative analysis outputs
- Conclusion
- References

---

## How to Run the Project

You can run this project locally (Python) or in a hosted environment (e.g., Google Colab).

Option A — Run in Google Colab (recommended)
1. Open a new Colab notebook.
2. Copy-paste the full pipeline code from the repository notebook/script into a notebook cell.
3. Run the cell. The code will:
   - Download and load the dataset,
   - Perform full EDA,
   - Preprocess and split the data,
   - Train multiple models,
   - Evaluate and visualize results,
   - Provide a `predict_cpu_burst(...)` function to get predictions from all models.

Option B — Run locally (Python)
1. Ensure you have Python 3.9+ installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -U pip
   pip install pandas numpy matplotlib seaborn missingno scikit-learn xgboost lightgbm
   ```
4. Run the script or Jupyter Notebook:
   - Script:
     ```bash
     python main.py
     ```
   - Notebook:
     ```bash
     pip install notebook
     jupyter notebook
     ```
     Open the notebook (e.g., cpu_burst_prediction.ipynb) and run all cells.

5. Optional: If you encounter build issues on xgboost/lightgbm (especially on Apple Silicon/Windows), you may temporarily disable those two models in the code or install prebuilt wheels following the libraries’ docs.

Outputs
- The pipeline prints summaries to the console and renders plots inline (Notebook/Colab).
- Comparative tables are displayed as DataFrames.
- The prediction utility prints and plots per-model predictions for any given sample(s).

---

## Dataset Description

Source
- Dropbox file: https://www.dropbox.com/s/ikyxo0zew514a0b/processes_datasets.csv

Observed download log (example)
```text
--2025-11-27 12:17:02--  https://www.dropbox.com/s/ikyxo0zew514a0b/processes_datasets.csv
Resolving www.dropbox.com (www.dropbox.com)... 162.125.5.18, 2620:100:601d:18::a27d:512
Connecting to www.dropbox.com (www.dropbox.com)|162.125.5.18|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://www.dropbox.com/scl/fi/h7expu9n3e04pddcl494y/processes_datasets.csv?rlkey=3lcfn9mx2ndnswqa5gb8ct0kb [following]
--2025-11-27 12:17:02--  https://www.dropbox.com/scl/fi/h7expu9n3e04pddcl494y/processes_datasets.csv?rlkey=3lcfn9mx2ndnswqa5gb8ct0kb
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com/cd/0/inline/C18JUOXu1JVsrwgs6DeTuCUZHyw4MO9dzA69W1-5zm2jEPgzVZ0j2fcoAqenSyzaV7XvTexiNv4eoNY1TlVf2hmUmGcq1zR02GUS-9O9mDQjqH48_ecAYgC19_Rw0I36QJo/file# [following]
--2025-11-27 12:17:03--  https://uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com/cd/0/inline/C18JUOXu1JVsrwgs6DeTuCUZHyw4MO9dzA69W1-5zm2jEPgzVZ0j2fcoAqenSyzaV7XvTexiNv4eoNY1TlVf2hmUmGcq1zR02GUS-9O9mDQjqH48_ecAYgC19_Rw0I36QJo/file
Resolving uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com (uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com)... 162.125.5.15, 2620:100:601d:15::a27d:50f
Connecting to uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com (uc1c3fc953daf1d2067900766c33.dl.dropboxusercontent.com)|162.125.5.15|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 49425441 (47M) [text/plain]
Saving to: ‘processes_datasets.csv.1’

processes_datasets. 100%[===================>]  47.14M  53.3MB/s    in 0.9s    

2025-11-27 12:17:04 (53.3 MB/s) - ‘processes_datasets.csv.1’ saved [49425441/49425441]
```

Shape and columns (observed)
- Rows: 404,176
- Columns: 29
- Head sample:
  ```
  JobID  SubmitTime   WaitTime   RunTime  NProcs  AverageCPUTimeUsed  Used Memory  ReqNProcs  ReqTime:  ReqMemory  ...  JobStructureParams  UsedNetwork  UsedLocalDiskSpace  UsedResources  ReqPlatform  ReqNetwork  ReqLocalDiskSpace  ReqResources  VOID  ProjectID
  1      1136070024   203761     138467   1       138371              98652        1          259200    -1         ...  -1                   -1           -1                   -1             -1           -1           -1                 -1            -1    -1
  2      1136070690   0          11       1       4                   35848        1          259200    -1         ...  -1                   -1           -1                   -1             -1           -1           -1                 -1            -1    -1
  ...
  ```

Notes
- Many “Req*”, “Used*”, “JobStructure*”, “VOID”, “ProjectID” fields use -1 to indicate missing or not-applicable values.
- Column names may include whitespace or punctuation (e.g., `ReqTime:`). The pipeline strips whitespace to harmonize names.
- Target variable: RunTime (the CPU burst length we aim to predict).
- The dataset is large but manageable on a modern laptop (≈47 MB CSV, ~400k rows).
- ID-like categorical columns include: UserID, QueueID, GroupID, ExecutableID, OrigSiteID, LastRunSiteID.

---

## Algorithms / Models

We implement and compare the following supervised regression models:

Linear and Regularized Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet

Tree-Based and Ensemble Models
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

Boosted GBDT Frameworks
- XGBoost Regressor (optional)
- LightGBM Regressor (optional)

Kernel/Instance-Based/NN
- Support Vector Regression (RBF kernel)
- K-Nearest Neighbors Regressor
- Multi-Layer Perceptron Regressor (feed-forward NN)

Evaluation Metrics
- R², Adjusted R²
- RMSE, MAE
- MAPE
- Explained Variance
- Training time
- Overfitting analysis (Train vs Test R² gap)

---

## Implementation Steps

The code follows these end‑to‑end steps aligned with your requested structure:

1) Import all necessary libraries
- pandas, numpy, matplotlib, seaborn, missingno
- scikit-learn preprocessing, model selection, metrics
- Model libraries (sklearn, xgboost, lightgbm)
- Warnings and plotting defaults

2) Load the data from the data source
- Downloads the CSV via wget in Colab or reads local file
- Prints dataset shape and previews head/tail

3) Apply EDA (descriptive + visualization)
- .info(), .describe(), data types, unique counts
- Missing values matrix and bar plot (missingno)
- Target distribution, numeric histograms, box plots
- Correlation matrix heatmap
- Selected categorical distributions (top 5 categorical columns)
- Outlier inspection via box plots

4) Preprocessing (incl. Train/Test split)
- Strip column whitespace
- Drop highly-missing, low-signal columns (JobStructure*, UsedNetwork, UsedLocalDiskSpace, UsedResources, ReqPlatform, ReqNetwork, ReqLocalDiskSpace, ReqResources, VOID, ProjectID) if present
- Replace sentinel -1 with NaN and impute (median for numeric, mode for categorical)
- Convert ID columns (UserID, QueueID, GroupID, ExecutableID, OrigSiteID, LastRunSiteID) to numeric ranks (dense ranking)
- Optional outlier capping via IQR
- 70/30 split into Train/Test
- Feature scaling with StandardScaler (and RobustScaler prepared)

5) Feature Engineering
- Example engineered features (e.g., memory ratios, interactions if signals exist)
- Maintain consistent feature order and scalers for inference

6) Initialize all models (separately)
- Reasonable baseline hyperparameters for all models
- Toggle XGBoost/LightGBM if unavailable

7) Train all models (separately)
- Time each fit
- Maintain trained model registry for later evaluation and prediction

8) Evaluate and visualize results (Train and Test)
- Compute R², Adjusted R², RMSE, MAE, MAPE, EVS
- Visual comparisons (R², RMSE, MAE bar charts)
- Actual vs Predicted, Residuals, Error distributions for top models

9) Comparative analysis across models
- Best-by-metric tables
- Overfitting analysis (Train/Test R² gap)
- Composite ranking with normalized metrics and optional training time weighting
- Complexity vs performance visualization

10) Prediction function
- `predict_cpu_burst(sample)` accepts dict/Series/DataFrame
- Applies feature alignment and scaling
- Returns per-model predictions and summary statistics
- Produces visual comparison plot for single/multiple samples

---

## Hardware / Software Environment

Recommended
- CPU: Any recent 4+ core CPU (8+ threads preferred)
- RAM: 8 GB minimum (16 GB recommended)
- Disk: ~1 GB free for data and caches
- OS: Windows 10/11, macOS 12+, Ubuntu 20.04+

Software
- Python: 3.9–3.12
- Packages:
  - pandas, numpy
  - matplotlib, seaborn, missingno
  - scikit-learn
  - xgboost (optional)
  - lightgbm (optional)
- Jupyter/Colab for EDA and plots

Notes
- XGBoost and LightGBM may require platform-specific installation steps; if unavailable, comment those models out—the pipeline continues with the remaining models.
- The dataset is ~47 MB; memory footprint during modeling is higher due to multiple copies and models. Close other apps if you experience memory pressure.

---

## Comparative Analysis Outputs

The pipeline produces:
- Training and testing metrics for each model (DataFrames).
- Bar charts comparing Train vs Test R², RMSE, MAE.
- Top-3 models’ scatter plots (Actual vs Predicted), residuals, and error histograms.
- Overfitting analysis table and plot.
- Overall ranking considering accuracy metrics and optional training time.

How to interpret
- Prefer models with high Test R² and low RMSE/MAE.
- Check residual plots for systematic bias.
- Compare Train vs Test R² to detect overfitting.
- If latency is critical (e.g., online scheduling), weigh training/inference time more heavily.

---

## Conclusion

- Accurate CPU burst prediction can substantially aid SJF and SRTF schedulers by providing better runtime estimates, ultimately improving average waiting time and turnaround time.
- In general:
  - Gradient-boosted trees (GBDTs such as XGBoost/LightGBM/Sklearn-GBR) and Random Forests tend to provide strong baselines for tabular data like this dataset.
  - Linear/regularized models are fast and interpretable but may underfit if relationships are nonlinear.
  - SVR/MLP can perform well but may require careful tuning and scaling.
- The included pipeline:
  - Implements robust preprocessing and feature handling,
  - Benchmarks multiple models fairly,
  - Provides clear visual diagnostics,
  - Exposes a unified prediction function for downstream integration.

Future work
- Hyperparameter tuning (RandomizedSearchCV/Optuna).
- Feature selection/importance analysis (permutation, SHAP).
- Time-aware validation if temporal drift exists.
- Integration into a simulator to quantify the improvement in SJF/SRTF scheduling metrics using predicted vs baseline (e.g., exponential averaging) burst times.

---

## References

- Paper 1: “Comparative Analysis and Implementation of AI Algorithms and NN Model in Process Scheduling Algorithm”
- Paper 2: “Predicting CPU Burst Times with ML to Enhance Shortest Job First (SJF) and Shortest Remaining Time First (SRTF) CPU Scheduling”
  - Emmanuel Effah, Stephen Julius Atsu, Zerubbabel Abeeku Brew, Joseph Kofi Mensah, Evans Adjei Quaicoe, Michael Owusu Ansah, Anthony Yeful, Rowland P. Baffoe
  - Departments: University of Mines and Technology (Ghana), University of Cape Coast (Ghana)
- Paper 3: “CPU Burst-Time Estimation using Machine Learning”
  - Prathamesh Samal, Sagar Jha
  - CSED, Thapar Institute of Engineering and Technology, Patiala, India
  - 2022 IEEE Delhi Section Conference (DELCON) | DOI: 10.1109/DELCON54057.2022.9753639

---

## Acknowledgements

Thanks to the authors of the above works for the insights motivating ML-based CPU burst prediction and its application to SJF/SRTF scheduling improvements.

---
