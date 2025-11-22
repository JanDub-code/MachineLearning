# CRITICAL TASKS

## 1. Model Validation & Tuning (PRIORITY)
- **Cross Validation**: Implement TimeSeriesSplit or KFold (depending on data nature) to robustly validate model performance.
- **Grid Search**: Perform hyperparameter tuning to find optimal model parameters.
- **Goal**: Prove that the selected parameters are indeed the best and not just arbitrary choices.

## 2. Data Pipeline
- Ensure data collection is robust and reproducible (Completed).
- Verify data integrity before training.

## 3. Documentation
- Clearly state the validation strategy in the project description.
- Document the results of Grid Search (best params, scores).
