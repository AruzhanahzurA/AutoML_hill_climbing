# AutoML Hill Climbing

This repository contains a small AutoML benchmarking project with two regression datasets and three AutoML/machine learning approaches.

## Repository contents

- `Project1_last.ipynb` — main analysis notebook containing the full workflow, preprocessing, model search, and evaluation
- `cleaned_all_phones.csv` — phone pricing dataset used for the first regression problem
- `coffee_shop_revenue.csv` — coffee shop revenue dataset used for the second regression problem
- `requirements.txt` — Python package dependencies required to run the notebook

## AutoML approaches included

1. **Custom Local Search AutoML**
   - Implements a random-restart hill-climbing search over algorithm choice and discrete hyperparameter values
   - Uses 5-fold cross-validation and RMSE as the objective
   - Searches across four base regressors:
     - `RandomForestRegressor`
     - `GradientBoostingRegressor`
     - `KNeighborsRegressor`
     - `SVR`
   - The search space includes discrete values for:
     - Random forest: `n_estimators`, `max_features`, `max_depth`, `min_samples_split`
     - Gradient boosting: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
     - KNN: `n_neighbors`, `weights`, `p`
     - SVR: `C`, `epsilon`, `gamma`, `kernel`

2. **TPOT**
   - Uses `TPOTRegressor` with default AutoML pipeline search settings
   - Optimizes regression pipelines using genetic programming and internal 5-fold CV
   - Evaluated on both datasets for comparison with the custom local search

3. **FLAML**
   - Uses `flaml.AutoML` for fast regression tuning
   - Restricted to the learner list: `lgbm`, `rf`, `xgboost`
   - Uses `rmse` as the metric and `holdout` evaluation for faster search

## Custom Local Search architecture

- The notebook implements a random-restart hill-climbing search over both algorithm selection and discrete hyperparameters.
- Each candidate configuration is evaluated with 5-fold cross-validation and RMSE.
- Neighbor moves are defined by:
  - changing one discrete hyperparameter by one step in its predefined ordered list, or
  - switching to a different algorithm and sampling a fresh parameter configuration for it.
- The algorithm keeps moving to better neighbors until it sees a sequence of non-improving steps.
- When the local search cannot improve further, it performs another random restart, effectively making a new jump to a different configuration.

## Comparison results in the notebook

The notebook compares the three systems on both datasets using test RMSE and runtime.

- **PhonePrices**
  - `MyAutoML_LocalSearch`: RMSE `226.93`, time `3600 s`, best model: `RandomForestRegressor`
  - `TPOT`: RMSE `220.40`, time `~3694 s`
  - `FLAML`: RMSE `243.99`, time `~1401 s`, best estimator: `xgboost`

- **CoffeeShop**
  - `MyAutoML_LocalSearch`: RMSE `214.31`, time `3600 s`, best model: `GradientBoostingRegressor`
  - `TPOT`: RMSE `205.25`, time `~3605 s`
  - `FLAML`: RMSE `213.31`, time `~3156 s`, best estimator: `xgboost`

### Observations

- `TPOT` produced the best test RMSE on both datasets in the notebook runs.
- The custom hill-climbing search was competitive and produced the best model in the search space for each dataset, but it was slightly behind TPOT in final test error.
- `FLAML` was faster on the phone dataset and still competitive on the coffee dataset, but it delivered higher RMSE for PhonePrices and slightly higher RMSE for CoffeeShop compared to the other approaches.

## Data preparation

- Each dataset is preprocessed with shared transformation logic
- Numeric columns are median-imputed
- Categorical columns are imputed with the most frequent value and one-hot encoded
- For the custom hill-climbing search, the preprocessed feature matrix is passed directly to the model pipeline

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open `Project1_last.ipynb` in Jupyter Notebook or JupyterLab

3. Run the notebook cells from top to bottom to execute the data preparation, custom AutoML, TPOT, and FLAML experiments

## Contributors

- [AruzhanahzurA](https://github.com/AruzhanahzurA)
- [@Ajkemon](https://github.com/Ajkemon)

## Notes

- This repo is intentionally simple and centered around the notebook workflow.
- If the project expands, consider separating code into `src/`, notebooks into `notebooks/`, and data into `data/raw/`.
