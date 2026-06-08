# AutoML Hill Climbing

This repository contains a small AutoML benchmarking project with two regression datasets and three AutoML/machine learning approaches.

## Repository contents

- `Project1_AutoML.ipynb` — main analysis notebook containing the full workflow, preprocessing, model search, and evaluation
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
   - Uses `TPOTRegressor` optimized with the `linear-light` search space built for faster linear/lightweight pipeline configurations
   - Optimizes regression pipelines using genetic programming and internal 5-fold CV

3. **FLAML**
   - Uses `flaml.AutoML` for zero-configuration, cost-frugal regression tuning
   - Runs across its default search pool (allowing it to dynamically select complex estimators like CatBoost and XGBoost variants)
   - Uses `rmse` as the optimization metric

## Custom Local Search architecture

- The notebook implements a random-restart hill-climbing search over both algorithm selection and discrete hyperparameters.
- Each candidate configuration is evaluated with 5-fold cross-validation and RMSE.
- Neighbor moves are defined by:
  - changing one discrete hyperparameter by one step in its predefined ordered list, or
  - switching to a different algorithm and sampling a fresh parameter configuration for it.
- The algorithm keeps moving to better neighbors until it sees a sequence of non-improving steps.
- When the local search cannot improve further, it performs another random restart, effectively making a new jump to a different configuration.

## Comparison results in the notebook

The notebook compares the three systems on both datasets using test RMSE and runtime (with a configured time budget of 3600 seconds per framework).

- **PhonePrices**
  - `MyAutoML_LocalSearch`: RMSE `225.15`, time `3600.6 s`, best model: `RandomForestRegressor`
  - `FLAML`: RMSE `225.33`, time `3600.7 s`, best estimator: `xgb_limitdepth`
  - `TPOT`: RMSE `228.10`, time `3611.7 s`, config: `linear-light`

- **CoffeeShop**
  - `FLAML`: RMSE `205.65`, time `3601.4 s`, best estimator: `catboost`
  - `TPOT`: RMSE `206.40`, time `3603.2 s`, config: `linear-light`
  - `MyAutoML_LocalSearch`: RMSE `214.31`, time `3600.6 s`, best model: `GradientBoostingRegressor`

### Observations

- On the **PhonePrices** dataset, the custom random-restart hill climbing search achieved the highest performance (lowest out-of-sample Test RMSE), narrowly outperforming FLAML's depth-limited XGBoost model. 
- On the **CoffeeShop** dataset, **FLAML** achieved the best generalization score using a `catboost` model, closely followed by TPOT, while the custom local search engine came in third.
- All systems reliably respected the allocated ~60-minute optimization window.

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
