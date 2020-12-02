# Preprocessing_PipeLine_ML

Data Cleaning pipeline for preprocessing data of three Regression ML models and model evaluation


## Pre-processing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


cont_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])




cate_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder())])






preprocessor = ColumnTransformer(
    transformers = [
        ('continuus', cont_pipeline, continuous_features),
        ('categorical', cate_pipeline, cat_features)
        
    ]
)
```

## Model Evaluation

```python
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


# print abs error
print('validating model')
abs_err = np.abs(regressor.predict(x_test) - y_test)
abs_err_rf = np.abs(test_pred_rf - y_test)
abs_err_xgboost = np.abs(test_pred_xgboost - y_test)


# print couple perf metrics
for q in [10, 50, 90]:
    print('LinearRegression_AE-at-' + str(q) + 'th-percentile: '
          + str(np.percentile(a=abs_err, q=q)))
    print('RandomForest_AE-at-' + str(q) + 'th-percentile: '
          + str(np.percentile(a=abs_err_rf, q=q)))
    print('XGBoost_AE-at-' + str(q) + 'th-percentile: '
      + str(np.percentile(a=abs_err_xgboost, q=q)))
```
**validating model**

R2 score of XGBoost = 0.81


Square Root of MSE of XGBoost: 191694.38631413673

MAE of XGBoost: 135442.6762381243

R2 score of Random Forest= 0.79

Square Root of MSE of Random Forest: 199853.71768196145

MAE of Random Forest: 139813.3231220977

R2 score of Linear Regression = 0.61

Square Root of MSE of Linear Regression: 272007.94185983855

MAE of Linear Regression: 209399.89834611764

## Cumulative Distribution Function (CDF)

CDF is used to determine the probability that a random observation that is taken from the population will be less than or equal to a certain value. We can also use this information to determine the probability that an observation will be greater than a certain value, or between two values. It will enable us to determine the best performing ML model.

![Index](https://github.com/haxamxam/Preprocessing_PipeLine_ML/blob/main/.ipynb_checkpoints/index.png)

