#!/usr/bin/env python
# coding: utf-8

# Section 1 – Regression with Abalone dataset (Kaggle)

# Step 1 – Set up notebook and imports

# In[1]:


# Adds the parent directory or current directory to the path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor




# Step 2 – Load sample_submission, train, and test

# In[2]:


# Step 2: Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

train.head(), test.head(), sample_submission.head()


# Step 3 – Inspect data and define features/target

# In[3]:


# Step 3: Basic inspection
train.info()
train.describe(include="all")

# Define target and features
target_col = "Rings"          # adjust if your competition uses a different target name
X = train.drop(columns=[target_col])
y = train[target_col]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

categorical_cols, numeric_cols


# Step 4 – Build preprocessing pipeline (encode categoricals)

# In[4]:


# Step 4: Preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# Step 5 – Train/validation split for model evaluation

# In[5]:


# Step 5: Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6 – First model: Linear Regression

# In[6]:


# Step 6: Linear Regression model
linreg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

linreg_model.fit(X_train, y_train)

y_valid_pred_lr = linreg_model.predict(X_valid)
rmse_lr = mean_squared_error(y_valid, y_valid_pred_lr, squared=False)
rmse_lr


# Step 7 – Second model: Random Forest Regression

# In[7]:


# Step 7: Random Forest model
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_valid_pred_rf = rf_model.predict(X_valid)
rmse_rf = mean_squared_error(y_valid, y_valid_pred_rf, squared=False)
rmse_rf


# Step 8 – Investigate regression assumptions
# Use residuals from the validation set.

# In[8]:


# Step 8: Assumption checks for Random Forest (or Linear Regression)
import matplotlib.pyplot as plt
import seaborn as sns

residuals_rf = y_valid - y_valid_pred_rf

# Linearity & residual patterns
plt.figure(figsize=(6,4))
plt.scatter(y_valid_pred_rf, residuals_rf, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (RF)")
plt.show()

# Distribution of residuals
plt.figure(figsize=(6,4))
sns.histplot(residuals_rf, kde=True)
plt.title("Residual distribution (RF)")
plt.show()

# QQ plot for residuals
import statsmodels.api as sm
sm.qqplot(residuals_rf, line="45")
plt.title("QQ plot of residuals (RF)")
plt.show()


# Step 9 – Train chosen model on full training data and predict test

# In[9]:


# Step 9: Final model on full training data
final_model = rf_model  # or linreg_model, depending on your comparison

final_model.fit(X, y)

test_predictions = final_model.predict(test)


# Step 10 – Create submission file matching sample_submission format

# In[10]:


# Step 10: Submission file
submission = sample_submission.copy()
submission[target_col] = test_predictions

submission.head()

submission.to_csv("my_abalone_submission.csv", index=False)


# Section 2 – Conceptual Question #3 (ISLR Python)

# Markdown cell for Q3 answers
# 
# Part (b): Plug IQ = 110 and GPA = 4.0 into the given regression equation from the book and show the calculation.
# 
# Part (c): Explain why a “small” interaction coefficient does not automatically mean “no interaction” (you must consider its standard error and p‑value).

# ### Conceptual Question #3 (ISLR Python)
# 
# **(b)** Predicted salary for college graduate with IQ = 110 and GPA = 4.0:
# 
# 
# 
# \[
# \hat{Salary} = \beta_0 + \beta_1 \cdot \text{GPA} + \beta_2 \cdot \text{IQ} + \beta_3 \cdot \text{GPA}:\text{IQ} + \beta_4 \cdot \text{College} + \beta_5 \cdot \text{GPA}:\text{College}
# \]
# 
# 
# 
# (Then substitute the coefficients from the text and compute.)
# 
# **(c)** Even if the interaction coefficient is numerically small, we must look at its **standard error** and **t‑statistic**. A small coefficient with a very small standard error can still be highly significant, indicating evidence of interaction.
# 

# Section 3 – Applied Question #10 (Carseats data)

# Step 11 – Imports for Carseats analysis

# In[11]:


# Step 11: Additional imports for Carseats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


# Step 12 – Load Carseats dataset

# In[12]:


# Step 12: Load Carseats data
carseats = pd.read_csv("Carseats.csv")
carseats.head()
carseats.info()


# Step 13 – Fit full model: Sales ~ Price + Urban + US

# In[13]:


# Step 13: Full model
model_full = smf.ols("Sales ~ Price + Urban + US", data=carseats).fit()
print(model_full.summary())


# Sales = β0 + β1 * Price + β2 * UrbanYes + β3 * USYes
# 

# Identify significant predictors (H₀: βᵢ = 0)
# 
# From model_full.summary():
# 
#     -Look at p‑values for Price, UrbanYes, USYes.
#     -In Markdown, state which predictors have p‑value < 0.05 →      reject H₀.

# Step 14 – Fit reduced model with only significant predictors

# In[14]:


# Step 14: Reduced model
model_reduced = smf.ols("Sales ~ Price + US", data=carseats).fit()
print(model_reduced.summary())


# Step 15 – Compare fit of full vs reduced models
# Use 𝑅2, adjusted 𝑅2, and possibly ANOVA.

# In[15]:


# Step 15: Compare models
print("Full model R^2:", model_full.rsquared, "Adj R^2:", model_full.rsquared_adj)
print("Reduced model R^2:", model_reduced.rsquared, "Adj R^2:", model_reduced.rsquared_adj)

sm.stats.anova_lm(model_reduced, model_full)


# Step 16 – 95% confidence intervals for coefficients (reduced model)

# In[16]:


# Step 16: Confidence intervals
model_reduced.conf_int(alpha=0.05)


# Step 17 – Diagnostic plots for reduced model

# In[17]:


# Step 17: Diagnostic plots
residuals = model_reduced.resid
fitted = model_reduced.fittedvalues

# Residuals vs fitted
plt.figure(figsize=(6,4))
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (Carseats reduced model)")
plt.show()

# Histogram of residuals
sns.histplot(residuals, kde=True)
plt.title("Residual distribution")
plt.show()

# QQ plot
sm.qqplot(residuals, line="45")
plt.title("QQ plot of residuals")
plt.show()

# Leverage plot
sm.graphics.influence_plot(model_reduced, criterion="cooks")
plt.show()


# In[ ]:




