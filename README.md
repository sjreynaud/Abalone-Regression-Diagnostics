Assignment Overview
This assignment contains three major components:

1. Abalone Regression (Kaggle Dataset)
The goal is to predict abalone age (Rings) using supervised regression models.
The notebook performs:

Data loading and inspection

Feature identification (categorical vs numeric)

Preprocessing using OneHotEncoder and ColumnTransformer

Train/validation split

Model training:

Linear Regression

Random Forest Regression

RMSE comparison

Diagnostic plots:

Residuals vs Predicted

Residual distribution

QQ plot

Final predictions saved to submissions/my_abalone_submission.csv

All code and outputs are based directly on the dataset shown in the attached document, including the columns:

“Sex”, “Length”, “Diameter”, “Height”, “Whole weight”, “Whole weight.1”, “Whole weight.2”, “Shell weight”, “Rings”

2. ISLR Conceptual Question #3
Completed in Markdown inside the notebook.

Topics addressed:

Plugging GPA = 4.0 and IQ = 110 into the regression equation

Understanding interaction terms

Why a small interaction coefficient does not imply no interaction

Importance of standard errors and p‑values

Explore concepts:

interaction terms

standard error

p‑value

*3. Carseats OLS Regression (Applied Question #10)
Using the Carseats dataset, the notebook performs:*

Full model: Sales ~ Price + Urban + US

Identification of significant predictors

Reduced model with only significant variables

Model comparison using R², Adjusted R², and ANOVA

95% confidence intervals

Diagnostic plots:

Residuals vs Fitted

Residual distribution

QQ plot

Leverage / Cook’s distance
