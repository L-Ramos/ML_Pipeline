# ML_Pipeline
This is a Machine Learning pipeline for training and testing models.


Methods.py: contains all classes for the classifiers. Training is based on Pipelines to prevent biased results due to Scaling or Imputation.
Hyperparameters.py: contains all hyperparameters for all classifiers, you can edit and add/remove you own
utils.py: contains functions for initializing classes.
example_main.py: A short example of how to use the code.

Remarks:

Only Random Forest and Logistic Regression were tested, so other classifiers might not work
SHAP visualization works only for RF, other kernels need to be added to other classifiers.
Features with multiple categories (like colors for instance), should be encoded outside the pipeline for now (to be fixed)
