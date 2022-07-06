# =============================================================================
# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================



# From 'sklearn' library, we need to import:
# 'datasets', for loading our data
# 'metrics', for measuring scores
# 'linear_model', which includes the LinearRegression() method
# From 'scipy' library, we need to import:
# 'stats', which includes the spearmanr() and pearsonr() methods for computing correlation
# Additionally, we need to import 
# 'pyplot' from package 'matplotlib' for our visualization purposes
# 'numpy', which implementse a wide variety of operations
# =============================================================================

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection
from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr
# =============================================================================


# Load diabetes data from 'datasets' class
# =============================================================================

diabetes = datasets.load_diabetes()

# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# =============================================================================

# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value (prediction target)

X = diabetes.data[:, np.newaxis, 0]
y = diabetes.target

# =============================================================================


# Create linear regression model. All models behave differently, according to
# their own, model-specific parameter values. In our case, however, the linear
# regression model does not have any substancial parameters to tune. Refer
# to the documentation of this technique for more information.
# =============================================================================

linearRegressionModel = LinearRegression()

# =============================================================================

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3, random_state=42)


# Let's train our model.
# =============================================================================

linearRegressionModel.fit(x_train, y_train)

# =============================================================================


# Ok, now let's predict the output for the test input set
# =============================================================================

y_predicted = linearRegressionModel.predict(x_test)

# =============================================================================



# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.
# =============================================================================

print("Pearson Correlation between the two variables : %.5f" % (pearsonr([i for j in X for i in j],y)[0]))
print("Mean Squared Error : %.2f" % (metrics.mean_squared_error(y_test, y_predicted)))
print("R^2 : %.5f" %(metrics.r2_score(y_test, y_predicted)))

# =============================================================================


# Plot results in a 2D plot (scatter() plot, line plot())
# =============================================================================

plt.scatter(x_test, y_test,  color='red')
plt.plot(x_test, y_predicted, color='black', linewidth=2)

plt.xlabel("Age")
plt.ylabel("Diabetes")
plt.xticks(())
plt.yticks(())

# Show plot
plt.show()

# =============================================================================
