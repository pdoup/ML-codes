# =============================================================================
# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'tree' package, for creating the DecisionTreeClassifier and using graphviz
# 'model_selection' package, which will help test our model.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
breastCancer = datasets.load_breast_cancer()

# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target


# DecisionTreeClassifier() is the core of this script. You can customize its functionality
# in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE
crit, m_depth = 'gini', 3

model = DecisionTreeClassifier(criterion=crit, max_depth=m_depth, random_state=42)

# =============================================================================

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, train_size=.7)



# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE
model.fit(x_train,y_train)

# =============================================================================

# Ok, now let's predict the output for the test input set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = model.predict(x_test)

# =============================================================================

# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Accuracy: %7.3f%%" %(metrics.accuracy_score(y_test, y_predicted) * 100))
print("Precision: %5.3f%%" %(metrics.precision_score(y_test, y_predicted) * 100))
print("Recall: %9.3f%%" %(metrics.recall_score(y_test, y_predicted) * 100))
print("F1 Score: %7.3f%%" %(metrics.f1_score(y_test, y_predicted) * 100))

# =============================================================================



# By using the 'plot_tree' function from the tree classifier we can visualize the trained model.
# There is a variety of parameters to configure, which can lead to a quite visually pleasant result.
# Make sure that you set the following parameters within the function:
feature_names = breastCancer.feature_names[:numberOfFeatures]
class_names = breastCancer.target_names
filled = True
# =============================================================================
plt.figure(figsize=(13,8), dpi=80)
plot_tree(model, feature_names=feature_names, class_names=class_names, filled=filled,
        fontsize=8, rounded=True)
plt.show()
