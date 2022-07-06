# Panagiotis Doupidis, 03/12/21

import numpy as np
from scipy.stats import spearmanr, normaltest

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

np.set_printoptions(precision=5)

# Read CSV File
steps, value = np.genfromtxt("GeorgeData.csv", delimiter=';', skip_header=True).T
steps = steps.astype(np.int64)

# First check whether each dataset is normally distributed to perform Pearson's test
# Set alpha = 0.05
alpha = 5e-2
chk_steps, chk_value = normaltest(steps), normaltest(value)

if chk_steps.pvalue < alpha:
    print("Rejecting Null hypothesis -> Data[Steps] is not normally distributed")
else:
    print("Null Hypothesis cannot be rejected -> Data[Steps] follows the normal distribution with 95% confidence")

if chk_value.pvalue < alpha:
    print("Rejecting Null hypothesis -> Data[Value] is not normally distributed")
else:
    print("Null Hypothesis cannot be rejected -> Data[Value] follows the normal distribution with 95% confidence")


# Not every dataset is normally distributed, so we'll use Spearman's test to find the correlation
corr, p = spearmanr(steps, value)
print("\nSpearman correlation coefficient between the 2 variables : {}".format(np.round(corr,5)))
print("Null Hypothesis (datasets are uncorrelated) cannot be rejected {0} > {1}".format(np.round(p,5),alpha))

# Plot the performance of the agent with respect to the number of steps
ax = plt.subplot(111)
ax.plot(steps, value, "bo--")
ax.fill_between(steps, 0, value, alpha=0.2)
ax.grid()
majorLocator = MultipleLocator(5e5)
ax.xaxis.set_major_locator(majorLocator)
ax.set_title("Agent's Performance ~ Number of Steps [Corr = {}]".format(np.round(corr,5)))
ax.set_xlabel("Agents steps")
ax.set_ylabel("Reward value")
plt.show()