import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")
N = 20000
n = 20
matrix = np.random.binomial(n=1, p=0.5, size=(N, n))
row_means = np.mean(matrix, axis=1)
eps_vals = np.linspace(0, 1, 50)

empirical_probabilities = []
Hoeffding_bds = []
for eps in eps_vals:
    condition = np.abs(row_means-0.5) > eps
    emp_probability = np.mean(condition)
    empirical_probabilities.append(emp_probability)
    Hoeffding_bds.append(2*np.exp(-2*n*(eps**2)))

plt.figure(figsize=(8, 6))
plt.plot(eps_vals, empirical_probabilities, marker='', linestyle='-', label='Empirical Probability')
plt.plot(eps_vals, Hoeffding_bds, marker='', linestyle='-', color='red', label='Hoeffding Bound')
plt.xlabel('Epsilon')
plt.ylabel('Probability')
#plt.title('Probability and its Hoeffding bound as a function of ε')
plt.legend()
plt.grid(True)

# Export the plot to a PDF file
plt.savefig('empirical_probabilities_and_hoeffding_bounds.pdf')

