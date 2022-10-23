# PCA
This code calculates the PCA of a given dataset and extracts the first and second component.
Then:
- The histograms of the two components are calculated and the negative logarithm of the probability is plotted.
- The negative logarithm of the kernel density estimation of the 2-dimensional PCA is plotted.
- The mutual information between PCA1-PCA2, and PCA1-PCA1^2 is calculated via H(x) + H(y) - H(x,y).

This last calculation is performed from the grids of p(x), p(y), and p(x,y), and the number of bins is determined via the Freedman-Diaconis rule.

# Usage
usage: python main.py [-h] input

example: python main.py data_coding_exercise.txt
