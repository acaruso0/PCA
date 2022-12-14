# PCA
This code calculates the PCA of a given dataset and extracts the first and second component.
Then:
- The histograms of the two components are generated and the negative logarithm (base 2) of the probability is plotted.
- The negative logarithm (base 2) of the kernel density estimation of the 2-dimensional PCA is plotted.
- The mutual information (in Shannons) between PCA1-PCA2 and PCA1-PCA1^2 is calculated via H(x) + H(y) - H(x,y).

This last calculation is performed from the histograms of p(x), p(y), and p(x,y), and the number of bins is determined via the Freedman-Diaconis rule.

Plots are saved in "output.pdf" and numerical results are saved in "output.dat".

# Usage
usage: python main.py [-h] input

example: python main.py data\_coding\_exercise.txt
