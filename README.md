# RS-error-based-learning

This repository contains all data and code used to create the master thesis 'Alleviating the Cold Start Problem in Recommender Systems Using Error-Based Learning' by Ricardo Laanen. 

This thesis evaluates the performances of several variations of error-based strategies in recommender systems. The data used for this research can be found in the zip file in this repository, and is made available by Dutch luxury department store De Bijenkorf. The dataset includes over 2.5 million user-item interactions with binary values, where 1 is a positive interaction, and 0 is a negative interaction. This data is implicit, meaning the users did not actually rate these items. Instead, their evaluation is based on whether they bought and kept the product, or bought and returned it instead.

The code is written in Python 3.11 and makes use of the Scikit-surprise package for the SVD-model which is used to evaluate the error-based strategies. An .rmd file using R version 4.3.2 is included for the creation of the graph included in the thesis.

Please be aware that the dataset must be reconstructed using all seven .zip files, for example by using 7-Zip (https://www.7-zip.org/).
