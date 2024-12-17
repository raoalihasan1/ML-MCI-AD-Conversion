# MCI to AD Conversion Research

## Overview
This repository contains the code for my research which looks at building a 
classification model for predicting whether a patient that is initially diagnosed to 
have MCI (Mild Cognitive Impairment) converts to AD (Alzheimer's Disease) over a 2 years
period from their baseline visit. This research uses data from the 
[ADNI](https://adni.loni.usc.edu/) dataset.

The data used for modelling includes the CSF & Plasma of participants as well as tests 
taken by them at each visit. The process followed in this research includes:
1. [Data Collection](#data-collection)
2. [Data Pre-Processing](#data-pre-processing)
3. [Model Training & Testing](#model-training--testing)
4. [Model Evaluation](#model-evaluation)

## Process Description

### <a name="data-collection"></a>Data Collection
In the data collection phase, I sourced the ADNI merge dataset which hosts most of the 
data required in a single CSV and merged this with individual samples retrieved in 
regards to the plasma samples from other datasets in ADNI.

### <a name="data-pre-processing"></a>Data Pre-Processing
In this stage, I filtered the dataset to get a subset of the features that were relevant
to this research, mapped non-numerical values into numerical form, filtered the dataset
further for data that is within 2 years of the baseline visit, and finally tested KNN 
and MICE forest imputation methods with statistical analysis to find which works better 
for our data, which in this case was MICE forest.

### <a name="model-training--testing"></a>Model Training & Testing
This phase began with splitting the dataset into training and testing sets, oversampling
the data due its highly skewed nature towards non-conversion. I tested 4 different 
oversampling methods and proceeded with the one that aligns best with our dataset which 
in this case was ADASYN. Then, I performed hyperparameter tuning, looking into various 
ML models with a variety of configurations. Finally, I created an ensemble model, 
testing all permutations of the ML models with all hyperparameter setting to produce a 
robust model which I performed statistical analysis on to test its performance on the 
test sets.

### <a name="model-evaluation"></a>Model Evaluation
Finally, I tested the model on the different datasets made in the preprocessing stage, 
looking at the different permutations of data to test the models general performance. 
This phase was completed with analysis into each feature and its importance/weighting in
determining the target variable, which is the diagnosis after the 2 year period. 