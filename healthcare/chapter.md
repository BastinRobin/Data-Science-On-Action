# Hospital Readmission for Patients with Diabetes

## Problem
Diabetes is a condition in which there is too much glucose (a type of sugar) in the blood. Over time, high blood glucose levels can damage the body's organs. Possible complications include damage to large (macrovascular) and small (microvascular) blood vessels, which can lead to heart attack, stroke, and problems with the kidneys, eyes, gums, feet and nerves. 

## Facts Exploration
Risk of most diabetes-related complications can be reduced by keeping blood pressure, blood glucose and cholesterol levels within recommended range. Also, being a healthy weight, eating healthily, reducing alcohol intake, and not smoking will help reduce your risk. Regular check-ups and screening are important to pick up any problems early.

As the healthcare system moves toward value-based care, CMS has created many programs to improve the quality of care of patients. One of these programs is called the Hospital Readmission Reduction Program ([HRRP](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program.html)), which reduces reimbursement to hospitals with above average readmissions. For those hospitals which are currently penalized under this program, one solution is to create interventions to provide additional assistance to patients with increased risk of readmission. But how do we identify these patients? We can use predictive modeling from data science to help prioritize patients.

One patient population that is at increased risk of hospitalization and readmission is that of diabetes. Diabetes is a medical condition that affects approximately 1 in 10 patients in the United States. According to Ostling et al, patients with diabetes have almost double the chance of being hospitalized than the general population ([Ostling et al 2017](https://clindiabetesendo.biomedcentral.com/articles/10.1186/s40842-016-0040-x)). Therefore, in this article, I will focus on predicting hospital readmission for patients with diabetes. In this usecase we will explore how to build a model predicting readmission in Python.

## How-To
Predict if a patient with diabetes will be readmitted to the hospital within 30 days.


## Data Sources:
The data that is used in this project originally comes from the UCI machine learning repository. The data consists of over 100000 hospital admissions from patients with diabetes from 130 US hospitals between 1999–2008.


## Data Modeling:
In this project, we will utilize Python to build the predictive model. Let’s begin by loading the data and exploring some of the columns. We can start using `scikit-learn` python machine learning library combined with `python-pandas` library for data wranging.
`In [1]:`
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# load the csv file
df = pd.read_csv('diabetic_data.csv')

print('Number of samples:',len(df))
```
__Output:__ Number of samples: 101766


From briefly, looking through the data columns, we can see there are some identification columns, some numerical columns, and some categorical (free-text) columns. These columns will be described in more detail below.

![Figure 1](img/fig1.png)

There is some missing data that are represented with a question mark (?). We will deal with this in the feature engineering section.

The most important column here is readmitted, which tells us if a patient was hospitalized within 30 days, greater than 30 days or not readmitted.

![Figure 2](img/fig2.png)

Another column that is important is `discharge_disposition_id`, which tells us where the patient went after the hospitalization. If we look at the IDs_mapping.csv provided by UCI we can see that 11,13,14,19,20,21 are related to death or hospice. We should remove these samples from the predictive model since they cannot be readmitted.

```python
df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]
```

Now let’s define an output variable for our binary classification. Here we will try to predict if a patient is likely to be re-admitted within 30 days of discharge.

```python
df['OUTPUT_LABEL'] = (df.readmitted == '<30').astype('int')
```

Let’s define a function to calculate the prevalence of population that is readmitted with 30 days.

![Figure 3](img/fig3.png)

Around 11% of the population is rehospitalized. This represented an imbalanced classification problem so we will address that below.

From further analysis of the columns, we can see there are a mix of categorical (non-numeric) and numerical data. A few things to point out,

-   `encounter_id` and `patient_nbr`: these are just identifiers and not useful variables
  
-   `age` and `weight`: are categorical in this data set
-   `admission_type_id`,`discharge_disposition_id`,`admission_source_id`: are numerical here, but are IDs (see IDs_mapping). They should be considered categorical.
-   `examide` and `citoglipton` only have 1 value, so we will not use these variables
`diag1`, `diag2`, `diag3` — are categorical and have a lot of values. We will not use these as part of this project, but you could group these ICD codes to reduce the dimension. We will use number_diagnoses to capture some of this information.
-   `medical_speciality` — has many categorical variables, so we should consider this when making features.


Now we are in need of creating new features to make the model more reliable, We can create new features from existing context of the dataset using Feature Engineering.

## Feature Engineering
In this section, we will create features for our predictive model. For each section, we will add new variables to the dataframe and then keep track of which columns of the dataframe we want to use as part of the predictive model features. We will break down this section into numerical features, categorical features and extra features.

In this data set, the missing numbers were filled with a question mark. Let’s replace it with a nan representation.

`In [10]:`
```python
# replace ? with nan
df = df.replace('?',np.nan)
```

### Numerical Features
The easiest type of features to use is numerical features. These features do not need any modification. The columns that are numerical that we will use are shown below

```python
cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses']
```

Let’s check if there are any missing values in the numerical data.

`In [12]: `
```python
df[cols_num].isnull().sum()
```
`Out[12]:`
```python
time_in_hospital      0
num_lab_procedures    0
num_procedures        0
num_medications       0
number_outpatient     0
number_emergency      0
number_inpatient      0
number_diagnoses      0
dtype: int64
```

### Categorical Features
The next type of features we want to create are categorical variables. Categorical variables are non-numeric data such as race and gender. To turn these non-numerical data into variables, the simplest thing is to use a technique called one-hot encoding, which will be explained below.

The first set of categorical data we will deal with are these columns:

`In [14]:`
```python
cols_cat = ['race', 'gender', 
       'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed','payer_code']
```
Of our categorical features, `race`, `payer_code`, and `medical_specialty` have missing data. Since these are categorical data, the best thing to do is to just add another categorical type for unknown using the `fillna` function.

`In [15]:`
```python
df['race'] = df['race'].fillna('UNK')
df['payer_code'] = df['payer_code'].fillna('UNK')
df['medical_specialty'] = df['medical_specialty'].fillna('UNK')
```

## Outcomes 

## Deployment 

## Conclusions

### Links:
- HRRP: https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program.html
- Ostling et al 2017: https://clindiabetesendo.biomedcentral.com/articles/10.1186/s40842-016-0040-x
- Dataset: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008