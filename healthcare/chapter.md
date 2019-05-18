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

In this project, we will utilize Python to build the predictive model. Let’s begin by loading the data and exploring some of the columns.

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

## Model Building:

## Outcomes 

## Deployment 

## Conclusions

### Links:
- HRRP: https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program.html
- Ostling et al 2017: https://clindiabetesendo.biomedcentral.com/articles/10.1186/s40842-016-0040-x
- Dataset: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008