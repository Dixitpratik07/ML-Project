#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Note :this Project is signed with a Non disclousre Agrrement with Innodatatics.Inc so I am not providing my dataset here 



#importing Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
import sweetviz as sv
import numpy as np

#calling the dataset
df= pd.read_excel("Putting location of dataset .xlsx")
df.head(10)
df.info()
# Attention: duration column must be index 0, event column index 1 in y
#show the all columns of dataset
df.columns

#EDA

my_report = sv.analyze(df)

my_report.show_html("Auto_EDA_Report.html")


#seprating Input and Output variables Y is target variable X is input variable
#in my project there are two outputs so i put it here 
df['Output1']=df['Output2'].astype(bool)

y = df.loc[:,['Output1', 'Output2']]
X = df.drop(['Output1','Output2'], axis=1)
X.head()
y = y.to_records(index=False)



#Split the data in test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y["Output"], random_state=1)

print ("X_train = ", X_train.shape)
print ("X_test = ", X_test.shape)
print ("Y_train = ", y_train.shape)
print ("Y_test = ", y_test.shape)

#Random Forest Survival 

rsf = RandomSurvivalForest(max_depth=3, random_state=1)
rsf.fit(X_train, y_train)

rsf.score(X_train, y_train)#0.96
rsf.score(X_test, y_test)#0.97


#Sample prediction
#Sorting out variables based on eGFR values, and taking top 3 and bottom 3 values as sample for prediction
X_test_sorted = X_test.sort_values(by=["any one dominating input feature"])
X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))

X_test_sel

surv_rsf = rsf.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(surv_rsf):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)


lower, upper = np.percentile(y["Output1"], [10, 90])
y_times = np.arange(lower, upper + 1)
print(y_times)

surv_test_rsf = rsf.predict_survival_function(X_test, return_array=False)

#To get Brier Score
T1, T2 = surv_test_rsf[0].x.min(),surv_test_rsf[0].x.max()
mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
times = y_times[~mask]

#it will gives risk score for the survival
rsf_surv_prob_test = np.row_stack([ fn(times) for fn in surv_test_rsf  ])
rsf_surv_prob_test


#This will shows survival curv with the survial probability within years 

    six_mon = rsf_surv_prob_test1.iloc[ :,20]*100
    two_yr = rsf_surv_prob_test1.iloc[ :,40]*100
    five_yr = rsf_surv_prob_test1.iloc[ :,-1]*100 
    
    plt.step(loaded_model.event_times_,surv[0], where="post", label=str(0))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    st.pyplot()
    
    return(('Probability of survival at 6 months in % :', six_mon.to_string(index=False)),
    ('Probability of survival at 2 years in % :', two_yr.to_string(index=False)),
    ('Probability of survival at 5 years :', five_yr.to_string(index=False))) 

