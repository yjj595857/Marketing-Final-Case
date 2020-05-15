#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn import metrics
from sklearn.metrics import auc, classification_report, f1_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

pd.set_option('display.max_columns', 500)


# ## Data Preprocessing

# In[101]:


# Read data
sub = pd.read_pickle(r'subscribers.dms')
sub = sub.set_index('subid')
print(sub.shape)
sub.head()


# In[102]:


# Read engagement data

eng = pd.read_pickle(r'engagement.dms')
print(eng.shape)
eng.head()


# In[103]:


# We have approximately 1% of null values, which happened when users didn't open the app on that day
eng.isnull().sum()


# In[104]:


# Fill the null values with 0
eng = eng.fillna(0)


# In[105]:


def period_analysis(df, period):
    
    def active(row):
        if row['app_opens'] != 0:
            return 1
        else:
            return 0
    
    df['duration'] = df.groupby('subid')['date'].transform('count')
    df[ period + 'active'] = df.apply(active,axis=1)
    df[ period + 'active_days'] = df.groupby('subid')[ period + 'active'].transform('count')
    df[ period + 'active_%'] = df[ period + 'active_days']/df['duration']

    # apps_open
    df[ period + 'appopen_count'] = df.groupby('subid')['app_opens'].transform('sum')
    df[ period + 'appopen_avg'] = df.groupby('subid')['app_opens'].transform('mean')

    # cust_service_mssgs
    df[ period + 'msg_count'] = df.groupby('subid')['cust_service_mssgs'].transform('sum')
    df[ period + 'msg_avg'] = df.groupby('subid')['cust_service_mssgs'].transform('mean')

    # num_videos_completed
    df[ period + 'vd_complt_count'] = df.groupby('subid')['num_videos_completed'].transform('sum')
    df[ period + 'vd_complt_avg'] = df.groupby('subid')['num_videos_completed'].transform('mean')

    # num_videos_more_than_30_seconds
    df[ period + 'vd_>30s_count'] = df.groupby('subid')['num_videos_more_than_30_seconds'].transform('sum')
    df[ period + 'vd_>30s_avg'] = df.groupby('subid')['num_videos_more_than_30_seconds'].transform('mean')

    # num_videos_rated
    df[ period + 'vd_rated_count'] = df.groupby('subid')['num_videos_rated'].transform('sum')
    df[ period + 'vd_rated_avg'] = df.groupby('subid')['num_videos_rated'].transform('mean')

    # num_series_started
    df[ period + 'srs_start_count'] = df.groupby('subid')['num_series_started'].transform('sum')
    df[ period + 'srs_start_avg'] = df.groupby('subid')['num_series_started'].transform('mean')

    df = df.drop([ period + 'active'],axis=1)


# In[106]:


## First, we want to look at user-wide analysis

eng_id = eng.copy()

period_analysis(eng_id,"")

eng_id['start_date'] = eng_id.groupby('subid')['date'].transform('first')
eng_id['end_date'] = eng_id.groupby('subid')['date'].transform('last')
eng_id['start_month'] = eng_id['start_date'].dt.month_name()
eng_id['end_month'] = eng_id['end_date'].dt.month_name()
eng_id = eng_id.drop(['active','start_date','end_date'],axis=1)

eng_id = eng_id.drop_duplicates('subid').set_index('subid')
eng_id = eng_id.iloc[:,-17:]


# In[107]:


## Then, we want to look at product usage in each payment period

# Period = 0
eng_pd_0 = eng.loc[eng['payment_period'] == 0]
period_analysis(eng_pd_0,"p0_")
eng_pd_0 = eng_pd_0.drop_duplicates('subid').set_index('subid')
eng_pd_0 = eng_pd_0.iloc[:,-14:]

# Period = 1
eng_pd_1 = eng.loc[eng['payment_period'] == 1]
period_analysis(eng_pd_1,"p1_")
eng_pd_1 = eng_pd_1.drop_duplicates('subid').set_index('subid')
eng_pd_1 = eng_pd_1.iloc[:,-14:]

# Period = 2
eng_pd_2 = eng.loc[eng['payment_period'] == 2]
period_analysis(eng_pd_2,"p2_")
eng_pd_2 = eng_pd_2.drop_duplicates('subid').set_index('subid')
eng_pd_2 = eng_pd_2.iloc[:,-14:]


# Period = 3
eng_pd_3 = eng.loc[eng['payment_period'] == 3]
period_analysis(eng_pd_3,"p3_")
eng_pd_3 = eng_pd_3.drop_duplicates('subid').set_index('subid')
eng_pd_3 = eng_pd_3.iloc[:,-14:]


# In[108]:


# Merge all periods of engagement data

eng_all = pd.merge(eng_id, eng_pd_0, left_index=True, right_index=True, how='left')
eng_all = pd.merge(eng_all, eng_pd_1, left_index=True, right_index=True, how='left')
eng_all = pd.merge(eng_all, eng_pd_2, left_index=True, right_index=True, how='left')
eng_all = pd.merge(eng_all, eng_pd_3, left_index=True, right_index=True, how='left')


# In[109]:


eng_all.shape


# In[110]:


# Read Customer Service Representatives data
rep = pd.read_pickle(r'customer_service_reps.dms')
print(rep.shape)
rep.head()


# In[111]:


# clean the outliers of variable "age"
def clean_age(data):
    age = data['age']
    if age <= 0 or age >= 100:
        return np.nan
    else:
        return age

# categorize revenue_net to better flag churn
def revenue_cat(data):
    revenue = data['revenue_net']
    if  revenue <= 0:
        return '<=0'
    else:
        return '>0'


## Prepare the data we're going to use for A/B Testing 

# select the UAE plan_type
uae = []
counts = sub.plan_type.value_counts()
for idx in counts.index:
    if 'uae' in idx and '7' not in idx:
        uae.append(idx)

rep = rep.drop_duplicates('subid').set_index('subid').drop('payment_period',axis=1)     
sub_rep = pd.merge(sub, rep, left_index=True, right_index=True, how='inner')
sub_rep_uae = sub_rep[sub_rep['plan_type'].isin(uae)]
sub_rep_uae['age'] = sub_rep_uae.apply(clean_age, axis=1)
sub_rep_uae['revenue_cat'] = sub_rep_uae.apply(revenue_cat, axis=1)
#sub_rep_uae.to_csv('sub_rep_uae.csv')


## Prepare the data we're going to use for Customer Segmentation and Churn Modeling
sub_eng = pd.merge(sub, eng_all, left_index=True, right_index=True, how='inner')
sub_eng_rep = pd.merge(sub_eng, rep, left_index=True, right_index=True, how='inner')
sub_eng_rep['revenue_cat'] = sub_eng_rep.apply(revenue_cat, axis=1)
sub_eng_rep['age'] = sub_eng_rep.apply(clean_age, axis=1)
#sub_eng_rep.to_csv('sub_eng_rep.csv')


# ## A/B Testing

# In[112]:


# Read and prepare data
sub_rep = pd.read_csv('sub_rep_uae.csv')

sub_rep = sub_rep.drop(['package_type','num_weekly_services_utilized','preferred_genre','intended_use',
                        'weekly_consumption_hour','num_ideal_streaming_services','age','male_TF','country',
                        'attribution_technical','attribution_survey','op_sys','months_per_bill_period',
                        'creation_until_cancel_days','initial_credit_card_declined','language','payment_type',
                        'customer_service_rep_id','billing_channel','cancel_before_trial_end'],axis=1)
sub_rep.head()


# In[113]:


print(sub_rep.shape)


# ### 14-day Trial VS No Trial
# For 14-day Trial users: Converted if Trial Completed and Revenue > 0
# <br>For No Trial users: Converted if Revenue > 0 (or paid_TF=True)

# In[114]:


sub_rep.groupby(['trial_completed_TF','refund_after_trial_TF','revenue_cat']).size()


# In[115]:


# Define the conversions of No Trial users
sub_rep_0_trial = sub_rep.loc[sub_rep['num_trial_days']==0]

def convert_trial_0(data):
    revenue = data['revenue_cat']
    if  revenue == '>0':
        return 1
    else:
        return 0

sub_rep_0_trial['conversion'] = sub_rep_0_trial.apply(convert_trial_0,axis=1)

# Define the conversions of 14-day Trial users
sub_rep_14_trial = sub_rep.loc[(sub_rep['num_trial_days']==14) & (sub_rep['plan_type']=='base_uae_14_day_trial')
                              & (sub_rep['retarget_TF']==True)]

def convert_trial_14(data):
    revenue = data['revenue_cat']
    trial = data['trial_completed_TF']
    if  revenue == '>0' and trial == True:
        return 1
    else:
        return 0

sub_rep_14_trial['conversion'] = sub_rep_14_trial.apply(convert_trial_14,axis=1)


# ### Hypotheses
# A: 14-day Trial
# <br>B: No Trial
# <br>H0: Conversion Rate A = B
# <br>H1: Conversion Rate A != B

# In[116]:


### Conduct A/B Testing
sub_rep_trial_AB = pd.concat([sub_rep_0_trial,sub_rep_14_trial])
sub_rep_trial_AB = sub_rep_trial_AB[['subid','num_trial_days','conversion']].reset_index()

# number of samples
n_A = sub_rep_trial_AB.query('num_trial_days=="14"').shape[0]
n_B = sub_rep_trial_AB.query('num_trial_days=="0"').shape[0]
print("Number of samples for A: {} and B: {}".format(n_A,n_B))

# number of conversions
conver_A = sub_rep_trial_AB.query('num_trial_days=="14" & conversion=="1"').shape[0]
conver_B = sub_rep_trial_AB.query('num_trial_days=="0" & conversion=="1"').shape[0]
print("Number of conversions for A: {} and B: {}".format(conver_A,conver_B))

# conversion rate
p_A = round(conver_A/n_A,2)
p_B = round(conver_B/n_B,2)
print("Conversion rate for A: {} and B: {}".format(p_A,p_B))

from scipy import stats as st
from scipy.stats import norm
z_alpha = st.norm.ppf(1-0.05/2)
z = (p_B-p_A) / np.sqrt(p_A*(1-p_A)/n_B)
print("Z_alpha: {} and Z:{}".format(z_alpha,z))

z > z_alpha


# In[117]:


# Calculate the optimal size
from scipy.stats import norm
z_beta = st.norm.ppf(1-0.2)

p_hat = (p_A+p_B)/2

n = (z_alpha*np.sqrt(2*p_hat*(1-p_hat))+z_beta*np.sqrt(p_A*(1-p_A)+p_B*(1-p_B)))**2 * 1/(p_B-p_A)**2 
print("Optimal sample size is %.2f" % n)


# In[118]:


# Conduct the test 10 times

def ab_test(N_concat): 
    test=[]

    n_A = N_concat.query('num_trial_days=="14"').shape[0]
    n_B = N_concat.query('num_trial_days=="0"').shape[0]
    
    convert_N_A = N_concat.query('num_trial_days=="14" & conversion=="1"').shape[0]
    convert_N_B = N_concat.query('num_trial_days=="0" & conversion=="1"').shape[0]
    
    p_N_A = round(convert_N_A / n_A,2)
    p_N_B = round(convert_N_B / n_B,2)
    
    z_alpha = st.norm.ppf(1-0.05/2)
    z = round((p_N_B-p_N_A) / np.sqrt(p_N_A*(1-p_N_A)/n_B),2)
    
    reject_H0 = z > z_alpha 
    
    test.append(p_N_B)
    test.append(z)
    test.append(reject_H0)
    
    return test

ab_dict={}

i=0
while i < 10:
    N_A = sub_rep_trial_AB[sub_rep_trial_AB["num_trial_days"]==14]
    N_B = sub_rep_trial_AB[sub_rep_trial_AB["num_trial_days"]==0].sample(n=33)
    N_concat = pd.concat([N_A,N_B])
    ab_dict[i]=ab_test(N_concat)
    i += 1

df = pd.DataFrame(ab_dict)
df = df.T
df.rename(columns={0:'p_B_sample', 1:'z_score', 2:'reject_H0'},inplace=True)
df = df.T
trials=[1,2,3,4,5,6,7,8,9,10]
df.columns = trials
df


# 10/10 trials we can reject H0: For retargeted customers, conversion rate of 14-day Trial = conversion rate of No Trial and accept that No Trial affects the converision rate.

# ### Base 14-day Trial VS High 14-day Trial
# For both 14-day Trial users: Converted if Trial Completed and Revenue > 0

# In[119]:


sub_rep_base= sub_rep.loc[(sub_rep['plan_type']=='base_uae_14_day_trial') & (sub_rep['retarget_TF']==False)]
sub_rep_high = sub_rep.loc[(sub_rep['plan_type']=='high_uae_14_day_trial') & (sub_rep['retarget_TF']==False)]

sub_rep_base['conversion'] = sub_rep_base.apply(convert_trial_14,axis=1)
sub_rep_high['conversion'] = sub_rep_high.apply(convert_trial_14,axis=1)


# ### Hypotheses
# A: Base 14-day Trial
# <br>B: High 14-day Trial
# <br>H0: Conversion Rate A = B
# <br>H1: Conversion Rate A != B

# In[120]:


### Conduct A/B Testing
sub_rep_trial_AB_2 = pd.concat([sub_rep_base,sub_rep_high])
sub_rep_trial_AB_2 = sub_rep_trial_AB_2[['subid','plan_type','conversion']].reset_index()

# number of samples
n_A = sub_rep_trial_AB_2.query('plan_type=="base_uae_14_day_trial"').shape[0]
n_B = sub_rep_trial_AB_2.query('plan_type=="high_uae_14_day_trial"').shape[0]
print("Number of samples for A: {} and B: {}".format(n_A,n_B))

# number of conversions
conver_A = sub_rep_trial_AB_2.query('plan_type=="base_uae_14_day_trial" & conversion=="1"').shape[0]
conver_B = sub_rep_trial_AB_2.query('plan_type=="high_uae_14_day_trial" & conversion=="1"').shape[0]
print("Number of conversions for A: {} and B: {}".format(conver_A,conver_B))

# conversion rate
p_A = round(conver_A/n_A,2)
p_B = round(conver_B/n_B,2)
print("Conversion rate for A: {} and B: {}".format(p_A,p_B))

from scipy import stats as st
from scipy.stats import norm
z_alpha = st.norm.ppf(1-0.05/2)
z = (p_B-p_A) / np.sqrt(p_A*(1-p_A)/n_B)
print("Z_alpha: {} and Z:{}".format(z_alpha,z))

z > z_alpha


# In[121]:


# Calculate the optimal size
from scipy.stats import norm
z_beta = st.norm.ppf(1-0.2)

p_hat = (p_A+p_B)/2

n = (z_alpha*np.sqrt(2*p_hat*(1-p_hat))+z_beta*np.sqrt(p_A*(1-p_A)+p_B*(1-p_B)))**2 * 1/(p_B-p_A)**2 
print("Optimal sample size is %.2f" % n)


# We failed to reject H0: Conversion rate of Base 14-day Trial = Conversion rate of High 14-day Trial. High 14-day Trial can not affect the conversion rate.

# ## Customer Segmentation

# In[122]:


# Read and prepare data

sub_eng_rep = pd.read_csv('sub_eng_rep.csv', index_col=0).reset_index()
sub_eng_rep = sub_eng_rep.drop(['monthly_price','account_creation_date_x','creation_until_cancel_days','cancel_before_trial_end',
                                'trial_end_date','initial_credit_card_declined','revenue_net','duration','attribution_survey',
                                'customer_service_rep_id','cancel_date','revenue_net_1month','last_payment',
                               'country','language','billing_channel','months_per_bill_period','account_creation_date_y'], axis=1)
for col in sub_eng_rep.columns:
    if 'avg' in col:
        sub_eng_rep = sub_eng_rep.drop(col,axis=1)

sub_eng_rep.head()


# In[123]:


# Drop columns with lots of null values and replace them with 0
thresh = pd.Series(sub_eng_rep.isnull().sum())
thresh = thresh[thresh > 10000]
null_index_list = thresh.index
sub_eng_rep = sub_eng_rep.drop(null_index_list, axis=1)

sub_eng_rep.fillna(0, inplace=True)


# In[124]:


# Get dummies of categorical values
sub_eng_rep_clus = pd.get_dummies(sub_eng_rep, columns=['plan_type','intended_use','attribution_technical',
                                                        'op_sys','start_month','end_month',
                                                          'revenue_cat'])
sub_eng_rep_clus = sub_eng_rep_clus.drop('subid',axis=1)
print(sub_eng_rep_clus.shape)
sub_eng_rep_clus.head()


# ### Elbow Test

# In[ ]:


from sklearn.cluster import KMeans

def get_wcss(df):
  wcss = []
  for i in range(1, 21):
      kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
      kmeans.fit(df)
      wcss.append(kmeans.inertia_)
  plt.plot(range(1, 21), wcss)
  plt.title('Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.show()

# determine k - look at wcss for each number of clustrs to determine how number of clusters
get_wcss(sub_eng_rep_clus)


# ### Predict Clusters

# In[ ]:


# Create k-means model
nclusters = 3

kmeans = KMeans(n_clusters=nclusters, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(sub_eng_rep_clus)
cluster = kmeans.fit_predict(sub_eng_rep_clus)


# In[ ]:


# Let's look at the centroids of each feature for each cluster
centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids.columns = sub_eng_rep_clus.columns

# Drop the features which have too little difference among clusters
for col in centroids.columns:
    if (centroids[col].diff()[1] < 0.1) and (centroids[col].diff()[2] < 0.1):
        centroids = centroids.drop(col,axis=1)
        
centroids
#centroids.to_csv('/Users/huangyurong/Desktop/Marketing Analytics/Final Case/cluster_summary.csv', index=False, header=True)


# In[ ]:


# Mark each customer with the cluster he/she is assigned to

cluster_result = pd.DataFrame({'subid': sub_eng_rep.subid.values})

cluster_result['cluster'] = np.where(cluster==0, "Medium Addicted",
                                             np.where(cluster==1, "Inactive and Churn",
                                                      "Heavily Addicted"))
cluster_result.head()


# ## Churn Forecasting

# In[ ]:


sub_eng_rep = pd.read_csv('sub_eng_rep.csv', index_col=0).reset_index()
sub_eng_rep = sub_eng_rep.drop(['monthly_price','account_creation_date_x','creation_until_cancel_days','cancel_before_trial_end',
                                'trial_end_date','initial_credit_card_declined','duration','attribution_survey','next_payment',
                                'customer_service_rep_id','cancel_date','revenue_net','revenue_net_1month','last_payment',
                               'country','language','billing_channel','months_per_bill_period','account_creation_date_y'], axis=1)


# ### Predict Whether Trial Users Will Become Full Subscribers
# Churn if failed to complete the trial
# <br> Churn if completed trial, requested refund, and revenue <= 0
# <br> Churn if completed trial, revenue <= 0 though did not request refund

# In[ ]:


sub_eng_rep_1 = sub_eng_rep.copy()
sub_eng_rep_1.groupby(['trial_completed_TF','refund_after_trial_TF','revenue_cat']).size()


# In[ ]:


def churn_1(data):
    trial = data['trial_completed_TF']
    refund = data['refund_after_trial_TF']
    revenue = data['revenue_cat']
    
    if trial == False:
        return 1
    elif (trial == True) & (refund == True) & (revenue != '>0'):
        return 1
    elif (trial == True) & (refund == False) & (revenue != '>0'):
        return 1
    else: 
        return 0

sub_eng_rep_1['churn'] = sub_eng_rep_1.apply(churn_1, axis=1)


# In[ ]:


sub_eng_rep_1.churn.value_counts()


# In[ ]:


# Drop columns which involved in the determination of churn 
sub_eng_rep_1 = sub_eng_rep_1.drop(['revenue_cat','paid_TF','trial_completed_TF','refund_after_trial_TF',], axis=1)

# Drop columns with lots of null values and replace them with 0
thresh = pd.Series(sub_eng_rep_1.isnull().sum())
thresh = thresh[thresh > 10000]
null_index_list = thresh.index
sub_eng_rep_1 = sub_eng_rep_1.drop(null_index_list, axis=1)
sub_eng_rep_1 = sub_eng_rep_1.drop('renew', axis=1)

print(sub_eng_rep_1.shape)
sub_eng_rep_1.head()


# In[ ]:


# Fill categorical null values with "other"
def intend_use(data):
    use = data['intended_use']
    if pd.isna(use) == True:
        use = 'other'
    return use
sub_eng_rep_1['intended_use'] = sub_eng_rep_1.apply(intend_use, axis=1)

def op_sys(data):
    sys = data['op_sys']
    if pd.isna(sys) == True:
        sys = 'other'
    return sys
sub_eng_rep_1['op_sys'] = sub_eng_rep_1.apply(op_sys, axis=1)

# Fill continuos null values with 0
sub_eng_rep_1.fillna(0, inplace=True)


# In[ ]:


# Get dummies of categorical values
sub_eng_rep_1 = pd.get_dummies(sub_eng_rep_1, columns=['intended_use','attribution_technical','op_sys','plan_type',
                                                      'start_month','end_month'])
print(sub_eng_rep_1.shape)
sub_eng_rep_1.head()


# In[ ]:


def model_evaluation(clf):
    clf.fit(X_train, y_train)
    #print("Coefficient of each independent variable is {}".format(clf.coef_))
    print("Mean train cross-validation score (5-folds): {:.4f}".format(np.mean(cross_val_score(clf, X_train, y_train, cv=5))))
    print("Mean test cross-validation score (5-folds): {:.4f}".format(np.mean(cross_val_score(clf, X_test, y_test, cv=5))))

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:,1]
    #print('Training accuracy: {:.4f}'.format(clf.score(X_train, y_train)))
    #print('Test accuracy: {:.4f}'.format(clf.score(X_test, y_test)))
    print('AUC: {:.4f}'.format(metrics.roc_auc_score(y_test,prob)))

    CM = confusion_matrix(y_test,pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    recall = tp/(tp+fn) 
    precision = tp/(tp+fp) 
    #print("Confusion matrix for the test set: {}".format(CM))
    #print("TN: {} / FP: {} / FN: {} / TP: {}".format(tn, fp, fn, tp))
    print("Recall: {:.4f} / Precision: {:.4f}".format(recall, precision))
    #print(classification_report(y_test, pred))


# In[ ]:


X = sub_eng_rep_1.drop(['churn'],axis=1)
y = sub_eng_rep_1['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

for c in [.01,.1,1,10,100]:
    clf = LogisticRegression(C=c)
    model_evaluation(clf)
    print("-"*60)


# In[ ]:


clf_best = LogisticRegression(C=1).fit(X_train, y_train)

cols = X_train.columns

# Calculate the correlation coefficient between each feature and the target feature
index = 0
for i in clf_best.coef_[0]:
    print("%s      %.5f" %(cols[index],i))
    index += 1


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Set parameters to help avoid overfitting
for dep in [3,4,5,6,7]:
        clfDT = DecisionTreeClassifier(max_depth=dep, max_features='sqrt', min_samples_leaf=50).fit(X_train, y_train)
        print("The result for Decision Tree Model with depth={}: ".format(dep))
        model_evaluation(clfDT)
        print("-"*60)


# In[ ]:


# Decision Tree Model with the best params
clf_DT_best = DecisionTreeClassifier(max_depth=7, max_features='sqrt', min_samples_leaf=50, random_state=0).fit(X_train, y_train)

def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    feat_importances = pd.Series(clf_DT_best.feature_importances_, index=feature_names)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    
plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf_DT_best, X_train.columns)
plt.show()


# In[ ]:


# Plot the decision tree 
from sklearn import tree
from IPython.display import SVG, display, Image
from graphviz import Source

graph = Source(tree.export_graphviz(clf_DT_best, out_file=None
   , feature_names=X.columns, class_names=['No Churn', 'Churn'] 
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# ## Revenue Modeling

# In[ ]:


churn = pd.DataFrame(data=clf_DT_best.predict_proba(X_test))
churn = churn.join(pd.DataFrame(y_test).reset_index())
churn = churn.merge(pd.DataFrame(sub_eng_rep_1['subid']).reset_index(),on='index')
churn = churn[['subid','churn',1]]
churn.columns = ['subid','y_test','y_pred']
churn


# In[ ]:


offer_accept_rate = 0.3

def receive(y_pred,thresh):
    if y_pred >= thresh:
        return 1
    else:
        return 0

import random
def accept(receive):
    if receive == 1:
        if random.random() <= offer_accept_rate:
            return 1
        else:
            return 0
    else:
        return 0
    
def renew(row,thresh):
    value = 0
    receive = 'receive_offer_'+str(thresh)
    accept = 'accept_offer_'+str(thresh)
    if (row.y_test == 0):
        if (row[receive] == 0) | ((row[receive] == 1) & (row[accept] == 0)):
            value = 1
    return value

def plan(row,thresh):
    accept = 'accept_offer_'+str(thresh)
    renew = 'renew_at_base_'+str(thresh)
    if row[accept] == 1:
        return 'offer'
    elif row[renew] == 1:
        return 'base'
    else:
        return 'churn'


# In[ ]:


def rev_model(churn, offer_accept_rate, thresh):
    
    rec = 'receive_offer_'+str(thresh)
    churn[rec] = churn.apply(lambda x: receive(x['y_pred'],thresh),axis=1)
    
    acc = 'accept_offer_'+str(thresh)
    churn[acc] = churn[rec].apply(accept)
    
    ren = 'renew_at_base_'+str(thresh)
    churn[ren] = churn.apply(lambda x: renew(x,thresh),axis=1)
    
    pl = 'plan_'+str(thresh)
    churn[pl] = churn.apply(lambda x: plan(x,thresh),axis=1)
    
    return churn


# In[ ]:


rev_model(churn,offer_accept_rate,0.5)

