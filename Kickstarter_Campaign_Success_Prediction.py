# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:34:43 2020

@author: Nupur
"""

##################################### ANALYSIS #####################################################################

#################################### REGRESSION TASK ################################################################

##Loading Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

## Importing Data ##
kickstarter = pd.read_excel(r"C:\Users\Nupur\Desktop\McGill Notes\Data Mining-Warut\Data\Kickstarter.xlsx")

## Data Preprocessing ##

#removing observations which have state other than successful and failed
def stage1(kickstarter):
    kickstarter = kickstarter[(kickstarter["state"]=="failed")|(kickstarter["state"]=="successful")]
    return kickstarter
kickstarter = stage1(kickstarter)

## Checking for missing data 
kickstarter.isnull().sum()

# replacing missing categories with "Missing"

def stage2(kickstarter):
    kickstarter["category"].fillna("Missing",inplace = True)
    return kickstarter

kickstarter = stage2(kickstarter)

## replacing missing values of launch_to_state_change_days with a constant vaue of zero
def stage3(kickstarter):
    from sklearn.impute import SimpleImputer
    imputer2 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value = 0)
    kickstarter.iloc[:,44] =imputer2.fit_transform(kickstarter.iloc[:,44].values.reshape(-1,1))
    return kickstarter
kickstarter = stage3(kickstarter)

## checking again for missing values if any
kickstarter.isnull().sum()

## Exploring Dataset ##
#columns and data types
kickstarter.info()

#basic statistics of the data
describe = kickstarter.describe()

#distribution of data across state
percent_dist = round(kickstarter["state"].value_counts()/len(kickstarter["state"])*100,2)
print(percent_dist)
percent_dist.plot(kind="bar")

#distribution of projects across various categories
kickstarter.groupby(['state','category']).size().unstack(0).plot.bar()

#distribution of usd_pledged amount for different status of the projects
kickstarter.groupby("state")["usd_pledged"].mean().plot(kind="bar")

#distribution of goal amount for different status of the projects
kickstarter.groupby("state")["goal"].mean().plot(kind="bar")

# distribution of state and usd pledged for different countries
kickstarter.groupby(["state","country"]).size().unstack(0).plot.bar()
kickstarter.groupby("country")["usd_pledged"].mean().plot(kind="bar")

#average usd_pledged amount for different categories
kickstarter.groupby("category")["usd_pledged"].mean().plot(kind="bar")

#number of backers for each category and state
kickstarter.groupby("state")["backers_count"].sum().plot(kind="bar")
kickstarter.groupby(["category","state"])["backers_count"].sum()

#name_len distribution for different status of projects
kickstarter.groupby(["state","name_len_clean"]).size().unstack(0).plot.bar()
kickstarter.groupby(["name_len_clean"])["usd_pledged"].mean().plot(kind="bar")
kickstarter.groupby(["state","blurb_len_clean"]).size().unstack(0).plot.bar()
kickstarter.groupby(["blurb_len_clean"])["usd_pledged"].mean().plot(kind="bar")

##number of projects for each category below
kickstarter["disable_communication"].value_counts()
kickstarter.groupby(["staff_pick","state"]).size()
kickstarter["category"].value_counts()
kickstarter.groupby(["spotlight","state"]).size()

##Data Cleansing##
##removing variables which don't have predictive power (project_id,name,deadline,state_changed_at,created_at,launched_at)##
#removing state variable as we cannot know whether a project will be successful or not beforehand and if we already know the state of the project then there is no point of prediction.
#removing backers_count variable as well because we would not know the number of backers before prediction.
#removing pledged amount as we need to predict the usd_pledged and before the project is launched we would not know the how much amount has been pledged and if we already know the pledged amount then there is no point of predicting usd_pledged amount as it would then just be a conversion to usd currency.
#removing staff_pick and spotlight as before the project is launched we can't know whether it is staff_picked or spotlight
#removed variables related to state changes as we are only focussing on successful and failed states.
#removed disable_communication as it contains only one field of False which doesn't give much information for the prediction purpose. 

kickstarter2 = kickstarter.drop(columns=["project_id","name","disable_communication","deadline","state_changed_at","created_at","launched_at","currency","state_changed_at_weekday",
                                        "state_changed_at_month","state_changed_at_day","state_changed_at_yr","state_changed_at_hr","launch_to_state_change_days"])

kickstarter2.columns
kickstarter_copy_corr = pd.get_dummies(kickstarter2,columns=["state","staff_pick","spotlight"])

##taking a look at the correlation of the variables
data_corr = kickstarter_copy_corr.corr()

ax = sns.heatmap(
   data_corr, 
    vmin=-1, vmax=1, center=0,
    cmap = "Blues",square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

##taking a look at the correlation of the variables
from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["pledged"],kickstarter_copy_corr["usd_pledged"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["state_failed"],kickstarter_copy_corr["spotlight_False"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["state_successful"],kickstarter_copy_corr["spotlight_True"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["name_len"],kickstarter_copy_corr["name_len_clean"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["deadline_yr"],kickstarter_copy_corr["created_at_yr"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["deadline_yr"],kickstarter_copy_corr["launched_at_yr"])
print(corr)

from scipy.stats import pearsonr
corr = pearsonr(kickstarter_copy_corr["created_at_yr"],kickstarter_copy_corr["launched_at_yr"])
print(corr)

# since name_len,created_at_yr,launched_at_yr are highly correalted therefore I am removing these variables also to avoid multicollinearity issues.
# removing blurb_len and keeping only blurb_len_clean for the prediction purpose.

kickstarter2 = kickstarter2.drop(columns=["state","spotlight","staff_pick","pledged","backers_count","name_len","blurb_len","created_at_yr","launched_at_yr"])

##dummifying categorical variables
kickstarter2.info()

kickstarter2 = pd.get_dummies(kickstarter2, columns = ["country","category","deadline_weekday","created_at_weekday","launched_at_weekday"])

kickstarter2.columns

kickstarter2 = kickstarter2.reset_index()
kickstarter2 = kickstarter2.drop(columns=["index"])
kickstarter2.columns

##Creating X and Y variables
X = kickstarter2.drop(columns=["usd_pledged"])
y= kickstarter2["usd_pledged"]

#standardising the variables
X_copy = kickstarter2.drop(columns=["usd_pledged"])
X1 = X_copy
X1.info()
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X1.iloc[:,0:16]= standardizer.fit_transform(X1.iloc[:,0:16])
X1.head()

##Feature Selection##
##LASSO Feature Selection##

coef = []
alphas = [0.01,0.05,0.1,0.6,1,6,10]
for i in alphas:
    from sklearn.linear_model import Lasso 
    model = Lasso(alpha=i, max_iter=10000, random_state=0)
    model.fit(X1,y)
    coef.append(model.coef_)
model_ldf= pd.DataFrame(coef,index =alphas,columns = X1.columns)
#at alpha=10, useless predictors=["country_IE","country_LU","country_SG","category_Academic","category_Blues","category_Comedy","category_Immersive","category_Makerspaces","category_Shorts","category_Thrillers","category_Webseries","deadline_weekday_Wednesday","created_at_weekday_Wednesday","launched_at_weekday_Thursday"])

#Building  Model##
## Lasso Model##
#After feature selection from Lasso I got to know that the 14 predictors were useless so these have been removed from the list of predictors
X2 = X1.drop(columns = ["country_IE","country_LU","country_SG","category_Academic","category_Blues","category_Comedy","category_Immersive","category_Makerspaces","category_Shorts","category_Thrillers","category_Webseries","deadline_weekday_Wednesday","created_at_weekday_Wednesday","launched_at_weekday_Thursday"])
X2.head()
#splitting the data into training and testing dataset
from sklearn.model_selection import train_test_split
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y,test_size=0.30,random_state=0)

# finding optimal alpha value
coef1=[]
alphas_ = [0.01,0.05,0.1,0.6,1,6,10]
for i in alphas:
    from sklearn.linear_model import Lasso
    lasso=Lasso(alpha=i,max_iter=10000,random_state=0)
    model_la = lasso.fit(X2_train,y2_train)
    y_test_pred_la = model_la.predict(X2_test)
    from sklearn.metrics import mean_squared_error
    print('Alpha=',i,'/ MSE=',mean_squared_error(y2_test,y_test_pred_la))

# Fitting the model at Alpha=10
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
lasso = Lasso(alpha=10,random_state=0)
model2 = lasso.fit(X2_train,y2_train)
y_test_pred2 = model2.predict(X2_test)
mse = mean_squared_error(y2_test,y_test_pred2)
print("Lasso Model MSE: {}".format(mse))
#17987217067.035107

##Ridge Regression Model##

XR = X.drop(columns = ["country_IE","country_LU","country_SG","category_Academic","category_Blues","category_Comedy","category_Immersive","category_Makerspaces","category_Shorts","category_Thrillers","category_Webseries","deadline_weekday_Wednesday","created_at_weekday_Wednesday","launched_at_weekday_Thursday"])
XR.head()
#from sklearn.model_selection import train_test_split
XRR_train, XRR_test , yRR_train , yRR_test = train_test_split(XR, y, test_size = 0.30, random_state =0)

#finding optimal alpha
coef2=[]
alphas_ = [0.01,0.05,0.1,0.6,1,6,10]
for i in alphas_:
    from sklearn.linear_model import Ridge
    r=Ridge(alpha=i,max_iter=10000,random_state=0)
    model_r = r.fit(XRR_train,yRR_train)
    y_test_pred_r = model_r.predict(XRR_test)
    from sklearn.metrics import mean_squared_error
    print('Alpha=',i,'/ MSE=',mean_squared_error(yRR_test,y_test_pred_r))

#Fitting the Ridge Model at Alpha=10

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10,random_state=0)
model3 = ridge.fit(XRR_train,yRR_train)
y_test_pred3 = model3.predict(XRR_test)
mse2 = mean_squared_error(yRR_test,y_test_pred3)
print("Ridge Model MSE: {}".format(mse2))
#17985709314.015327

##Linear Regression Model##

from sklearn.model_selection import train_test_split
XLR_train, XLR_test , yLR_train , yLR_test = train_test_split(XR, y, test_size = 0.30, random_state =0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model_lm = lm.fit(XLR_train,yLR_train)
y_test_pred = model_lm.predict(XLR_test)
from sklearn.metrics import mean_squared_error
mse4 = mean_squared_error(yLR_test,y_test_pred)
print("Linear Regression Model MSE: {}".format(mse4))
#17990169403.310646

## Random Forest Model ##
from sklearn.model_selection import train_test_split
XRF_train, XRF_test , yRF_train , yRF_test = train_test_split(XR, y, test_size = 0.30, random_state =0)
XR.head()
#finding optimal depth and estimators of the tree
depth = [3,4,6,8,10]
for i in depth:
    from sklearn.ensemble import RandomForestRegressor
    randomforest = RandomForestRegressor(random_state=0, max_depth=i)
    model_rf_d = randomforest.fit(XRF_train,yRF_train)
    y_test_pred_rf_d = model_rf_d.predict(XRF_test)
    from sklearn.metrics import mean_squared_error
    print('Alpha=',i,'/ MSE=',mean_squared_error(yRF_test,y_test_pred_rf_d))
 
estimators = [100,200,300,400,500]
for i in estimators:
    from sklearn.ensemble import RandomForestRegressor
    randomforest = RandomForestRegressor(random_state=0, n_estimators=i)
    model_rf = randomforest.fit(XRF_train,yRF_train)
    y_test_pred_rf = model_rf.predict(XRF_test)
    from sklearn.metrics import mean_squared_error
    print('Alpha=',i,'/ MSE=',mean_squared_error(yRF_test,y_test_pred_rf))    
      
#Fitting Model with depth=10 and estimators=500
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0,n_estimators=500,max_depth=10)
modelRF = rf.fit(XRF_train,yRF_train)
y_test_pred4 = modelRF.predict(XRF_test)
mse_rf = mean_squared_error(yRF_test,y_test_pred4)
print("Random Forest MSE:{}".format(mse_rf))
#16549774340.416492

##K-NN Model ##
X2.head()
from sklearn.model_selection import train_test_split
Xknn_train, Xknn_test , yknn_train , yknn_test = train_test_split(X2, y, test_size = 0.30, random_state =0)

#finding optimal k
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor

for i  in range(1,21):
    knn = KNeighborsRegressor(n_neighbors=i)
    model_knn = knn.fit(Xknn_train,yknn_train)
    y_test_pred_knn = model_knn.predict(Xknn_test)
    print('Alpha=',i,'/ MSE=',mean_squared_error(yknn_test,y_test_pred_knn))    
    
#fitting the K-NN model on optimal k=20
knn = KNeighborsRegressor(n_neighbors=20, weights="uniform")
model_knn = knn.fit(Xknn_train,yknn_train)
y_test_pred_knn = model_knn.predict(Xknn_test)
mse_knn = mean_squared_error(yknn_test,y_test_pred_knn)
print("K-NN Model MSE:{}".format(mse_knn)) 
#18481403713.129467

##CART Model##

from sklearn.model_selection import train_test_split
Xct_train, Xct_test , yct_train , yct_test = train_test_split(XR, y, test_size = 0.30, random_state =0)

#finding optimal depth
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

for i in range(1,11):
    model_ct= DecisionTreeRegressor(max_depth=i)
    scores = cross_val_score(estimator=model_ct,X=X,y=y,cv=10)
    print(i,':',np.average(scores))


#Fitting CART Model for optimal depth
cart = DecisionTreeRegressor(max_depth=3)
model_ct= cart.fit(Xct_train,yct_train)
y_test_pred_ct = model_ct.predict(Xct_test)    
mse_ct= mean_squared_error(yct_test,y_test_pred_ct)
print("CART Model MSE:{}".format(mse_ct))
#17697405256.45186

##Gradient Boosting Model ##
from sklearn.model_selection import train_test_split
Xgb_train,Xgb_test,ygb_train,ygb_test=train_test_split(XR,y,test_size=0.30,random_state=0)

#finding optimal min sample split and estimators
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

for i in range(2,11):
    gbt = GradientBoostingRegressor(random_state=0,min_samples_split=i)
    model_gbt = gbt.fit(Xgb_train,ygb_train)
    scores2 = cross_val_score(estimator=model_gbt,X=X,y=y,cv=5)
    print(i,":",np.average(scores2))
    

estimator_ = [100,200,300,400,500]
for i in estimator_:
    gbt_e = GradientBoostingRegressor(random_state=0,n_estimators=i)
    model_gbt_e = gbt_e.fit(Xgb_train,ygb_train)
    scores3 = cross_val_score(estimator=model_gbt_e,X=X,y=y,cv=5)
    print(i,":",np.average(scores3))
    
#Fitting the model on optimal depth and estimators
gbt_m = GradientBoostingRegressor(n_estimators=100,min_samples_split=3,random_state=0)
model_gbt_m = gbt_m.fit(Xgb_train,ygb_train)
y_test_pred_gbt = model_gbt_m.predict(Xgb_test)
mse_gbt = mean_squared_error(ygb_test,y_test_pred_gbt)
print("GBT Model MSE:{}".format(mse_gbt))
#16829626372.334356

#MLP Model#

from sklearn.model_selection import train_test_split
Xmlp_train,Xmlp_test,ymlp_train,ymlp_test = train_test_split(X2,y,test_size=0.30,random_state=0)

#finding optimal hidden layer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

for i in range(2,21):
    model_mlp= MLPRegressor(hidden_layer_sizes=(i),max_iter=1000, random_state=0)
    scores = cross_val_score(estimator=model_mlp,X=X,y=y,cv=5)
    print(i,':',np.average(scores))

ann = MLPRegressor(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
model_ann = ann.fit(Xmlp_train,ymlp_train)
y_test_pred_ann = model_ann.predict(Xmlp_test)
mse_ann=mean_squared_error(ymlp_test,y_test_pred_ann)
print("MLP Model MSE:{}".format(mse_ann))
#17992377014.74005

###################################################################################################################
########################################### CLASSIFICATION TASK ########################################################################

##Classification Task##

##Data Preprocessing##
##Import Data##
kickstarter = pd.read_excel(r"C:\Users\Nupur\Desktop\McGill Notes\Data Mining-Warut\Kickstarter.xlsx")
kickstarter.info()
kickstarter.isnull().sum()

#removing observations which have state other than successful and failed

def stage1(kickstarter):
    kickstarter = kickstarter[(kickstarter["state"]=="failed")|(kickstarter["state"]=="successful")]
    return kickstarter
kickstarter = stage1(kickstarter)
# replacing missing categories with "Missing"
def stage2(kickstarter):
    kickstarter["category"].fillna("Missing",inplace = True)
    return kickstarter
kickstarter = stage2(kickstarter)
## replacing missing values of launch_to_state_change_days with a constant vaue of zero
def stage3(kickstarter):
    from sklearn.impute import SimpleImputer
    imputer2 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value = 0)
    kickstarter.iloc[:,44] =imputer2.fit_transform(kickstarter.iloc[:,44].values.reshape(-1,1))
    return kickstarter
kickstarter = stage3(kickstarter)
## checking again for missing values if any
kickstarter.isnull().sum()

##Exploring Data##
kickstarter["disable_communication"].value_counts()
kickstarter["country"].value_counts()
kickstarter["staff_pick"].value_counts()
kickstarter["category"].value_counts()
kickstarter["spotlight"].value_counts()

##Data Cleansing##
##removing variables which don't have predictive power (project_id,name,deadline,state_changed_at,created_at,launched_at)##
#removing state variable as we cannot know whether a project will be successful or not beforehand and if we already know the state of the project then there is no point of prediction.
#removing backers_count variable as well because we would not know the number of backers before prediction.
#removing pledged amount as we need to predict the usd_pledged and before the project is launched we would not know the how much amount has been pledged and if we already know the pledged amount then there is no point of predicting usd_pledged amount as it would then just be a conversion to usd currency.
#removing staff_pick and spotlight as before the project is launched we can't know whether it is staff_picked or spotlight
#removed variables related to state changes as we are only focussing on successful and failed states.
#removed disable_communication as it contains only one field of False which doesn't give much information for the prediction purpose. 

#list(kickstarter.columns)
kickstarter3 = kickstarter.drop(columns=["project_id","name","disable_communication","deadline","state_changed_at","created_at","launched_at","currency","state_changed_at_weekday",
                                         "state_changed_at_month","state_changed_at_day","state_changed_at_yr","state_changed_at_hr","launch_to_state_change_days", "pledged","spotlight","staff_pick","currency","backers_count","usd_pledged","static_usd_rate"])


# since name_len,created_at_yr,launched_at_yr,launched_at_hr are highly correalted therefore I am removing these variables also to avoid multicollinearity issues.
# removing blurb_len and keeping only blurb_len_clean for the prediction purpose.
kickstarter3 = kickstarter3.drop(columns=["name_len","blurb_len","created_at_yr","launched_at_yr"])

##dummifying categorical variables
kickstarter3.info()

kickstarter3 = pd.get_dummies(kickstarter3, columns = ["country","category","deadline_weekday","created_at_weekday","launched_at_weekday"])
   
X_class = kickstarter3.drop(columns=["state"])
y_class = kickstarter3["state"]

X_class.info()
X_class.head()
kickstarter3.info()

#dummifying categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_class=le.fit_transform(y_class)

#standardising the variables
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X2_class = kickstarter3.drop(columns=["state"])
X2_class.info()
X2_class.iloc[:,0:15] = standardizer.fit_transform(X2_class.iloc[:,0:15])

##Feature Selection##

## Random Forest Model Feature Selection##

feature = []
depth = [3,4,6,8,10]
for i in depth:
    from sklearn.ensemble import RandomForestClassifier
    randomforest = RandomForestClassifier(random_state=0, max_depth=i)
    model_rf_class = randomforest.fit(X_class,y_class)
    feature.append(model_rf_class.feature_importances_)
model_rf_class_df = pd.DataFrame(feature,index=depth,columns=X_class.columns)

##Building Model##
# Random Forest Model#

#eliminated all the less important variables from training data

X_class_rf = X_class.drop(columns=["category_Experimental","launched_at_weekday_Tuesday",
                                   "country_US","category_Shorts","country_GB","category_Hardware",
                                   "deadline_weekday_Sunday","launched_at_weekday_Friday",
                                   "created_at_weekday_Wednesday","deadline_weekday_Friday",
                                   "category_Places","created_at_weekday_Thursday",
                                   "created_at_weekday_Tuesday","created_at_weekday_Monday",
                                  "deadline_weekday_Monday","launched_at_weekday_Thursday",
                                   "created_at_weekday_Saturday","category_Sound","deadline_weekday_Wednesday",
                                   "country_CA","deadline_weekday_Thursday","launched_at_weekday_Monday",
                                   "created_at_weekday_Sunday","country_AU","deadline_weekday_Tuesday",
                                   "country_IT","launched_at_weekday_Wednesday","category_Gadgets",
                                   "category_Wearables","launched_at_weekday_Sunday","category_Apps",
                                   "launched_at_weekday_Saturday","category_Robots","category_Flight",
                                   "category_Immersive","country_DE","country_FR","category_Spaces",
                                   "category_Blues","country_NL","country_CH","country_ES","category_Makerspaces",
                                   "country_NZ","country_HK","country_SE","country_DK","country_MX",
                                   "country_SG","country_IE","country_BE","country_NO","country_AT",
                                   "category_Thrillers","category_Academic","country_LU","category_Comedy",
                                   "category_Webseries"])

X_class_rf_copy = X_class.drop(columns=["category_Experimental","launched_at_weekday_Tuesday",
                                   "country_US","category_Shorts","country_GB","category_Hardware",
                                   "deadline_weekday_Sunday","launched_at_weekday_Friday",
                                   "created_at_weekday_Wednesday","deadline_weekday_Friday",
                                  "created_at_weekday_Tuesday","created_at_weekday_Monday",
                                  "deadline_weekday_Saturday","created_at_weekday_Friday",
                                   "deadline_weekday_Monday","launched_at_weekday_Thursday",
                                   "created_at_weekday_Saturday","category_Sound","deadline_weekday_Wednesday",
                                   "country_CA","deadline_weekday_Thursday","launched_at_weekday_Monday",
                                   "created_at_weekday_Sunday","country_AU","deadline_weekday_Tuesday",
                                   "country_IT","launched_at_weekday_Wednesday","category_Gadgets",
                                   "category_Wearables","launched_at_weekday_Sunday","category_Apps",
                                   "launched_at_weekday_Saturday","category_Robots","category_Flight",
                                   "category_Immersive","country_DE","country_FR","category_Spaces",
                                   "category_Blues","country_NL","country_CH","country_ES","category_Makerspaces",
                                   "country_NZ","country_HK","country_SE","country_DK","country_MX",
                                   "country_SG","country_IE","country_BE","country_NO","country_AT",
                                   "category_Thrillers","category_Academic","country_LU","category_Comedy",
                                   "category_Webseries"])
X2_class_rf = X_class_rf_copy
X2_class_rf.info()
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X2_class_rf.iloc[:,0:15]=standardizer.fit_transform(X2_class_rf.iloc[:,0:15])

#split the data into training and testing
from sklearn.model_selection import train_test_split
Xrf_train, Xrf_test , yrf_train , yrf_test = train_test_split(X_class_rf, y_class, test_size = 0.30, random_state =0)

#finding optimal depth and estimators
from sklearn.model_selection import cross_val_score
for i in range(1,11):
    model_ranf =RandomForestClassifier(random_state=0,max_features=i)
    scores = cross_val_score(estimator=model_ranf,X=X_class,y=y_class,cv=10)
    print(i,":",np.average(scores))
    
estimators_ = [100,200,300,400,500]
for i in estimators_:
    model_e =RandomForestClassifier(random_state=0,n_estimators=i)
    scores_e = cross_val_score(estimator=model_e,X=X_class,y=y_class,cv=10)
    print(i,":",np.average(scores))

#fitting Random Forest Model on the optimal depth and estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
rf = RandomForestClassifier(random_state=0,n_estimators=100, max_features=10)
modelRF2 = rf.fit(Xrf_train,yrf_train)
y_test_pred_rfclass = modelRF2.predict(Xrf_test)
accuracy_score(yrf_test,y_test_pred_rfclass)
#0.7286442838929027
precision_score(yrf_test,y_test_pred_rfclass)
#0.6291970802919709
recall_score(yrf_test,y_test_pred_rfclass)
# 0.5285101164929491

##Logistic Regression Model##

#split data into training and testing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Xlo_train, Xlo_test , ylo_train , ylo_test = train_test_split(X_class_rf, y_class, test_size = 0.30, random_state =0)

from sklearn import metrics
logit = LogisticRegression(random_state=0)
model_logit = logit.fit(Xlo_train,ylo_train)
y_test_pred_logit = model_logit.predict(Xlo_test)
metrics.accuracy_score(ylo_test,y_test_pred_logit)
#0.6578835529111772
precision_score(ylo_test,y_test_pred_logit)
#0.6329113924050633
recall_score(ylo_test,y_test_pred_logit)
#0.030656039239730228

## K-NN Model ##
#data split into training and testing
X2_class.head()
from sklearn.model_selection import train_test_split
XKNN_train, XKNN_test , yKNN_train , yKNN_test = train_test_split(X2_class_rf, y_class, test_size = 0.30, random_state =0)

#finding optimal k
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

for i  in range(1,11):
    KNN = KNeighborsClassifier(n_neighbors=i)
    model_KNN = KNN.fit(XKNN_train,yKNN_train)
    y_test_pred_KNN = model_KNN.predict(XKNN_test)
    print(accuracy_score(yKNN_test,y_test_pred_KNN))
    
#fitting K-NN model on the optimal k

KNN = KNeighborsClassifier(n_neighbors=7, weights="uniform")
model_KNN = KNN.fit(XKNN_train,yKNN_train)
y_test_pred_KNN = model_KNN.predict(XKNN_test)
metrics.accuracy_score(yKNN_test,y_test_pred_KNN)
#0.6572460688482787
precision_score(yKNN_test,y_test_pred_KNN)
#0.5091649694501018
recall_score(yKNN_test,y_test_pred_KNN)
#0.30656039239730226

## CART Model##

#data split into training and testing
from sklearn.model_selection import train_test_split
XCT_train, XCT_test , yCT_train , yCT_test = train_test_split(X_class_rf, y_class, test_size = 0.30, random_state =0)

#finding optimal max_depth

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
for i in range(1,11):
    model_ct = DecisionTreeClassifier(max_depth=i,random_state=0)
    scores = cross_val_score(estimator=model_ct,X=X_class,y=y_class,cv=10)
    print(i,':',np.average(scores))

#fitting the CART model on the optimal depth
from sklearn.tree import DecisionTreeClassifier
Cart = DecisionTreeClassifier(max_depth=8, random_state=0)
model_CT= Cart.fit(XCT_train,yCT_train)
y_test_pred_CT = model_CT.predict(XCT_test)    
metrics.accuracy_score(yCT_test,y_test_pred_CT)
##0.7033574160645983
metrics.precision_score(yCT_test,y_test_pred_CT)
#0.5667234525837592
metrics.recall_score(yCT_test,y_test_pred_CT)
#0.6118945432250154

##Gradient Boosting Model##
from sklearn.model_selection import train_test_split
XGB_train,XGB_test,yGB_train,yGB_test=train_test_split(X_class_rf,y_class,test_size=0.30,random_state=0)

#finding optimal min sample split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

for i in range(2,11):
    GBT = GradientBoostingClassifier(random_state=0,min_samples_split=i)
    model_GBT = GBT.fit(XGB_train,yGB_train)
    scores2 = cross_val_score(estimator=model_GBT,X=X_class,y=y_class,cv=5)
    print(i,":",np.average(scores2))
    

estimator_ = [100,200,300,400,500]
for i in estimator_:
    GBT_e = GradientBoostingClassifier(random_state=0,n_estimators=i)
    model_GBT_e = GBT_e.fit(XGB_train,yGB_train)
    scores3 = cross_val_score(estimator=model_GBT_e,X=X_class,y=y_class,cv=5)
    print(i,":",np.average(scores3))

GBT = GradientBoostingClassifier(n_estimators=200,min_samples_split=10,random_state=0)
model_GBT = GBT.fit(XGB_train,yGB_train)
y_test_pred_GBT = model_GBT.predict(XGB_test)
metrics.accuracy_score(yGB_test,y_test_pred_GBT)
#0.7411814704632385
metrics.precision_score(yGB_test,y_test_pred_GBT)
#0.6558490566037736
metrics.recall_score(yGB_test,y_test_pred_GBT)
#0.5328019619865113

##MLP Model##
#data split into training and testing
from sklearn.model_selection import train_test_split
XMLP_train,XMLP_test,yMLP_train,yMLP_test = train_test_split(X2_class_rf,y_class,test_size=0.30,random_state=0)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

#finding optimal hidden layer

from sklearn.model_selection import GridSearchCV
ANN = GridSearchCV(MLPClassifier(max_iter=1000, random_state=0, verbose=False),{"hidden_layer_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]},cv=5)
model_ANN = ANN.fit(XMLP_train,yMLP_train)
df = pd.DataFrame(ANN.cv_results_)
ANN.best_params_
ANN.best_score_

ANN = GridSearchCV(MLPClassifier(max_iter=1000, random_state=0, verbose=False),{"hidden_layer_sizes":[5]},cv=5)
model_ANN = ANN.fit(XMLP_train,yMLP_train)
y_test_pred_ANN = model_ANN.predict(XMLP_test)
metrics.accuracy_score(yMLP_test,y_test_pred_ANN)
#0.7163195920101998
metrics.precision_score(yMLP_test,y_test_pred_ANN)
#0.6149068322981367
metrics.recall_score(yMLP_test,y_test_pred_ANN)
#0.4855916615573268

# =============================================================================
# 
# =============================================================================
