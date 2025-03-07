#!/usr/bin/env python
# coding: utf-8

# ### Something great happened because somebody did not have money and had some extra space, that great idea was Airbnb
# 
# 
# # Scope for this report
# 1) Introduction to our project
# 
# 2) Data Cleaning
# 
# 3) Data Exploratory Analysis
# 
# 4) Residual Plots
# 
# 5) Model Building  
#  
# 6) Synopsis and Conclusion
# 
# 
# 
# # 1. Introduction
# 
# Airbnb has been an online marketplace since 2008 to arrange or offer accommodation, mainly homes, or tourism experiences. NYC is the most populous city in the U.S. and one of the world's most popular tourism and business locations.
# 
# Data from Airbnb NYC 2019 provides operation reporting and metrics. This project focuses on the gleaning patterns and other relevant information about airbnb listing in NYC.The goals of this project are to answere questions such as:
# How do prices vary with respect to neighbourhood?
# Rental property types and rental amentities.
# The data that we being using is Air.csv , it is a detailed dataset with 56 attributes ou of which will be considering price,neighbourhood,roomtype,minimum nights and reviews per month.
# We initailly explored the data to gain some initial insights of the various features from the dataset.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


# First, data content will be examined. 

# In[3]:


nyc_data = pd.read_csv('Air.csv')


# In[4]:


import pandas as pd
import pandas_profiling
pandas_profiling.ProfileReport(nyc_data)


# In[5]:


nyc_data.info()


# In[6]:


nyc_data.head(10)


# In[7]:


nyc_data.isnull().sum()


# In[8]:


modifiedAirbnb=nyc_data.dropna()


# In[9]:


modifiedAirbnb.isnull().sum()


# Above table shows that, there are some missing data for some features. They will be detailed later. 

# # 2. Data Exploratory Analysis
# 
# The first graph is about the ' ' price '' and ' ' room type '' relationship. The price of ' ' shared room '' is always less than $2,000. On the other hand, both the ' ' private room '' and the ' ' full home '' have the highest price in that way.me. 

# In[10]:


plt.figure(figsize=(15,12))
sns.scatterplot(x='room_type', y='price', data=nyc_data)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price",size=15, weight='bold')


# The graph below has the details of'' price'' and'' room type'' based on'' neighborhood group'' are shown below. In the same area that is'' Manhattan'' is the highest price of'' Private Room'' and'' Entire Home / Apt'.' Even, in'' Private Room'' and'' Entire Home / Apt'' Brooklyn has very high prices. On the other hand, the highest price for'' shared room'' is in the Queens area.

# In[11]:


plt.figure(figsize=(20,15))
sns.scatterplot(x="room_type", y="price",
            hue="neighbourhood_group", size="neighbourhood_group",
            sizes=(50, 200), palette="Dark2", data=nyc_data)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price vs Neighbourhood Group",size=15, weight='bold')


# Another graph is about ``price`` vs ``number of reviews`` based on ``neighborhood group``. It shows us the lowest prices have higher reviews than the higher prices. It shows negative correlation between ``price`` and ``number of reviews``. Also ``Manhattan``, ``Brooklyn`` and ``Queens`` areas have higher reviews than others.

# In[12]:


plt.figure(figsize=(20,15))
sns.set_palette("Set1")

sns.lineplot(x='price', y='number_of_reviews', 
             data=nyc_data[nyc_data['neighbourhood_group']=='Brooklyn'],
             label='Brooklyn')
sns.lineplot(x='price', y='number_of_reviews', 
             data=nyc_data[nyc_data['neighbourhood_group']=='Manhattan'],
             label='Manhattan')
sns.lineplot(x='price', y='number_of_reviews', 
             data=nyc_data[nyc_data['neighbourhood_group']=='Queens'],
             label='Queens')
sns.lineplot(x='price', y='number_of_reviews', 
             data=nyc_data[nyc_data['neighbourhood_group']=='Staten Island'],
             label='Staten Island')
sns.lineplot(x='price', y='number_of_reviews', 
             data=nyc_data[nyc_data['neighbourhood_group']=='Bronx'],
             label='Bronx')
plt.xlabel("Price", size=13)
plt.ylabel("Number of Reviews", size=13)
plt.title("Price vs Number of Reviews vs Neighbourhood Group",size=15, weight='bold')


# Before examining ``price`` feature, categorical variables will be mapped with help of ``cat.code``. This will assist to make easier and comprehensible data analysis. 

# In[13]:


nyc_data['neighbourhood_group']= nyc_data['neighbourhood_group'].astype("category").cat.codes
nyc_data['neighbourhood'] = nyc_data['neighbourhood'].astype("category").cat.codes
nyc_data['room_type'] = nyc_data['room_type'].astype("category").cat.codes
nyc_data.info()


# In[14]:


plt.figure(figsize=(10,10))
sns.distplot(nyc_data['price'], fit=norm)
plt.title("Price Distribution Plot",size=15, weight='bold')


# The above distribution graph shows that there is a right-skewed distribution on ``price``. This means there is a positive skewness. Log transformation will be used to make this feature less skewed. This will help to make easier interpretation and better statistical analysis
# 
# Since division by zero is a problem, ``log+1`` transformation would be better.

# In[15]:


nyc_data['price_log'] = np.log(nyc_data.price+1)


# With help of log transformation, now, price feature have normal distribution. 

# In[16]:


plt.figure(figsize=(12,10))
sns.distplot(nyc_data['price_log'], fit=norm)
plt.title("Log-Price Distribution Plot",size=15, weight='bold')


# In below graph, the good fit indicates that normality is a reasonable approximation.

# In[17]:


plt.figure(figsize=(7,7))
stats.probplot(nyc_data['price_log'], plot=plt)
plt.show()


# Now it is time to prepare data for modeling. First, non-nominal data and old ``price`` feature will be eliminated.

# In[18]:


nyc_model = nyc_data.drop(columns=['name','id' ,'host_id','host_name', 
                                   'last_review','price'])
nyc_model.isnull().sum()


# ``Number of reviews`` feature has some missing data. For this feature, missing data will be replaced with mean. Since the data is more symmetric, mean replacement would be better. 

# In[19]:


mean = nyc_model['reviews_per_month'].mean()
nyc_model['reviews_per_month'].fillna(mean, inplace=True)
nyc_model.isnull().sum()


# Now it is time to make more details about data. A correlation table will be created and the Pearson method will be used.

# In[20]:


plt.figure(figsize=(15,12))
palette = sns.diverging_palette(20, 220, n=256)
corr=nyc_model.corr(method='pearson')
sns.heatmap(corr, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(11, 0))
plt.title("Correlation Matrix",size=15, weight='bold')


# The correlation table shows that there is no strong relationship between price and other features. This indicates no feature needed to be taken out of data. This relationship will be detailed with Residual Plots and Multicollinearity.

# ## Residual Plots
# 
# Residual Plot is strong method to detect outliers, non-linear data and detecting data for regression models. The below charts show the residual plots for each feature with the ``price``. 
# 
# An ideal Residual Plot, the red line would be horizontal. Based on the below charts, most features are non-linear. On the other hand, there are not many outliers in each feature. This result led to underfitting. Underfitting can occur when input features do not have a strong relationship to target variables or over-regularized. For avoiding underfitting new data features can be added or regularization weight could be reduced.
# 
# In this kernel, since the input feature data could not be increased, Regularized Linear Models will be used for regularization and polynomial transformation will be made to avoid underfitting. 

# In[21]:


nyc_model_x, nyc_model_y = nyc_model.iloc[:,:-1], nyc_model.iloc[:,-1]


# In[22]:


f, axes = plt.subplots(5, 2, figsize=(15, 20))
sns.residplot(nyc_model_x.iloc[:,0],nyc_model_y, lowess=True, ax=axes[0, 0], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,1],nyc_model_y, lowess=True, ax=axes[0, 1],
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,2],nyc_model_y, lowess=True, ax=axes[1, 0], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,3],nyc_model_y, lowess=True, ax=axes[1, 1], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,4],nyc_model_y, lowess=True, ax=axes[2, 0], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,5],nyc_model_y, lowess=True, ax=axes[2, 1], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,6],nyc_model_y, lowess=True, ax=axes[3, 0], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,7],nyc_model_y, lowess=True, ax=axes[3, 1], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,8],nyc_model_y, lowess=True, ax=axes[4, 0], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.residplot(nyc_model_x.iloc[:,9],nyc_model_y, lowess=True, ax=axes[4, 1], 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.setp(axes, yticks=[])
plt.tight_layout()


# ## Multicollinearity
# 
# Multicollinearity will help to measure the relationship between explanatory variables in multiple regression. If there is multicollinearity occurs, these highly related input variables should be eliminated from the model.
# 
# In this kernel, multicollinearity will be control with ``Eigen vector values`` results. 

# In[23]:


#Eigen vector of a correlation matrix.
multicollinearity, V=np.linalg.eig(corr)
multicollinearity


# None one of the eigenvalues of the correlation matrix is close to zero. It means that there is no multicollinearity exists in the data.

# ## Feature Selection and GridSearch
# 
# First, ``Standard Scaler`` technique will be used to normalize the data set. Thus, each feature has 0 mean and 1 standard deviation. 

# In[24]:


scaler = StandardScaler()
nyc_model_x = scaler.fit_transform(nyc_model_x)


# Secondly, data will be split in a 70–30 ratio

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(nyc_model_x, nyc_model_y, test_size=0.3,random_state=42)


# Now it is time to build a ``feature importance`` graph. For this ``Extra Trees Classifier`` method will be used. In the below code, ``lowess=True`` makes sure the lowest regression line is drawn.

# In[26]:


lab_enc = preprocessing.LabelEncoder()

feature_model = ExtraTreesClassifier(n_estimators=50)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))

plt.figure(figsize=(7,7))
feat_importances = pd.Series(feature_model.feature_importances_, index=nyc_model.iloc[:,:-1].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# The above graph shows the feature importance of dataset. According to that, ``neighborhood group`` and ``room type`` have the lowest importance on the model. Under this result, the model building will be made in 2 phases. In the first phase, models will be built within all features and in the second phase, models will be built without ``neighborhood group`` and ``room type`` features.  

# # 3. Model Building
# 
# ## Phase 1 - With All Features
# 
# Correlation matrix, Residual Plots and Multicollinearity results show that underfitting occurs on the model and there is no multicollinearity on the independent variables. Avoiding underfitting will be made with ``Polynomial Transformation`` since no new features can not be added or replaced with the existing ones.  
# 
# In model building section, `Linear Regression`, `Ridge Regression`, `Lasso Regression`, and `ElasticNet Regression` models will be built. These models will be used to avoiding plain ``Linear Regression`` and show the results with a little of regularization. 
# 
# First, `GridSearchCV` algorithm will be used to find the best parameters and tuning hyperparameters for each model. In this algorithm ``5-Fold Cross Validation`` and ``Mean Squared Error Regression Loss`` metrics will be used. 

# In[27]:


### Linear Regression ###

def linear_reg(input_x, input_y, cv=5):
    ## Defining parameters
    model_LR= LinearRegression()

    parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_LR = GridSearchCV(estimator=model_LR,  
                         param_grid=parameters,
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_LR.fit(input_x, input_y)
    best_parameters_LR = grid_search_LR.best_params_  
    best_score_LR = grid_search_LR.best_score_ 
    print(best_parameters_LR)
    print(best_score_LR)



# In[28]:


### Ridge Regression ###

def ridge_reg(input_x, input_y, cv=5):
    ## Defining parameters
    model_Ridge= Ridge()

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_Ridge = GridSearchCV(estimator=model_Ridge,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_Ridge.fit(input_x, input_y)
    best_parameters_Ridge = grid_search_Ridge.best_params_  
    best_score_Ridge = grid_search_Ridge.best_score_ 
    print(best_parameters_Ridge)
    print(best_score_Ridge)
    
# ridge_reg(nyc_model_x, nyc_model_y)


# In[29]:


### Lasso Regression ###

def lasso_reg(input_x, input_y, cv=5):
    ## Defining parameters
    model_Lasso= Lasso()

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_lasso = GridSearchCV(estimator=model_Lasso,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_lasso.fit(input_x, input_y)
    best_parameters_lasso = grid_search_lasso.best_params_  
    best_score_lasso = grid_search_lasso.best_score_ 
    print(best_parameters_lasso)
    print(best_score_lasso)

# lasso_reg(nyc_model_x, nyc_model_y)


# In[30]:


### ElasticNet Regression ###

def elastic_reg(input_x, input_y,cv=5):
    ## Defining parameters
    model_grid_Elastic= ElasticNet()

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_elastic = GridSearchCV(estimator=model_grid_Elastic,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_elastic.fit(input_x, input_y)
    best_parameters_elastic = grid_search_elastic.best_params_  
    best_score_elastic = grid_search_elastic.best_score_ 
    print(best_parameters_elastic)
    print(best_score_elastic)

# elastic_reg(nyc_model_x, nyc_model_y)


# ### Polynomial Transformation
# The polynomial transformation will be made with a second degree which adding the square of each feature.

# In[31]:


Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)


# ### Model Prediction

# In[32]:


##Linear Regression
lr = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
lr.fit(X_train, y_train)
lr_pred= lr.predict(X_test)

#Ridge Model
ridge_model = Ridge(alpha = 0.01, normalize = True)
ridge_model.fit(X_train, y_train)             
pred_ridge = ridge_model.predict(X_test) 

#Lasso Model
Lasso_model = Lasso(alpha = 0.001, normalize =False)
Lasso_model.fit(X_train, y_train)
pred_Lasso = Lasso_model.predict(X_test) 

#ElasticNet Model
model_enet = ElasticNet(alpha = 0.01, normalize=False)
model_enet.fit(X_train, y_train) 
pred_test_enet= model_enet.predict(X_test)


# ## Phase 2 - Without All Features
# 
# All steps from Phase 1, will be repeated in this Phase. The difference is, ``neighbourhood_group`` and ``room_type`` features will be eliminated.

# In[33]:


nyc_model_xx= nyc_model.drop(columns=['neighbourhood_group', 'room_type'])


# In[34]:


nyc_model_xx, nyc_model_yx = nyc_model_xx.iloc[:,:-1], nyc_model_xx.iloc[:,-1]
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(nyc_model_xx, nyc_model_yx, test_size=0.3,random_state=42)


# In[35]:


scaler = StandardScaler()
nyc_model_xx = scaler.fit_transform(nyc_model_xx)


# In[36]:


### Linear Regression ###
# linear_reg(nyc_model_xx, nyc_model_yx, cv=4)


# In[37]:


### Ridge Regression ###
# ridge_reg(nyc_model_xx, nyc_model_yx, cv=4)


# In[38]:


### Lasso Regression ###
# lasso_reg(nyc_model_xx, nyc_model_yx, cv=4)


# In[39]:


### ElasticNet Regression ###
# elastic_reg(nyc_model_xx, nyc_model_yx, cv=4)


# ### K-Fold Cross Validation

# In[40]:


kfold_cv=KFold(n_splits=4, random_state=42, shuffle=False)
for train_index, test_index in kfold_cv.split(nyc_model_xx,nyc_model_yx):
    X_train_x, X_test_x = nyc_model_xx[train_index], nyc_model_xx[test_index]
    y_train_x, y_test_x = nyc_model_yx[train_index], nyc_model_yx[test_index]


# ### Polynomial Transformation

# In[41]:


Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_x = Poly.fit_transform(X_train_x)
X_test_x = Poly.fit_transform(X_test_x)


# ### Model Prediction

# In[42]:


###Linear Regression
lr_x = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
lr_x.fit(X_train_x, y_train_x)
lr_pred_x= lr_x.predict(X_test_x)

###Ridge
ridge_x = Ridge(alpha = 0.01, normalize = True)
ridge_x.fit(X_train_x, y_train_x)           
pred_ridge_x = ridge_x.predict(X_test_x) 

###Lasso
Lasso_x = Lasso(alpha = 0.001, normalize =False)
Lasso_x.fit(X_train_x, y_train_x)
pred_Lasso_x = Lasso_x.predict(X_test_x) 

##ElasticNet
model_enet_x = ElasticNet(alpha = 0.01, normalize=False)
model_enet_x.fit(X_train_x, y_train_x) 
pred_train_enet_x= model_enet_x.predict(X_train_x)
pred_test_enet_x= model_enet_x.predict(X_test_x)


# # 4. Model Comparison
# 
# In this part, 3 metrics will be calculated for evaluating predictions.
# 
# * ``Mean Absolute Error (MAE)``    shows the difference between predictions and actual values.
# 
# * ``Root Mean Square Error (RMSE)`` shows how accurately the model predicts the response.
# 
# *                   ``R^2``  will be calculated to find the goodness of fit measure.

# In[43]:


print('-------------Lineer Regression-----------')

print('--Phase-1--')
print('MAE: %f'% mean_absolute_error(y_test, lr_pred))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, lr_pred)))   
print('R2 %f' % r2_score(y_test, lr_pred))

print('--Phase-2--')
print('MAE: %f'% mean_absolute_error(y_test_x, lr_pred_x))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, lr_pred_x)))   
print('R2 %f' % r2_score(y_test_x, lr_pred_x))

print('---------------Ridge ---------------------')

print('--Phase-1--')
print('MAE: %f'% mean_absolute_error(y_test, pred_ridge))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred_ridge)))   
print('R2 %f' % r2_score(y_test, pred_ridge))

print('--Phase-2--')
print('MAE: %f'% mean_absolute_error(y_test_x, pred_ridge_x))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, pred_ridge_x)))   
print('R2 %f' % r2_score(y_test_x, pred_ridge_x))

print('---------------Lasso-----------------------')

print('--Phase-1--')
print('MAE: %f' % mean_absolute_error(y_test, pred_Lasso))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test, pred_Lasso)))
print('R2 %f' % r2_score(y_test, pred_Lasso))

print('--Phase-2--')
print('MAE: %f' % mean_absolute_error(y_test_x, pred_Lasso_x))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x, pred_Lasso_x)))
print('R2 %f' % r2_score(y_test_x, pred_Lasso_x))

print('---------------ElasticNet-------------------')

print('--Phase-1 --')
print('MAE: %f' % mean_absolute_error(y_test,pred_test_enet)) #RMSE
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_test_enet))) #RMSE
print('R2 %f' % r2_score(y_test, pred_test_enet))

print('--Phase-2--')
print('MAE: %f' % mean_absolute_error(y_test_x,pred_test_enet_x)) #RMSE
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x,pred_test_enet_x))) #RMSE
print('R2 %f' % r2_score(y_test_x, pred_test_enet_x))


# The results show that all models have similar prediction results. Phase 1 and 2 have a great difference for each metric. All metric values are increased in Phase 2 it means, the prediction error value is higher in that Phase and model explainability are very low the variability of the response data around mean.
# 
# * The MAE value of 0 indicates no error on the model. In other words, there is a perfect prediction. The above results show that all predictions have great error especially in phase 2. 
# * RMSE gives an idea of how much error the system typically makes in its predictions. The above results show that all models with each phase have significant errors.
# * R2 represents the proportion of the variance for a dependent variable that's explained by an independent variable. The above results show that, in phase 1, 52% of data fit the regression model while in phase 2, 20% of data fit the regression model. 

# In[44]:


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(30, 20))
fig.suptitle('True Values vs Predictions')

ax1.scatter(y_test, lr_pred)
ax1.set_title('Linear Regression - Phase-1')

ax2.scatter(y_test, pred_ridge)
ax2.set_title('Ridge - Phase-1')

ax3.scatter(y_test, pred_Lasso)
ax3.set_title('Lasso - Phase-1')

ax4.scatter(y_test, pred_test_enet)
ax4.set_title('ElasticNet - Phase-1')

ax5.scatter(y_test_x, lr_pred_x)
ax5.set_title('Linear Regression - Phase-2')

ax6.scatter(y_test_x, pred_ridge_x)
ax6.set_title('Ridge - Phase-2')

ax7.scatter(y_test_x, pred_Lasso_x)
ax7.set_title('Lasso - Phase-2')

ax8.scatter(y_test_x, pred_test_enet_x)
ax8.set_title('ElasticNet - Phase-2')

for ax in fig.get_axes():
    ax.set(xlabel='True Values', ylabel='Predictions')


# The last graph is about the difference between True Values vs Prediction for Phase 1 and Phase 2. The great difference between the two phases has been seen in 'Linear Regression' and 'ElasticNet Regression' models. 

# # 5. Synopsis and Conclusion
# 
# In conclusion, we began our project with important data cleaning methods, making the data fit for work. Relationships between various attributes have then been found and graphs have been plotted for important relations like room type, price and neighborhood group. Wherever needed, transformations have been made for making data more efficient followed by a correlation matrix to exemplify the relationship between attributes. Residual plots and multicollinearity have been plotted and calculated, followed by feature selection to show feature importance of each attribute on a relative scale which will help with the model building process. Correlation matrix, Residual Plots and Multicollinearity results show that underfitting occurs on the model and there is no multicollinearity on the independent variables. Avoiding underfitting will be made with Polynomial Transformations since no new features can be added or replaced with the existing ones.
# 
# The model building has been completed in two phases, first phase including all features and attributes. In model building section: Linear Regression, Ridge Regression, Lasso Regression and ElasticNet Regression have been used. GridSearchCV algorithm will be used to find the best parameters for each model. The second phase of the model building process uses the main attributes, excluding the ones which are least important according to our feature selection models.
# 
# Finally, model comparisons have been made to compare the models: using evaluators like Mean Absolute Error, Root Mean Square Error and R^2 for each of the features. For each metric, the graphs for the built models are shown and the difference between the models in each phase is highlighted.
# 
# On using each of the various methods and features as explained above, we have tried to make predictions with different prediction models and compared the important metrics on evaluation scales. A number of models have been used to avoid plain linear regression analysis and show the results with a little of regularization. With the models being built, we consider our goal to be achieved.

# # References:

# www.w3schools.com
# 
# www.analytics.com
# 
# www.github.com
# 
# www.stackoverflow.com

# # THANKYOU FOR READING 
