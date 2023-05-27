#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm


# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[3]:


boston_df.head()


# ## Task 4: Generate Descriptive Statistics and Visualizations

# #### 1. For the "Median value of owner-occupied homes" provide a boxplot

# In[69]:


ax = sns.boxplot(y='MEDV', data=boston_df)
ax.set_title('Median value of owner-occupied homes')
plt.show()


# #### 2. Provide a  bar plot for the Charles river variable

# In[21]:


ax2 = sns.countplot(x = 'CHAS', data = boston_df)
ax2.set_title('Number of homes near the Charles River')


# Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)

# In[29]:


boston_df.loc[(boston_df['AGE'] <= 35), 'age_group'] = ' 35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35)&(boston_df['AGE'] < 70), 'age_group'] = 'betwwen 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'age_group'] = '70 years and older'


# In[31]:


ax3 = sns.boxplot(x='MEDV', y ='age_group', data=boston_df)
plt.show()


# Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?

# In[37]:


ax4 = sns.scatterplot(x='INDUS', y= 'NOX', data=boston_df)
ax4.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')
plt.show()


# Create a histogram for the pupil to teacher ratio variable

# In[49]:


ax5 = sns.countplot(x = 'PTRATIO', data = boston_df)
ax5.set_title('Pupil to teacher ratio per town')


# Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

# In[53]:


boston_df.loc[(boston_df['CHAS'] == 1), 'chas_distant'] = 'near'
boston_df.loc[(boston_df['CHAS'] == 0), 'chas_distant'] = 'far'
boston_df.head()


# In[55]:


#levane test
scipy.stats.levene(boston_df[boston_df['chas_distant'] == 'near']['MEDV'],
                   boston_df[boston_df['chas_distant'] == 'far']['MEDV'], center='mean')


# In[59]:


#ttest_ind
scipy.stats.ttest_ind(boston_df[boston_df['chas_distant'] == 'near']['MEDV'],
                   boston_df[boston_df['chas_distant'] == 'far']['MEDV'], equal_var = False)


# Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

# In[64]:


thity_five_lower = boston_df[boston_df['age_group'] == ' 35 years and younger']['MEDV']
seventy = boston_df[boston_df['age_group'] == 'betwwen 35 and 70 years']['MEDV']
seventy_older = boston_df[boston_df['age_group'] == '70 years and older']['MEDV']


# In[66]:


X = pd.get_dummies(boston_df[['age_group']])
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[63]:


from statsmodels.formula.api import ols
lm = ols('MEDV ~ AGE', data = boston_df).fit()
table = sm.stats.anova_lm(lm)
print(table)


# Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

# In[67]:


scipy.stats.pearsonr(boston_df['INDUS'], boston_df['NOX'])


# What is the impact of an additional weighted distance  to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

# In[68]:


## insert code here
## X is the input variables (or independent variables)
X = boston_df['DIS']
## y is the target/dependent variable
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# In[ ]:




