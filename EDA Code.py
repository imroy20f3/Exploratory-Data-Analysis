#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


df = pd.read_csv('TitanicDataset2.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.sample(5)


# In[7]:


df.info()


# In[10]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# In[14]:


df.describe()


# In[36]:


sns.countplot(x='Survived', data=df)
plt.show()
#df['Survived'].value_counts().plot(kind ='bar')


# In[37]:


sns.countplot(x='Pclass', data=df)
plt.show()


# In[38]:


sns.countplot(x='Sex', data=df)
plt.show()


# In[39]:


sns.countplot(x='Embarked', data=df)
plt.show()


# In[41]:


df['Survived'].value_counts().plot(kind ='pie',autopct ='%.2f')


# In[42]:


df['Embarked'].value_counts().plot(kind ='pie',autopct ='%.2f')


# In[43]:


df['Pclass'].value_counts().plot(kind ='pie',autopct ='%.2f')


# In[44]:


df['Sex'].value_counts().plot(kind ='pie',autopct ='%.2f')


# In[52]:


plt.hist(df['Age']) 
#plt.hist(df['Age'],bins=25)


# In[53]:


sns.distplot(df['Age'])


# In[54]:


sns.boxplot(df['Fare'])


# In[56]:


sns.boxplot(df['Age'])


# In[58]:


sns.boxplot(x='Fare', data=df)


# In[59]:


sns.boxplot(x='Age', data=df)


# In[60]:


df['Age'].min()


# In[61]:


df['Age'].max()


# In[62]:


df['Fare'].min()


# In[63]:


df['Fare'].max()


# In[64]:


df['Fare'].mean()


# In[65]:


df['Age'].mean()


# In[66]:


df['Age'].median()


# In[67]:


df['Age'].std()


# In[68]:


df['Fare'].median()


# In[69]:


df['Fare'].std()


# In[71]:


flights = sns.load_dataset('flights')


# In[72]:


iris = sns.load_dataset('iris')


# In[73]:


tips = sns.load_dataset('tips')


# In[76]:


sns.scatterplot(x='total_bill', y='tip', data=tips)


# In[81]:


sns.scatterplot(x='total_bill', y='tip', data=tips)


# In[82]:


sns.scatterplot(x='total_bill', y='tip',hue = 'sex', data=tips)


# In[85]:


sns.scatterplot(x='total_bill', y='tip',style = 'smoker', data=tips)


# In[86]:


sns.scatterplot(x='total_bill', y='tip',hue = 'sex',size = 'size', data=tips)


# In[87]:


sns.barplot(x='Pclass', y='Age', data = df)


# In[91]:


sns.barplot(x='Pclass', y='Fare',hue ='Sex', data = df)


# In[92]:


sns.barplot(x='Pclass', y='Age',hue ='Sex', data = df)


# In[96]:


sns.boxplot(x='Sex',y='Age',hue='Survived', data=df)


# In[104]:


sns.distplot(df[df['Survived']==0]['Age'],hist = False)
sns.distplot(df[df['Survived']==1]['Age'],hist = False)


# In[110]:


sns.kdeplot(df[df['Survived']==0]['Age'])
sns.kdeplot(df[df['Survived']==1]['Age'])


# In[109]:


sns.kdeplot(df[df['Survived'] == 0]['Age'], label='Not Survived')
sns.kdeplot(df[df['Survived'] == 1]['Age'], label='Survived')
plt.legend()


# In[112]:


sns.heatmap(pd.crosstab(df['Pclass'],df['Survived']))


# In[115]:


df.groupby('Pclass')[['Survived']].mean() * 100


# In[117]:


df.groupby('Embarked')[['Survived']].mean() * 100


# In[119]:


df.groupby('Age')[['Survived']].mean() * 100


# In[116]:


(df.groupby('Pclass')[['Survived']].mean() * 100).plot(kind='bar')


# In[121]:


(df.groupby('Embarked')[['Survived']].mean() * 100).plot(kind='bar')


# In[123]:


pd.crosstab(df['SibSp'],df['Survived'])


# In[127]:


sns.clustermap(pd.crosstab(df['Parch'],df['Survived']))


# In[128]:


iris.head()


# In[131]:


sns.pairplot(iris,hue='species')


# In[130]:


sns.pairplot(df)


# In[133]:


flights.head()


# In[141]:


new=flights.groupby('year')['passengers'].sum().reset_index()


# In[143]:


sns.lineplot(x='year',y = 'passengers',data=new)


# In[144]:


flights.pivot_table(values='passengers',index='month',columns='year')


# In[145]:


sns.heatmap(flights.pivot_table(values='passengers',index='month',columns='year'))


# In[146]:


sns.clustermap(flights.pivot_table(values='passengers',index='month',columns='year'))


# In[ ]:




