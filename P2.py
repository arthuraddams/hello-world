
# coding: utf-8

# # 泰坦尼克数据分析
# ### 介绍背景及提出问题 --> 数据处理 --> 数据分析 —> 结论和限制

# ## 一、该数据集来自 Kaggle，包括泰坦尼克号上 2224 名乘客和船员中 891 名的人口学数据和乘客基本信息。提出问题：影响乘客生还率的因素有哪些？通过分析可以得出什么结论？结论是否可信？

# ## 二、数据处理

# ### 1.使用的是python3版本，首先引入工具包，导入并查看数据

# In[38]:


import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv("E:\\UDA\\titanic.csv",engine="python",encoding = "gbk")
data.head(10)


# ### 2.查看数据的完整程度，可知Age、Cabin、Embarked三个数据都有缺失

# In[28]:


data.info()


# ### 3.年龄字段缺失177个，使用年龄中位数填充年龄缺失值,用到fillna函数

# In[53]:


age_med1 = data.Age.median()
data.Age.fillna(age_med1, inplace=True)
data.Age.describe()


# ### 4.上船点缺失两个值，使用出现频率最高的值填充上船点缺失值

# In[77]:


#查看原始数据中上船点的分布
data.groupby('Embarked')['PassengerId'].count()


# In[81]:


#用出现频率最高的值填充缺失值
data.Embarked.fillna('S', inplace=True)
data.groupby('Embarked')['PassengerId'].count()


# In[6]:


#统计总体生还率
survived_num = data.Survived.sum()
no_survived_num = 891 - survived_num
print("共计891人，生还 %d 人，未生还%d 人。" % (survived_num,no_survived_num))
print("生还率: ",survived_num/891)


# ## 三、数据分析

# ### 1.舱位与生还率的关系

# In[15]:


#动态设置matplotlibrc，解决中文支持问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# In[26]:


#统计不同舱位乘客数量
data.groupby('Pclass')['PassengerId'].count()


# In[94]:


#绘图
pclass.plot(kind = 'bar')
plt.title('不同舱位的乘客数量分布')
plt.xlabel('舱位')
plt.ylabel('人数')
plt.show()


# In[96]:


#查看不同舱位乘客占比
(data.groupby('Pclass').agg('size')/len(data)).sort_values(ascending = True)


# In[60]:


#统计不同舱位乘客的生还数量
data.groupby('Pclass')['Survived'].sum()


# In[52]:


#统计不同舱位乘客的生还率
print(data[['Pclass','Survived']].groupby('Pclass').mean())
survived_by_pclass = data.groupby('Pclass')['Survived'].mean()
survived_by_pclass.plot(kind = 'bar')
plt.title('不同舱位的乘客生还率对比')
plt.xlabel('舱位')
plt.ylabel('生还率')
plt.show()


# ### 分析可知,三等舱乘客数量最多，其次是一等舱，二等舱最少；生还率方面一等舱生还率最高，二等舱次之，三等舱最低。

# ### 2.性别与生还率的关系

# In[57]:


# 统计不同性别的乘客数量
data.groupby('Sex')['PassengerId'].count()


# In[59]:


# 统计不同性别的乘客生还数量
data.groupby('Sex')['Survived'].sum()


# In[98]:


#统计不同性别乘客生还率
print(data[['Sex','Survived']].groupby('Sex').mean())
survived_by_gender = data.groupby('Sex')['Survived'].mean()
survived_by_gender.plot(kind = 'bar')
plt.show()


# ### 分析可知,女性生还率远高于男性。

# ### 3.上船点与生还率的关系

# In[111]:


#查看不同上船点的上船的乘客数量
print('不同上船点的上船的乘客数量:')
print(data.groupby('Embarked')['PassengerId'].count())
print('-------------')
#查看不同上船点的上船的乘客生还数量
print('不同上船点的上船的乘客生还数量:')
print(data.groupby('Embarked')['Survived'].sum())
print('-------------')
#查看上船点占比
print('不同上船点上船乘客的比例:')
print((data.groupby('Embarked').agg('size')/len(data)).sort_values(ascending = True))


# In[112]:


Embarked = data.groupby('Embarked')['PassengerId'].count()
Embarked.plot(kind = 'pie')
plt.title('不同上船点上船的乘客分布')
plt.show()


# In[110]:


#统计不同上船点乘客的生还率
print(data[['Embarked','Survived']].groupby('Embarked').mean())
survived_by_Embarked = data.groupby('Embarked')['Survived'].mean()
survived_by_Embarked.plot(kind = 'bar')
plt.show()


# ### 分析可知，S上船点上船的乘客数量和生还的乘客数量都是最多的，但是生还率最高的C登船点。

# ### 4 .年龄与生还率的关系

# In[75]:


#统计不同年龄段的乘客数量
bins = [0,10,20,30,40,50,60,70,100]
data['Age_group'] = pd.cut(data['Age'], bins)
age = data.groupby('Age_group')['PassengerId'].count()
plt.title('不同年龄段乘客数量分布')
plt.xlabel('年龄段')
plt.ylabel('乘客数量')
age.plot(kind = "bar")
plt.show()


# In[74]:


#统计不同年龄段乘客的生还数量
bins = [0, 10, 20,30,40,50,60,70, 100]
data['Age_group'] = pd.cut(data['Age'], bins)
age = data.groupby('Age_group')['Survived'].sum()
plt.title('不同年龄段乘客的生还率分布')
plt.xlabel('年龄段')
plt.ylabel('生还数量')
age.plot(kind = "bar")
plt.show()


# In[55]:


#划分年龄组进行统计生还率
bins = [0, 10, 20,30,40,50,60,70, 100]
data['Age_group'] = pd.cut(data['Age'], bins)
age = data.groupby('Age_group')['Survived'].mean()
plt.title('不同年龄段乘客的生还率分布')
plt.xlabel('年龄段')
plt.ylabel('生还率')
age.plot(kind = "bar")
plt.show()


# ### 分析可知,年龄在20~30岁之间的乘客数量最多，生还数量也是最多的，而生还率最高的年龄段是0~10岁，即儿童生还率最高。

# ### 5.综合考虑性别和舱位考虑生还率

# In[120]:


#不同舱位乘客的性别分布
data.pivot_table(values='Embarked',index=['Pclass'],columns='Sex',aggfunc=np.size)


# In[115]:


#不同舱位生还乘客的性别分布
data.pivot_table(values='Survived',index=['Pclass'],columns='Sex',aggfunc=np.sum)


# In[113]:


#不同舱位生还乘客的生还率
data.pivot_table(values='Survived',index=['Pclass'],columns='Sex',aggfunc=np.mean)


# ### 分析可知,在任意舱位女性的生还率都高于男性。

# ## 四、综合以上得出结论，泰坦尼克号上生还率最高的是女性和儿童;其中舱位等级越高，生还的可能性越大;登船地点对生还率的影响不明显。限制：1.部分数据因为缺失值过多，未进行分析；2.填充年龄时采用的是中位数，有可能导致数据失真，影响分析结果。
