#!/usr/bin/env python
# coding: utf-8

# # 프로젝트1. 손글씨 분류

# In[1]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


digits = load_digits()


# In[3]:


digits_data = digits.data
digits_label = digits.target


# In[4]:


digits.target_names


# In[5]:


print(digits.DESCR)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                   digits_label,
                                                   test_size = 0.2,
                                                   random_state = 7)


# In[7]:


X_train.shape, y_train.shape


# In[8]:


X_test.shape, y_test.shape


# ## Decision Tree 모델

# In[9]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state = 32)
print(decision_tree._estimator_type)


# In[10]:


decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[12]:


print(classification_report(y_test, y_pred))


# ## Random Forest 모델

# In[13]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state = 32)
print(random_forest._estimator_type)


# In[14]:


random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[16]:


print(classification_report(y_test, y_pred))


# ## SVM 모델

# In[17]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[18]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)


# In[19]:


# 평가 지표


# In[20]:


print(classification_report(y_test, y_pred))


# ## SGD Classifier 모델

# In[21]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[22]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)


# In[23]:


# 평가 지표


# In[24]:


print(classification_report(y_test, y_pred))


# ## Logistic Regression 모델

# In[25]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)

print(logistic_model._estimator_type)


# In[26]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)


# In[27]:


# 평가지표


# In[28]:


print(classification_report(y_test, y_pred))


# ## 모델 평가지표 선택 이유
# 
# 손글씨가 있는 사진을 보고 정답을 맞추는 경우엔, 전체 데이터 중 맞은 데이터가 많을수록 좋은 모델이라고 판단을 하여 '정확도'로 모델 평가를 진행하였다.

# ## 회고
# 
# 회고는 프로젝트 (3) 유방암 여부 진단 하단에 기재하였습니다.

# In[ ]:




