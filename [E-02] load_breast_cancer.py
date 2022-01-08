#!/usr/bin/env python
# coding: utf-8

# # 프로젝트3. 유방암 여부

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


breast_cancer = load_breast_cancer()


# In[3]:


breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target


# In[4]:


breast_cancer.target_names


# In[5]:


print(breast_cancer.DESCR)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data,
                                                   breast_cancer_label,
                                                   test_size = 0.2,
                                                   random_state = 7)


# In[7]:


X_train.shape, y_train.shape


# In[8]:


X_test.shape, y_test.shape


# ### 1) Decision Tree 모델

# In[9]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state = 32)
print(decision_tree._estimator_type)


# In[10]:


decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)


# In[11]:


from sklearn.metrics import recall_score

recall_score(y_test, y_pred)


# In[12]:


print(classification_report(y_test, y_pred))


# ### 2) Random Forest 모델

# In[13]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state = 32)
print(random_forest._estimator_type)


# In[14]:


random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)


# In[15]:


from sklearn.metrics import recall_score

recall_score(y_test, y_pred)


# In[16]:


print(classification_report(y_test, y_pred))


# ### 3) SVM 모델

# In[17]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[18]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)


# In[19]:


from sklearn.metrics import recall_score

recall_score(y_test, y_pred)


# In[20]:


print(classification_report(y_test, y_pred))


# ### 4) SGD Classifier 모델

# In[21]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[22]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)


# In[23]:


from sklearn.metrics import recall_score

recall_score(y_test, y_pred)


# In[24]:


print(classification_report(y_test, y_pred))


# ### 5) Logistic Regression 모델

# In[25]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)

print(logistic_model._estimator_type)


# In[26]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)


# In[27]:


from sklearn.metrics import recall_score

recall_score(y_test, y_pred)


# In[28]:


print(classification_report(y_test, y_pred))


# ## 모델 평가지표 선택 이유
# 유방암과 같은 암의 양성, 음성 여부를 오진하게 될 경우 매우 위험하다.
# 그 중 음성인 경우에 양성으로 판단하는 경우보다 양성인데 음성으로 오진하는 경우가 더 위험하다. 그렇기에 '재현율(Recall)'이 높을수록 해당 경우가 적다는 뜻이므로 '재현율'로 모델 평가를 진행하였다

# ## <회고>
# 
# 이번 프로젝트를 진행하면서 가장 어려웠던 점은 모델학습에서 다양한 모델들을 공부하는 것이었다. 그리고 학습시키는 모델에 따라 정확도 등 수치가 달라져 여러 모델을 학습시킨 후 추가로 평가하는 것이 중요하다고 깨달았다.
# 프로젝트를 하며 가장 고민이 많았던 부분은 평가문항 3번, 모델의 평가 지표 선택이다. 각 프로젝트 별 분류할 때 중요시 여기는 부분을 잘 고려해서, 그 부분을 잘 커버한 모델인지 평가하는 것이 제일 중요하다고 생각했기 때문이다. 그래서 각 프로젝트의 특성을 위주로 생각을 해보았지만 아직도 명확히 알지 못해 평가 지표를 선택한 것에 대한 확신이 없다. 그래서 루브릭 평가 지표를 달성하지 못하였을 때의 이유가 이것이 되지 않을까 싶다.
# 이번 프로젝트, 프로젝트 전 LMS를 통해 '사이킷런'에 대해 처음 알게 되었다. 개념 공부부터 프로젝트까지 해보니 사이킷런의 활용도에 대해 알게 되어서 좋았다. 그리고 앞으로도 계속 사용할 것이기에 현재 이해가 되지 않고 부족한 부분들을 찾아보며 추가적으로 공부를 할 것이다.
