#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[56]:


test_df = pd.read_csv("C:/Users/Alejandro/Downloads/testing_set.csv",low_memory=False)
train_df = pd.read_csv("C:/Users/Alejandro/Downloads/training_set.csv",low_memory=False)

train_df.shape,test_df.shape


# In[57]:


train_df.info()


# In[58]:


train_df = train_df.drop(columns=['train_idx','client_ID'])


# In[59]:


test_df.info()


# In[60]:


test_df = test_df.drop(columns=['test_idx','client_ID'])


# In[93]:


train_df.label.unique()


# In[62]:


train_df.martial_status.value_counts()


# In[63]:


plt.figure(figsize=(6,6))

sns.histplot(data=train_df,x=train_df.martial_status)


# In[64]:


plt.figure(figsize=(10,10))

sns.scatterplot(data=train_df,x="age",y='mean_spending',hue='martial_status')


# In[65]:



X = train_df.drop('label', axis=1)
y = train_df['label']

categorical_features = ['gender', 'location']
X_encoded = pd.get_dummies(X, columns=categorical_features)

label_encoder = LabelEncoder()
X_encoded['age'] = label_encoder.fit_transform(X['age'])
X_encoded['martial_status'] = label_encoder.fit_transform(X['martial_status'])

X_encoded


# In[66]:


numeric_features = ['products_count', 'monthly_count', 'mean_spending']
scaler = StandardScaler()
X_encoded[numeric_features] = scaler.fit_transform(X[numeric_features])


# In[67]:


sns.histplot(data=X_encoded,x='mean_spending')


# In[69]:


from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC())
]


param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Decision Tree': {'max_depth': [None, 5, 10]},
    'Random Forest': {'n_estimators': [100, 200, 300]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_model = None
best_f1_macro = 0


for name, model in models:

    grid_search = GridSearchCV(model, param_grid[name], scoring='f1_macro', cv=5)
    grid_search.fit(X_train, y_train)
    

    best_model_cv = grid_search.best_estimator_
    f1_macro_cv = grid_search.best_score_
    

    y_pred = best_model_cv.predict(X_test)
    f1_macro_val = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
    

    print(f"Model: {name}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"F1-Score (macro) - CV: {f1_macro_cv:.4f}")
    print(f"F1-Score (macro) - Validation: {f1_macro_val:.4f}")
    print("-------------------------------------------")
    

    if f1_macro_val > best_f1_macro:
        best_model = best_model_cv
        best_f1_macro = f1_macro_val


# In[74]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


# In[70]:


test_df.head(5)


# In[71]:


categorical_features = ['gender', 'location']
test_encoded = pd.get_dummies(test_df, columns=categorical_features)

label_encoder = LabelEncoder()
test_encoded['age'] = label_encoder.fit_transform(test_df['age'])
test_encoded['martial_status'] = label_encoder.fit_transform(test_df['martial_status'])

test_encoded


# In[76]:


y_pred = best_model.predict(test_encoded)


# In[102]:


import json
predictions_dict = {"target": {}}


for i, pred in enumerate(y_pred):
    predictions_dict["target"][str(i)] = int(pred)

predictions_json = json.dumps(predictions_dict)

with open('predictions.json', 'w') as f:
    f.write(predictions_json)


# In[ ]:




