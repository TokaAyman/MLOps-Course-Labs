#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

url = "http://localhost:8000/predict"
data = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Female",
    "Age": 30,
    "Tenure": 3,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())

