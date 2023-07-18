## SYRIATEL CUSTOMER CHURN
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/718bce74-01a3-4c18-903d-c7a8bdb1aa9c)

### PROJECT BACKGROUND

SyriaTel is a telecommunications company facing the challenge of customer churn.
According to Forbes Advisor an article by Monique Danao, published 2nd March 2023 at 11.00am, Customer churn rate, "refers to the rate at which subscribers or customers stop transacting with your business." https://www.forbes.com/advisor/business/churn-rate/

Churn can have significant financial implications for SyriaTel, including the loss of recurring revenue, increased customer acquisition costs, and potential negative impact on the company's reputation.

To address this issue, SyriaTel have requested **CodeTribe3** researchers to build a churn prediction system that can identify customers likely to churn in the near future.

### BUSINESS PROBLEM

Syriatel, a mobile telecommunications provider, focuses on attracting new customers and improving customer retention to grow revenue. To achieve this, they prioritize long-term customer relationships over acquiring new customers. Therefore, churn prediction plays a crucial role in their strategy. The goal of this project is to develop an accurate model for predicting customer churn and identify the significant features for churn prediction. By identifying potential churners, Syriatel can take proactive measures to prevent customers from leaving.

### DATA UNDERSTANDING

For model building we used data from Churn in Telecom's dataset aquired from kaggle. This data file is available in the project repo in the folder "data".

The dataset utilised in this research project contains information about customer attributes, call usage, charges and customer service interactions with the churn column acting as our target variable.

The dataset contains 3333 rows(number of entries) and 20 columns.

The column names and their respective descriptions can be found_ (link to data understanding)_

#### EXPLORATORY DATA ANALYSIS

Within the dataset of 3,333 customers, around 14.5% (483 customers) have terminated their contract with SyriaTel, resulting in a loss of customer base. This distribution indicates an imbalance between the two binary classes. It is essential to handle this data imbalance prior to modeling to ensure accurate predictions. Failing to address this imbalance could lead to inaccurate model outcomes.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/850341cc-3230-445d-8ef3-8037327b95f5)

The features in the dataframe exhibit varying scaling and non-normal distribution. To ensure consistency and comparability, it is necessary to scale and normalize the features.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/f7936dca-26d7-4671-a49e-8fadf4fee853)

Most features in the dataset show a low correlation with each other. However, a notable exception is the perfect positive correlation between total charge and total minutes at different times. This correlation is expected, as the charge for a call is directly influenced by its duration in minutes.

When it comes to churn prediction, there is a weak positive correlation between total day minutes, total day charge, customer service calls, and churn. In contrast, the remaining features demonstrate a negligible correlation with churn.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/5a06d570-f1e1-40f3-8b00-692e7e22b31a)

### DATA PREPARATION FOR MODELLING

**Label Encoding:**

Label Encoding is used to convert label variables in the "international plan," "voice mail plan," and "churn" columns into numeric form. "Yes" and "No" in "international plan" and "voice mail plan" are encoded as 1 and 0, respectively, while "False" and "True" in "churn" are encoded as 0 and 1.

**One-Hot Encoding:**

To handle categorical variables, one-hot encoding is applied to the "State" column. This technique converts categorical variables into multiple binary columns, making them compatible with the algorithm.

**Scaling:**

Scaling is essential for accurate predictions in machine learning, especially for algorithms sensitive to feature scaling. The data is scaled to ensure consistency and comparability among features. In this case, the StandardScaler method is used, which standardizes features by subtracting the mean and dividing by the standard deviation. This scaling technique brings the features to a zero mean and unit variance.

**Data Splitting:**

The data is split into training and test sets to evaluate the model's performance on unseen data.

**Handling Class Imbalance using SMOTE:**

To address class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is employed. SMOTE oversamples the minority class by synthesizing new examples from existing ones, ensuring better model performance on the minority class without adding new information. It is only applied to the training data, not the test data, to evaluate the model's ability to generalize to unseen data.

### MODELLING

**Model Development with Lazy Predict**

Lazy Predict is a tool that simplifies the initial stages of building and evaluating machine learning models without requiring extensive coding. It provides a comprehensive overview of how different models perform on a given dataset. With Lazy Predict, we can quickly preview the performance of various models before selecting specific ones to work with. This helps in streamlining the model development process and facilitates efficient decision-making.

<img width="407" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/354149fb-080a-47e7-a813-e3c1fc44de97">

### EVALUATION
We employed various evaluation metrics to assess the performance of different models and determine the most suitable candidate for hyperparameter tuning.

The evaluation metrics used include accuracy, precision, recall, F1-score, ROC curve and AUC value, and confusion matrices. These metrics provided insights into overall performance, classification accuracy, visualization of performance, and evaluation of target variable classification.

By considering these evaluation metrics, we identified the best-performing model and guided the subsequent hyperparameter tuning process.

**RESULTS**

**Decision Tree**

The Decision Tree model achieved an accuracy of 92% with a precision of 0.93, recall of 0.92, and F1-score of 0.92. It performed well in predicting the majority class (0) but had lower performance in predicting the minority class (1).

<img width="301" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/9e964684-cf49-4ad7-b0f2-6925f04a5da0">


**Random Forest**

The Random Forest model achieved an accuracy of 93% with a precision of 0.93, recall of 0.93, and F1-score of 0.93. It showed improved performance compared to the Decision Tree model, achieving balanced precision and recall for both classes.

<img width="326" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/e7dc83ac-ebaa-452c-a5a5-aa44f410e0ad">


**KNN (K-Nearest Neighbors)**

The KNN model achieved an accuracy of 74% with a precision of 0.82, recall of 0.74, and F1-score of 0.77. It had lower performance compared to the Decision Tree and Random Forest models, particularly in predicting the minority class (1).

<img width="317" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/716851a4-76fd-4685-a1ad-beffacf64ecb">

**XGBoost**

The XGBoost model achieved the highest accuracy of 96% with a precision of 0.95, recall of 0.96, and F1-score of 0.95. It demonstrated the best overall performance among the models, with balanced precision and recall for both classes.

Overall, the XGBoost model outperformed the other models, providing the highest accuracy and F1-score, making it the recommended choice for predictive modeling.

<img width="299" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/fb3add9f-ca5e-4c10-bed5-8ba662fed371">


### CONCLUSION

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/97736dd2-4ebb-46b8-847a-250b2e126c64)

### RECOMMENDATIONS

## SYRIATEL CUSTOMER CHURN
---
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/718bce74-01a3-4c18-903d-c7a8bdb1aa9c)

### PROJECT BACKGROUND
---

>SyriaTel is a telecommunications company facing the challenge of customer churn.
>According to Forbes Advisor an article by Monique Danao, published 2nd March 2023 at 11.00am, Customer churn rate, "refers to the rate at which subscribers or customers stop transacting with your business." https://www.forbes.com/advisor/business/churn-rate/.

>Churn can have significant financial implications for SyriaTel, including the loss of recurring revenue, increased customer acquisition costs, and potential negative impact on the company's reputation.

>To address this issue, SyriaTel have requested **CodeTribe3** researchers to build a churn prediction system that can identify customers likely to churn in the near future.

### BUSINESS PROBLEM
---

> Syriatel, a mobile telecommunications provider, focuses on attracting new customers and improving customer retention to grow revenue. To achieve this, they prioritize long-term customer relationships over acquiring new customers. Therefore, churn prediction plays a crucial role in their strategy. The goal of this project is to develop an accurate model for predicting customer churn and identify the significant features for churn prediction. By identifying potential churners, Syriatel can take proactive measures to prevent customers from leaving.

### DATA UNDERSTANDING
---
> For model building we used data from Churn in Telecom's dataset aquired from kaggle. This data file is available in the project repo in the folder "data".

> The dataset utilised in this research project contains information about customer attributes, call usage, charges and customer service interactions with the churn column acting as our target variable.

> The dataset contains 3333 rows(number of entries) and 20 columns.

|Column Name |Data Type| Description
|---|---|---|
|account length|The number of days the customer has been an active customer|Int64|
|area code|The area code of the customer's phone number|Int64|
|phone number|The customer's phone number|object|
|international plan|Indicates whether the customer has an international calling plan|object|
|voice mail plan|Indicates whether the customer has a voicemail plan|object|
|number vmail messages| : Represents the number of voicemail messages the customer has|Int64|
|total day minutes|The total number of minutes the customer has used during the day|Float64|
|total day calls|The total number of calls the customer has made during the day|Int64|
|total day charge|The total charge in dollars for the day's usage|Float64|
|total eve minutes|The total number of minutes the customer has used during the evening|Float64|
|total eve calls|The total number of calls the customer has made in the evening|Int64|
|total eve charge|The total charge in dollars for the evening's usage|Float64|
|total night minutes|The total number of minutes the customer has used during the night|Float64|
|total night calls|The total number of calls the customer has made during the night|Int64|
|total night charge|The total charge in dollars for the night's usage|Float64|
|total intl minutes|The total number of international minutes the customer has used|Float64|
|total intl calls|The total number of international calls the customer has made|Int64|
|total intl charge|The total charge in dollars for the international usage|Float64|
|customer service calls|The number of customer service calls made by the customer|Int64|
|churn|Indicating whether the customer has churned or not|Bool|


#### EXPLORATORY DATA ANALYSIS
---
> Within the dataset of 3,333 customers, around 14.5% (483 customers) have terminated their contract with SyriaTel, resulting in a loss of customer base. This distribution indicates an imbalance between the two binary classes. It is essential to handle this data imbalance prior to modeling to ensure accurate predictions. Failing to address this imbalance could lead to inaccurate model outcomes.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/850341cc-3230-445d-8ef3-8037327b95f5)

> The features in the dataframe exhibit varying scaling and non-normal distribution. To ensure consistency and comparability, it is necessary to scale and normalize the features.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/f7936dca-26d7-4671-a49e-8fadf4fee853)

> Most features in the dataset show a low correlation with each other. However, a notable exception is the perfect positive correlation between total charge and total minutes at different times. This correlation is expected, as the charge for a call is directly influenced by its duration in minutes.

> When it comes to churn prediction, there is a weak positive correlation between total day minutes, total day charge, customer service calls, and churn. In contrast, the remaining features demonstrate a negligible correlation with churn.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/5a06d570-f1e1-40f3-8b00-692e7e22b31a)

### DATA PREPARATION FOR MODELLING
---
- **Label Encoding:**

>Label Encoding is used to convert label variables in the "international plan," "voice mail plan," and "churn" columns into numeric form. "Yes" and "No" in "international plan" and "voice mail plan" are encoded as 1 and 0, respectively, while "False" and "True" in "churn" are encoded as 0 and 1.


- **One-Hot Encoding:**

>To handle categorical variables, one-hot encoding is applied to the "State" column. This technique converts categorical variables into multiple binary columns, making them compatible with the algorithm.

- **Scaling:**

>Scaling is essential for accurate predictions in machine learning, especially for algorithms sensitive to feature scaling. The data is scaled to ensure consistency and comparability among features. In this case, the StandardScaler method is used, which standardizes features by subtracting the mean and dividing by the standard deviation. This scaling technique brings the features to a zero mean and unit variance.

- **Data Splitting:**

>The data is split into training and test sets to evaluate the model's performance on unseen data.

- **Handling Class Imbalance using SMOTE:**

>To address class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is employed. SMOTE oversamples the minority class by synthesizing new examples from existing ones, ensuring better model performance on the minority class without adding new information. It is only applied to the training data, not the test data, to evaluate the model's ability to generalize to unseen data.

### MODELLING
---

- **Model Development with Lazy Predict**

> Lazy Predict is a tool that simplifies the initial stages of building and evaluating machine learning models without requiring extensive coding. It provides a comprehensive overview of how different models perform on a given dataset. With Lazy Predict, we can quickly preview the performance of various models before selecting specific ones to work with. This helps in streamlining the model development process and facilitates efficient decision-making.
<img width="407" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/354149fb-080a-47e7-a813-e3c1fc44de97">

### EVALUATION
---
> We employed various evaluation metrics to assess the performance of different models and determine the most suitable candidate for hyperparameter tuning.

> The evaluation metrics used include accuracy, precision, recall, F1-score, ROC curve and AUC value, and confusion matrices. These metrics provided insights into overall performance, classification accuracy, visualization of performance, and evaluation of target variable classification.

>By considering these evaluation metrics, we identified the best-performing model and guided the subsequent hyperparameter tuning process.

**RESULTS**
---

- **Decision Tree**

>The Decision Tree model achieved an accuracy of 92% with a precision of 0.93, recall of 0.92, and F1-score of 0.92. It performed well in predicting the majority class (0) but had lower performance in predicting the minority class (1).
<img width="301" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/9e964684-cf49-4ad7-b0f2-6925f04a5da0">


- **Random Forest**

>The Random Forest model achieved an accuracy of 93% with a precision of 0.93, recall of 0.93, and F1-score of 0.93. It showed improved performance compared to the Decision Tree model, achieving balanced precision and recall for both classes.
<img width="326" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/e7dc83ac-ebaa-452c-a5a5-aa44f410e0ad">


- **KNN (K-Nearest Neighbors)**

>The KNN model achieved an accuracy of 74% with a precision of 0.82, recall of 0.74, and F1-score of 0.77. It had lower performance compared to the Decision Tree and Random Forest models, particularly in predicting the minority class (1).
<img width="317" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/716851a4-76fd-4685-a1ad-beffacf64ecb">

- **XGBoost**

>The XGBoost model achieved the highest accuracy of 96% with a precision of 0.95, recall of 0.96, and F1-score of 0.95. It demonstrated the best overall performance among the models, with balanced precision and recall for both classes.

>Overall, the XGBoost model outperformed the other models, providing the highest accuracy and F1-score, making it the recommended choice for predictive modeling.
<img width="299" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/fb3add9f-ca5e-4c10-bed5-8ba662fed371">


### CONCLUSION
---

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/97736dd2-4ebb-46b8-847a-250b2e126c64)

### RECOMMENDATIONS
---
