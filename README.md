
## SYRIATEL CUSTOMER CHURN
---
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/718bce74-01a3-4c18-903d-c7a8bdb1aa9c)

### PROJECT BACKGROUND
---
>SyriaTel is a telecommunications company facing the challenge of customer churn.
>According to Forbes Advisor an article by Monique Danao, published 2nd March 2023 at 11.00am, Customer churn rate, "refers to the rate at which subscribers or customers stop transacting with your business." https://www.forbes.com/advisor/business/churn-rate/.

>Churn has significant financial implications for SyriaTel, including the loss of recurring revenue, increased customer acquisition costs, and potential negative impact on the company's reputation.

>To address this issue, SyriaTel requested **CodeTribe3** researchers to build a churn prediction system that can identify customers likely to churn in the near future.

### BUSINESS PROBLEM
---
> Syriatel, a mobile telecommunications provider, focuses on attracting new customers and improving customer retention to grow revenue. To achieve this, they prioritize long-term customer relationships over acquiring new customers. Therefore, churn prediction plays a crucial role in their strategy. The goal of this project is to develop an accurate model for predicting customer churn and identify the significant features for churn prediction. By identifying potential churners, Syriatel can take proactive measures to prevent customers from leaving.
>

### LIBRARIES USED
>To conduct above mentioned analysis and predictions we used Python language and employed various libraries as shown below;

|Library|
|---|
|numpy as np
pandas as pd
scipy.stats as stats 
seaborn as sns 
statsmodels.api as sm 
matplotlib.pyplot as plt
sklearn.model_selection import train_test_split 
warnings 
sklearn.model_selection import train_test_split
sklearn.preprocessing import StandardScaler 
sklearn.preprocessing import LabelEncoder 
sklearn.preprocessing import OneHotEncoder
sklearn.linear_model import LogisticRegression
sklearn.tree import DecisionTreeClassifier 
sklearn.neighbors import KNeighborsClassifier

### DATA UNDERSTANDING
---
> For model building **CodeTribe3** used data from Churn in Telecom's dataset aquired from kaggle. This data file is available in the project repo.
> The dataset utilised in this research project contains information about customer attributes, call usage, charges and customer service interactions with the churn column acting as our target variable.
> The dataset contains 3333 rows(number of entries) and 20 columns.

|Column Name |Data Type| Description
|---|---|---|
|Account length|The number of days the customer has been an active customer|Int64|
|Area code|The area code of the customer's phone number|Int64|
|Phone number|The customer's phone number|object|
|International plan|Indicates whether the customer has an international calling plan|object|
|Voice mail plan|Indicates whether the customer has a voicemail plan|object|
|Number vmail messages|  Represents the number of voicemail messages the customer has|Int64|
|Total day minutes|The total number of minutes the customer has used during the day|Float64|
|Total day calls|The total number of calls the customer has made during the day|Int64|
|Total day charge|The total charge in dollars for the day's usage|Float64|
|Total eve minutes|The total number of minutes the customer has used during the evening|Float64|
|Total eve calls|The total number of calls the customer has made in the evening|Int64|
|Total eve charge|The total charge in dollars for the evening's usage|Float64|
|Total night minutes|The total number of minutes the customer has used during the night|Float64|
|Total night calls|The total number of calls the customer has made during the night|Int64|
|Total night charge|The total charge in dollars for the night's usage|Float64|
|Total intl minutes|The total number of international minutes the customer has used|Float64|
|Total intl calls|The total number of international calls the customer has made|Int64|
|Total intl charge|The total charge in dollars for the international usage|Float64|
|Customer service calls|The number of customer service calls made by the customer|Int64|
|Churn|Indicating whether the customer has churned or not|Bool|

#### DATA CLEANING

> The initial step in data analysis and modeling involves cleaning and preparing the dataset to ensure its quality and reliability. The following steps were taken:

 - Dropping Unnecessary Columns: We removed irrelevant columns, such as "phone number," that did not contribute to the analysis and modeling process.

- Handling Missing Values: There were no missing values (NANs) in the dataset.

- Checking for Duplicate Entries: We checked for duplicate entries in the unique identifier column (id) to eliminate any duplicated data that may affect the analysis.

- Dealing with Outliers: Outliers are extreme values that deviate significantly from the majority of the data and can impact the analysis. To mitigate their influence, we employed the Z-score technique to identify and handle outliers, ensuring their minimal impact on the results.

>By performing these steps, we ensured that the data is clean, reliable, and ready for subsequent analysis and modeling tasks.

#### EXPLORATORY DATA ANALYSIS
---
> Within the cleaned dataset of, around 13.9% of the customers had terminated their contract with SyriaTel, resulting in a loss of customer base. This distribution indicated an imbalance between the two binary classes. It was essential to handle this data imbalance prior to modeling to ensure accurate predictions. Failing to address this imbalance could lead to inaccurate model outcomes.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/04c5b50f-8055-4f46-b7a0-ddec2d898eff)

> The features in the dataframe exhibited varying scaling and non-normal distribution. To ensure consistency and comparability, it was necessary to scale and normalize the features.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/f7936dca-26d7-4671-a49e-8fadf4fee853)

> Most features in the dataset showed a low correlation with each other. However, a notable exception is the perfect positive correlation between total charge and total minutes at different times. This correlation is expected, as the charge for a call is directly influenced by its duration in minutes.

> When it comes to churn prediction, there was a weak positive correlation between total day minutes, total day charge, customer service calls, and churn. In contrast, the remaining features demonstrate a negligible correlation with churn.
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/5a06d570-f1e1-40f3-8b00-692e7e22b31a)

### DATA PREPARATION FOR MODELLING
---
- **Label Encoding:**

>We used label encoding to convert label variables in the "international plan," "voice mail plan," and "churn" columns into numeric form. "Yes" and "No" in "international plan" and "voice mail plan" are encoded as 1 and 0, respectively, while "False" and "True" in "churn" are encoded as 0 and 1.

- **One-Hot Encoding:**

>We used one hot encoding to handle categorical variables, applied to the "State" column. This technique converts categorical variables into multiple binary columns, making them compatible with the algorithm.

- **Scaling:**

>Scaling is essential for accurate predictions in machine learning, especially for algorithms sensitive to feature scaling. The data was scaled to ensure consistency and comparability among features. In this case, the StandardScaler method was used, which standardized features by subtracting the mean and dividing by the standard deviation. This scaling technique brings the features to a zero mean and unit variance.

- **Data Splitting:**

>The data was split into training and test sets to evaluate the model's performance on unseen data.

- **Handling Class Imbalance using SMOTE:**

>To address class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was employed. SMOTE oversamples the minority class by synthesizing new examples from existing ones, ensuring better model performance on the minority class without adding new information. It is only applied to the training data, not the test data, to evaluate the model's ability to generalize to unseen data.

### MODELLING
---
- **Model Development with Lazy Predict**

> We used Lazy Predict, a tool that simplifies the initial stages of building and evaluating machine learning models. It provides a comprehensive overview of model performance on a given dataset, allowing for quick previews before selecting specific models. This streamlines the model development process and supports efficient decision-making.
<img width="367" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/31f5878c-414f-4a12-abf9-a6768c4eed76">

**Models**
---

- **Decision Tree**

>The Decision Tree model achieved an accuracy of 91% with a precision of 0.92, recall of 0.91, and F1-score of 0.92. It performed well in predicting the majority class (0) but had lower performance in predicting the minority class (1).
<img width="268" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/49a88b32-d339-4c3e-a367-7cf533c31375">

- **Random Forest**

>The Random Forest model achieved an accuracy of 94% with a precision of 0.93, recall of 0.93, and F1-score of 0.93. It showed improved performance compared to the Decision Tree model, achieving balanced precision and recall for both classes.
<img width="287" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/21925ea3-62cf-41eb-ab4e-fe5528750501">

- **KNN (K-Nearest Neighbors)**

>The KNN model achieved an accuracy of 73% with a precision of 0.82, recall of 0.73, and F1-score of 0.76. It had lower performance compared to the Decision Tree and Random Forest models, particularly in predicting the minority class (1).
<img width="263" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/cb2849b6-45c5-4f35-8459-60d3f1d3706d">

- **XGBoost**

>The XGBoost model achieved the highest accuracy of 96% with a precision of 0.96, recall of 0.96, and F1-score of 0.96. It demonstrated the best overall performance among the models, with balanced precision and recall for both classes.
<img width="266" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/d3739256-5371-4b71-ab54-21c6e727f451">

### MODEL EVALUATION AND SELECTION
---
> We employed various evaluation metrics to assess the performance of different models and determine the most suitable candidate for hyperparameter tuning.

> The evaluation metrics used include accuracy, precision, recall, F1-score, ROC curve and AUC value, and confusion matrices. These metrics provided insights into overall performance, classification accuracy, visualization of performance, and evaluation of target variable classification.

>By considering these evaluation metrics, the XGBoost model outperformed the other models, providing the highest accuracy and F1-score, making it the recommended choice for predictive modeling.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/2e6c0592-e7fb-4f52-a5ea-2c684098c53d)

**Model Optimization**
>We  enhanced  XGBoost model's performance by utilizing the top 10 most relevant features. We identified these features by sorting them based on their importance using indices.

<img width="284" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/921f9b93-72bd-4b54-9a94-056b5da1c06b">

> The most important features identified in predicting churn were customer service calls, total day minutes, total day charge, voice mail plan, total eve charge, and total int calls. These features played a crucial role in accurately predicting churned customers.

<img width="257" alt="image" src="https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/ea220ac7-762e-491b-a946-43590166761d">

>Compared to the previous version of the XGBoost model, the optimized version achieved slightly better performance in terms of precision, recall, and F1-score for churned customers. The model demonstrated high accuracy and effectively balanced correctly identifying churned customers while minimizing false positives.

### CONCLUSION
---
- **Relationship between Customer Service Calls and Churn**

>Customer service calls are positively related to churn.
Factors that may customer service calls and churn include customer satisfaction rate, customer experience, issue resolution, service quality, and proactive engagement.

- **Relationship between Time of Day and Churn**

>Total day minutes, total evening minutes, and total night minutes show a positive relationship with churn.The specific time of day does not directly influence churn.

- **Relationship between Area Code and Churn**

>Area code is not directly related to churn.Area code distribution may vary, but it does not directly influence reasons for churn.


### RECOMMENDATIONS
---

- Enhance Network Coverage: Invest in expanding and improving network infrastructure to provide reliable service across all regions.

- Personalized Customer Experience: Utilize customer data and analytics to understand individual preferences and behavior. Create tailored marketing messages, offers, and service suggestions to enhance the customer experience.

- Proactive Customer Support: Implement predictive issue detection and resolution to address problems before they escalate. Promptly resolve customer complaints and inquiries to demonstrate responsiveness.

- Introduce Value-Added Services and Offers: Implement loyalty programs and exclusive offers to incentivize customer loyalty. Provide perks such as discounted plans, free upgrades, or access to premium content.

- Regular Communication: Maintain regular communication with customers through personalized emails, SMS, or in-app messages. Keep customers informed about new services, features, and promotions.

- Customer Feedback and Surveys: Actively seek feedback and conduct surveys to understand customer pain points and areas for improvement.

- Community Engagement: Foster a sense of community among customers through forums, online communities, or social media platforms to encourage interaction and build loyalty.

- Proactive Churn Prediction: Utilize data analytics to predict potential churners and implement targeted retention efforts.

>These strategies can help SyrialTel effectively manage customer churn and proactively retain customers.
