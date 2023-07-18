### SYRIATEL CUSTOMER CHURN
![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/718bce74-01a3-4c18-903d-c7a8bdb1aa9c)

## PROJECT BACKGROUND

SyriaTel is a telecommunications company facing the challenge of customer churn.
According to Forbes Advisor an article by Monique Danao, published 2nd March 2023 at 11.00am, Customer churn rate, "refers to the rate at which subscribers or customers stop transacting with your business." https://www.forbes.com/advisor/business/churn-rate/

Churn can have significant financial implications for SyriaTel, including the loss of recurring revenue, increased customer acquisition costs, and potential negative impact on the company's reputation.

To address this issue, SyriaTel have requested **CodeTribe3** researchers to build a churn prediction system that can identify customers likely to churn in the near future.

## BUSINESS PROBLEM

Syriatel, a mobile telecommunications provider, focuses on attracting new customers and improving customer retention to grow revenue. To achieve this, they prioritize long-term customer relationships over acquiring new customers. Therefore, churn prediction plays a crucial role in their strategy. The goal of this project is to develop an accurate model for predicting customer churn and identify the significant features for churn prediction. By identifying potential churners, Syriatel can take proactive measures to prevent customers from leaving.

## DATA UNDERSTANDING

For model building we used data from Churn in Telecom's dataset aquired from kaggle. This data file is available in the project repo in the folder "data".

The dataset utilised in this research project contains information about customer attributes, call usage, charges and customer service interactions with the churn column acting as our target variable.

The dataset contains 3333 rows(number of entries) and 20 columns.

The column names and their respective descriptions are:

_account length_: The number of days the customer has been an active customer; _(dtype : Int64)_

_area code_: The area code of the customer's phone number; _(dtype : Int64)_

_phone number_: The customer's phone number; _(dtype : object)_

_international plan_: Indicates whether the customer has an international calling plan; _(dtype : object)_

_voice mail plan_: Indicates whether the customer has a voicemail plan; _(dtype : object)_

_number vmail messages_ : Represents the number of voicemail messages the customer has; _(dtype : Int64)_

_total day minutes_: The total number of minutes the customer has used during the day; _(dtype : Float64)_

_total day calls_: The total number of calls the customer has made during the day; _(dtype : Int64)_

_total day charge_: The total charge in dollars for the day's usage; _(dtype : Float64)_

_total eve minutes_: The total number of minutes the customer has used during the evening; _(dtype : Float64)_

_total eve calls_: The total number of calls the customer has made in the evening; _(dtype : Int64)_

_total eve charge_: The total charge in dollars for the evening's usage; _(dtype : Float64)_

_total night minutes_: The total number of minutes the customer has used during the night; _(dtype : Float64)_

_total night calls_: The total number of calls the customer has made during the night; _(dtype : Int64)_

_total night charge_: The total charge in dollars for the night's usage; _(dtype : Float64)_

_total intl minutes_: The total number of international minutes the customer has used; _(dtype : Float64)_

_total intl calls_: The total number of international calls the customer has made; _(dtype : Int64)_

_total intl charge_: The total charge in dollars for the international usage; _(dtype : Float64)_

_customer service calls_: The number of customer service calls made by the customer; _(dtype : Int64)_

_churn_: Indicating whether the customer has churned or not; _(dtype : Bool)_

# EXPLORATORY DATA ANALYSIS

Within the dataset of 3,333 customers, around 14.5% (483 customers) have terminated their contract with SyriaTel, resulting in a loss of customer base. This distribution indicates an imbalance between the two binary classes. It is essential to handle this data imbalance prior to modeling to ensure accurate predictions. Failing to address this imbalance could lead to inaccurate model outcomes.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/850341cc-3230-445d-8ef3-8037327b95f5)

The features in the dataframe exhibit varying scaling and non-normal distribution. To ensure consistency and comparability, it is necessary to scale and normalize the features.

![image](https://github.com/Muthoni-Kahura/Customer_Churn_Prediction_model_by_CodeTribe3/assets/128212536/f7936dca-26d7-4671-a49e-8fadf4fee853)





