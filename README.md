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

account length: The number of days the customer has been an active customer.

area code: The area code of the customer's phone number.

phone number: The customer's phone number.

international plan: Indicates whether the customer has an international calling plan.<br>
voice mail plan: Indicates whether the customer has a voicemail plan.
number vmail messages : Represents the number of voicemail messages the customer has.
total day minutes: The total number of minutes the customer has used during the day.
total day calls: The total number of calls the customer has made during the day.
total day charge: The total charge in dollars for the day's usage.
total eve minutes: The total number of minutes the customer has used during the evening.
total eve calls: The total number of calls the customer has made in the evening.
total eve charge: The total charge in dollars for the evening's usage.
total night minutes: The total number of minutes the customer has used during the night.
total night calls: The total number of calls the customer has made during the night.
total night charge: The total charge in dollars for the night's usage.
total intl minutes: The total number of international minutes the customer has used.
total intl calls: The total number of international calls the customer has made.
total intl charge: The total charge in dollars for the international usage.
customer service calls: The number of customer service calls made by the customer.
churn: Indicating whether the customer has churned or not.
The data is categorised by 4 data types:

Bool - 1 column (churn)
Float64 - 8 columns (total day minutes, total day charge, total eve minutes, total eve charge, total night minutes, total night charge - total intl minutes, total intl charge)
Int64 - 8 columns (account length, area code, number vmail messages, total day calls, total eve calls, total night calls, total intl calls, customer service calls)
Object - 3 columns (phone number, international plan, voice mail plan)
