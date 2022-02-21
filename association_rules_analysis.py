# importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import datetime as dt
import calendar
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


# reading csv
df = pd.read_csv("D:\github\deneme\deneme5/bread basket.csv")

# examining the dataset
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.info)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
check_df(df)


# creating new variables with date_time variable
def time_variables(dataframe, colname):
    dataframe[colname] = pd.to_datetime(df[colname])
    dataframe['date'] = dataframe[colname].dt.date
    dataframe['time'] = dataframe[colname].dt.time
    dataframe['month'] = dataframe[colname].dt.month
    dataframe['month'] = dataframe['month'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                                    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                                     'August', 'September', 'October', 'November', 'December'])
    dataframe['day'] = dataframe[colname].dt.day
    dataframe['weekday'] = dataframe[colname].dt.weekday
    dataframe['weekday'] = dataframe['weekday'].replace([0, 1, 2, 3, 4, 5, 6],
                                                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                         'Saturday', 'Sunday'])

    return dataframe

time_variables(df, 'date_time')

######### DATA VISUALIZATION ###########


# top 20 Items purchased by customers
plt.figure(figsize=(15,5))
sns.barplot(x = df.Item.value_counts().head(20).index, y = df.Item.value_counts().head(20).values, palette = 'hls')
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'red', size = 20)
plt.show()

# number of orders received each month
monthTran = df.groupby('month')['Transaction'].count().reset_index()
monthTran.loc[:,"monthorder"] = [4,8,12,2,1,7,6,3,5,11,10,9]
monthTran.sort_values("monthorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = monthTran, x = "month", y = "Transaction")
plt.xlabel('Months', size = 15)
plt.ylabel('Orders per month', size = 15)
plt.title('Number of orders received each month', color = 'red', size = 20)
plt.show()

# number of orders received each day
weekTran = df.groupby('weekday')['Transaction'].count().reset_index()
weekTran.loc[:,"weekorder"] = [4,0,5,6,3,1,2]
weekTran.sort_values("weekorder",inplace=True)

plt.figure(figsize=(12,5))
sns.barplot(data = weekTran, x = "weekday", y = "Transaction")
plt.xlabel('Week Day', size = 15)
plt.ylabel('Orders per day', size = 15)
plt.title('Number of orders received each day', color = 'red', size = 20)
plt.show()

#making transaction data
df_invoice_product = df.groupby(["Transaction", "Item"])["Item"].count().unstack().fillna(0).applymap(lambda x: 1 if  x>0 else 0)
df_invoice_product.shape

# using the 'apriori algorithm' with min_support=0.01
frequent_itemsets = apriori(df_invoice_product, min_support=0.01, use_colnames=True, low_memory=True)
frequent_itemsets.sort_values("support", ascending=False)

# Making the rules from frequent itemset
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# selecting lift grander than 1 and arranging  the data and from highest to lowest with respect to 'support'
rules[rules["lift"]>1].sort_values("support",ascending = False)
