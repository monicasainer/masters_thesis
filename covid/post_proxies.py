# %% [markdown]
# # Empirical Analysis

# %% [markdown]
# ## Extracting the data

# %% [markdown]
# Import the necessary libraries

# %%
import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings


# %%
warnings.filterwarnings('ignore')
path = os.getenv("ROOT_PATH")
sys.path.append(path)
print(path)


# %% [markdown]
# The list of **all the current components** of the OMX Stockholm PI index can be found [here](https://indexes.nasdaqomx.com/Index/Weighting/OMXSPI) by the end of the day of 16th February 2024.
#
# The list of **large-caps** of the OMX Stockholm PI index can be found [here](https://indexes.nasdaqomx.com/Index/Weighting/OMXSLCPI) by the end of the day of 16th February 2024.
#
# The list of **mid-caps** of the OMX Stockholm PI index can be found [here](https://indexes.nasdaqomx.com/Index/Weighting/OMXSMCPI) by the end of the day of 16th February 2024.
#
# The list of **small-caps** of the OMS Stockholm PI index can be found [here](https://indexes.nasdaqomx.com/Index/Weighting/OMXSSCPI) by the end of the day of 16th February 2024.
#

# %% [markdown]
# In the following steps we're charging the name of all the components and turn them into a list.
#
# The latter will be used to fetch the data - adjusted closed price and volume - from Yahoo Finance. And save accordingly in a file called `raw_data.csv`

# %%
tickers= pd.read_excel(f"{path}/raw_data/post_covid/Weightings_20240216_OMXSPI.xlsx",header=0)
# If error shows up run: !pip3 install xlrd


# %%
tickers.head()


# %%
tickers_list=tickers['Security-Symbol'].to_list()


# %%
data = yf.download(tickers_list, start="2014-03-06",end='2020-03-14')


# %%
data.head()


# %%
data.to_csv(f"{path}/raw_data/post_covid/raw_data.csv")


# %% [markdown]
# During the following cells we are going to create different lists with the names of the companies considered large-caps, mid-caps, and small caps.

# %%
l_caps=pd.read_excel(f"{path}/raw_data/post_covid/large_caps.xlsx")
l_caps_list=l_caps['Security-Symbol'].to_list()


# %%
m_caps=pd.read_excel(f"{path}/raw_data/post_covid/mid_caps.xlsx")
m_caps_list=m_caps['Security-Symbol'].to_list()


# %%
s_caps=pd.read_excel(f"{path}/raw_data/post_covid/small_caps.xlsx")
s_caps_list=s_caps['Security-Symbol'].to_list()


# %%
len(l_caps_list)+len(m_caps_list)+len(s_caps_list)


# %%
len(tickers_list)


# %% [markdown]
# There is one company that we cannot classify as large, mid or small-cap.
#
# It'll be pointed out in the following steps.

# %% [markdown]
# ## Cleaning data

# %% [markdown]
# After downloading the data in the file `raw_data.csv` you must open it in Microsoft Excel.
# In the **first row** we can find the number of the metric fetched.
# In the **second row** we can find the names of the different companies.
# In the **first column** we can find the dates we have exported.
#
# To clean up the dataset, delete those columns where the first row differs from `adjClose` and `volume`.
# As soon as this is done, cut those columns where the first row is `volume` and paste them in a new spreadsheet (not tab).
# Remove the first row as it doesn't add useful information at the moment. Call `volumes` to this new spreadsheet and save it as a .csv file.
#
# Come back to the initial spreadsheet called `raw_data.csv`.
# Since we only have `adjClose` prices, remove the first row.
# Rename the spreadsheet as `price` and save it as a .csv file
#
#

# %%
df_price = pd.read_excel(f'{path}/raw_data/post_covid/price.xlsx')


# %%
print(f"Number of companies in the sample: {df_price.shape[1]-2}") # Excluding the 'Date' and '^OMXSPI' columns.


# %%
null_percentage_dict={'Company':[],'Null_percentage':[],'Type':[]}

for column in df_price.columns[1:-1]:
    company_name=column
    null_percentage = df_price[company_name].isnull().mean()*100
    null_percentage_dict['Company'].append(company_name)
    null_percentage_dict['Null_percentage'].append(null_percentage)
    if company_name in l_caps_list:
        null_percentage_dict['Type'].append("l-cap")
    elif company_name in m_caps_list:
        null_percentage_dict['Type'].append("m-cap")
    elif company_name in s_caps_list:
        null_percentage_dict['Type'].append("s-cap")
    else: null_percentage_dict['Type'].append("non-registered")

df_null_percentage=pd.DataFrame.from_dict(null_percentage_dict)


# %%
df_null_percentage[df_null_percentage['Type']=="non-registered"]


# %%
df_null_percentage=df_null_percentage.sort_values(by="Null_percentage",ascending=False)

df_null_percentage.head()


# %%
df_null_percentage.to_excel(f'{path}/raw_data/post_covid/null_percentage.xlsx')


# %%
fig = px.bar(df_null_percentage, x='Company', y='Null_percentage', color='Type',
             labels={'Null_percentage': 'Null_percentage'},
             title='Null Percentage of Companies by Cap Classification',
             hover_data=['Company', 'Null_percentage', 'Type'])
fig.update_layout(barmode='group', xaxis_title='Company', yaxis_title='Null_percentage')
fig.show()


# %% [markdown]
# Components over time of the OMXSPI [here](https://indexes.nasdaqomx.com/Index/Weighting/OMXSPI)
#
# Methodology of the index [here](https://indexes.nasdaqomx.com/docs/Methodology_Nordic_AllShare.pdf)

# %% [markdown]
# ## Proxy A:
#
# Daily raw stock returns with absolute values exceeding 8% and 10%

# %%
threshold_8_percent_large = 0.08
threshold_10_percent_large = 0.10

threshold_10_percent_mid = 0.10
threshold_12_percent_mid = 0.12

threshold_12_percent_small = 0.13
threshold_14_percent_small =  0.15

proxy_a_df = pd.DataFrame(df_price['Date'].copy())


# %%
for column in l_caps_list:
    stock_returns = df_price[column].pct_change()

    # Create proxy columns based on the defined thresholds for large-caps
    proxy_a_df[f'{column}_Increase_small_thres'] = (stock_returns > threshold_8_percent_large).astype(int)
    proxy_a_df[f'{column}_Decrease_small_thres'] = (stock_returns < -threshold_8_percent_large).astype(int)
    proxy_a_df[f'{column}_Increase_large_thres'] = (stock_returns > threshold_10_percent_large).astype(int)
    proxy_a_df[f'{column}_Decrease_large_thres'] = (stock_returns < -threshold_10_percent_large).astype(int)

for column in m_caps_list:
    stock_returns = df_price[column].pct_change()

    # Create proxy columns based on the defined thresholds for mid-caps
    proxy_a_df[f'{column}_Increase_small_thres'] = (stock_returns > threshold_10_percent_mid).astype(int)
    proxy_a_df[f'{column}_Decrease_small_thres'] = (stock_returns < -threshold_10_percent_mid).astype(int)
    proxy_a_df[f'{column}_Increase_large_thres'] = (stock_returns > threshold_12_percent_mid).astype(int)
    proxy_a_df[f'{column}_Decrease_large_thres'] = (stock_returns < -threshold_12_percent_mid).astype(int)

for column in s_caps_list:
    stock_returns = df_price[column].pct_change()

    # Create proxy columns based on the defined thresholds for small-caps
    proxy_a_df[f'{column}_Increase_small_thres'] = (stock_returns > threshold_12_percent_small).astype(int)
    proxy_a_df[f'{column}_Decrease_small_thres'] = (stock_returns < -threshold_12_percent_small).astype(int)
    proxy_a_df[f'{column}_Increase_large_thres'] = (stock_returns > threshold_14_percent_small).astype(int)
    proxy_a_df[f'{column}_Decrease_large_thres'] = (stock_returns < -threshold_14_percent_small).astype(int)


index_returns = df_price['^OMXSPI'].pct_change()
proxy_a_df['Market_Return_Increase'] = (index_returns > 0).astype(int)
proxy_a_df['Market_Return_Decrease'] = (index_returns < 0).astype(int)

proxy_a_df.to_excel(f'{path}/raw_data/post_covid/proxy_a.xlsx')


# %% [markdown]
# ## Proxy B:
#
# Daily raw stock returns with absolute values exceeding 3 and 4 standard deviations.

# %%
price_2013 = yf.download(tickers_list, start="2013-03-06", end='2020-03-14') #Downloading prior year's prices to calculate standard deviations ovet the las 250 trading days
price_2013.to_csv(f'{path}/raw_data/post_covid/price_2013.csv')


# %%

threshold_3_std = 3
threshold_4_std = 4
window_size = 250 # Number of trading days for calculating the rolling standard deviation
df_price_2013 = pd.read_excel(f"{path}/raw_data/post_covid/price_2013.xlsx")

proxy_b_df = pd.DataFrame(df_price['Date'].copy())


# %%
for column in df_price_2013.columns[1:-1]:  # Exclude the 'Date' and '^OMXSPI' columns
    stock_returns = df_price_2013[column].pct_change()

    result_df = pd.DataFrame({
            'Date': df_price_2013['Date'],
            'Stock_Returns': stock_returns
        })


    rolling_std = stock_returns.rolling(window=window_size).std()


    result_df[f'{column}_Increase_3std'] = (stock_returns > threshold_3_std * rolling_std).astype(int)
    result_df[f'{column}_Decrease_3std'] = (stock_returns < - threshold_3_std * rolling_std).astype(int)
    result_df[f'{column}_Increase_4std'] = (stock_returns > threshold_4_std * rolling_std).astype(int)
    result_df[f'{column}_Decrease_4std'] = (stock_returns < - threshold_4_std * rolling_std).astype(int)
    result_df = result_df[result_df['Date']>='2014-03-06']
    result_df.drop(columns=['Stock_Returns'],inplace=True)
    proxy_b_df=pd.merge(proxy_b_df,result_df,left_on='Date', right_on='Date', how='left')
proxy_b_df.head()

index_returns = df_price['^OMXSPI'].pct_change()
proxy_b_df['Market_Return_Increase'] = (index_returns > 0).astype(int)
proxy_b_df['Market_Return_Decrease'] = (index_returns < 0).astype(int)

proxy_b_df.to_excel(f'{path}/raw_data/post_covid/proxy_b.xlsx')


# %% [markdown]
# ## Proxy C
#
# Daily abnormal stock returns with absolute values exceeding 8% and 10% using Market Model Adjusted Returns

# %% [markdown]
# The risk-free interest rate corresponds to the 1-month Swedish Treasury Bills. It has been obtained from [Sveriges Riskbank]('https://www.riksbank.se/en-gb/statistics/interest-rates-and-exchange-rates/search-interest-rates-and-exchange-rates/?s=g6-SETB1MBENCHC&fs=2#riksbank-seriesform')

# %%
threshold_8_percent_large = 0.08
threshold_10_percent_large = 0.10

threshold_10_percent_mid = 0.10
threshold_12_percent_mid = 0.12

threshold_13_percent_small = 0.13
threshold_15_percent_small =  0.15

window_size = 250  # Number of trading days for estimating beta


risk_free_rate_df= pd.read_excel(f"{path}/raw_data/post_covid/risk_free.xlsx")
risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)'].fillna(method='ffill', inplace=True)
risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)']= (1 + risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)']) ** (1/250) - 1

proxy_c_df = pd.DataFrame(df_price['Date'].copy())


for column in l_caps_list:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013[column].pct_change()

    # Market returns (e.g., using OMXSPI as a proxy for the market)
    market_returns = df_price_2013['^OMXSPI'].pct_change()


    result_df = pd.DataFrame({
    'Date': df_price_2013['Date'],
    'Stock_Returns': stock_returns,
    'Market_Returns': market_returns
})
    # Beta is calculated as the covariance of the stock's returns with the market returns divided by the variance of the market returns over the preceding 250 trading days.
    result_df['beta'] = result_df['Stock_Returns'].rolling(window=window_size).cov(result_df['Market_Returns']).div(result_df['Market_Returns'].rolling(window=window_size).var())
    result_df = pd.merge(result_df,risk_free_rate_df, left_on='Date', right_on='Date', how='left')


    # Ri = Rf + beta * (Rm-Rf) + ei --> Ri - [Rf + beta * (Rm - Rf)]
    result_df['MMAR'] = result_df['Swedish Treasury Bills (SE TB 1 Month)']+ result_df['beta'] * (result_df['Market_Returns']- result_df['Swedish Treasury Bills (SE TB 1 Month)'])

    result_df[f'{column}_ARs'] = result_df['Stock_Returns'] - result_df['MMAR']


    result_df.drop(columns=['Stock_Returns','Market_Returns','beta','Swedish Treasury Bills (SE TB 1 Month)','MMAR'],inplace=True)

    proxy_c_df = pd.merge(proxy_c_df,result_df, left_on='Date',right_on='Date',how='left')

    proxy_c_df[f'{column}_Increase_small_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_8_percent_large).astype(int)
    proxy_c_df[f'{column}_Decrease_small_thres'] = (proxy_c_df[f'{column}_ARs'] < -threshold_8_percent_large).astype(int)
    proxy_c_df[f'{column}_Increase_large_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_10_percent_large).astype(int)
    proxy_c_df[f'{column}_Decrease_large_thres'] = (proxy_c_df[f'{column}_ARs'] < - threshold_10_percent_large).astype(int)
    proxy_c_df.drop(columns=[f'{column}_ARs'],inplace=True)

for column in m_caps_list:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013[column].pct_change()

    # Market returns (e.g., using OMXSPI as a proxy for the market)
    market_returns = df_price_2013['^OMXSPI'].pct_change()


    result_df = pd.DataFrame({
    'Date': df_price_2013['Date'],
    'Stock_Returns': stock_returns,
    'Market_Returns': market_returns
})
    # Beta is calculated as the covariance of the stock's returns with the market returns divided by the variance of the market returns over the preceding 250 trading days.
    result_df['beta'] = result_df['Stock_Returns'].rolling(window=window_size).cov(result_df['Market_Returns']).div(result_df['Market_Returns'].rolling(window=window_size).var())
    result_df = pd.merge(result_df,risk_free_rate_df, left_on='Date', right_on='Date', how='left')


    # Ri = Rf + beta * (Rm-Rf) + ei --> Ri - [Rf + beta * (Rm - Rf)]
    result_df['MMAR'] = result_df['Swedish Treasury Bills (SE TB 1 Month)']+ result_df['beta'] * (result_df['Market_Returns']- result_df['Swedish Treasury Bills (SE TB 1 Month)'])

    result_df[f'{column}_ARs'] = result_df['Stock_Returns'] - result_df['MMAR']


    result_df.drop(columns=['Stock_Returns','Market_Returns','beta','Swedish Treasury Bills (SE TB 1 Month)','MMAR'],inplace=True)

    proxy_c_df = pd.merge(proxy_c_df,result_df, left_on='Date',right_on='Date',how='left')

    proxy_c_df[f'{column}_Increase_small_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_10_percent_mid).astype(int)
    proxy_c_df[f'{column}_Decrease_small_thres'] = (proxy_c_df[f'{column}_ARs'] < -threshold_10_percent_mid).astype(int)
    proxy_c_df[f'{column}_Increase_large_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_15_percent_small).astype(int)
    proxy_c_df[f'{column}_Decrease_large_thres'] = (proxy_c_df[f'{column}_ARs'] < - threshold_15_percent_small).astype(int)
    proxy_c_df.drop(columns=[f'{column}_ARs'],inplace=True)

for column in s_caps_list:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013[column].pct_change()

    # Market returns (e.g., using OMXSPI as a proxy for the market)
    market_returns = df_price_2013['^OMXSPI'].pct_change()


    result_df = pd.DataFrame({
    'Date': df_price_2013['Date'],
    'Stock_Returns': stock_returns,
    'Market_Returns': market_returns
})
    # Beta is calculated as the covariance of the stock's returns with the market returns divided by the variance of the market returns over the preceding 250 trading days.
    result_df['beta'] = result_df['Stock_Returns'].rolling(window=window_size).cov(result_df['Market_Returns']).div(result_df['Market_Returns'].rolling(window=window_size).var())
    result_df = pd.merge(result_df,risk_free_rate_df, left_on='Date', right_on='Date', how='left')


    # Ri = Rf + beta * (Rm-Rf) + ei --> Ri - [Rf + beta * (Rm - Rf)]
    result_df['MMAR'] = result_df['Swedish Treasury Bills (SE TB 1 Month)']+ result_df['beta'] * (result_df['Market_Returns']- result_df['Swedish Treasury Bills (SE TB 1 Month)'])

    result_df[f'{column}_ARs'] = result_df['Stock_Returns'] - result_df['MMAR']


    result_df.drop(columns=['Stock_Returns','Market_Returns','beta','Swedish Treasury Bills (SE TB 1 Month)','MMAR'],inplace=True)

    proxy_c_df = pd.merge(proxy_c_df,result_df, left_on='Date',right_on='Date',how='left')

    proxy_c_df[f'{column}_Increase_small_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_13_percent_small).astype(int)
    proxy_c_df[f'{column}_Decrease_small_thres'] = (proxy_c_df[f'{column}_ARs'] < -threshold_13_percent_small).astype(int)
    proxy_c_df[f'{column}_Increase_large_thres'] = (proxy_c_df[f'{column}_ARs'] > threshold_15_percent_small).astype(int)
    proxy_c_df[f'{column}_Decrease_large_thres'] = (proxy_c_df[f'{column}_ARs'] < - threshold_15_percent_small).astype(int)
    proxy_c_df.drop(columns=[f'{column}_ARs'],inplace=True)

index_returns = df_price['^OMXSPI'].pct_change()
proxy_c_df['Market_Return_Increase'] = (index_returns > 0).astype(int)
proxy_c_df['Market_Return_Decrease'] = (index_returns < 0).astype(int)

proxy_c_df.to_excel(f"{path}/raw_data/post_covid/proxy_c.xlsx")


# %%
