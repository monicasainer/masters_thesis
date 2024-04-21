# %%
import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
from io import StringIO

# %%
warnings.filterwarnings('ignore')
path = os.getenv("ROOT_PATH")

# %% [markdown]
# > After analysing different periods of time by checking different proxies, we reached some conclusions:
# >
# > - **Before Covid:** We find significant price reversals when the **price decreases**.
# > - **During and after Covid:** We find statistically significant price reversals when the **price increases**.
# >
# > Therefore, we are going to create regressions for:
# > - Events before covid where the large price change was negative, and their corresponding temporal windows.
# > - Events during and after covid where the large price change was positive, and their corresponding temporal windows.
# > In order to control the effect of different variables in the abnormal stock returns.
#

# %% [markdown]
# > Variables that we need:
# > - ARit: Abnormal stock return from the event i until the end of the post-event window t.
# > - MR0_dumi: Dummy variable that takes the value 1  if the market return corresponding to day 0 for event i is positive/negative, and 0 otherwise
# > - MCapi: Algorithm of the firm’s market capitalization on the day of the event i.
# > - betai: CAPM beta calculated between day -250 and -1 of the event i.
# > - SR_volati: Standard deviation of the stock returns calculated between day -250 and -1 of the event i.
# > - |SR0|i : Absolut stock return during day 0 of event i.
# > - ABVOL0i: Difference between the stock trading volume on day 0 and its average volume from days -250 to -1.

# %% [markdown]
# 1. **ARit**

# %%
AR_df_post = pd.read_excel(f"{path}/raw_data/post_covid/ARs_df.xlsx")

# %% [markdown]
# 2. **MR_dum**

# %%
proxy_a_df_post = pd.read_excel(f"{path}/raw_data/post_covid/proxy_a.xlsx")
mr_dum_post = proxy_a_df_post[['Date','Market_Return_Increase']]

# %% [markdown]
# 3. **MCap**

# %% [markdown]
# Since we didn't fetch the market capitalization earlier, we need to do it now:

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
price_2013_post=pd.read_excel(f"{path}/raw_data/post_covid/price_2013.xlsx",header=0)
# tickers = yf.Tickers(tickers_list) #, start="2014-03-06", end='2020-03-14')
for t in ['8TRA.ST','AAK.ST']: #tickers_list:
    ticker= yf.Ticker(f'{t}')
    quarterly_income_stmt = ticker.quarterly_income_stmt
    print(quarterly_income_stmt.columns)
    # all_dates = ticker.quarterly_income_stmt.columns.to_list
    # print(all_dates)
    # total_shares = ticker.quarterly_income_stmt[date]['Basic Average Shares']

# %% [markdown]
# 4. **Beta**

# %%
df_price_2013_post=pd.read_excel(f"{path}/raw_data/post_covid/price_2013.xlsx",header=0)
df_price_2013_post['market_returns'] =df_price_2013_post['^OMXSPI'].pct_change()
window_size=250
beta_post={'Date':df_price_2013_post['Date']}
for column in df_price_2013_post.columns[1:-2]:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013_post[column].pct_change()
    result_df = pd.DataFrame({
    'Date': df_price_2013_post['Date'],
    'Stock_Returns': stock_returns,
    'Market_Returns': df_price_2013_post['market_returns']})

    result_df['beta'] = result_df['Stock_Returns'].rolling(window=window_size).cov(result_df['Market_Returns']).div(result_df['Market_Returns'].rolling(window=window_size).var())
    result_df['normalized_beta'] = (result_df['beta'] - result_df['beta'].mean())/result_df['beta'].std()

    beta_post[f'beta_{column}']= result_df['normalized_beta']
beta_post=pd.DataFrame.from_dict(beta_post)
beta_post

# %% [markdown]
# 5. **SR_vola**

# %%
df_price_2013_post=pd.read_excel(f"{path}/raw_data/main/price_2013.xlsx",header=0)
df_sr_vola_post_dict={'Date':df_price_2013_post['Date']}
window_size=250
for column in df_price_2013_post.columns[1:-1]:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013_post[column].pct_change()
    result_df = pd.DataFrame({
    'Date': df_price_2013_post['Date'],
    'Stock_Returns': stock_returns})

    result_df['sr_vola']= stock_returns.rolling(window=window_size).std()
    result_df['normalized_sr_vola']= (result_df['sr_vola']-result_df['sr_vola'].mean())/result_df['sr_vola'].std()


    df_sr_vola_post_dict[f'sr_vola_{column}']=result_df['normalized_sr_vola']
df_sr_vola_post=pd.DataFrame.from_dict(df_sr_vola_post_dict)
df_sr_vola_post.head(400)

# %% [markdown]
# 6.  **|SRo|i**

# %%
df_price_2013_post=pd.read_excel(f"{path}/raw_data/main/price_2013.xlsx",header=0)
df_abs_sr_post_dict={'Date':df_price_2013_post['Date']}
for column in df_price_2013_post.columns[1:-1]:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = df_price_2013_post[column].pct_change()
    df_abs_sr_post_dict[f'abs_sr_{column}'] = abs(stock_returns)
df_abs_sr_post = pd.DataFrame.from_dict(df_abs_sr_post_dict)
df_abs_sr_post.head()

# %% [markdown]
# 7. **ABVOL**

# %%
volume_post = pd.read_excel(f'{path}/raw_data/main/volume.xlsx')
window_size=250
df_ab_vol_post_dict = {'Date':volume_post['Date']}

for column in volume_post.columns[1:]:
    rolling_mean = volume_post[column].rolling(window=window_size, min_periods=1).mean()
    rolling_std = volume_post[column].rolling(window=window_size, min_periods=1).std()
    df_ab_vol_post_dict[f'ab_vol_{column}']=(volume_post[column] - rolling_mean) / rolling_std
df_ab_vol_post = pd.DataFrame.from_dict(df_ab_vol_post_dict)
df_ab_vol_post.head(400)

# %% [markdown]
# Now it's time to merge all the variables together.
#
# Additionaly, once at a time, we will join one proxy, in order to be able to stand out the events.
#

# %%
total_variables_post = pd.merge(AR_df_post,mr_dum_post,on='Date',how='left')
total_variables_post = pd.merge(total_variables_post,beta_post,on='Date',how='left')
total_variables_post = pd.merge(total_variables_post,df_sr_vola_post,on='Date',how='left')
total_variables_post = pd.merge(total_variables_post,df_abs_sr_post,on='Date',how='left')
total_variables_post = pd.merge(total_variables_post,df_ab_vol_post,on='Date',how='left')
total_variables_post.drop(columns=['Unnamed: 0','Swedish Treasury Bills (SE TB 1 Month)','market_returns'],inplace=True)
total_variables_post

# %% [markdown]
# # Regression Model

# %%
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# %% [markdown]
# ## post-Covid

# %% [markdown]
# ### Proxy A

# %% [markdown]
# #### Small Threshold / 2-days window

# %%
total_df_a_post = pd.merge(total_variables_post,proxy_a_df_post, on='Date', how='left')


# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_two_dict = {}
a_post_two_dict['ARs'] =[]
a_post_two_dict['Market_Return'] =[]
a_post_two_dict['beta']=[]
a_post_two_dict['sr_vola']=[]
a_post_two_dict['abs_sr']=[]
a_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        a_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_two_dict['ab_vol'].append(event_day_ab_vol)

a_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_two_dict.items() })

a_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_two_df['ARs']
X= a_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_two_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Small Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_five_dict = {}
a_post_five_dict['ARs'] =[]
a_post_five_dict['Market_Return'] =[]
a_post_five_dict['beta']=[]
a_post_five_dict['sr_vola']=[]
a_post_five_dict['abs_sr']=[]
a_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_5_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        a_post_five_dict['ARs'].extend(next_5_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_five_dict['ab_vol'].append(event_day_ab_vol)

a_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_five_dict.items() })

a_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_five_df['ARs']
X= a_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_five_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Small Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_20_dict = {}
a_post_20_dict['ARs'] =[]
a_post_20_dict['Market_Return'] =[]
a_post_20_dict['beta']=[]
a_post_20_dict['sr_vola']=[]
a_post_20_dict['abs_sr']=[]
a_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_20_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        a_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_20_dict['ab_vol'].append(event_day_ab_vol)

a_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_20_dict.items() })

a_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_20_df['ARs']
X= a_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_20_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Large Threshold / 2-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_two_dict = {}
a_post_two_dict['ARs'] =[]
a_post_two_dict['Market_Return'] =[]
a_post_two_dict['beta']=[]
a_post_two_dict['sr_vola']=[]
a_post_two_dict['abs_sr']=[]
a_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        a_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_two_dict['ab_vol'].append(event_day_ab_vol)

a_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_two_dict.items() })

a_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_two_df['ARs']
X= a_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_two_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Large Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_five_dict = {}
a_post_five_dict['ARs'] =[]
a_post_five_dict['Market_Return'] =[]
a_post_five_dict['beta']=[]
a_post_five_dict['sr_vola']=[]
a_post_five_dict['abs_sr']=[]
a_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_five_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 5
        next_five_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        a_post_five_dict['ARs'].extend(next_five_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_five_dict['ab_vol'].append(event_day_ab_vol)

a_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_five_dict.items() })

a_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_five_df['ARs']
X= a_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_five_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Large Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
a_post_20_dict = {}
a_post_20_dict['ARs'] =[]
a_post_20_dict['Market_Return'] =[]
a_post_20_dict['beta']=[]
a_post_20_dict['sr_vola']=[]
a_post_20_dict['abs_sr']=[]
a_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_a_post[(total_df_a_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_20_days = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_a_post.loc[total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 1: total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        a_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_a_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_a_post.loc[idx_event_day, 'Market_Return_Decrease']
        a_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_a_post.loc[idx_event_day, f'beta_{t}']
        a_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_a_post.loc[idx_event_day, f'sr_vola_{t}']
        a_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_a_post.loc[idx_event_day, f'abs_sr_{t}']
        a_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_a_post.loc[idx_event_day, f'ab_vol_{t}']
        a_post_20_dict['ab_vol'].append(event_day_ab_vol)

a_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in a_post_five_dict.items() })

a_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= a_post_20_df['ARs']
X= a_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/a_post_20_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# ### Proxy B

# %% [markdown]
# #### Small Threshold / 2-days window

# %%
proxy_b_df_post = pd.read_excel(f"{path}/raw_data/post_covid/proxy_b.xlsx")
total_df_b_post = pd.merge(total_variables_post,proxy_b_df_post, on='Date', how='left')

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_two_dict = {}
b_post_two_dict['ARs'] =[]
b_post_two_dict['Market_Return'] =[]
b_post_two_dict['beta']=[]
b_post_two_dict['sr_vola']=[]
b_post_two_dict['abs_sr']=[]
b_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_3std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        b_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_two_dict['ab_vol'].append(event_day_ab_vol)

b_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_two_dict.items() })

b_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_two_df['ARs']
X= b_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_two_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Small Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_five_dict = {}
b_post_five_dict['ARs'] =[]
b_post_five_dict['Market_Return'] =[]
b_post_five_dict['beta']=[]
b_post_five_dict['sr_vola']=[]
b_post_five_dict['abs_sr']=[]
b_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_3std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_5_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        b_post_five_dict['ARs'].extend(next_5_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_five_dict['ab_vol'].append(event_day_ab_vol)

b_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_five_dict.items() })

b_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_five_df['ARs']
X= b_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_five_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Small Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_20_dict = {}
b_post_20_dict['ARs'] =[]
b_post_20_dict['Market_Return'] =[]
b_post_20_dict['beta']=[]
b_post_20_dict['sr_vola']=[]
b_post_20_dict['abs_sr']=[]
b_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_3std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_20_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        b_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_20_dict['ab_vol'].append(event_day_ab_vol)

b_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_20_dict.items() })

b_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_20_df['ARs']
X= b_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_20_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Large Threshold / 2-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_two_dict = {}
b_post_two_dict['ARs'] =[]
b_post_two_dict['Market_Return'] =[]
b_post_two_dict['beta']=[]
b_post_two_dict['sr_vola']=[]
b_post_two_dict['abs_sr']=[]
b_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_4std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        b_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_b_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_two_dict['ab_vol'].append(event_day_ab_vol)

b_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_two_dict.items() })

b_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_two_df['ARs']
X= b_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_two_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# #### Large Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_five_dict = {}
b_post_five_dict['ARs'] =[]
b_post_five_dict['Market_Return'] =[]
b_post_five_dict['beta']=[]
b_post_five_dict['sr_vola']=[]
b_post_five_dict['abs_sr']=[]
b_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_4std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_five_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 5
        next_five_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        b_post_five_dict['ARs'].extend(next_five_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_five_dict['ab_vol'].append(event_day_ab_vol)

b_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_five_dict.items() })

b_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_five_df['ARs']
X= b_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_five_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# #### Large Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
b_post_20_dict = {}
b_post_20_dict['ARs'] =[]
b_post_20_dict['Market_Return'] =[]
b_post_20_dict['beta']=[]
b_post_20_dict['sr_vola']=[]
b_post_20_dict['abs_sr']=[]
b_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_b_post[(total_df_b_post[f'{t}_Decrease_4std'] == 1) & (total_df_b_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_20_days = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_b_post.loc[total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 1: total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        b_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_b_post.index[total_df_b_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_b_post.loc[idx_event_day, 'Market_Return_Decrease']
        b_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_b_post.loc[idx_event_day, f'beta_{t}']
        b_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_b_post.loc[idx_event_day, f'sr_vola_{t}']
        b_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_b_post.loc[idx_event_day, f'abs_sr_{t}']
        b_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_b_post.loc[idx_event_day, f'ab_vol_{t}']
        b_post_20_dict['ab_vol'].append(event_day_ab_vol)

b_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in b_post_five_dict.items() })

b_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= b_post_20_df['ARs']
X= b_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/b_post_20_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# ### Proxy C

# %% [markdown]
# #### Small Threshold / 2-days window

# %%
proxy_c_df_post = pd.read_excel(f"{path}/raw_data/post_covid/proxy_c.xlsx")
total_df_c_post = pd.merge(total_variables_post,proxy_c_df_post, on='Date', how='left')

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_two_dict = {}
c_post_two_dict['ARs'] =[]
c_post_two_dict['Market_Return'] =[]
c_post_two_dict['beta']=[]
c_post_two_dict['sr_vola']=[]
c_post_two_dict['abs_sr']=[]
c_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        c_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_two_dict['ab_vol'].append(event_day_ab_vol)

c_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_two_dict.items() })

c_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_two_df['ARs']
X= c_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_two_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# #### Small Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_five_dict = {}
c_post_five_dict['ARs'] =[]
c_post_five_dict['Market_Return'] =[]
c_post_five_dict['beta']=[]
c_post_five_dict['sr_vola']=[]
c_post_five_dict['abs_sr']=[]
c_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_5_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        c_post_five_dict['ARs'].extend(next_5_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_five_dict['ab_vol'].append(event_day_ab_vol)

c_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_five_dict.items() })

c_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_five_df['ARs']
X= c_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_five_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Small Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_20_dict = {}
c_post_20_dict['ARs'] =[]
c_post_20_dict['Market_Return'] =[]
c_post_20_dict['beta']=[]
c_post_20_dict['sr_vola']=[]
c_post_20_dict['abs_sr']=[]
c_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_small_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_5_days
        idx_next_20_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        c_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_5_days_returns)

        idx_event_day = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_20_dict['ab_vol'].append(event_day_ab_vol)

c_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_20_dict.items() })

c_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_20_df['ARs']
X= c_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_20_df_small.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()

# %% [markdown]
# #### Large Threshold / 2-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_two_dict = {}
c_post_two_dict['ARs'] =[]
c_post_two_dict['Market_Return'] =[]
c_post_two_dict['beta']=[]
c_post_two_dict['sr_vola']=[]
c_post_two_dict['abs_sr']=[]
c_post_two_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_2_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 2, f'{t}_ARs']
        c_post_two_dict['ARs'].extend(next_2_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_c_post.index[total_df_a_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_two_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_two_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_two_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_two_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_two_dict['ab_vol'].append(event_day_ab_vol)

c_post_two_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_two_dict.items() })

c_post_two_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_two_df['ARs']
X= c_post_two_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_two_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# #### Large Threshold / 5-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_five_dict = {}
c_post_five_dict['ARs'] =[]
c_post_five_dict['Market_Return'] =[]
c_post_five_dict['beta']=[]
c_post_five_dict['sr_vola']=[]
c_post_five_dict['abs_sr']=[]
c_post_five_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_five_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 5
        next_five_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 5, f'{t}_ARs']
        c_post_five_dict['ARs'].extend(next_five_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_five_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_five_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_five_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_five_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_five_dict['ab_vol'].append(event_day_ab_vol)

c_post_five_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_five_dict.items() })

c_post_five_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_five_df['ARs']
X= c_post_five_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_five_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# #### Large Threshold / 20-days window

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
c_post_20_dict = {}
c_post_20_dict['ARs'] =[]
c_post_20_dict['Market_Return'] =[]
c_post_20_dict['beta']=[]
c_post_20_dict['sr_vola']=[]
c_post_20_dict['abs_sr']=[]
c_post_20_dict['ab_vol']=[]
for t in tickers_list:

    rows_with_condition = total_df_c_post[(total_df_c_post[f'{t}_Decrease_large_thres'] == 1) & (total_df_c_post['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']
        # Index_next_2_days
        idx_next_20_days = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns =total_df_c_post.loc[total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 1: total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0] + 20, f'{t}_ARs']
        c_post_20_dict['ARs'].extend(next_20_days_returns.values)
        # print(next_2_days_returns)

        idx_event_day = total_df_c_post.index[total_df_c_post['Date'] == event_date].to_numpy()[0]
        event_day_mr = total_df_c_post.loc[idx_event_day, 'Market_Return_Decrease']
        c_post_20_dict['Market_Return'].append(event_day_mr)

        event_day_beta = total_df_c_post.loc[idx_event_day, f'beta_{t}']
        c_post_20_dict['beta'].append(event_day_beta)

        event_day_sr_vola = total_df_c_post.loc[idx_event_day, f'sr_vola_{t}']
        c_post_20_dict['sr_vola'].append(event_day_sr_vola)

        event_day_abs_sr = total_df_c_post.loc[idx_event_day, f'abs_sr_{t}']
        c_post_20_dict['abs_sr'].append(event_day_abs_sr)

        event_day_ab_vol = total_df_c_post.loc[idx_event_day, f'ab_vol_{t}']
        c_post_20_dict['ab_vol'].append(event_day_ab_vol)

c_post_20_df = pd.DataFrame({ key:pd.Series(value) for key, value in c_post_five_dict.items() })

c_post_20_df.dropna(inplace=True)

independent_vars=['Market_Return', 'beta', 'sr_vola', 'abs_sr', 'ab_vol']

y= c_post_20_df['ARs']
X= c_post_20_df[[f'{var}' for var in independent_vars]]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
model_summary = model.summary()
string_buf = StringIO()
string_buf.write(model_summary.as_text())

save_path = f'{path}/raw_data/post_covid/c_post_20_df_large.txt'

with open(save_path, 'w') as file:
    file.write(string_buf.getvalue())

# Close the buffer
string_buf.close()


# %% [markdown]
# ## another way to select data
#

# %%
tickers_total= pd.read_excel(f"{path}/raw_data/main/Weightings_20240216_OMXSPI.xlsx",header=0)
tickers_list=tickers_total['Security-Symbol'].to_list()
tickers_list.remove('NOKIA-SEK.ST')
resultados = pd.DataFrame()

# Iterar sobre cada empostsa
for company in tickers_list:
    # Seleccionar filas para la empostsa actual donde proxy_a_{empostsa} y mercado_decrease son ambos 1
    selection = total_df_a_post[(total_df_a_post[f'{company}_Decrease_small_thres'] == 1) & (total_df_a_post['Market_Return_Decrease'] == 1)]

    # Si hay al menos una fila seleccionada para esta empostsa
    if not selection.empty:
        # Obtener los índices de las filas seleccionadas
        indices_seleccionados = selection.index

        # Iterar sobre los índices seleccionados
        for indice in indices_seleccionados:
            # Obtener las dos siguientes filas
            filas_siguientes = total_df_a_post.iloc[indice+1:indice+3]

            # Concatenar las filas seleccionadas al DataFrame de resultados
            resultados = pd.concat([resultados, filas_siguientes])
    # else:
    #     # Si no hay filas seleccionadas para esta empostsa, agregar filas con valores nulos al DataFrame de resultados
    #     n_filas_nulas = pd.DataFrame(index=range(2), columns=total_df_a_post.columns)
    #     resultados = pd.concat([resultados, n_filas_nulas])

# Mostrar los resultados
resultados.to_excel(f'{path}/raw_data/post_covid/resultados.xlsx')


# %%
resultados=pd.read_excel(f'{path}/raw_data/post_covid/resultados.xlsx')
exog = resultados[['Market_Return_Increase_y'] + [f'beta_{company}'  for company in tickers_list]
                  + [f'sr_vola_{company}'  for company in tickers_list]
                  + [f'abs_sr_{company}'  for company in tickers_list]
                  + [f'ab_vol_{company}'  for company in tickers_list]]

exog = exog.fillna(0)  # Drop rows with missing values
print(exog)

# Selecting the endogenous variable
endog = resultados[[f'{company}_ARs' for company in tickers_list]]

# PanelOLS regression
# model = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
# results = model.fit()
exog = sm.add_constant(exog)
model = sm.OLS(endog, exog).fit()
print(model.summary())

# Access coefficients and significance
print("Results for the entire sample:")
print(results.summary)

# %%
resultados['beta'] = resultados[[f'beta_{t}' for t in tickers_list]].mean(axis=1)
resultados['sr_vola'] = resultados[[f'sr_vola_{t}' for t in tickers_list]].mean(axis=1)
resultados['abs_sr'] = resultados[[f'abs_sr_{t}' for t in tickers_list]].mean(axis=1)
resultados['ab_vol'] = resultados[[f'ab_vol_{t}' for t in tickers_list]].mean(axis=1)
resultados['ARs'] = resultados[[f'{t}_ARs' for t in tickers_list]].mean(axis=1)

# %%
exog = resultados[['Market_Return_Increase_y','sr_vola','abs_sr','ab_vol']]

exog = exog.fillna(0)  # Drop rows with missing values
# print(exog)

# Selecting the endogenous variable
endog = resultados['ARs']


model = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
results = model.fit()

# %%
X = resultados[['beta', 'sr_vola', 'abs_sr', 'ab_vol']]
y = resultados['ARs']  # Assuming 'ARs' is your dependent variable

# Adding intercept term
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# %%
