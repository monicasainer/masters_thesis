# %%
import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
from scipy.stats import ttest_1samp


# %%
warnings.filterwarnings('ignore')
path = os.getenv("ROOT_PATH")
sys.path.append(path)
print(path)


# %% [markdown]
# ## Statistical Significance. Table 1. Proxies A and B

# %% [markdown]
# The first step is creating a table with the daily ARs calculated using CAPM for each company

# %%
window_size = 250
risk_free_rate_df= pd.read_excel(f"{path}/raw_data/pre_covid/risk_free.xlsx")
risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)'].fillna(method='ffill', inplace=True)
risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)']= (1 + risk_free_rate_df['Swedish Treasury Bills (SE TB 1 Month)']) ** (1/250) - 1
df_price_2013 = pd.read_excel(f"{path}/raw_data/pre_covid/price_2013.xlsx")

ARs_df = pd.merge(df_price_2013,risk_free_rate_df,left_on='Date',right_on='Date',how='left')
ARs_df['market_returns'] = df_price_2013['^OMXSPI'].pct_change()
ARs_df.drop(columns='^OMXSPI',inplace=True)


# %%
ARs_df.head()


# %%
for column in df_price_2013.columns[1:-1]:  # Exclude the 'Date' and '^OMXSPI' column

    stock_returns = ARs_df[column].pct_change()

    result_df = pd.DataFrame({
    'Date': ARs_df['Date'],
    'Stock_Returns': stock_returns,
    'Market_Returns': ARs_df['market_returns']})

    result_df['beta'] = result_df['Stock_Returns'].rolling(window=window_size).cov(result_df['Market_Returns']).div(result_df['Market_Returns'].rolling(window=window_size).var())
    result_df = pd.merge(result_df,ARs_df[['Date','Swedish Treasury Bills (SE TB 1 Month)']], left_on='Date',right_on='Date', how='left')

    # Ri = Rf + beta * (Rm-Rf) + ei --> Ri - [Rf + beta * (Rm - Rf)]
    result_df['MMAR'] = result_df['Swedish Treasury Bills (SE TB 1 Month)']+ result_df['beta'] * (result_df['Market_Returns'] - result_df['Swedish Treasury Bills (SE TB 1 Month)'])

    ARs_df[f'{column}_ARs'] = result_df['Stock_Returns'] - result_df['MMAR']
    ARs_df.drop(columns=f'{column}',inplace=True)


# %%
ARs_df.to_excel(f'{path}/raw_data/pre_covid/ARs_df.xlsx')


# %% [markdown]
# ### **Proxy A :**
#
# Abnormal stock returns following large stock price increases and decreases

# %%
proxy_a_df = pd.read_excel(f'{path}/raw_data/pre_covid/proxy_a.xlsx')
df_price = pd.read_excel (f'{path}/raw_data/pre_covid/price.xlsx')
df_price.drop(columns='NOKIA-SEK.ST', inplace=True)
total_a_df = pd.merge(proxy_a_df,ARs_df,on='Date',how='left')


# %%
total_a_df.head()


# %% [markdown]
# #### Price increases:

# %% [markdown]
# ##### Testing events individually:

# %%

# Create a dictionary to store the result for each threshold
results_dict_a_eight_increase = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_a_ten_increase = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for 8% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Increase_small_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

        # Average_next_day
            avg_next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0, alternative='two-sided').pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0, alternative='two-sided').pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0, alternative='two-sided').pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0, alternative='two-sided').pvalue


        results_dict_a_eight_increase['Company'].append(i)
        results_dict_a_eight_increase['Event_Date'].append(event_date)
        results_dict_a_eight_increase['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_eight_increase['P_Value_1_Day'].append(p_value_1_day)
        results_dict_a_eight_increase['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_eight_increase['P_Value_2_Days'].append(p_value_2_days)
        results_dict_a_eight_increase['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_eight_increase['P_Value_5_Days'].append(p_value_5_days)
        results_dict_a_eight_increase['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_a_eight_increase['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_a_eight_increase = pd.DataFrame(results_dict_a_eight_increase)


# Calculate average ARs and p-values for 10% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Increase_large_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

        # Average_next_day
            avg_next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0, alternative='two-sided').pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0, alternative='two-sided').pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0, alternative='two-sided').pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0, alternative='two-sided').pvalue

        results_dict_a_ten_increase['Company'].append(i)
        results_dict_a_ten_increase['Event_Date'].append(event_date)
        results_dict_a_ten_increase['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_ten_increase['P_Value_1_Day'].append(p_value_1_day)
        results_dict_a_ten_increase['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_ten_increase['P_Value_2_Days'].append(p_value_2_days)
        results_dict_a_ten_increase['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_ten_increase['P_Value_5_Days'].append(p_value_5_days)
        results_dict_a_ten_increase['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_a_ten_increase['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_a_ten_increase = pd.DataFrame(results_dict_a_ten_increase)


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold 8%
results_dict_a_eight_increase_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 8% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Increase_small_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

            # Average_next_day
            next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_a_eight_increase_alt['Company'].append(i)
        results_dict_a_eight_increase_alt['Event_Date'].append(event_date)
        results_dict_a_eight_increase_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_eight_increase_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_eight_increase_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_eight_increase_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_a_eight_increase_alt = pd.DataFrame(results_dict_a_eight_increase_alt)

results_significance_a_eight_increase_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_a_eight_increase_alt_result['Avg_ARs_%'].append(results_significance_a_eight_increase_alt['Avg_Next_Day_Returns'].mean())
results_significance_a_eight_increase_alt_result['Avg_ARs_%'].append(results_significance_a_eight_increase_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_a_eight_increase_alt_result['Avg_ARs_%'].append(results_significance_a_eight_increase_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_a_eight_increase_alt_result['Avg_ARs_%'].append(results_significance_a_eight_increase_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_increase_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_increase_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_increase_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_increase_alt_result['p_value'].append(result_20)
results_significance_a_eight_increase_alt_result=pd.DataFrame(results_significance_a_eight_increase_alt_result)

# Create a dictionary to store the result for threshold 10%

results_dict_a_ten_increase_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 10% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Increase_large_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

            # Average_next_day
            next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_a_ten_increase_alt['Company'].append(i)
        results_dict_a_ten_increase_alt['Event_Date'].append(event_date)
        results_dict_a_ten_increase_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_ten_increase_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_ten_increase_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_ten_increase_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_a_ten_increase_alt = pd.DataFrame(results_dict_a_ten_increase_alt)

results_significance_a_ten_increase_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_a_ten_increase_alt_result['Avg_ARs_%'].append(results_significance_a_ten_increase_alt['Avg_Next_Day_Returns'].mean())
results_significance_a_ten_increase_alt_result['Avg_ARs_%'].append(results_significance_a_ten_increase_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_a_ten_increase_alt_result['Avg_ARs_%'].append(results_significance_a_ten_increase_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_a_ten_increase_alt_result['Avg_ARs_%'].append(results_significance_a_ten_increase_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_increase_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_increase_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_increase_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_increase_alt_result['p_value'].append(result_20)
results_significance_a_ten_increase_alt_result=pd.DataFrame(results_significance_a_ten_increase_alt_result)
print(results_significance_a_eight_increase_alt_result)
print(results_significance_a_ten_increase_alt_result)


# %% [markdown]
# #### Price decreases:

# %% [markdown]
# ##### Testing events individually:

# %%
# Create a dictionary to store the result for each threshold
results_dict_a_eight_decrease = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_a_ten_decrease = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for 8% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Decrease_small_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

        # Average_next_day
            avg_next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue



        results_dict_a_eight_decrease['Company'].append(i)
        results_dict_a_eight_decrease['Event_Date'].append(event_date)
        results_dict_a_eight_decrease['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_eight_decrease['P_Value_1_Day'].append(p_value_1_day)
        results_dict_a_eight_decrease['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_eight_decrease['P_Value_2_Days'].append(p_value_2_days)
        results_dict_a_eight_decrease['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_eight_decrease['P_Value_5_Days'].append(p_value_5_days)
        results_dict_a_eight_decrease['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_a_eight_decrease['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_a_eight_decrease = pd.DataFrame(results_dict_a_eight_decrease)


# Calculate average ARs and p-values for 10% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Decrease_large_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

        # Average_next_day
            avg_next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue

        results_dict_a_ten_decrease['Company'].append(i)
        results_dict_a_ten_decrease['Event_Date'].append(event_date)
        results_dict_a_ten_decrease['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_ten_decrease['P_Value_1_Day'].append(p_value_1_day)
        results_dict_a_ten_decrease['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_ten_decrease['P_Value_2_Days'].append(p_value_2_days)
        results_dict_a_ten_decrease['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_ten_decrease['P_Value_5_Days'].append(p_value_5_days)
        results_dict_a_ten_decrease['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_a_ten_decrease['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_a_ten_decrease = pd.DataFrame(results_dict_a_ten_decrease)


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold 8%
results_dict_a_eight_decrease_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 8% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Decrease_small_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

            # Average_next_day
            next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_a_eight_decrease_alt['Company'].append(i)
        results_dict_a_eight_decrease_alt['Event_Date'].append(event_date)
        results_dict_a_eight_decrease_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_eight_decrease_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_eight_decrease_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_eight_decrease_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_a_eight_decrease_alt = pd.DataFrame(results_dict_a_eight_decrease_alt)

results_significance_a_eight_decrease_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_a_eight_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_eight_decrease_alt['Avg_Next_Day_Returns'].mean())
results_significance_a_eight_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_eight_decrease_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_a_eight_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_eight_decrease_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_a_eight_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_eight_decrease_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_decrease_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_decrease_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_decrease_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_eight_decrease_alt_result['p_value'].append(result_20)
results_significance_a_eight_decrease_alt_result=pd.DataFrame(results_significance_a_eight_decrease_alt_result)

# Create a dictionary to store the result for threshold 10%

results_dict_a_ten_decrease_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 10% threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_a_df[total_a_df[f'{i}_Decrease_large_thres'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_a_df.loc[total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 1: total_a_df.index[total_a_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_a_df) and idx_next_2_days < len(total_a_df) and \
           idx_next_5_days < len(total_a_df) and idx_next_20_days < len(total_a_df):

            # Average_next_day
            next_day_returns = total_a_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_a_ten_decrease_alt['Company'].append(i)
        results_dict_a_ten_decrease_alt['Event_Date'].append(event_date)
        results_dict_a_ten_decrease_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_a_ten_decrease_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_a_ten_decrease_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_a_ten_decrease_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_a_ten_decrease_alt = pd.DataFrame(results_dict_a_ten_decrease_alt)

results_significance_a_ten_decrease_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_a_ten_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_ten_decrease_alt['Avg_Next_Day_Returns'].mean())
results_significance_a_ten_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_ten_decrease_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_a_ten_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_ten_decrease_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_a_ten_decrease_alt_result['Avg_ARs_%'].append(results_significance_a_ten_decrease_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_decrease_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_decrease_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_decrease_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_a_ten_decrease_alt_result['p_value'].append(result_20)
results_significance_a_ten_decrease_alt_result=pd.DataFrame(results_significance_a_ten_decrease_alt_result)
print(results_significance_a_eight_decrease_alt_result.head())
print(results_significance_a_ten_decrease_alt_result.head())


# %% [markdown]
# ## Proxy B:
#
# Daily raw stock returns with absolute values exceeding 3 and 4 standard deviations.

# %%
proxy_b_df = pd.read_excel(f'{path}/raw_data/pre_covid/proxy_b.xlsx')
df_price = pd.read_excel (f'{path}/raw_data/pre_covid/price.xlsx')
total_b_df = pd.merge(proxy_b_df,ARs_df,on='Date',how='left')


# %% [markdown]
# #### Price increases:

# %% [markdown]
# ##### Testing events individually:

# %%
# Create a dictionary to store the result for each threshold
results_dict_three_increase = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_four_increase = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for 3 std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Increase_3std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

        # Average_next_day
            avg_next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue

        results_dict_three_increase['Company'].append(i)
        results_dict_three_increase['Event_Date'].append(event_date)
        results_dict_three_increase['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_three_increase['P_Value_1_Day'].append(p_value_1_day)
        results_dict_three_increase['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_three_increase['P_Value_2_Days'].append(p_value_2_days)
        results_dict_three_increase['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_three_increase['P_Value_5_Days'].append(p_value_5_days)
        results_dict_three_increase['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_three_increase['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_b_three_increase = pd.DataFrame(results_dict_three_increase)


# Calculate average ARs and p-values for 4 std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition =total_b_df[total_b_df[f'{i}_Increase_4std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

        # Average_next_day
            avg_next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue

        results_dict_four_increase['Company'].append(i)
        results_dict_four_increase['Event_Date'].append(event_date)
        results_dict_four_increase['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_four_increase['P_Value_1_Day'].append(p_value_1_day)
        results_dict_four_increase['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_four_increase['P_Value_2_Days'].append(p_value_2_days)
        results_dict_four_increase['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_four_increase['P_Value_5_Days'].append(p_value_5_days)
        results_dict_four_increase['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_four_increase['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_b_four_increase = pd.DataFrame(results_dict_four_increase)


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold 3std
results_dict_b_three_increase_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 3std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Increase_3std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

            # Average_next_day
            next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_b_three_increase_alt['Company'].append(i)
        results_dict_b_three_increase_alt['Event_Date'].append(event_date)
        results_dict_b_three_increase_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_b_three_increase_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_b_three_increase_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_b_three_increase_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_b_three_increase_alt = pd.DataFrame(results_dict_b_three_increase_alt)

results_significance_b_three_increase_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_b_three_increase_alt_result['Avg_ARs_%'].append(results_significance_b_three_increase_alt['Avg_Next_Day_Returns'].mean())
results_significance_b_three_increase_alt_result['Avg_ARs_%'].append(results_significance_b_three_increase_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_b_three_increase_alt_result['Avg_ARs_%'].append(results_significance_b_three_increase_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_b_three_increase_alt_result['Avg_ARs_%'].append(results_significance_b_three_increase_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_increase_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_increase_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_increase_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_increase_alt_result['p_value'].append(result_20)
results_significance_b_three_increase_alt_result=pd.DataFrame(results_significance_b_three_increase_alt_result)

# Create a dictionary to store the result for threshold 4std
results_dict_b_four_increase_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 4std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Increase_4std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

            # Average_next_day
            next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_b_four_increase_alt['Company'].append(i)
        results_dict_b_four_increase_alt['Event_Date'].append(event_date)
        results_dict_b_four_increase_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_b_four_increase_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_b_four_increase_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_b_four_increase_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_b_four_increase_alt = pd.DataFrame(results_dict_b_four_increase_alt)

results_significance_b_four_increase_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_b_four_increase_alt_result['Avg_ARs_%'].append(results_significance_b_four_increase_alt['Avg_Next_Day_Returns'].mean())
results_significance_b_four_increase_alt_result['Avg_ARs_%'].append(results_significance_b_four_increase_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_b_four_increase_alt_result['Avg_ARs_%'].append(results_significance_b_four_increase_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_b_four_increase_alt_result['Avg_ARs_%'].append(results_significance_b_four_increase_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_increase_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_increase_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_increase_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_increase_alt_result['p_value'].append(result_20)
results_significance_b_four_increase_alt_result=pd.DataFrame(results_significance_b_four_increase_alt_result)
print(results_significance_b_three_increase_alt_result.head())
print(results_significance_b_four_increase_alt_result.head())


# %% [markdown]
# #### Price decreases:

# %% [markdown]
# ##### Testing events individually:

# %%
# Create a dictionary to store the result for each threshold
results_dict_three_decrease = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_four_decrease = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for 3std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Decrease_3std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

        # Average_next_day
            avg_next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue

        results_dict_three_decrease['Company'].append(i)
        results_dict_three_decrease['Event_Date'].append(event_date)
        results_dict_three_decrease['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_three_decrease['P_Value_1_Day'].append(p_value_1_day)
        results_dict_three_decrease['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_three_decrease['P_Value_2_Days'].append(p_value_2_days)
        results_dict_three_decrease['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_three_decrease['P_Value_5_Days'].append(p_value_5_days)
        results_dict_three_decrease['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_three_decrease['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_b_three_decrease = pd.DataFrame(results_dict_three_decrease)


# Calculate average ARs and p-values for 4std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Decrease_4std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

        # Average_next_day
            avg_next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            p_value_1_day = ttest_1samp(avg_next_day_returns,0).pvalue


            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()
            p_value_2_days = ttest_1samp(next_2_days_returns, 0).pvalue

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()
            p_value_5_days = ttest_1samp(next_5_days_returns, 0).pvalue

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()
            p_value_20_days = ttest_1samp(next_20_days_returns, 0).pvalue


        results_dict_four_decrease['Company'].append(i)
        results_dict_four_decrease['Event_Date'].append(event_date)
        results_dict_four_decrease['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_four_decrease['P_Value_1_Day'].append(p_value_1_day)
        results_dict_four_decrease['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_four_decrease['P_Value_2_Days'].append(p_value_2_days)
        results_dict_four_decrease['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_four_decrease['P_Value_5_Days'].append(p_value_5_days)
        results_dict_four_decrease['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_four_decrease['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_b_four_decrease = pd.DataFrame(results_dict_four_decrease)


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold 3std
results_dict_b_three_decrease_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 3std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Decrease_3std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

            # Average_next_day
            next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_b_three_decrease_alt['Company'].append(i)
        results_dict_b_three_decrease_alt['Event_Date'].append(event_date)
        results_dict_b_three_decrease_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_b_three_decrease_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_b_three_decrease_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_b_three_decrease_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_b_three_decrease_alt = pd.DataFrame(results_dict_b_three_decrease_alt)

results_significance_b_three_decrease_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_b_three_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_three_decrease_alt['Avg_Next_Day_Returns'].mean())
results_significance_b_three_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_three_decrease_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_b_three_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_three_decrease_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_b_three_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_three_decrease_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_decrease_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_decrease_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_decrease_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_three_decrease_alt_result['p_value'].append(result_20)
results_significance_b_three_decrease_alt_result=pd.DataFrame(results_significance_b_three_decrease_alt_result)

# Create a dictionary to store the result for threshold 4std
results_dict_b_four_decrease_alt = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for 4std threshold
for i in df_price.columns[1:-1]:

    rows_with_condition = total_b_df[total_b_df[f'{i}_Decrease_4std'] == 1]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_b_df.loc[total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 1: total_b_df.index[total_b_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_b_df) and idx_next_2_days < len(total_b_df) and \
           idx_next_5_days < len(total_b_df) and idx_next_20_days < len(total_b_df):

            # Average_next_day
            next_day_returns = total_b_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_b_four_decrease_alt['Company'].append(i)
        results_dict_b_four_decrease_alt['Event_Date'].append(event_date)
        results_dict_b_four_decrease_alt['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_b_four_decrease_alt['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_b_four_decrease_alt['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_b_four_decrease_alt['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_b_four_decrease_alt = pd.DataFrame(results_dict_b_four_decrease_alt)

results_significance_b_four_decrease_alt_result = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_b_four_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_four_decrease_alt['Avg_Next_Day_Returns'].mean())
results_significance_b_four_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_four_decrease_alt['Avg_Next_2_Days_Returns'].mean())
results_significance_b_four_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_four_decrease_alt['Avg_Next_5_Days_Returns'].mean())
results_significance_b_four_decrease_alt_result['Avg_ARs_%'].append(results_significance_b_four_decrease_alt['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_decrease_alt_result['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_decrease_alt_result['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_decrease_alt_result['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_b_four_decrease_alt_result['p_value'].append(result_20)
results_significance_b_four_decrease_alt_result=pd.DataFrame(results_significance_b_four_decrease_alt_result)
print(results_significance_b_three_decrease_alt_result.head())
print(results_significance_b_four_decrease_alt_result.head())


# %%
results_significance_a_eight_increase.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_eight_increase.xlsx')
results_significance_a_eight_decrease.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_eight_decrease.xlsx')
results_significance_a_ten_increase.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_ten_increase.xlsx')
results_significance_a_ten_decrease.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_ten_decrease.xlsx')

results_significance_b_three_increase.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_three_increase.xlsx')
results_significance_b_three_decrease.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_three_decrease.xlsx')
results_significance_b_four_increase.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_four_increase.xlsx')
results_significance_b_four_decrease.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_four_decrease.xlsx')


# %%
results_significance_a_eight_increase_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_eight_increase_alt_result.xlsx')
results_significance_a_eight_decrease_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_eight_decrease_alt_result.xlsx')
results_significance_a_ten_increase_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_ten_increase_alt_result.xlsx')
results_significance_a_ten_decrease_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_a_ten_decrease_alt_result.xlsx')

results_significance_b_three_increase_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_three_increase_alt_result.xlsx')
results_significance_b_three_decrease_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_three_decrease_alt_result.xlsx')
results_significance_b_four_increase_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_four_increase_alt_result.xlsx')
results_significance_b_four_decrease_alt_result.to_excel(f'{path}/raw_data/pre_covid/t1_results_significance_b_four_decrease_alt_result.xlsx')


# %%
