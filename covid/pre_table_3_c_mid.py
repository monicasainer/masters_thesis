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
# ## Statistical Significance - Table 3c - Large

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


# %% [markdown]
# ### **Proxy C :**
#
# Daily abnormal stock returns with absolute values exceeding small_thres and large_thres using Market Model Adjusted Returns by market sign and market capitalization.

# %%
proxy_c_df = pd.read_excel(f'{path}/raw_data/pre_covid/proxy_c.xlsx')
df_mid = pd.read_excel (f'{path}/raw_data/pre_covid/mid_caps.xlsx')
total_c_df = pd.merge(proxy_c_df,ARs_df,on='Date',how='left')


# %%
total_c_df.head()


# %% [markdown]
# #### Price increases with market increases:

# %% [markdown]
# ##### Testing events individually:

# %%

# Create a dictionary to store the result for each threshold
results_dict_c_eight_increase_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_c_ten_increase_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_small_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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


        results_dict_c_eight_increase_m['Company'].append(i)
        results_dict_c_eight_increase_m['Event_Date'].append(event_date)
        results_dict_c_eight_increase_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_increase_m['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_eight_increase_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_increase_m['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_eight_increase_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_increase_m['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_eight_increase_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_eight_increase_m['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_eight_increase_m = pd.DataFrame(results_dict_c_eight_increase_m)


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_large_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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

        results_dict_c_ten_increase_m['Company'].append(i)
        results_dict_c_ten_increase_m['Event_Date'].append(event_date)
        results_dict_c_ten_increase_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_increase_m['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_ten_increase_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_increase_m['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_ten_increase_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_increase_m['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_ten_increase_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_ten_increase_m['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_ten_increase_m = pd.DataFrame(results_dict_c_ten_increase_m)
print(results_significance_c_eight_increase_m.head())
print(results_significance_c_ten_increase_m.head())


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold small_thres
results_dict_c_eight_increase_alt_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_small_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_eight_increase_alt_m['Company'].append(i)
        results_dict_c_eight_increase_alt_m['Event_Date'].append(event_date)
        results_dict_c_eight_increase_alt_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_increase_alt_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_increase_alt_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_increase_alt_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_eight_increase_alt_m = pd.DataFrame(results_dict_c_eight_increase_alt_m)

results_significance_c_eight_increase_alt_result_m = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_eight_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m['Avg_Next_Day_Returns'].mean())
results_significance_c_eight_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m['Avg_Next_2_Days_Returns'].mean())
results_significance_c_eight_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m['Avg_Next_5_Days_Returns'].mean())
results_significance_c_eight_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m['p_value'].append(result_20)
results_significance_c_eight_increase_alt_result_m=pd.DataFrame(results_significance_c_eight_increase_alt_result_m)

# Create a dictionary to store the result for threshold large_thres

results_dict_c_ten_increase_alt_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_large_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_ten_increase_alt_m['Company'].append(i)
        results_dict_c_ten_increase_alt_m['Event_Date'].append(event_date)
        results_dict_c_ten_increase_alt_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_increase_alt_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_increase_alt_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_increase_alt_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_ten_increase_alt_m = pd.DataFrame(results_dict_c_ten_increase_alt_m)

results_significance_c_ten_increase_alt_result_m = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_ten_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m['Avg_Next_Day_Returns'].mean())
results_significance_c_ten_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m['Avg_Next_2_Days_Returns'].mean())
results_significance_c_ten_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m['Avg_Next_5_Days_Returns'].mean())
results_significance_c_ten_increase_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m['p_value'].append(result_20)
results_significance_c_ten_increase_alt_result_m=pd.DataFrame(results_significance_c_ten_increase_alt_result_m)

print(results_significance_c_eight_increase_alt_result_m.head())
print(results_significance_c_ten_increase_alt_result_m.head())


# %% [markdown]
# #### Price increases with market decreases:

# %% [markdown]
# ##### Testing events individually:

# %%

# Create a dictionary to store the result for each threshold
results_dict_c_eight_increase_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_c_ten_increase_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_small_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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


        results_dict_c_eight_increase_m_dec['Company'].append(i)
        results_dict_c_eight_increase_m_dec['Event_Date'].append(event_date)
        results_dict_c_eight_increase_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_increase_m_dec['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_eight_increase_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_increase_m_dec['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_eight_increase_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_increase_m_dec['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_eight_increase_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_eight_increase_m_dec['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_eight_increase_m_dec = pd.DataFrame(results_dict_c_eight_increase_m_dec)


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_large_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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

        results_dict_c_ten_increase_m_dec['Company'].append(i)
        results_dict_c_ten_increase_m_dec['Event_Date'].append(event_date)
        results_dict_c_ten_increase_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_increase_m_dec['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_ten_increase_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_increase_m_dec['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_ten_increase_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_increase_m_dec['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_ten_increase_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_ten_increase_m_dec['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_ten_increase_m_dec = pd.DataFrame(results_dict_c_ten_increase_m_dec)
print(results_significance_c_eight_increase_m_dec.head())
print(results_significance_c_ten_increase_m_dec.head())


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold small_thres
results_dict_c_eight_increase_alt_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_small_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_eight_increase_alt_m_dec['Company'].append(i)
        results_dict_c_eight_increase_alt_m_dec['Event_Date'].append(event_date)
        results_dict_c_eight_increase_alt_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_increase_alt_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_increase_alt_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_increase_alt_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_eight_increase_alt_m_dec = pd.DataFrame(results_dict_c_eight_increase_alt_m_dec)

results_significance_c_eight_increase_alt_result_m_dec = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_eight_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m_dec['Avg_Next_Day_Returns'].mean())
results_significance_c_eight_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m_dec['Avg_Next_2_Days_Returns'].mean())
results_significance_c_eight_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m_dec['Avg_Next_5_Days_Returns'].mean())
results_significance_c_eight_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_increase_alt_m_dec['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m_dec['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m_dec['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m_dec['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_increase_alt_result_m_dec['p_value'].append(result_20)
results_significance_c_eight_increase_alt_result_m_dec=pd.DataFrame(results_significance_c_eight_increase_alt_result_m_dec)

# Create a dictionary to store the result for threshold large_thres

results_dict_c_ten_increase_alt_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Increase_large_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_ten_increase_alt_m_dec['Company'].append(i)
        results_dict_c_ten_increase_alt_m_dec['Event_Date'].append(event_date)
        results_dict_c_ten_increase_alt_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_increase_alt_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_increase_alt_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_increase_alt_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_ten_increase_alt_m_dec = pd.DataFrame(results_dict_c_ten_increase_alt_m_dec)

results_significance_c_ten_increase_alt_result_m_dec = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_ten_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m_dec['Avg_Next_Day_Returns'].mean())
results_significance_c_ten_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m_dec['Avg_Next_2_Days_Returns'].mean())
results_significance_c_ten_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m_dec['Avg_Next_5_Days_Returns'].mean())
results_significance_c_ten_increase_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_increase_alt_m_dec['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m_dec['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m_dec['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m_dec['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_increase_alt_result_m_dec['p_value'].append(result_20)
results_significance_c_ten_increase_alt_result_m_dec=pd.DataFrame(results_significance_c_ten_increase_alt_result_m_dec)

print(results_significance_c_eight_increase_alt_result_m_dec.head())
print(results_significance_c_ten_increase_alt_result_m_dec.head())


# %% [markdown]
# #### Price decreases with market increases:

# %% [markdown]
# ##### Testing events individually:

# %%

# Create a dictionary to store the result for each threshold
results_dict_c_eight_decrease_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_c_ten_decrease_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_small_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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


        results_dict_c_eight_decrease_m['Company'].append(i)
        results_dict_c_eight_decrease_m['Event_Date'].append(event_date)
        results_dict_c_eight_decrease_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_decrease_m['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_eight_decrease_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_decrease_m['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_eight_decrease_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_decrease_m['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_eight_decrease_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_eight_decrease_m['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_eight_decrease_m = pd.DataFrame(results_dict_c_eight_decrease_m)


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_large_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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

        results_dict_c_ten_decrease_m['Company'].append(i)
        results_dict_c_ten_decrease_m['Event_Date'].append(event_date)
        results_dict_c_ten_decrease_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_decrease_m['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_ten_decrease_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_decrease_m['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_ten_decrease_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_decrease_m['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_ten_decrease_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_ten_decrease_m['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_ten_decrease_m = pd.DataFrame(results_dict_c_ten_decrease_m)
print(results_significance_c_eight_decrease_m.head())
print(results_significance_c_ten_decrease_m.head())


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold small_thres
results_dict_c_eight_decrease_alt_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_small_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_eight_decrease_alt_m['Company'].append(i)
        results_dict_c_eight_decrease_alt_m['Event_Date'].append(event_date)
        results_dict_c_eight_decrease_alt_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_decrease_alt_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_decrease_alt_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_decrease_alt_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_eight_decrease_alt_m = pd.DataFrame(results_dict_c_eight_decrease_alt_m)

results_significance_c_eight_decrease_alt_result_m = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_eight_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m['Avg_Next_Day_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m['Avg_Next_2_Days_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m['Avg_Next_5_Days_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m['p_value'].append(result_20)
results_significance_c_eight_decrease_alt_result_m=pd.DataFrame(results_significance_c_eight_decrease_alt_result_m)

# Create a dictionary to store the result for threshold large_thres

results_dict_c_ten_decrease_alt_m = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_large_thres'] == 1) & (total_c_df['Market_Return_Increase'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_ten_decrease_alt_m['Company'].append(i)
        results_dict_c_ten_decrease_alt_m['Event_Date'].append(event_date)
        results_dict_c_ten_decrease_alt_m['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_decrease_alt_m['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_decrease_alt_m['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_decrease_alt_m['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_ten_decrease_alt_m = pd.DataFrame(results_dict_c_ten_decrease_alt_m)

results_significance_c_ten_decrease_alt_result_m = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_ten_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m['Avg_Next_Day_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m['Avg_Next_2_Days_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m['Avg_Next_5_Days_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m['p_value'].append(result_20)
results_significance_c_ten_decrease_alt_result_m=pd.DataFrame(results_significance_c_ten_decrease_alt_result_m)

print(results_significance_c_eight_decrease_alt_result_m.head())
print(results_significance_c_ten_decrease_alt_result_m.head())


# %% [markdown]
# #### Price decreases with market decreases:

# %% [markdown]
# ##### Testing events individually:

# %%

# Create a dictionary to store the result for each threshold
results_dict_c_eight_decrease_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

results_dict_c_ten_decrease_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'P_Value_1_Day': [],
                'Avg_Next_2_Days_Returns': [],
                'P_Value_2_Days': [],
                'Avg_Next_5_Days_Returns': [],
                'P_Value_5_Days': [],
                'Avg_Next_20_Days_Returns': [],
                'P_Value_20_Days': []}

# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_small_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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


        results_dict_c_eight_decrease_m_dec['Company'].append(i)
        results_dict_c_eight_decrease_m_dec['Event_Date'].append(event_date)
        results_dict_c_eight_decrease_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_decrease_m_dec['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_eight_decrease_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_decrease_m_dec['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_eight_decrease_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_decrease_m_dec['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_eight_decrease_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_eight_decrease_m_dec['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_eight_decrease_m_dec = pd.DataFrame(results_dict_c_eight_decrease_m_dec)


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_large_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1
        idx_event_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0]

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

        # Average_next_day
            avg_next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
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

        results_dict_c_ten_decrease_m_dec['Company'].append(i)
        results_dict_c_ten_decrease_m_dec['Event_Date'].append(event_date)
        results_dict_c_ten_decrease_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_decrease_m_dec['P_Value_1_Day'].append(p_value_1_day)
        results_dict_c_ten_decrease_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_decrease_m_dec['P_Value_2_Days'].append(p_value_2_days)
        results_dict_c_ten_decrease_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_decrease_m_dec['P_Value_5_Days'].append(p_value_5_days)
        results_dict_c_ten_decrease_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)
        results_dict_c_ten_decrease_m_dec['P_Value_20_Days'].append(p_value_20_days)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns} , P value: {p_value_1_day} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns},P value: {p_value_2_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns},P value: {p_value_5_days} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns},P value: {p_value_20_days} ')

results_significance_c_ten_decrease_m_dec = pd.DataFrame(results_dict_c_ten_decrease_m_dec)
print(results_significance_c_eight_decrease_m_dec.head())
print(results_significance_c_ten_decrease_m_dec.head())


# %% [markdown]
# ##### Testing events aggregately:

# %%
# Create a dictionary to store the result for threshold small_thres
results_dict_c_eight_decrease_alt_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for small_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_small_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_eight_decrease_alt_m_dec['Company'].append(i)
        results_dict_c_eight_decrease_alt_m_dec['Event_Date'].append(event_date)
        results_dict_c_eight_decrease_alt_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_eight_decrease_alt_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_eight_decrease_alt_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_eight_decrease_alt_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_eight_decrease_alt_m_dec = pd.DataFrame(results_dict_c_eight_decrease_alt_m_dec)

results_significance_c_eight_decrease_alt_result_m_dec = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_eight_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m_dec['Avg_Next_Day_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m_dec['Avg_Next_2_Days_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m_dec['Avg_Next_5_Days_Returns'].mean())
results_significance_c_eight_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_eight_decrease_alt_m_dec['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m_dec['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m_dec['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m_dec['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_eight_decrease_alt_result_m_dec['p_value'].append(result_20)
results_significance_c_eight_decrease_alt_result_m_dec=pd.DataFrame(results_significance_c_eight_decrease_alt_result_m_dec)

# Create a dictionary to store the result for threshold large_thres

results_dict_c_ten_decrease_alt_m_dec = {'Company': [],
                'Event_Date': [],
                'Avg_Next_Day_Returns': [],
                'Avg_Next_2_Days_Returns': [],
                'Avg_Next_5_Days_Returns': [],
                'Avg_Next_20_Days_Returns': []}
Next_Day_Returns=[]
Next_2_Days_Returns=[]
Next_5_Days_Returns=[]
Next_20_Days_Returns=[]


# Calculate average ARs and p-values for large_thres threshold
for i in df_mid['Security-Symbol']:

    rows_with_condition = total_c_df[(total_c_df[f'{i}_Decrease_large_thres'] == 1) & (total_c_df['Market_Return_Decrease'] == 1)]

    for index, row in rows_with_condition.iterrows():

        # Date with large price change
        event_date = row['Date']

        # Index_next_day
        idx_next_day = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1

        # Index_next_2_days
        idx_next_2_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2
        next_2_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 2, f'{i}_ARs']
        list_2=next_2_days_returns.dropna().to_list()
        Next_2_Days_Returns.extend(list_2)

        # Index_next_5_days
        idx_next_5_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5
        next_5_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 5, f'{i}_ARs']
        list_5=next_5_days_returns.dropna().to_list()
        Next_5_Days_Returns.extend(list_5)

        # Index_next_20_days
        idx_next_20_days = total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20
        next_20_days_returns = total_c_df.loc[total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 1: total_c_df.index[total_c_df['Date'] == event_date].to_numpy()[0] + 20, f'{i}_ARs']
        list_20=next_20_days_returns.dropna().to_list()
        Next_20_Days_Returns.extend(list_20)


        if idx_next_day < len(total_c_df) and idx_next_2_days < len(total_c_df) and \
           idx_next_5_days < len(total_c_df) and idx_next_20_days < len(total_c_df):

            # Average_next_day
            next_day_returns = total_c_df.at[idx_next_day, f'{i}_ARs']
            Next_Day_Returns.append(next_day_returns)

            # Average_next_2_days
            avg_next_2_days_returns = next_2_days_returns.mean()

            # Average_next_5_days
            avg_next_5_days_returns = next_5_days_returns.mean()

            # Average_next_20_days
            avg_next_20_days_returns = next_20_days_returns.mean()


        results_dict_c_ten_decrease_alt_m_dec['Company'].append(i)
        results_dict_c_ten_decrease_alt_m_dec['Event_Date'].append(event_date)
        results_dict_c_ten_decrease_alt_m_dec['Avg_Next_Day_Returns'].append(avg_next_day_returns)
        results_dict_c_ten_decrease_alt_m_dec['Avg_Next_2_Days_Returns'].append(avg_next_2_days_returns)
        results_dict_c_ten_decrease_alt_m_dec['Avg_Next_5_Days_Returns'].append(avg_next_5_days_returns)
        results_dict_c_ten_decrease_alt_m_dec['Avg_Next_20_Days_Returns'].append(avg_next_20_days_returns)



        print(f'Company: {i}, Date: {event_date}, Avg Next Day Returns: {avg_next_day_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 2 Days Returns: {avg_next_2_days_returns} ')
        print(f'Company: {i}, Date: {event_date}, Avg Next 5 Days Returns: {avg_next_5_days_returns}')
        print(f'Company: {i}, Date: {event_date}, Avg Next 20 Days Returns: {avg_next_20_days_returns}')

results_significance_c_ten_decrease_alt_m_dec = pd.DataFrame(results_dict_c_ten_decrease_alt_m_dec)

results_significance_c_ten_decrease_alt_result_m_dec = {'Window':['1','1-2','1-5','1-20'],
                                                    'Avg_ARs_%':[],
                                                    'p_value':[]}
results_significance_c_ten_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m_dec['Avg_Next_Day_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m_dec['Avg_Next_2_Days_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m_dec['Avg_Next_5_Days_Returns'].mean())
results_significance_c_ten_decrease_alt_result_m_dec['Avg_ARs_%'].append(results_significance_c_ten_decrease_alt_m_dec['Avg_Next_20_Days_Returns'].mean())

result_1=ttest_1samp(Next_Day_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m_dec['p_value'].append(result_1)

result_2=ttest_1samp(Next_2_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m_dec['p_value'].append(result_2)
result_5=ttest_1samp(Next_5_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m_dec['p_value'].append(result_5)
result_20=ttest_1samp(Next_20_Days_Returns,popmean=0,alternative='two-sided').pvalue
results_significance_c_ten_decrease_alt_result_m_dec['p_value'].append(result_20)
results_significance_c_ten_decrease_alt_result_m_dec=pd.DataFrame(results_significance_c_ten_decrease_alt_result_m_dec)

print(results_significance_c_eight_decrease_alt_result_m_dec.head())
print(results_significance_c_ten_decrease_alt_result_m_dec.head())


# %%
results_significance_c_eight_increase_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_increase_m_mid.xlsx')
results_significance_c_ten_increase_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_increase_m_mid.xlsx')

results_significance_c_eight_increase_alt_result_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_increase_alt_result_m_mid.xlsx')
results_significance_c_ten_increase_alt_result_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_increase_alt_result_m_mid.xlsx')

results_significance_c_eight_increase_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_increase_m_dec_mid.xlsx')
results_significance_c_ten_increase_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_increase_m_dec_mid.xlsx')

results_significance_c_eight_increase_alt_result_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_increase_alt_result_m_dec_mid.xlsx')
results_significance_c_ten_increase_alt_result_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_increase_alt_result_m_dec_mid.xlsx')

results_significance_c_eight_decrease_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_decrease_m_mid.xlsx')
results_significance_c_ten_decrease_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_decrease_m_mid.xlsx')

results_significance_c_eight_decrease_alt_result_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_decrease_alt_result_m_mid.xlsx')
results_significance_c_ten_decrease_alt_result_m.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_decrease_alt_result_m_mid.xlsx')

results_significance_c_eight_decrease_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_decrease_m_dec_mid.xlsx')
results_significance_c_ten_decrease_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_decrease_m_dec_mid.xlsx')

results_significance_c_eight_decrease_alt_result_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_eight_decrease_alt_result_m_dec_mid.xlsx')
results_significance_c_ten_decrease_alt_result_m_dec.to_excel(f'{path}/raw_data/pre_covid/t3_results_significance_c_ten_decrease_alt_result_m_dec_mid.xlsx')


# %%


# %%
