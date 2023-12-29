# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:33:29 2023

@author: 18563
"""
import streamlit as st
import matplotlib.pyplot as plt
import boto3
import pandas as pd
import io
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import numpy as np

def get_data_from_s3(business_unit, start_date, end_date, agg_level, brand_filter, kw_filter, sku_filter):
    access_key = 'AKIAZK6N3K6M6BKSFGAF'
    secret_key = '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'
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    bucket = 'fw-advanced-analytics'
    
    current = start_date
    end = end_date
    
    keys = []

    # Loop until the current month is greater than the end month
    while current <= end:
        # Add the current month in 'YYYY-MM' format to the list
        date = current.strftime("%Y-%m")
        key = f'Zac/Walmart Incrementality Results/{business_unit} {date}.csv'
        
        keys.append(key)

        if current >= end:
            break

        # Move to the next month
        # Check for year end and increment year if necessary
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    dfs = []

# Loop over each key in the list
    for key in keys:
    # Get the object from S3
        response = s3.get_object(Bucket=bucket, Key=key)

    # Read the object's body into a DataFrame
        df = pd.read_csv(response['Body'], dtype={'sku': str})

    # Append the DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df[['date', 'sku', 'keyword','brand','spend', 'incremental_sales', 'sales']]
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    combined_df= combined_df.query(f"'{start_date_str}' <= date <= '{end_date_str}'")
    
    if brand_filter:
        combined_df = combined_df[combined_df['brand'].isin(brand_filter)]
    else:
        combined_df = combined_df.copy()
        
    if kw_filter:
        combined_df = combined_df[combined_df['keyword'].isin(kw_filter)]
    else:
        combined_df = combined_df.copy()
        
    if sku_filter:
        combined_df = combined_df[combined_df['sku'].isin(sku_filter)]
    else:
        combined_df = combined_df.copy()

    if agg_level == 'Daily': 
        output = combined_df
        output['% incremental'] = output.apply(
        lambda row: None if row['sales'] == 0 else row['incremental_sales'] / row['sales'], axis=1
        )
        output['incremental_roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['incremental_sales'] / row['spend'], axis=1
        )
        output['roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['sales'] / row['spend'], axis=1
        )

        output['date'] = output['date'].dt.date
    elif agg_level == 'Weekly':
        combined_df['firstdayofweek'] = combined_df['date'].apply(lambda x: x - pd.Timedelta(days=(x.weekday() + 1) % 7))
        output = combined_df.groupby(['firstdayofweek', 'sku', 'keyword', 'brand']).agg(
        incremental_sales=pd.NamedAgg(column='incremental_sales', aggfunc='sum'),
        sales=pd.NamedAgg(column='sales', aggfunc='sum'),
        spend=pd.NamedAgg(column='spend', aggfunc='sum')
        ).reset_index()
        output['% incremental'] = output.apply(
        lambda row: None if row['sales'] == 0 else row['incremental_sales'] / row['sales'], axis=1
        )
        output['incremental_roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['incremental_sales'] / row['spend'], axis=1
        )
        output['roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['sales'] / row['spend'], axis=1
        )
        output['firstdayofweek'] = output['firstdayofweek'].dt.date
        cols = output.columns.tolist()
        idx1, idx2 = cols.index('% incremental'), cols.index('spend')
        cols[idx2], cols[idx1] = cols[idx1], cols[idx2]
        output = output[cols]
    elif agg_level == 'Monthly': 
        combined_df['Month'] = combined_df['date'].dt.strftime("%Y-%m")
        output = combined_df.groupby(['Month', 'sku', 'keyword', 'brand']).agg(
        incremental_sales=pd.NamedAgg(column='incremental_sales', aggfunc='sum'),
        sales=pd.NamedAgg(column='sales', aggfunc='sum'),
        spend=pd.NamedAgg(column='spend', aggfunc='sum')
        ).reset_index()
        output['% incremental'] = output.apply(
        lambda row: None if row['sales'] == 0 else row['incremental_sales'] / row['sales'], axis=1
        )
        output['incremental_roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['incremental_sales'] / row['spend'], axis=1
        )
        output['roas'] = output.apply(
        lambda row: None if row['spend'] == 0 else row['sales'] / row['spend'], axis=1
        )
        cols = output.columns.tolist()
        idx1, idx2 = cols.index('% incremental'), cols.index('spend')
        cols[idx2], cols[idx1] = cols[idx1], cols[idx2]
        output = output[cols]
        

    return output

def plot_stacked_bar_chart(df, num_ticks):
    # Convert currency formatted strings back to numeric

    # Calculate non-incremental sales
    df['non_incremental_sales'] = df['sales'] - df['incremental_sales']

    # Apply filters if they exist
    if brand_filter:
        df = df[df['brand'].isin(brand_filter)]
    if kw_filter:
        df = df[df['keyword'].isin(kw_filter)]
    if sku_filter:
        df = df[df['sku'].isin(sku_filter)]

    # Different handling based on aggregation level
    if agg_level == 'Daily':
        df['date'] = pd.to_datetime(df['date'])
        grouped_df = df.groupby(['date']).agg(
            incremental_sales=pd.NamedAgg(column='incremental_sales', aggfunc='sum'),
            spend=pd.NamedAgg(column='spend', aggfunc='sum'),
            non_incremental_sales=pd.NamedAgg(column='non_incremental_sales', aggfunc='sum'),
            sales=pd.NamedAgg(column='sales', aggfunc='sum')
        ).reset_index()
        grouped_df['roas'] = np.where(grouped_df['spend'] != 0, grouped_df['sales'] / grouped_df['spend'], None)
        grouped_df['incremental_roas'] = np.where(grouped_df['spend'] != 0, grouped_df['incremental_sales'] / grouped_df['spend'], None)
        grouped_df = grouped_df.sort_values(by='date', ascending=False).head(num_ticks).iloc[::-1]
        x_label = 'Date (MM-DD)'
        x_values = grouped_df['date'].dt.strftime('%m-%d').tolist()
        grouped_df['date'] = grouped_df['date'].dt.strftime('%m-%d')

    elif agg_level == 'Weekly':
        df['firstdayofweek'] = pd.to_datetime(df['firstdayofweek'])
        grouped_df = df.groupby(['firstdayofweek']).agg(
            incremental_sales=pd.NamedAgg(column='incremental_sales', aggfunc='sum'),
            spend=pd.NamedAgg(column='spend', aggfunc='sum'),
            non_incremental_sales=pd.NamedAgg(column='non_incremental_sales', aggfunc='sum'),
            sales=pd.NamedAgg(column='sales', aggfunc='sum')
        ).reset_index()
        grouped_df['roas'] = np.where(grouped_df['spend'] != 0, grouped_df['sales'] / grouped_df['spend'], None)
        grouped_df['incremental_roas'] = np.where(grouped_df['spend'] != 0, grouped_df['incremental_sales'] / grouped_df['spend'], None)
        grouped_df = grouped_df.sort_values(by='firstdayofweek', ascending=False).head(num_ticks).iloc[::-1]
        x_label = 'First Day of Week (MM-DD)'
        x_values = grouped_df['firstdayofweek'].dt.strftime('%m-%d').tolist()
        grouped_df['firstdayofweek'] = grouped_df['firstdayofweek'].dt.strftime('%m-%d')

    elif agg_level == 'Monthly':
        df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
        grouped_df = df.groupby(['Month']).agg(
            incremental_sales=pd.NamedAgg(column='incremental_sales', aggfunc='sum'),
            spend=pd.NamedAgg(column='spend', aggfunc='sum'),
            non_incremental_sales=pd.NamedAgg(column='non_incremental_sales', aggfunc='sum'),
            sales=pd.NamedAgg(column='sales', aggfunc='sum')
        ).reset_index()
        grouped_df['roas'] = np.where(grouped_df['spend'] != 0, grouped_df['sales'] / grouped_df['spend'], None)
        grouped_df['incremental_roas'] = np.where(grouped_df['spend'] != 0, grouped_df['incremental_sales'] / grouped_df['spend'], None)
        grouped_df = grouped_df.sort_values(by='Month', ascending=False).head(num_ticks).iloc[::-1]
        x_label = 'Month (MM-YYYY)'
        x_values = grouped_df['Month'].dt.strftime('%m-%d').tolist()
        grouped_df['Month'] = grouped_df['Month'].dt.strftime('%m-%Y')

    # Apply dark theme
    plt.style.use('dark_background')

    fig_width = max(10, num_ticks / 2)
    # Create two subplots with different heights
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(fig_width, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Plotting with custom colors
    bars1 = ax.bar(x_values, grouped_df['non_incremental_sales'], color='#FF073A', label='Non-Incremental Sales')
    bars2 = ax.bar(x_values, grouped_df['incremental_sales'], bottom=grouped_df['non_incremental_sales'], color='#00BFFF', label='Incremental Sales')

    # Rotate date labels vertically and format y-axis labels as currency
    plt.xticks(rotation='vertical', color='white')  # Set text color to white for visibility
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

    # Adding data labels with white color for visibility
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'${height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=90 if num_ticks > 9 else 0,
                            color='white')

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel(x_label, color='white')
    ax.set_ylabel('Sales', color='white')
    ax.set_title('Sales Breakdown by Date', color='white')
    ax.legend()

        # Second plot (Time Series Graph)
    ax2.plot(x_values, grouped_df['roas'], label='ROAS', color='#FF073A', marker='o')
    ax2.plot(x_values, grouped_df['incremental_roas'], label='Incremental ROAS', color='#00BFFF', marker='o')

    # Add labels to the points
    for i in range(len(grouped_df)):
        ax2.annotate(f"${grouped_df['roas'].iloc[i]:,.2f}", 
                     (x_values[i], grouped_df['roas'].iloc[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     color='white')
        ax2.annotate(f"${grouped_df['incremental_roas'].iloc[i]:,.2f}", 
                     (x_values[i], grouped_df['incremental_roas'].iloc[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     color='white')

    # Formatting for the time series graph
    plt.xticks(rotation='vertical', color='white')
    ax2.set_xlabel(x_label, color='white')
    ax2.set_ylabel('ROAS', color='white')
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.2f}'))
    ax2.legend()
    ax2.grid(True)
    
    # Create a secondary y-axis for '% incrementality'
    ax_right = ax.twinx()

# Calculate '% incrementality', handling division by zero
    grouped_df['% incrementality'] = np.where(grouped_df['sales'] != 0, 
                                          grouped_df['incremental_sales'] / grouped_df['sales'], 
                                          None)

# Plotting the line on the secondary y-axis
    line, = ax_right.plot(x_values, grouped_df['% incrementality'], label='% Incrementality', color='white', marker='o')

# Set the limits and labels for the secondary y-axis
    ax_right.set_ylim(0, 1)
    ax_right.set_ylabel('% Incrementality', color='white')
    ax_right.tick_params(axis='y', labelcolor='white')

    ax.set_xticklabels(x_values, rotation='vertical', color='white')

    return fig
        

start_date_str = '2023-10-01'
end_date_str = '2023-10-31'

# Convert the string dates to datetime objects
default_start = datetime.strptime(start_date_str, '%Y-%m-%d')
default_end = datetime.strptime(end_date_str, '%Y-%m-%d')

business_unit = st.sidebar.selectbox('Business Unit', options=['Clorox', 'Mattel', 'Nestle', 'Revlon', 'McCormick'])
access_key = 'AKIAZK6N3K6M6BKSFGAF'
secret_key = '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'
s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
bucket = 'fw-advanced-analytics'
key = f'Zac/Walmart Incrementality Results/{business_unit} brands.csv'
response = s3.get_object(Bucket=bucket, Key=key)
brands = pd.read_csv(response['Body'])['brand'].tolist()

key = f'Zac/Walmart Incrementality Results/{business_unit} kws.csv'
response = s3.get_object(Bucket=bucket, Key=key)
kws = pd.read_csv(response['Body'])['keyword'].tolist()

key = f'Zac/Walmart Incrementality Results/{business_unit} SKUs.csv'
response = s3.get_object(Bucket=bucket, Key=key)
skus = pd.read_csv(response['Body'])['sku'].tolist()


min_date = datetime(2023,9,1)
max_date = datetime(2023,11,30)
start_date = st.sidebar.date_input("Start date", default_start,min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", default_end,min_value=min_date, max_value=max_date)

agg_level = st.sidebar.selectbox('Interval', options=['Daily', 'Weekly', 'Monthly'] )

brand_filter = st.sidebar.multiselect('Brands', options = brands)
kw_filter = st.sidebar.multiselect('Keywords', options = kws)
sku_filter = st.sidebar.multiselect('SKUs', options = skus)

num_ticks = st.sidebar.slider('Graph Lookback Window:', min_value=1, max_value=31, value=5, step=1 )

# Validation
if start_date > end_date:
    st.error("Error: End date must fall after start date.")
    
df = get_data_from_s3(business_unit, start_date, end_date, agg_level, brand_filter, kw_filter, sku_filter)
display = df.copy()
def currency_formatter(number):
    return "${:,.2f}".format(number)
def percentage_formatter(number):
    return "{:.2%}".format(number)
formatted_display = display.style.format({
    'incremental_sales': currency_formatter,
    'sales': currency_formatter,
    'spend': currency_formatter, 
    'incremental_roas':currency_formatter, 
    'roas': currency_formatter, 
    '% incremental':percentage_formatter
})
chart = plot_stacked_bar_chart(df, num_ticks)
st.pyplot(chart)
st.dataframe(formatted_display)


