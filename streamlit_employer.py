from __future__ import annotations, division
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#%matplotlib inline


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(r'C:\Users\Admin\Desktop\PythonCodes\streamlit_employer_folder\united_kingdom_retailer.csv')
    data_2 = pd.read_csv(r'C:\Users\Admin\Desktop\PythonCodes\streamlit_employer_folder\customers.csv')
    return data, data_2

df, customers = load_data()



# Filtering 
filt_uk = df['Country'] == 'United Kingdom'
df = df.loc[filt_uk]
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['YearMonth'] = df['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))
df = df[ df['YearMonth'] < '2011-12'] # 2011-12 contains incomplete data





# Revenue Per Month calculation and visualization
df['Revenue'] = df['UnitPrice'] * df['Quantity']
df_revenue = df.groupby(['YearMonth'])['Revenue'].sum().reset_index()

data_revenue = [go.Scatter( x=df_revenue['YearMonth'],  y=df_revenue['Revenue'], mode='lines')]

layout_revenue = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Monthly Revenue', 
    plot_bgcolor='white'
     )

plot_revenue = go.Figure(data=data_revenue, layout=layout_revenue)

# Revenue Growth Per Month calculation and visualization
df_revenue['RevenueGrowth'] = df_revenue['Revenue'].pct_change().apply(lambda x: round(x, 3))

data_growth = [ go.Scatter(x = df_revenue['YearMonth'], y = df_revenue['RevenueGrowth'], mode='lines')]

layout_growth = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Monthly Revenue Growth', 
    plot_bgcolor='white',
)

plot_growth = go.Figure(data=data_growth, layout=layout_growth)
# New Customer Ratio Over Time calculation and visualization
first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
first_purchase.rename({'InvoiceDate' : 'First Purchase'}, axis=1, inplace=True)
first_purchase['First Purchase'] = first_purchase['First Purchase'].apply(lambda x: x.strftime('%Y-%m'))
df = pd.merge(df, first_purchase, on='CustomerID')
df.loc[ df['YearMonth'] <= df['First Purchase'], 'CustomerType'] = 'New'
df.loc[ df['YearMonth'] > df['First Purchase'], 'CustomerType'] = 'Returning' 
customer_type_revenue = df.groupby(['YearMonth', 'CustomerType'])['Revenue'].sum().reset_index()
customer_type_revenue = customer_type_revenue[customer_type_revenue['YearMonth'] > '2010-12']

new_customer = customer_type_revenue[customer_type_revenue['CustomerType'] == 'New']
returning_customer = customer_type_revenue[customer_type_revenue['CustomerType'] == 'Returning']

data_customer_type = [go.Scatter(x = new_customer['YearMonth'], y = new_customer['Revenue'], name = 'New', mode='lines'),
                      go.Scatter(x = returning_customer['YearMonth'], y = returning_customer['Revenue'], name = 'Returning', mode='lines')]

layout_customer_type = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'New vs Returning Customers', 
    plot_bgcolor='white',
)
# Monthly Active Customers
active_customers = df.groupby('YearMonth')['CustomerID'].nunique().reset_index()

data_active = [go.Bar(x=active_customers['YearMonth'], y=active_customers['CustomerID'], text=active_customers['CustomerID'], textposition='auto')]

layout_active = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Monthly Active Customers', 
    plot_bgcolor='white',
    autosize=True
)

plot_active = go.Figure(data=data_active, layout=layout_active)
# Monthly Order Count
monthly_sales = df.groupby('YearMonth')['Quantity'].sum().reset_index()

monthly_sales['Quantity'] = monthly_sales['Quantity'].apply(lambda x: round(x, 2))

data_sales = [go.Bar(x = monthly_sales['YearMonth'],y = monthly_sales['Quantity'], 
        text=monthly_sales['Quantity'], texttemplate='%{text:.2s}', textposition='auto')]

layout_sales = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Monthly Order Count', 
    plot_bgcolor='white',
)

plot_sales = go.Figure(data=data_sales, layout=layout_sales)

plot_customer_type = go.Figure(data=data_customer_type, layout=layout_customer_type)
# Customer Ratio Bar calculation and visualization 
filt_new = df['CustomerType'] == 'New'
filt_returning = df['CustomerType'] == 'Returning'
customer_ratio = df.loc[filt_new].groupby(['YearMonth'])['CustomerID'].nunique() / df.loc[filt_returning].groupby(['YearMonth'])['CustomerID'].nunique()
customer_ratio = customer_ratio.reset_index()
customer_ratio = customer_ratio[customer_ratio['YearMonth'] > '2010-12']
customer_ratio.rename(columns = {'CustomerID' : 'NewUserRatio'}, inplace = True)

customer_new = df['CustomerType'] == 'New'

customer_ratio['NewUserRatio'] = customer_ratio['NewUserRatio'].apply(lambda x: round(x, 2))

data_ratio = [go.Bar(x = customer_ratio['YearMonth'], y = customer_ratio['NewUserRatio'], text=customer_ratio['NewUserRatio'], 
                        textposition='auto')]

layout_ratio = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Ratio of New Customers to Returning Customers', 
    plot_bgcolor='white',
    autosize=True
)

plot_ratio = go.Figure(data=data_ratio, layout=layout_ratio)
# Monthly Retention Rate calculation and visualization
user_revenue = df.groupby(['CustomerID', 'YearMonth'])['Revenue'].sum().reset_index()
user_retention = pd.crosstab(user_revenue['CustomerID'], user_revenue['YearMonth']).reset_index()

months = user_retention.columns[2:]
retention_array = []

for i in range(len(months)-1): 
  retention_data = {}
  selected_month = months[i+1]
  prev_month = months[i]
  retention_data['YearMonth'] =  selected_month
  retention_data['TotalUserCount'] = user_retention[selected_month].sum()
  retention_data['RetainedUserCount'] = user_retention[(user_retention[selected_month] > 0) & (user_retention[prev_month] > 0)][selected_month].sum()
  retention_array.append(retention_data)

retention_df = pd.DataFrame(retention_array)
retention_df['RetentionRate'] = retention_df['RetainedUserCount']/retention_df['TotalUserCount']

retention_df['RetentionRate'] = retention_df['RetentionRate'].apply(lambda x: round(x, 3))

data_retention = [go.Scatter(x = retention_df['YearMonth'], y = retention_df['RetentionRate'], mode='lines')]

layout_retention = go.Layout(
    xaxis = dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    ticks='outside', tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    title = 'Monthly Retention Rate', 
    plot_bgcolor='white'
    )

plot_retention = go.Figure(data=data_retention, layout=layout_retention)


# Pie
data_pie=[go.Pie(labels=df['CustomerType'], values=df['Revenue'], hole=.5)]
layout_pie = go.Layout(title='Revenue by Customer Type')

plot_pie = go.Figure(data=data_pie, layout=layout_pie)

# Header plots
active_customers_header = active_customers['CustomerID'].sum()
sales_header = df['Quantity'].sum()
revenue_header = round(df['Revenue'].sum())
retention_header = round(retention_df['RetentionRate'].mean(), 3)












########################################################################################################


# ML 

kmeans_1 = KMeans(init='k-means++', n_clusters=4, max_iter=1000, n_init=10)
kmeans_2 = KMeans(init='k-means++', n_clusters=4, max_iter=1000, n_init=10)
kmeans_3 = KMeans(init='k-means++', n_clusters=4, max_iter=1000, n_init=10)

def full_function(df):

  filt_uk = df['Country'] == 'United Kingdom'
  df = df.loc[filt_uk]
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
  df['YearMonth'] = df['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))
  df = df[ df['YearMonth'] < '2011-12']
  df.drop('Description', inplace=True, axis=1)
  df.drop('StockCode', inplace=True, axis=1) 
  
  def recency_func(df):
    customers = pd.DataFrame(df['CustomerID'].unique())
    customers.columns = ['CustomerID']
    max_purchase = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    max_purchase.columns = ['CustomerID', 'MaxPurchaseDate'] 
    max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
    customers = pd.merge(customers, max_purchase[['CustomerID', 'Recency']], on='CustomerID')
    customers['RecencyCluster'] = kmeans_1.fit_predict(customers[['Recency']])
    return customers

  customers = recency_func(df)

  def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

  customers = order_cluster('RecencyCluster', 'Recency', customers, False)

  def revenue_func(df, customers):
    df['Revenue'] = df['UnitPrice'] * df['Quantity']
    revenue = df.groupby('CustomerID')['Revenue'].sum().reset_index()
    customers = pd.merge(customers, revenue, on='CustomerID')
    customers['RevenueCluster'] = kmeans_2.fit_predict(customers[['Revenue']])
    return customers

  customers = revenue_func(df, customers)

  customers = order_cluster('RevenueCluster', 'Revenue', customers, True)

  def frequency_func(df, customers):
    frequency = df.groupby('CustomerID')['InvoiceDate'].count().reset_index()
    frequency.columns = ['CustomerID','Frequency']
    customers = pd.merge(customers, frequency, on='CustomerID')    
    customers['FrequencyCluster'] = kmeans_3.fit_predict(customers[['Frequency']])    
    return customers

  customers = frequency_func(df, customers)

  customers = order_cluster('FrequencyCluster', 'Frequency', customers, True)

  customers['OverallScore'] = customers['RecencyCluster'] + customers['FrequencyCluster'] + customers['RevenueCluster']

  customers['Segment'] = 'Low Value'
  customers.loc[customers['OverallScore'] >= 3,'Segment'] = 'Medium Value' 
  customers.loc[customers['OverallScore'] >= 5,'Segment'] = 'High Value'
  
  customers['CustomerID'] = customers['CustomerID'].astype(int)

  return customers

customers = full_function(df)

customers_low_segment = customers[customers['Segment'] == 'Low Value']
customers_mid_segment = customers[customers['Segment'] == 'Medium Value']
customers_high_segment = customers[customers['Segment'] == 'High Value']

def select_customer(customers):
  random_customer = customers.sample()
  customer_id = random_customer['CustomerID'].to_string(index=False)
  recency = random_customer['Recency'].to_string(index=False)
  recencycluster = random_customer['RecencyCluster'].to_string(index=False)
  revenue = random_customer['Revenue'].to_string(index=False)
  revenuecluster = random_customer['RevenueCluster'].to_string(index=False)
  frequency = random_customer['Frequency'].to_string(index=False)
  frequencycluster = random_customer['FrequencyCluster'].to_string(index=False)
  overallscore = random_customer['OverallScore'].to_string(index=False)
  segment = random_customer['Segment'].to_string(index=False)

  return customer_id, recency, recencycluster, revenue, revenuecluster, frequency, frequencycluster, overallscore, segment

customer_id, recency, recencycluster, revenue, revenuecluster, frequency, frequencycluster, overallscore, segment = select_customer(customers)


########################################################################
def manual_segmentation(recency_input, revenue_input, frequency_input):
  segment_df = {'Recency': [recency_input],
                'Revenue': [revenue_input],
                'Frequency': [frequency_input]}

  segment_df = pd.DataFrame(segment_df)

  segment_df['Recency'] = segment_df['Recency'].astype(int)
  segment_df['Revenue'] = segment_df['Revenue'].astype(int)
  segment_df['Frequency'] = segment_df['Frequency'].astype(int)

  def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

  segment_df['RecencyCluster'] = kmeans_1.predict(segment_df[['Recency']])
  segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

  segment_df['RevenueCluster'] = kmeans_2.predict(segment_df[['Revenue']])
  segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

  segment_df['FrequencyCluster'] = kmeans_3.predict(segment_df[['Frequency']])   
  segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

  segment_df['OverallScore'] = segment_df['RecencyCluster'] + segment_df['FrequencyCluster'] + segment_df['RevenueCluster']
    
  segment_df['Segment'] = 'Low Value'
  segment_df.loc[segment_df['OverallScore'] >= 3,'Segment'] = 'Medium Value' 
  segment_df.loc[segment_df['OverallScore'] > 5,'Segment'] = 'High Value'

  return segment_df

segment_df = manual_segmentation(10, 10000, 50)
#########################################################################


























ss1, ss2, ss3 = st.beta_columns(3)
ss1.write('')
ss2.write('')
select_box = ss3.selectbox('Navigation', ['Home Page', 'Instructions', 'Retail Analysis', 'Customer Segmentation'])

if select_box == 'Home Page':
  st.title('')
  st.title('')
  st.title('')
  st.markdown("<h1 style='text-align: center; color: black;' >{fname}</h1>".format(fname='UK Retailer Analysis and Customer Segmentation'), unsafe_allow_html=True)
  st.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Development & Analysis by Jaafar Hoteit'), unsafe_allow_html=True)

if select_box == 'Instructions':
  st.header('Why Segmentation?')
  st.write('''Customers can't be treated with the same importance, content and approach. 
              Customers want business that understand them well and will pursue better options if presented.
              This is because customers have different personalities and needs and thus our approach must be adapted to their actions.''')

  st.header('What Method is Being Used?''')
  st.write('''There are different Customer Segmentation approaches. For example, calculating churn probability segmentation is one way to increase retention rate.
              However, I will be implementing a common segmentation technique called RFM using K-Means Clustering, an unsupervised machine learning algorithm.''')
  st.write('''We will calculate Recency, Frequency and Revenue (Monetary Value)  and 
              apply unsupervised machine learning to identify different clusters for each of them.''')

  st.header('What is RFM?')
  st.write('''RFM stands for Recency - Frequency - Monetary Value (Revenue).''')
  st.write('''Low Value RFM: Customers who show less activity than others, not very frequent buyer and generate very low revenue.''')
  st.write('''Mid Value RM: These customers often use the platform, somewhat frequently and generate moderate revenue.''')
  st.write('''High Value RM: This most important group. The customers have high activity, revenue, and frequency.''')

  st.header('How is Recency Calculated?')
  st.write('''We first identify most recent purchase date of the customer and see how many days he/she has been inactive for. 
              After having the number of inactive days for the customer, we apply K-Means clustering to provide the customer a recency score and ultimately
              a recency cluster.''')

  st.header('How is Frequency Calculated?')
  st.write('''To create frequency clusters, we need to find total number orders for each customer, and then apply K-Means clustering to provide the customer a frequency
            score and ultimately a frequency cluster.''')

  st.header('How is Revenue Calculated and Clustered?')
  st.write('''The customer's revenue is calculted then provided a monetary score (revenue) or monetary value using K-Means clustering and ultimately a revenue cluster.''')

  st.header('How Customers Segmented?')
  st.write('''The scoring above summarizes the customers; with score 8 is our best customers whereas 0 is the worst. ''')
  st.write('Customers with an overall score between 0-2 belong to the low value segment.')
  st.write('Customers with an overall score between 3-5 belong to the medium value segment.')
  st.write('Customers with an overall score between 6-8 belong to the high value segment.')

if select_box == 'Retail Analysis':

    st1, st2 = st.beta_columns(2)

# First, second and third lines of plotting 
#if st1.button('Data Exploration'):
    h1a, h2a, h3a,  h4a = st.beta_columns(4)
    h1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Active Customers'), unsafe_allow_html=True)
    h2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Total Orders'), unsafe_allow_html=True)
    h3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Total Revenue'), unsafe_allow_html=True)
    h4a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Avg. Retention Rate'), unsafe_allow_html=True)


    h1, h2, h3, h4 = st.beta_columns(4)
#h0.header('')
    h1.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=active_customers_header), unsafe_allow_html=True)
    h2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=sales_header), unsafe_allow_html=True)
    h3.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=revenue_header), unsafe_allow_html=True)
    h4.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=retention_header), unsafe_allow_html=True)

    st.header('')

    active, pie = st.beta_columns(2)
    active.write(plot_active)
    pie.write(plot_pie)

    sales, ratio = st.beta_columns(2)
    sales.write(plot_sales)
    ratio.write(plot_ratio)

    revenue, growth = st.beta_columns(2)
    revenue.write(plot_revenue)
    growth.write(plot_growth)

    customer_type, retention = st.beta_columns(2)
    customer_type.write(plot_customer_type)
    retention.write(plot_retention)








if select_box == 'Customer Segmentation':

  select_box_2 = st.selectbox('Choose Preferred Method', ['Random Customer Segmentation', 'Manual Customer Segmentation'])


  if select_box_2 == 'Random Customer Segmentation':
    
    auto1, auto2 = st.beta_columns(2)

    if auto1.button('Segment'):

      c3a, c1a, c2a = st.beta_columns(3)
      c3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Customer ID'), unsafe_allow_html=True)    
      c1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Recency'), unsafe_allow_html=True)
      c2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Recency Cluster'), unsafe_allow_html=True)

    
      c3, c1, c2 = st.beta_columns(3)
      c3.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=customer_id), unsafe_allow_html=True)    
      c1.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=recency), unsafe_allow_html=True)
      c2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=recencycluster), unsafe_allow_html=True)


      l1a, l2a, l3a = st.beta_columns(3)
      l1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Overall Score'), unsafe_allow_html=True)
      l2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Frequency'), unsafe_allow_html=True)
      l3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Frequency Cluster'), unsafe_allow_html=True)

      l1, l2, l3 = st.beta_columns(3)
      l1.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=overallscore), unsafe_allow_html=True)
      l2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=frequency), unsafe_allow_html=True)
      l3.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=frequencycluster), unsafe_allow_html=True)

      a1a, a2a, a3a = st.beta_columns(3)
      a1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Segment'), unsafe_allow_html=True)
      a2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Revenue'), unsafe_allow_html=True)
      a3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Revenue Cluster'), unsafe_allow_html=True)

      a1, a2, a3 = st.beta_columns(3)
      a1.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=segment), unsafe_allow_html=True)
      a2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=revenue), unsafe_allow_html=True)
      a3.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=revenuecluster), unsafe_allow_html=True)


  if select_box_2 == 'Manual Customer Segmentation':

    manual1, manual2 = st.beta_columns(2)

    #if manual1.button('Segment'):
    manual_st1, manual_st2, manual_st3 = st.beta_columns(3)

    recency_input = manual_st1.text_input('Enter Customer Recency Score', '')
    revenue_input = manual_st2.text_input('Enter Customer Revenue Generated by Customer', '')
    frequency_input = manual_st3.text_input('Enter Customer Frequency Score', '')


   #recency_input = int(recency_input, base=10)
  #revenue_input = int(revenue_input, base=10)
  #frequency_input = int(frequency_input, base=10)

    def manual_segmentation(recency_input, revenue_input, frequency_input):
    

      segment_df = {'Recency': [recency_input],
                'Revenue': [revenue_input],
                'Frequency': [frequency_input]}

      segment_df = pd.DataFrame(segment_df)

      segment_df['Recency'] = segment_df['Recency'].astype(int)
      segment_df['Revenue'] = segment_df['Revenue'].astype(int)
      segment_df['Frequency'] = segment_df['Frequency'].astype(int)

      def order_cluster(cluster_field_name, target_field_name, df, ascending):
        new_cluster_field_name = 'new_' + cluster_field_name
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
        df_new['index'] = df_new.index
        df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
        df_final = df_final.drop([cluster_field_name],axis=1)
        df_final = df_final.rename(columns={"index":cluster_field_name})
        return df_final

      segment_df['RecencyCluster'] = kmeans_1.predict(segment_df[['Recency']])
      segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

      segment_df['RevenueCluster'] = kmeans_2.predict(segment_df[['Revenue']])
      segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

      segment_df['FrequencyCluster'] = kmeans_3.predict(segment_df[['Frequency']])   
      segment_df = order_cluster('RecencyCluster', 'Recency', segment_df, False)

      segment_df['OverallScore'] = segment_df['RecencyCluster'] + segment_df['FrequencyCluster'] + segment_df['RevenueCluster']

      segment_df['Segment'] = 'Low Value'
      segment_df.loc[segment_df['OverallScore'] >= 3,'Segment'] = 'Medium Value' 
      segment_df.loc[segment_df['OverallScore'] > 5,'Segment'] = 'High Value'

      return segment_df


    if st.button('Segment '):
        segment_df = manual_segmentation(recency_input, revenue_input, frequency_input)

        def manual_select(segment_df):
          random_customer = segment_df.sample()
          #customer_id = random_customer['CustomerID'].to_string(index=False)
          recency = random_customer['Recency'].to_string(index=False)
          recencycluster = random_customer['RecencyCluster'].to_string(index=False)
          revenue = random_customer['Revenue'].to_string(index=False)
          revenuecluster = random_customer['RevenueCluster'].to_string(index=False)
          frequency = random_customer['Frequency'].to_string(index=False)
          frequencycluster = random_customer['FrequencyCluster'].to_string(index=False)
          overallscore = random_customer['OverallScore'].to_string(index=False)
          segment = random_customer['Segment'].to_string(index=False)

          return customer_id, recency, recencycluster, revenue, revenuecluster, frequency, frequencycluster, overallscore, segment

        customer_id, recency, recencycluster, revenue, revenuecluster, frequency, frequencycluster, overallscore, segment = manual_select(segment_df)

        st.write('')

        if segment == 'High Value':
          st.markdown("<h1 style='text-align: center; color: black;' >{fname}</h1>".format(fname='High Value Customer, Improve Retention'), unsafe_allow_html=True)

        if segment == 'Medium Value':
          st.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Medium Value Customer, Improve Retention and Increase Frequency'), unsafe_allow_html=True)

        if segment == 'Low Value':
          st.markdown("<h1 style='text-align: center; color: black;' >{fname}</h1>".format(fname='Low Value Customer, Increase Frequency'), unsafe_allow_html=True)

        st.write('')

        c3a, c1a, c2a = st.beta_columns(3)
        c3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Customer ID'), unsafe_allow_html=True)    
        c1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Recency'), unsafe_allow_html=True)
        c2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Recency Cluster'), unsafe_allow_html=True)

    
        c3, c1, c2 = st.beta_columns(3)
        c3.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=customer_id), unsafe_allow_html=True)    
        c1.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=recency), unsafe_allow_html=True)
        c2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=recencycluster), unsafe_allow_html=True)


        l1a, l2a, l3a = st.beta_columns(3)
        l1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Overall Score'), unsafe_allow_html=True)
        l2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Frequency'), unsafe_allow_html=True)
        l3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Frequency Cluster'), unsafe_allow_html=True)

        l1, l2, l3 = st.beta_columns(3)
        l1.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=overallscore), unsafe_allow_html=True)
        l2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=frequency), unsafe_allow_html=True)
        l3.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=frequencycluster), unsafe_allow_html=True)

        a1a, a2a, a3a = st.beta_columns(3)
        a1a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Segment'), unsafe_allow_html=True)
        a2a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Revenue'), unsafe_allow_html=True)
        a3a.markdown("<h2 style='text-align: center; color: black;' >{fname}</h2>".format(fname='Revenue Cluster'), unsafe_allow_html=True)

        a1, a2, a3 = st.beta_columns(3)
        a1.markdown("<h1 style='text-align: center; color: #EF553B;' >{fname}</h1>".format(fname=segment), unsafe_allow_html=True)
        a2.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=revenue), unsafe_allow_html=True)
        a3.markdown("<h1 style='text-align: center; color: #636EFA;' >{fname}</h1>".format(fname=revenuecluster), unsafe_allow_html=True)







    #st.header(customer_id)
    #st.header(recency)
    #st.header(recencycluster)
    #st.header(revenue)
    #st.header(revenuecluster)
    #st.header(frequency)
    #st.header(frequencycluster)
    #st.header(overallscore)
    #st.header(segment)







# ML PLOTS

#Revenue vs Frequency
data_segments_1 = [ 
                go.Scatter( x = customers_low_segment['Frequency'], y = customers_low_segment['Revenue'], mode='markers', name='Low', 
                marker= dict(size= 7, line= dict(width=1), color= 'blue', opacity= 0.8 ) ),
                go.Scatter( x = customers_mid_segment['Frequency'], y = customers_mid_segment['Revenue'], mode='markers', name='Mid', 
                marker= dict(size= 9, line= dict(width=1), color= 'green', opacity= 0.5 ) ),
                go.Scatter( x = customers_high_segment['Frequency'], y = customers_high_segment['Revenue'], mode='markers', name='High', 
                marker= dict(size= 11, line= dict(width=1), color= 'red', opacity= 0.9 ) )]

layout_segments_1 = go.Layout(title='Customer Segmentation by Frequency and Revenue', width=1400,     
    xaxis= dict(title='Frequency', showline=False, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=True, showticklabels=False),
    plot_bgcolor='white',
    autosize=False, 
    margin=dict(l=0)
    )

plot_segments_1 = go.Figure(data=data_segments_1, layout=layout_segments_1)

 
#Revenue vs Recency

data_segments_2 = [
                go.Scatter( x = customers_low_segment['Recency'], y = customers_low_segment['Revenue'], mode='markers', name='Low', 
                marker= dict(size= 7,line= dict(width=1), color= 'blue', opacity= 0.8)),
                go.Scatter( x = customers_mid_segment['Recency'], y = customers_mid_segment['Revenue'], mode='markers', name='Mid', 
                marker= dict(size= 9, line= dict(width=1), color= 'green', opacity= 0.5 ) ),
                go.Scatter( x = customers_high_segment['Recency'], y = customers_high_segment['Revenue'], mode='markers', name='High', 
                marker= dict(size= 11, line= dict(width=1), color= 'red', opacity= 0.9 ) )]

layout_segments_2 = go.Layout(title='Customer Segmentation by Recency and Revenue', width=1400,     
    xaxis= dict(title='Recency', showline=False, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=True, showticklabels=False),
    plot_bgcolor='white',
    autosize=False, 
    margin=dict(l=0) )
plot_segments_2 = go.Figure(data=data_segments_2, layout=layout_segments_2)


# Revenue vs Frequency

data_segments_3 = [
                go.Scatter( x = customers_low_segment['Recency'], y = customers_low_segment['Frequency'], mode='markers', name='Low', 
                marker= dict(size= 7, line= dict(width=1), color= 'blue', opacity= 0.8 )),
                go.Scatter( x = customers_mid_segment['Recency'], y = customers_mid_segment['Frequency'], mode='markers', name='Mid', 
                marker= dict(size= 9, line= dict(width=1), color= 'green', opacity= 0.5 ) ),
                go.Scatter( x = customers_high_segment['Recency'], y = customers_high_segment['Frequency'], mode='markers', name='High', 
                marker= dict(size= 11, line= dict(width=1), color= 'red', opacity= 0.9 ) )]

layout_segments_3 = go.Layout(title='Customer Segmentation by Recency and Frequency', width=1400,     
    xaxis= dict(title='Recency', showline=False, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=1, 
    tickfont=dict(family='Arial', color='rgb(82, 82, 82)')),
    yaxis = dict(showgrid=False, zeroline=False, showline=True, showticklabels=False),
    plot_bgcolor='white',
    autosize=False, 
    margin=dict(l=0))
plot_segments_3 = go.Figure(data=data_segments_3, layout=layout_segments_3)


#if st2.button('Customer Segmentation'):
#    st_receny, st_frequency = st.beta_columns(2)
#    st_receny.write(plot_recency)
#    st_frequency.write(plot_frequency)
#    st.write(plot_revenue)
#    st.write(plot_segments_1)
#    st.write(plot_segments_2)
#    st.write(plot_segments_3)



