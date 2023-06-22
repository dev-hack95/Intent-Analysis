import folium
from folium.plugins import HeatMap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

df = pd.read_csv("data/01_raw/e_commerce.csv")
geolocation = pd.read_csv("data/01_raw/olist_geolocation_dataset.csv")

st.set_page_config(
    page_title="Viz Dashboard",
    layout="wide"
)

st.title("# Churn Analysis")

#map = folium.Map(location=geolocation[['geolocation_lat', 'geolocation_lng']].values.tolist()[0])
#for i in range(0, len(geolocation)):
#    folium.CircleMarker(
#        location=geolocation[['geolocation_lat', 'geolocation_lng']].values.tolist(),
#        radius=100,
#        color='#69b3a2',
#        fill=True,
#        fill_color='#69b3a2'
#    ).add_to(map)
#coordinates = geolocation[['geolocation_lat', 'geolocation_lng']].values.tolist()
#heat_layer = HeatMap(coordinates)
#heat_layer.add_to(map)
#st_data = st_folium(map, width=1200, height=500)

#st.dataframe(df.head())

def value_count_norm(df, feature):
    value_count_norm = df[feature].value_counts()
    value_count_norm1 = df[feature].value_counts(normalize=True) * 100
    value_count_norm1_concat = pd.concat([value_count_norm, value_count_norm1], axis=1)
    value_count_norm1_concat.columns = ['Count', 'Frequncy']
    return value_count_norm1_concat

def create_donut_plot(df, feature, col):
    match feature:
        case 'order_status_x' | 'order_status_y' | 'payment_type':
            value_counts = value_count_norm(df, feature).index
            fig_1 = px.pie(names=value_counts,
                           values=value_count_norm(df, feature).iloc[:, 1],
                           hole=0.7)
            fig_1.update_traces(
                   title_font = dict(size=25,family='Verdana', 
                                     color='darkred'), textfont_size=20,)
            
            #fig_1.update_layout(labels)
            
            return col.plotly_chart(fig_1, use_container_width=True)
        
        case 'payment_installments' | 'payment_sequential':
            single_payment = [x for x in df[feature] if x == 1]
            installment_payment = [x for x in df[feature] if x > 1]
            
            single_payment_count = len(single_payment)
            installment_payment_count = len(installment_payment)
            
            labels = ['Single Pyament', 'Installments']
            sizes = [single_payment_count, installment_payment_count]

            fig_2 = px.pie(names=labels,
                           values=sizes,
                           hole=0.7)
            fig_2.update_traces(
                   title_font = dict(size=25,family='Verdana', 
                                     color='darkred'),
                   textinfo='percent', textfont_size=20)

            return col.plotly_chart(fig_2, use_container_width=True)
        
        case 'late_orders_x' | 'late_orders_y':
            early_deliveries = [x for x in df[feature] if x > 0]
            on_time_deliveries = [x for x in df[feature] if x == 0]
            late_deliveries = [x for x in df[feature] if x < 0]
            
            labels = ['early_deliveries', 'on_time_deliveries', 'late_deleveries']
            sizes = [len(early_deliveries), len(on_time_deliveries),  len(late_deliveries)]
            
            fig_3 = px.pie(names=labels,
                           values=sizes,
                           hole=0.7)
            fig_3.update_traces(
                   title_font = dict(size=25,family='Verdana', 
                                     color='darkred'),
                   hoverinfo='label+percent',
                   textinfo='percent', textfont_size=20)

            return col.plotly_chart(fig_3, use_container_width=True)
        
        case 'days_from_last_purchase':
            single_purchase = [x for x in df[feature] if x == 1]
            more_than_one_purchase = [x for x in df[feature] if x > 1]
            
            total = len(single_purchase) + len(more_than_one_purchase)
            lables = ['single_purchase', 'more_than_one_purchase']
            sizes = [len(single_purchase)/total, len(more_than_one_purchase)/total]
            
            fig_4 = px.pie(names=lables,
                           values=sizes,
                           hole=0.7)
            
            fig_4.update_traces(
                title_font = dict(size=25, family="Verdana",
                                  color='darkred'),
                hoverinfo = 'label+percent',
                textinfo = 'percent', textfont_size=20,
            )
        
            return col.plotly_chart(fig_4, use_container_width=True)
            
        
        case _:
            print("Error")

def impact_of_late_orders(df, feature):
    match feature:
        case 'late_orders_x':
            #late_deliveries = [x for x in df[feature] if x < 0]
            late_deliveries_df = df[df[feature] < 0]
            
            fig = px.histogram(x=late_deliveries_df['review_score'])
            return st.plotly_chart(fig)
        
        case _:
            print("error")

def create_bar_chart(df, feature, col):
    match feature:
        case 'customer_state' | 'seller_state':
            df_bar = df.groupby(feature).agg({'price': 'sum'})
            df_bar = df_bar['price'].sort_values(ascending=True)
            df_bar = pd.DataFrame(df_bar)
            fig = px.bar(df_bar, x='price', y=df_bar.index, orientation='h')
            col.plotly_chart(fig)
        
        case _:
            print("Error")

st.header("Transctions")
col1, col2 = st.columns(2)
create_donut_plot(df, 'payment_type', col1)
create_donut_plot(df, 'payment_installments', col2)


st.header("Orders")
col1, col2 = st.columns(2)
create_donut_plot(df, 'late_orders_x', col1)
create_donut_plot(df, 'order_status_x', col2)

st.header("Payment Sequentials")
col1, col2 = st.columns(2)
create_donut_plot(df, 'payment_sequential', col1)
create_donut_plot(df, 'days_from_last_purchase', col2)


st.header("Impact of Late Orders on review score (distribution)")
impact_of_late_orders(df, 'late_orders_x')


st.header("Overall review score distribution")
fig = px.histogram(x=df['review_score'])
st.plotly_chart(fig)

st.header('State wise sales')
col1, col2 = st.columns(2)
create_bar_chart(df, 'customer_state', col1)
create_bar_chart(df, 'seller_state', col2)

