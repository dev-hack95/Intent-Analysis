import numpy as np
import pandas as pd
from datetime import datetime
import logging


log = logging.getLogger(__name__)

def merge_data(customers: pd.DataFrame, orders_items: pd.DataFrame,
               orders_payments: pd.DataFrame, orders_reviews: pd.DataFrame,
               orders: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    log.info("Start Merging data")
    try:
        df_1 = (
            orders
            .merge(orders_items, on='order_id')
            .merge(orders_payments, on='order_id')
            .merge(customers, on='customer_id'))
    
        df = (
            df_1
            .merge(sellers, on='seller_id')
            .merge(orders, on='order_id')
            .merge(products, on='product_id')
            .merge(orders_reviews, on='order_id'))
 
        return df
       
    except Exception as err:
        log.error("Error While merging data: ", err)

    finally:
        log.info("Merging of data is completed")


def convert_to_datetime(df, feature):
    match feature:
        case 'order_delivered_customer_date_x' | 'order_delivered_customer_date_y' | 'order_estimated_delivery_date_x'| 'order_estimated_delivery_date_y' | 'order_approved_at_x' | 'order_approved_at_y':
            
            df[feature] = pd.to_datetime(df[feature])
            return df


def count_days_form_last_purchase(merged_data: pd.DataFrame, customer_id):
    customer_data = merged_data[merged_data['customer_unique_id'] == customer_id]
    dates = customer_data['order_purchase_timestamp_x']
    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]
    dates_ymd = [date.date() for date in dates]
    if len(set(dates_ymd)) == 1:
        unique_dates = [dates_ymd[0]]
    else:
        unique_dates = list(dict.fromkeys(dates_ymd))


    if len(unique_dates) >= 2:
        days_from_last_purchase = (unique_dates[-2] - unique_dates[-1]).days
        return np.abs(days_from_last_purchase)
    else:
        return 1
    
def how_many_time_customer_purchase(merged_data: pd.DataFrame, customer_id):
    customer_data = merged_data[merged_data['customer_unique_id'] == customer_id]
    return customer_data.shape[0]

def amount_of_purchase_made_by_customer(merged_data: pd.DataFrame, customer_id):
    customer_data = merged_data[merged_data['customer_unique_id'] == customer_id]
    spend = customer_data['price'].sum()
    return spend
    

def rfm_analysis(merged_data: pd.DataFrame) -> pd.DataFrame:
    log.info("Performing Recency-Frequency-Monetry Analysis")

    """This code let you tranfrom object type data to datetime format"""
    # Object ---------------> Datetime
    try:
        convert_to_datetime(merged_data, 'order_delivered_customer_date_x')
        convert_to_datetime(merged_data, 'order_delivered_customer_date_y')
        convert_to_datetime(merged_data, 'order_estimated_delivery_date_x')
        convert_to_datetime(merged_data, 'order_estimated_delivery_date_y')
    except Exception as error:
        print("Unable to convert object to datetime format")
        print("Error: ", error)


    try:
        """Calculting late orders"""
        merged_data['late_orders_x'] = (merged_data['order_estimated_delivery_date_x'] - merged_data['order_delivered_customer_date_x']).dt.days
        merged_data['late_orders_y'] = (merged_data['order_estimated_delivery_date_y'] - merged_data['order_delivered_customer_date_y']).dt.days
    except Exception as error:
        print('Error occured while calculating the number of late order.')
        print("Error: ", error)

    # RFM Analysis
    # Chunking

    recency_feature = 'customer_unique_id'
    recency_rows = 0
    recency = []
    for chunk in pd.read_csv("data/02_intermediate/merged_data.csv", chunksize=5000):
        recency_rows += len(chunk)
        customer_ids = chunk[recency_feature].unique()
        for customer_id in customer_ids:
            days = count_days_form_last_purchase(merged_data, customer_id)
            recency.append({'customer_unique_id': customer_id, 'count_days_form_last_purchase': days})
        print("{0} rows processed".format(recency_rows))


    days_form_last_purchase = pd.DataFrame(recency)

    # Frequncy

    frequency_feature = 'customer_unique_id'
    frequency_rows = 0
    frequency = []
    for chunk in pd.read_csv("data/02_intermediate/merged_data.csv", chunksize=5000):
        frequency_rows += len(chunk)
        customer_ids = chunk[frequency_feature].unique()
        for customer_id in customer_ids:
            days = how_many_time_customer_purchase(merged_data, customer_id)
            frequency.append({'customer_unique_id': customer_id, 'how_many_time_customer_purchase': days})
        print("{0} rows processed".format(frequency_rows))

    how_many_time_customer_purchase_dataframe = pd.DataFrame(frequency)

    # Monetary

    monetary_feature = 'customer_unique_id'
    monetary_rows = 0
    money = []

    for chunk in pd.read_csv("data/02_intermediate/merged_data.csv", chunksize=5000):
        monetary_rows += len(chunk)
        customer_ids = chunk[monetary_feature].unique()
        for customer_id in customer_ids:
            spends = amount_of_purchase_made_by_customer(merged_data, customer_id)
            money.append({'customer_unique_id': customer_id, 'amount_of_purchase_made_by_customer': spends})
        
        print("{0} rows processed".format(monetary_rows))

    purchase_made_by_customer  = pd.DataFrame(money)

    merged_data = (
        merged_data
        .merge(days_form_last_purchase, on='customer_unique_id')
        .merge(how_many_time_customer_purchase_dataframe, on='customer_unique_id')
        .merge(purchase_made_by_customer, on='customer_unique_id')
        )
    

    return merged_data
