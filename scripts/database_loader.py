# scripts/database_handler_customer.py

import pandas as pd
import mysql.connector
from mysql.connector import Error

def create_connection(db_name=None):
    try:
        if db_name:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="honest",
                database=db_name
            )
        else:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="honest"
            )
        if conn.is_connected():
            print(" Connected to MySQL")
        return conn
    except Error as e:
        print(f" Error connecting to MySQL: {e}")
        return None

def create_database(conn, db_name):
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        conn.commit()
        print(f" Database '{db_name}' created or already exists.")
    except Error as e:
        print(f" Error creating database: {e}")
        conn.rollback()

def create_customer_table(conn):
    try:
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS customer (
            customer_id VARCHAR(255) PRIMARY KEY,
            age INT,
            tenure INT,
            balance DECIMAL(10, 2),
            num_of_products INT,
            is_active_member INT,
            estimated_salary DECIMAL(10, 2),
            churn INT,
            gender_female INT,
            gender_male INT,
            has_cr_card_0 INT,
            has_cr_card_1 INT
        ) ENGINE=InnoDB;
        """
        cursor.execute(create_table_query)
        conn.commit()
        print(" Table 'customer' created or already exists.")
    except Error as e:
        print(f" Error creating table: {e}")
        conn.rollback()

def load_and_prepare_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f" Loaded CSV with {df.shape[0]} rows")

        # Clean missing values
        df.dropna(subset=['CustomerID', 'Gender', 'HasCrCard'], inplace=True)

        # One-hot encode
        df = pd.get_dummies(df, columns=['Gender', 'HasCrCard'])

        # Rename to match DB schema
        df.rename(columns={
        'CustomerID': 'customer_id',
        'Age': 'age',
        'Tenure': 'tenure',
        'Balance': 'balance',
        'NumOfProducts': 'num_of_products',
        'IsActiveMember': 'is_active_member',
        'EstimatedSalary': 'estimated_salary',
        'Churn': 'churn',
        'Gender_Female': 'gender_female',
        'Gender_Male': 'gender_male',
        'HasCrCard_0.0': 'has_cr_card_0',
        'HasCrCard_1.0': 'has_cr_card_1'
        }, inplace=True)


        # Replace NaNs with None for SQL
        df = df.where(pd.notnull(df), None)

        # Ensure proper types
        df['customer_id'] = df['customer_id'].astype(str)
        df['is_active_member'] = df['is_active_member'].astype(int)
        df['churn'] = df['churn'].astype(int)

        return df

    except Exception as e:
        print(f" Error loading/processing CSV: {e}")
        return None

def insert_customer_data(conn, df):
    try:
        cursor = conn.cursor()
        records_to_insert = [tuple(row) for row in df.to_numpy()]
        sql_insert = """
        INSERT INTO customer (
            customer_id, age, tenure, balance, num_of_products, is_active_member,
            estimated_salary, churn, gender_female, gender_male, has_cr_card_0, has_cr_card_1
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            age=VALUES(age), tenure=VALUES(tenure), balance=VALUES(balance),
            num_of_products=VALUES(num_of_products), is_active_member=VALUES(is_active_member),
            estimated_salary=VALUES(estimated_salary), churn=VALUES(churn),
            gender_female=VALUES(gender_female), gender_male=VALUES(gender_male),
            has_cr_card_0=VALUES(has_cr_card_0), has_cr_card_1=VALUES(has_cr_card_1);
        """
        cursor.executemany(sql_insert, records_to_insert)
        conn.commit()
        print(f" Inserted {len(records_to_insert)} records into the table.")
    except Error as e:
        print(f" Error inserting data: {e}")
        conn.rollback()

# Run full pipeline
if __name__ == '__main__':
    db_name = "churn_customer_db"
    file_path = "churn_cleaned.csv"

    conn = create_connection()
    if conn:
        create_database(conn, db_name)
        conn.close()

    conn = create_connection(db_name)
    if conn:
        create_customer_table(conn)
        df = load_and_prepare_csv(file_path)
        if df is not None:
            insert_customer_data(conn, df)
        conn.close()
        print("\n Pipeline completed.")