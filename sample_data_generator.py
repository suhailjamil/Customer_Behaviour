import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(num_customers=500, num_transactions=5000):
    np.random.seed(42)

    customer_ids = [f'CUST{i:04d}' for i in range(1, num_customers + 1)]

    genders = ['Male', 'Female', 'Non-binary', None]
    gender_weights = [0.48, 0.48, 0.02, 0.02]

    customer_data = {
        'customer_id': customer_ids,
        'age': np.random.randint(18, 75, size=num_customers),
        'gender': np.random.choice(genders, size=num_customers, p=gender_weights)
    }

    missing_age_indices = np.random.choice(
        range(num_customers), size=int(num_customers * 0.05), replace=False)
    customer_data['age'] = np.array(customer_data['age'], dtype=float)
    customer_data['age'][missing_age_indices] = np.nan

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    transaction_data = []

    for _ in range(num_transactions):
        customer_idx = np.random.randint(0, num_customers)
        customer_id = customer_ids[customer_idx]

        days_offset = np.random.randint(0, 365)
        purchase_date = start_date + timedelta(days=days_offset)

        age = customer_data['age'][customer_idx]
        if np.isnan(age):
            base_amount = np.random.uniform(20, 200)
        else:
            base_amount = np.random.uniform(20, 100) + (age / 10)

        categories = ['Electronics', 'Clothing',
                      'Home', 'Beauty', 'Food', 'Books']
        category = np.random.choice(categories)

        if category in ['Electronics', 'Home']:
            purchase_amount = base_amount * np.random.uniform(1.5, 3.0)
        else:
            purchase_amount = base_amount

        purchase_amount = round(
            purchase_amount * np.random.uniform(0.8, 1.2), 2)

        transaction = {
            'transaction_id': f'TXN{len(transaction_data) + 1:06d}',
            'customer_id': customer_id,
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'purchase_amount': purchase_amount,
            'category': category
        }

        transaction_data.append(transaction)

    customers_df = pd.DataFrame(customer_data)
    transactions_df = pd.DataFrame(transaction_data)

    df = pd.merge(transactions_df, customers_df, on='customer_id')

    df.to_csv('customer_data.csv', index=False)
    print(
        f"Generated sample data with {num_customers} customers and {num_transactions} transactions")
    print("Data saved to customer_data.csv")

    return df


if __name__ == "__main__":
    generate_sample_data()
