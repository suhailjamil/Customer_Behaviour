import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans

plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)


def load_data():
    try:
        df = pd.read_csv('customer_data.csv')
        print(
            f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("Error: customer_data.csv file not found")
        return None


def clean_data(df):
    print("\n--- Cleaning Data ---")
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")

    critical_columns = ['customer_id', 'purchase_date', 'purchase_amount']
    df = df.dropna(subset=critical_columns)

    df['age'] = df['age'].fillna(df['age'].median())
    df['gender'] = df['gender'].fillna('Unknown')

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_month'] = df['purchase_date'].dt.month
    df['purchase_day'] = df['purchase_date'].dt.day_name()

    print(f"Data shape after cleaning: {df.shape}")
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")

    return df


def analyze_purchase_patterns(df):
    print("\n--- Purchase Pattern Analysis ---")

    monthly_purchases = df.groupby('purchase_month')[
        'purchase_amount'].agg(['sum', 'mean', 'count'])
    monthly_purchases.columns = ['Total Sales',
                                 'Average Purchase', 'Number of Purchases']
    print("\nMonthly Purchase Summary:")
    print(monthly_purchases)

    plt.figure(figsize=(12, 6))
    monthly_purchases['Total Sales'].plot(kind='bar', color='skyblue')
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_sales.png')

    day_purchases = df.groupby('purchase_day')[
        'purchase_amount'].agg(['sum', 'mean', 'count'])
    day_order = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_purchases = day_purchases.reindex(day_order)
    print("\nDay of Week Purchase Summary:")
    print(day_purchases)

    plt.figure(figsize=(12, 6))
    day_purchases['count'].plot(kind='bar', color='lightgreen')
    plt.title('Number of Purchases by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Purchases')
    plt.tight_layout()
    plt.savefig('day_purchases.png')

    return monthly_purchases, day_purchases


def customer_segmentation(df):
    print("\n--- Customer Segmentation ---")

    customer_metrics = df.groupby('customer_id').agg({
        'purchase_amount': ['sum', 'mean', 'count'],
        'purchase_date': ['min', 'max']
    })

    customer_metrics.columns = ['total_spent', 'avg_purchase',
                                'purchase_count', 'first_purchase', 'last_purchase']

    latest_date = df['purchase_date'].max()
    customer_metrics['recency'] = (
        latest_date - customer_metrics['last_purchase']).dt.days

    customer_metrics['tenure'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
    customer_metrics['tenure'] = customer_metrics['tenure'].clip(lower=1)

    customer_metrics['purchase_frequency'] = customer_metrics['purchase_count'] / \
        (customer_metrics['tenure'] / 30)
    customer_metrics['clv'] = customer_metrics['avg_purchase'] * \
        customer_metrics['purchase_frequency'] * 12

    print("\nCustomer Metrics Summary:")
    print(customer_metrics.describe())

    features = ['total_spent', 'purchase_count', 'recency']
    X = customer_metrics[features].copy()

    for feature in features:
        X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()

    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_metrics['cluster'] = kmeans.fit_predict(X)

    cluster_analysis = customer_metrics.groupby('cluster').agg({
        'total_spent': 'mean',
        'purchase_count': 'mean',
        'recency': 'mean',
        'purchase_frequency': 'mean',
        'clv': 'mean'
    })

    cluster_counts = customer_metrics['cluster'].value_counts().sort_index()
    cluster_analysis['customer_count'] = cluster_counts

    print("\nCustomer Segments:")
    print(cluster_analysis)

    plt.figure(figsize=(12, 8))

    x_feature = 'total_spent'
    y_feature = 'purchase_count'

    sns.scatterplot(
        x=x_feature,
        y=y_feature,
        hue='cluster',
        data=customer_metrics.reset_index(),
        palette='viridis',
        s=100,
        alpha=0.7
    )

    plt.title('Customer Segments')
    plt.xlabel(x_feature.replace('_', ' ').title())
    plt.ylabel(y_feature.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig('customer_segments.png')

    return customer_metrics, cluster_analysis


def demographic_analysis(df, customer_metrics):
    print("\n--- Demographic Analysis ---")

    customer_demo = df[['customer_id', 'age', 'gender']].drop_duplicates()
    analysis_df = pd.merge(customer_metrics.reset_index(),
                           customer_demo, on='customer_id')

    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    analysis_df['age_group'] = pd.cut(
        analysis_df['age'], bins=bins, labels=labels)

    age_analysis = analysis_df.groupby('age_group').agg({
        'total_spent': 'mean',
        'purchase_count': 'mean',
        'clv': 'mean',
        'customer_id': 'count'
    })

    age_analysis = age_analysis.rename(
        columns={'customer_id': 'customer_count'})
    print("\nPurchase Behavior by Age Group:")
    print(age_analysis)

    gender_analysis = analysis_df.groupby('gender').agg({
        'total_spent': 'mean',
        'purchase_count': 'mean',
        'clv': 'mean',
        'customer_id': 'count'
    })

    gender_analysis = gender_analysis.rename(
        columns={'customer_id': 'customer_count'})
    print("\nPurchase Behavior by Gender:")
    print(gender_analysis)

    plt.figure(figsize=(12, 6))
    age_analysis['total_spent'].plot(kind='bar', color='coral')
    plt.title('Average Total Spent by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Total Spent ($)')
    plt.tight_layout()
    plt.savefig('age_spending.png')

    plt.figure(figsize=(10, 6))
    gender_analysis['purchase_count'].plot(kind='bar', color='lightblue')
    plt.title('Average Purchase Count by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Purchase Count')
    plt.tight_layout()
    plt.savefig('gender_purchases.png')

    return age_analysis, gender_analysis


def generate_insights(monthly_purchases, day_purchases, cluster_analysis, age_analysis, gender_analysis):
    print("\n--- Key Business Insights ---")

    best_month = monthly_purchases['Total Sales'].idxmax()
    worst_month = monthly_purchases['Total Sales'].idxmin()
    best_day = day_purchases['sum'].idxmax()
    high_value_segment = cluster_analysis['total_spent'].idxmax()
    valuable_age = age_analysis['clv'].idxmax()

    insights = [
        f"1. The best performing month is month {best_month} with ${monthly_purchases.loc[best_month, 'Total Sales']:.2f} in sales.",
        f"2. The worst performing month is month {worst_month} with ${monthly_purchases.loc[worst_month, 'Total Sales']:.2f} in sales.",
        f"3. {best_day} is the day with highest purchase activity.",
        f"4. Customer segment {high_value_segment} represents high-value customers with average CLV of ${cluster_analysis.loc[high_value_segment, 'clv']:.2f}.",
        f"5. The {valuable_age} age group has the highest customer lifetime value at ${age_analysis.loc[valuable_age, 'clv']:.2f}."
    ]

    for insight in insights:
        print(insight)

    with open('customer_insights.txt', 'w') as f:
        f.write("# Customer Behavior Analysis Insights\n\n")
        for insight in insights:
            f.write(insight + "\n")

    print("\nInsights saved to customer_insights.txt")


def main():
    print("=== Customer Behavior Analysis ===")

    df = load_data()
    if df is None:
        return

    df = clean_data(df)
    monthly_purchases, day_purchases = analyze_purchase_patterns(df)
    customer_metrics, cluster_analysis = customer_segmentation(df)
    age_analysis, gender_analysis = demographic_analysis(df, customer_metrics)
    generate_insights(monthly_purchases, day_purchases,
                      cluster_analysis, age_analysis, gender_analysis)

    print("\nAnalysis complete! Check the output files for visualizations.")


if __name__ == "__main__":
    main()
