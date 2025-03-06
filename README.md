# Customer Behavior Analysis

A data analysis project to uncover customer behavior patterns, segment customers, and provide actionable marketing insights.

## Overview

This project analyzes customer transaction data to help businesses understand their customer base better. It includes:

- Purchase pattern analysis by month and day of week
- Customer segmentation using clustering
- Demographic analysis by age and gender
- Calculation of customer lifetime value
- Generation of actionable business insights

## Features

- Data cleaning and preprocessing
- Customer engagement metrics analysis
- Purchase frequency and recency patterns
- Customer lifetime value calculation
- Demographic segmentation analysis
- Key marketing insights generation

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

pip install -r requirements.txt

## Usage

First, generate sample data (if you don't have your own dataset):
python sample_data_generator.py

Then run the analysis:
python customer_analysis.py

## Output

The script generates several visualizations saved as PNG files:
- monthly_sales.png
- day_purchases.png
- customer_segments.png
- age_spending.png
- gender_purchases.png

It also creates a text file with key business insights:
- customer_insights.txt

## Data Format

The expected data format is a CSV file with the following columns:
- customer_id: Unique identifier for each customer
- purchase_date: Date of purchase
- purchase_amount: Amount spent on the purchase
- age: Customer age
- gender: Customer gender


