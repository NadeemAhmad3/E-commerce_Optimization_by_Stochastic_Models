import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.orders_path = os.path.join(dataset_path, "olist_orders_dataset.csv")
        self.items_path = os.path.join(dataset_path, "olist_order_items_dataset.csv")
        self.processed_path = "processed_log.csv"

    def load_and_process(self):
        """
        Phase 1: Data Preparation Pipeline
        Generates processed_log.csv with:
        1. Inter-arrival Time
        2. Service Time
        3. Cost of Delay (Price)
        4. Failure Flag
        """
        print("Loading raw datasets...")
        try:
            orders_df = pd.read_csv(self.orders_path)
            items_df = pd.read_csv(self.items_path)
        except FileNotFoundError as e:
            return None, f"Error: Could not find dataset files. {str(e)}"

        # Convert timestamps
        time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date']
        for col in time_cols:
            orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')

        # Sort by purchase time for inter-arrival calculation
        orders_df = orders_df.sort_values('order_purchase_timestamp')

        # 1. Inter-arrival Time (in minutes)
        # Time difference between row N and N-1
        orders_df['inter_arrival_time'] = orders_df['order_purchase_timestamp'].diff().dt.total_seconds() / 60.0
        orders_df['inter_arrival_time'] = orders_df['inter_arrival_time'].fillna(0) # First order

        # 2. Service Time (in days)
        # order_delivered_carrier_date minus order_approved_at
        orders_df['service_time'] = (orders_df['order_delivered_carrier_date'] - orders_df['order_approved_at']).dt.total_seconds() / (24 * 3600)
        
        # Filter out negative or unrealistic service times (data cleaning)
        orders_df = orders_df[orders_df['service_time'] > 0]

        # 3. Cost of Delay (Price)
        # We need to merge with items to get price. 
        # An order might have multiple items. We'll sum the price per order to get total order value/risk.
        order_value = items_df.groupby('order_id')['price'].sum().reset_index()
        merged_df = pd.merge(orders_df, order_value, on='order_id', how='inner')

        # 4. Failure Flag (is_defective)
        # Simulate 5% of orders as "Returns" / Defective
        np.random.seed(42) # For reproducibility
        num_rows = len(merged_df)
        merged_df['is_defective'] = np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])

        # Select only necessary columns for the dashboard
        final_df = merged_df[[
            'order_id', 
            'order_purchase_timestamp', 
            'inter_arrival_time', 
            'service_time', 
            'price', 
            'is_defective'
        ]].copy()

        # Rename price to cost_of_delay_risk for clarity as per prompt
        final_df.rename(columns={'price': 'cost_of_delay_risk'}, inplace=True)

        print(f"Processing complete. Saving {len(final_df)} rows to {self.processed_path}...")
        final_df.to_csv(self.processed_path, index=False)
        
        return final_df, "Success"

if __name__ == "__main__":
    loader = DataLoader()
    df, msg = loader.load_and_process()
    print(msg)
