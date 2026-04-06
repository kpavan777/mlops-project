import pandas as pd
import numpy as np
import os

def generate_data(path="data/churn.csv"):
    os.makedirs("data", exist_ok=True)

    np.random.seed(42)
    n = 5000

    data = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "contract_type": np.random.choice([0, 1, 2], n),
        "internet_service": np.random.choice([0, 1, 2], n),
        "support_calls": np.random.randint(0, 10, n),
    })

    data["churn"] = (
        ((data["tenure"] < 12) & (data["support_calls"] > 5)) |
        (data["monthly_charges"] > 90)
    ).astype(int)

    data.to_csv(path, index=False)
    print(f"Dataset saved at {path}")

if __name__ == "__main__":
    generate_data()
