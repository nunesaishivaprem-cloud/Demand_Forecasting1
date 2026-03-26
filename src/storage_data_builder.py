import pandas as pd
import numpy as np

np.random.seed(7)

records = 5200

time_index = pd.date_range(
    start="2022-01-01",
    end="2024-12-31",
    freq="h"
)

zones = ["southindia-dc1", "southindia-dc2", "asiaedge-tokyo", "asiaedge-singapore"]
storage_services = ["object", "block", "archive"]

dataset = {
    "event_time": np.random.choice(time_index, records),
    "zone": np.random.choice(zones, records),
    "service_mode": np.random.choice(storage_services, records),
    "data_volume_tb": np.random.gamma(3.5, 40, records),
    "transaction_load": np.random.poisson(4500, records),
    "response_time_ms": np.random.normal(120, 35, records),
    "billing_amount": np.random.uniform(80, 900, records),
    "uptime_percent": np.random.uniform(96.5, 100, records),
    "festival_flag": np.random.choice([0, 1], records)
}

df = pd.DataFrame(dataset)

# introducing real-world data issues
df.loc[df.sample(frac=0.12).index, "data_volume_tb"] = np.nan
df.loc[df.sample(frac=0.07).index, "billing_amount"] = np.nan

df.to_csv("data/storage_usage_raw.csv", index=False)

print("Raw historical storage dataset created")
