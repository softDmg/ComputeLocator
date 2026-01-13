from cluster_data_client import ClusterApiClient
import pandas as pd


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((f"{new_key}.start", v[0]))
            items.append((f"{new_key}.end", v[-1]))
        else:
            items.append((new_key, v))
    return dict(items)

def feature_engineer(df):
    df["states.duration.end"] = df["states.process_uptime_seconds.end"] - df["states.process_uptime_seconds.start"]
    df = df.drop(columns=["states.process_uptime_seconds.start", "states.process_uptime_seconds.end"])

    df["states.cpu_duration_second.end"] = df["states.process_cpu_seconds_total.end"] - df["states.process_cpu_seconds_total.start"]
    df = df.drop(columns=["states.process_cpu_seconds_total.start", "states.process_cpu_seconds_total.end"])

    df = df.loc[:, df.nunique() > 1]
    return df

cluster_data_client = ClusterApiClient(api_name="localhost")
input_data = feature_engineer(pd.DataFrame([flatten_dict(e) for e in cluster_data_client.getHistory()]))

# model training
cols = input_data.columns
X_cols = [c for c in cols if "input." in c or ".start" in c or c in ["pod", "function", "exec_duration_ns"] ]

Y_cols = [c for c in cols if ".end" in c]

print("Finished")

# model testing

# plotting results

