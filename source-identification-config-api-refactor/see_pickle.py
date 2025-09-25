import pandas as pd
import pickle
# load grouped_events object
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

df = pd.DataFrame(grouped_events)
df.to_csv(r'group_events.csv')