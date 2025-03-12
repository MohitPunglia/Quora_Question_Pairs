# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("train.csv")

# %%
df.head(10)

# %%
df.shape

# %%
new_df = df.sample(50000, random_state=2)

# %%
new_df.shape

# %%
new_df.head()

# %%
new_df.isnull().sum()
