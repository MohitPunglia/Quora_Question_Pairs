# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_csv("train.csv")


# %%
df.head()

# %%
df.shape

# %%
new_df = df.sample(30000, random_state=2)

# %%
new_df.shape

# %%
new_df.describe()

# %%
new_df.isnull().sum()  # checking for null values

# %%
new_df.duplicated().sum()  # checking for duplicate values
