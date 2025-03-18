# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("train.csv")

# %%
df.shape

# %%
df.sample(5)


# %%
df.info()

# %% [markdown]
# #Missing values

# %%
df.isnull().sum()

# %%
df.duplicated().sum()
