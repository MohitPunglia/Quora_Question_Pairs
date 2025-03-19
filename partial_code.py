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


# %%
# Distribution of the target variable
print(df["is_duplicate"].value_counts())
print((df["is_duplicate"].value_counts() / df["is_duplicate"].count()) * 100)
df["is_duplicate"].value_counts().plot(kind="bar")


# %%
# Number of unique questions and repeated questions

# Number of unique questions and repeated questions

qid_series = pd.Series(df["qid1"].tolist() + df["qid2"].tolist())
unique_qid_count = qid_series.nunique()
print("Number of unique Questions", unique_qid_count)
x = qid_series.value_counts() > 1
print("Number of repeated Questions", x[x].shape[0])

# %%
# Repeated questions histogram
plt.hist(qid_series.value_counts().values, bins=160)
plt.title("Log-Histogram of question appearance counts")
plt.yscale("log")
plt.show()


# %%
