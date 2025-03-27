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

# %%
print(new_df["is_duplicate"].value_counts())
print((new_df["is_duplicate"].value_counts() / new_df["is_duplicate"].count()) * 100)


# %%
qid = pd.Series(new_df["qid1"].tolist() + new_df["qid2"].tolist())
print("Total number of unique questions", np.unique(qid).shape[0])

x = qid.value_counts() > 1
print("Number of questions getting repeated", x[x].shape[0])


# %%
# feature Engineering --> need to create new features
new_df["q1_len"] = new_df["question1"].str.len()
new_df["q2_len"] = new_df["question2"].str.len()

# %%
new_df.head()
