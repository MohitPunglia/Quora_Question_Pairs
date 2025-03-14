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

# %%
ques_df = new_df[["question1", "question2"]]
ques_df.head()

# %%
from sklearn.feature_extraction.text import CountVectorizer

questions = list(ques_df["question1"]) + list(ques_df["question2"])

# %%
cv = CountVectorizer(max_features=5000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)

# %%
questions[1:10]

# %%
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.head()

# %%
temp_df.shape

# %%
temp_df["is_duplicate"] = new_df["is_duplicate"]

# %%
temp_df.head()

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    temp_df.iloc[:, 0:-1].values,
    temp_df.iloc[:, -1].values,
    test_size=0.2,
    random_state=2,
)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
