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

# %%
new_df["q1_num_words"] = new_df["question1"].apply(lambda x: len(x.split()))
new_df["q2_num_words"] = new_df["question2"].apply(lambda x: len(x.split()))

# %%
new_df.head()


# %%
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row["question1"].split()))
    w2 = set(map(lambda word: word.lower().strip(), row["question2"].split()))
    return len(w1 & w2)


# %%
new_df["common_words"] = new_df.apply(common_words, axis=1)
new_df.head()


# %%
def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row["question1"].split()))
    w2 = set(map(lambda word: word.lower().strip(), row["question2"].split()))
    return len(w1) + len(w2)


# %%
new_df["total_words"] = new_df.apply(total_words, axis=1)
new_df.head()

# %%
new_df["word_share"] = new_df["common_words"] / new_df["total_words"]
new_df.head()

# %%
sns.displot(new_df["common_words"])
plt.show()


# %%
ques_df = new_df[["question1", "question2"]]
ques_df.head()

# %%
final_df = new_df.drop(columns=["id", "qid1", "qid2", "question1", "question2"])
print(final_df.shape)
final_df.head()

# %%
from sklearn.feature_extraction.text import CountVectorizer

questions = list(ques_df["question1"]) + list(ques_df["question2"])

cv = CountVectorizer(max_features=5000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)


# %%
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape

# %%
final_df = pd.concat([final_df, temp_df], axis=1)
print(final_df.shape)
final_df.head()
