import pandas as pd

dataframe = pd.read_csv('movielens/ml-100k.ratings')
rating = dataframe.values[:, 2].astype(int)
print(sum(rating) / len(rating))
