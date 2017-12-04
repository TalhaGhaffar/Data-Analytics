import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime
import os
import nltk

from textblob import TextBlob

_dataset2_dir = "C:\\Users\\talha\\Documents\\DA\\da_project\\dataset2\\"
email_file = _dataset2_dir + "email_info.csv"
device_file = _dataset2_dir + "device_info.csv"

def get_polarity_and_subjectivity(text):
    blob = TextBlob(text)
    return pd.Series({"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity})

chunk_size = 100000
email_chunks = pd.read_csv(_dataset2_dir + "email_info.csv", index_col=0, chunksize=chunksize)

# breakAt = 3
numChunks = 0
appended_data = []

for chunk in email_chunks:
	print ("\n ============ Processing chunk # "+ str(numChunks) +" =============\n")
	pol_and_subj_df = chunk['content'].apply(get_polarity_and_subjectivity)
	appended_data.append(pol_and_subj_df)
	print (pol_and_subj_df.head())
	print (pol_and_subj_df.shape)
	numChunks+=1
	# if numChunks == breakAt:
	# 	break

print ("\n ============ Processing Finished =============\n")
appended_data = pd.concat(appended_data, axis=0)
print (appended_data.head())
print (appended_data.shape)
print (type(appended_data))
appended_data.to_pickle(_dataset2_dir+"email_sentiment_analysis_df")

print (numChunks)


# print (type(email_df))
# print (email_df.shape)
# email_content = email_df['content']
# pol_and_subj_df = email_df['content'].apply(get_polarity_and_subjectivity)

# print (pol_and_subj_df.head())

# device_chunks = pd.read_csv(device_file, index_col=0, chunksize=10000)

# for device_chunk in device_chunks:
# 	print (type(device_chunk))