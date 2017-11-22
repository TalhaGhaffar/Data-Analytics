import pandas as pd
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

# use naive bayes Analyzer. Gives better results than PatternAnalyzer
# The output data frame is sorted by the p_neg values

tb = Blobber(analyzer=NaiveBayesAnalyzer())

df = pd.read_csv('url_count_content_descending.csv', sep=',')

classification = [None] * df.shape[0]
p_pos = [None] * df.shape[0]
p_neg = [None] * df.shape[0]

index = 0

for rows in df['content']:
    blob = tb(rows)
    classification[index] = blob.sentiment.classification
    p_pos[index] = blob.sentiment.p_pos
    p_neg[index] = blob.sentiment.p_neg
    index += 1
    print(index)

# df = df.drop('content', axis=1)
df['classification'] = classification
df['p_pos'] = p_pos
df['p_neg'] = p_neg
df.sort_values('p_neg', ascending=False, inplace=True)
df.reset_index(drop=True)
df.to_csv('url_naive_bayes_sentiment.csv', index=False)
