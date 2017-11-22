import pandas as pd
from textblob import Blobber
from textblob.sentiments import PatternAnalyzer

# use Pattern Analyzer.
# The output data frame is sorted by the Polarity values

tb = Blobber(analyzer=PatternAnalyzer())

df = pd.read_csv('url_count_content_descending.csv', sep=',')

polarity = [None] * df.shape[0]
subjectivity = [None] * df.shape[0]

index = 0

for rows in df['content']:
    blob = tb(rows)
    polarity[index] = blob.sentiment.polarity
    subjectivity[index] = blob.sentiment.subjectivity
    index += 1
    print(index)

# df = df.drop('content', axis=1)
df['polarity'] = polarity
df['subjectivity'] = subjectivity
df.sort_values('polarity', ascending=True, inplace=True)
df.reset_index(drop=True)
df.to_csv('url_pattern_analyzer_sentiment.csv', index=False)
