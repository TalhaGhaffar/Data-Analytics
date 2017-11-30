import pandas as pd
import numpy as np
import os
import seaborn as sns

# Global Variables
neg_threshold = 0.6

http_info_file = 'C:/Users/hkuad/Desktop/Subjects/DA/DataSets/NewData/DataSet2/http_info.csv'

url_sentiment_file = 'url_naive_bayes_sentiment.csv'

user_url_sentiment_csv = 'url_user_sentiment_list.csv'

psychometric_analysis_file = 'C:/Users/hkuad/Desktop/Subjects/DA/DataSets/NewData/DataSet2/psychometric_info.csv'

psychometric_url_visit_analysis_file = 'url_psychometric_user_analysis.csv'


def generate_url_user_sentiment_list():
    url_df = pd.read_csv(url_sentiment_file, sep=',', usecols=[0, 5])

    user_df = pd.DataFrame()

    url_df = url_df.loc[url_df['p_neg'] > neg_threshold]

    iteration = 1

    # id, date, user, pc, url, content
    for df in pd.read_csv(http_info_file, sep=',', usecols=[2, 4], chunksize=2000000):
        user_df = user_df.append(pd.merge(df, url_df, on='url'), ignore_index=True)

        print(iteration)

        iteration += 1

    user_df.to_csv(user_url_sentiment_csv, index=False)

    print(user_df.shape)

    return user_df


def aggregate_user_url_visits():
    global user_df

    user_df = user_df.groupby('user').agg({'url': np.size, 'p_neg': np.mean})

    user_df.reset_index(inplace=True)

    user_df.rename(columns={'user': 'user_id', 'url': 'sum_neg_url_hits', 'p_neg': 'avg_pneg_neg_urls'}, inplace=True)

    # user_df.sort_values('sum_neg_url_hits', ascending=False, inplace=True)

    return user_df


def publish_psychometric_url_visit_pattern():
    psychometric_df = pd.read_csv(psychometric_analysis_file, sep=',')

    psychometric_df = pd.merge(psychometric_df, user_df, on='user_id')

    psychometric_df.sort_values('sum_neg_url_hits', ascending=False, inplace=True)

    psychometric_df.reset_index(inplace=True, drop=True)

    psychometric_df.to_csv(psychometric_url_visit_analysis_file, index=False)


# Entry point for execution of main program
if os.path.exists(user_url_sentiment_csv):
    user_df = pd.read_csv(user_url_sentiment_csv, sep=',')
else:
    user_df = generate_url_user_sentiment_list()

aggregate_user_url_visits()

publish_psychometric_url_visit_pattern()
