 # -*- coding: utf-8 -*-

"""
@author: sabineherberth

"""
# =============================================================================
# Import packages 
# =============================================================================

import tweepy
import time
import pandas as pd

# =============================================================================
# Retrieving tweets with Twitter API Academic Access 
# =============================================================================

bearer_token = 'insert_bearer_token'

client = tweepy.Client(bearer_token, wait_on_rate_limit=True)

armut_tweets = []
for response in tweepy.Paginator(client.search_all_tweets, 
                                 query = '(#IchBinArmutsbetroffen OR #IchBinArmutbetroffen OR #ichbinarmutsbetroffen OR #ichbinarmutbetroffen) -is:retweet',
                                 user_fields = ['username', 'public_metrics', 'description', 'location'],
                                 tweet_fields = ['created_at', 'geo', 'public_metrics', 'text'],
                                 expansions = 'author_id',
                                 start_time = '2022-05-01T00:00:00Z',
                                 end_time = '2022-12-16T00:00:00Z',
                              max_results=500):
    time.sleep(1)
    armut_tweets.append(response)

result = []
user_dict = {}
# Loop through each response object
for response in armut_tweets:
    # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
    for user in response.includes['users']:
        user_dict[user.id] = {'username': user.username, 
                              'followers': user.public_metrics['followers_count'],
                              'tweets': user.public_metrics['tweet_count'],
                              'description': user.description,
                              'location': user.location
                             }
    for tweet in response.data:
        # For each tweet, find the author's information
        author_info = user_dict[tweet.author_id]
        # Put all of the information we want to keep in a single dictionary for each tweet
        result.append({'author_id': tweet.author_id, 
                       'username': author_info['username'],
                       'author_followers': author_info['followers'],
                       'author_tweets': author_info['tweets'],
                       'author_description': author_info['description'],
                       'author_location': author_info['location'],
                       'text': tweet.text,
                       'created_at': tweet.created_at,
                       'retweets': tweet.public_metrics['retweet_count'],
                       'replies': tweet.public_metrics['reply_count'],
                       'likes': tweet.public_metrics['like_count'],
                       'quote_count': tweet.public_metrics['quote_count']
                      })

# Change this list of dictionaries into a dataframe
df = pd.DataFrame(result)

# =============================================================================
# Export tweets to csv
# =============================================================================

path = '/Users/sabineherberth/Documents/02_UNI/Master/Faecher/Masterarbeit/Twitter_API/Datensatz/'
df.to_csv(path + 'tweets_ichbinarmutsbetroffen_no_retweets.csv')

#print (df[0:50])


