#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:59:25 2022

@author: sabineherberth
"""

import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "((#IchBinArmutsbetroffen OR #IchBinArmutbetroffen OR #ichbinarmutsbetroffen OR #ichbinarmutbetroffen) until:2022-12-16 since:2022-05-01)"
tweets = []
limit = 200000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.id, tweet.username, tweet.content, tweet.url, 
                       tweet.user, tweet.replyCount, tweet.retweetCount, 
                       tweet.likeCount, tweet.source, tweet.retweetedTweet, 
                       tweet.quotedTweet])
        
df = pd.DataFrame(tweets, columns=['Date', 'ID', 'Username', 'Tweet', 'URL', 
                                   'User', 'replycount', 'retweetcount', 
                                   'likecount', 'source', 'retweeted_tweet', 'quoted_tweet' ])

#print(df)

# to save to csv

path = '/Users/sabineherberth/Documents/02_UNI/Master/Faecher/Masterarbeit/Twitter_Data/'
df.to_csv(path + 'tweets_ichbinarmutsbetroffen.csv')

#df.to_csv('tweets.csv')