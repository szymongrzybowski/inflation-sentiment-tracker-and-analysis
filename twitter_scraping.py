import snscrape.modules.twitter as sntwitter
import pandas as pd

keyword = 'inflation'
maxTweets = 10

tweets_list = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:2022-02-01 until:2022-08-30 -filter:links -filter:replies').get_items()):
    if i > maxTweets:
        break
    tweets_list.append([tweet.date, tweet.id, tweet.user.username, tweet.content])

tweets_df = pd.DataFrame(tweets_list, columns = ['Datetime', 'Tweet Id', 'Username', 'Text'])
print(tweets_df)
