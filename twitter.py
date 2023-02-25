import tweepy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TwitterAPI:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        # set up authentication credentials for Twitter API
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def search_tweets(self, query, lang='en', count=100):
        # search for tweets matching a given query
        tweets = tweepy.Cursor(self.api.search_tweets, q=query, lang=lang).items(count)
        return tweets

    def get_user_tweets(self, user, count=100):
        # get tweets from a specific user's timeline
        tweets = tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode='extended').items(count)
        return tweets

    @staticmethod
    def preprocess_tweet_text(tweet_text):
        # preprocess tweet text by removing URLs, mentions, punctuation, and stop words, and stemming words
        # based on NLTK library
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()

        # remove URLs, mentions, and non-alphanumeric characters
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        tweet_text = re.sub(r'@[A-Za-z0-9]+', '', tweet_text)
        tweet_text = re.sub(r'\W+', ' ', tweet_text)

        # tokenize text into individual words
        words = word_tokenize(tweet_text)

        # remove stop words and stem words
        words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]

        # rejoin words into a cleaned text string
        cleaned_text = ' '.join(words)
        return cleaned_text

    def get_user_tweets(self, user, min_followers=0, count=100):
      # get tweets from a specific user's timeline, filtered by number of followers
      tweets = []
      for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode='extended').items(count):
          if tweet.user.followers_count >= min_followers:
              tweets.append(tweet)
      return tweets
