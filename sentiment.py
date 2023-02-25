# Step 1: connect to twitter and finnhub API
import tweepy
import finnhub

# Set up Twitter API credentials
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Set up Finnhub API credentials
finnhub_client = finnhub.Client(api_key="YOUR_API_KEY")


# Step 2: Query the Twitter API to obtain tweet data for a set of relevant companies or stocks
# Set up list of companies or stocks to query
companies = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']

# Define start date for tweet data
start_date = '2021-01-01'

# Define search query and filter parameters
search_query = ' OR '.join(companies)
search_params = {
    'q': search_query,
    'count': 100,
    'result_type': 'recent',
    'lang': 'en',
    'since_id': start_date,
    'tweet_mode': 'extended'
}

# Query Twitter API for tweets matching search query
tweet_data = []
for tweet in tweepy.Cursor(api.search_tweets, **search_params).items():
    tweet_data.append({
        'id': tweet.id_str,
        'created_at': tweet.created_at,
        'text': tweet.full_text,
        'user': tweet.user.screen_name
    })

# Step 3: Preprocess the tweet data
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Define preprocessing functions
def clean_text(text):
    # Remove URLs, mentions, hashtags, and punctuation
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase and remove whitespace
    text = text.lower().strip()
    return text

def tokenize_text(text):
    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Clean and tokenize tweet text
cleaned_tweets = [clean_text(tweet['text']) for tweet in tweet_data]
tokenized_tweets = [tokenize_text(tweet) for tweet in cleaned_tweets]

# Vectorize tweet text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_tweets)

# Step 4: Use the Finnhub API to obtain sentiment data
# Query Finnhub API for sentiment data for each company or stock
sentiment_data = []
for company in companies:
    sentiment = finnhub_client.news_sentiment(company)
    sentiment_data.append({
        'company': company,
        'sentiment': sentiment['companyNewsScore'],
        'date': sentiment['buzz']['articles'][0]['datetime']
    })

    
# Step 5: Combine the tweet data and sentiment data for each company or stock into a single dataset
# Create a dictionary mapping company to sentiment
sentiment_dict = {d['company']: d['sentiment'] for d in sentiment_data}

# Combine tweet data and sentiment data into a single dataset
dataset = []
for tweet, tokens in zip(tweet_data, tokenized_tweets):
    # Check if tweet mentions a relevant company or stock
    for company in companies:
        if company.lower() in tweet['text'].lower():
            # Add sentiment score for the company to the tweet data
            dataset.append({
                'id': tweet['id'],
                'created_at': tweet['created_at'],
                'text': tweet['text'],
                'user': tweet['user'],
                'company': company,
                'sentiment': sentiment_dict[company],
                'tokens': tokens
            })
            break

# Step 6: Select relevant features for sentiment analysis
from sklearn.feature_extraction.text import TfidfVectorizer

# Select relevant features for sentiment analysis
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_tweets)
features = tfidf_vectorizer.get_feature_names()


# Step 7: Train a Naive Bayes Classifier model on the dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X = X_tfidf
y = [d['sentiment'] for d in dataset]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes Classifier model on the training set
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# Step 8: Evaluate the performance of the Naive Bayes Classifier model
from sklearn.metrics import classification_report

# Evaluate the performance of the Naive Bayes Classifier model on the testing set
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Step 9: Use the trained model to predict sentiment for new tweet data
# Preprocess new tweet data
new_tweet_text = 'This is a new tweet about AAPL.'
cleaned_new_tweet = clean_text(new_tweet_text)
tokenized_new_tweet = tokenize_text(cleaned_new_tweet)

# Vectorize new tweet data
new_tweet_vector = tfidf_vectorizer.transform([cleaned_new_tweet])

# Predict sentiment for new tweet data using the trained model
new_tweet_sentiment = nb_model.predict(new_tweet_vector)[0]
print(new_tweet_sentiment)


# Step 10: Backtest the performance of the model on historical data
import pandas as pd

# Load historical stock price data for a relevant company or stock
company = 'AAPL'
start_date = '2021-01-01'
end_date = '2022-01-01'
df = finnhub_client.stock_candles(company, 'D', start_date, end_date)

# Define function to get sentiment for a given date and company
def get_sentiment_for_date(sentiment_data, company, date):
    for d in sentiment_data:
        if d['company'] == company and d['date'][:10] == date:
            return d['sentiment']
    return None

# Add sentiment data to the stock price data
df['date'] = pd.to_datetime(df['t'], unit
# Add sentiment data to the stock price data
df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
df['sentiment'] = [get_sentiment_for_date(sentiment_data, company, str(date)) for date in df['date']]

# Drop rows with missing sentiment data
df = df.dropna(subset=['sentiment'])

# Convert sentiment data to binary labels
df['label'] = [1 if s == 'positive' else 0 for s in df['sentiment']]

# Select relevant features for the backtesting model
X = df[['o', 'h', 'l', 'c']].values
y = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Evaluate the performance of the Logistic Regression model on the testing set
y_pred = logreg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 11: Use the sentiment prediction model to make buy/sell decisions for a given stock
# Load latest stock price data for the relevant company or stock
df_latest = finnhub_client.stock_candles(company, 'D', start_date, end_date)

# Add sentiment data to the latest stock price data
df_latest['date'] = pd.to_datetime(df_latest['t'], unit='s').dt.date
df_latest['sentiment'] = [get_sentiment_for_date(sentiment_data, company, str(date)) for date in df_latest['date']]

# Drop rows with missing sentiment data
df_latest = df_latest.dropna(subset=['sentiment'])

# Convert sentiment data to binary labels
df_latest['label'] = [1 if s == 'positive' else 0 for s in df_latest['sentiment']]

# Use the sentiment prediction model to make buy/sell decisions for the stock
latest_features = df_latest[['o', 'h', 'l', 'c']].values
latest_predictions = logreg_model.predict(latest_features)

# Calculate the total return on investment for the buy/sell decisions
total_return = 0
shares_owned = 0
for i, row in df_latest.iterrows():
    if latest_predictions[i] == 1 and shares_owned == 0:
        shares_owned = 10000 / row['o']
        total_return -= 10000
    elif latest_predictions[i] == 0 and shares_owned > 0:
        total_return += shares_owned * row['o']
        shares_owned = 0
print(total_return)

# Step 12: Visualize the results of the buy/sell decisions over time

import matplotlib.pyplot as plt

# Visualize the results of the buy/sell decisions over time
df_latest['total_return'] = 0
shares_owned = 0
for i, row in df_latest.iterrows():
    if latest_predictions[i] == 1 and shares_owned == 0:
        shares_owned = 10000 / row['o']
        df_latest.at[i, 'total_return'] -= 10000
    elif latest_predictions[i] == 0 and shares_owned > 0:
        df_latest.at[i, 'total_return'] += shares_owned * row['o']
        shares_owned = 0
df_latest['cumulative_return'] = df_latest['total_return'].cumsum

                            
