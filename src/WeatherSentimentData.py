from os import path
import pandas as pd
import numpy as np

class WeatherSentimentData:
    def __init__(self, data_path):
        if not path.exists(path.join(data_path, 'weather-sentiment.csv')):
            return None
        df = pd.read_csv(path.join(data_path, 'weather-sentiment.csv'))

        df = df[['is_the_category_correct_for_this_tweet', 'sentiment', 'tweet_text',
                'what_emotion_does_the_author_express_specifically_about_the_weatherconfidence']]
        df.columns = ['category_correct', 'sentiment', 'tweet_text', 'confidence']
        df = df[df.category_correct == 'Yes']
        self.full_data = df[['sentiment', 'tweet_text', 'confidence']]

        mask = np.random.rand(len(df)) < 0.85
        self.train_data = self.full_data[mask]
        self.test_data = self.full_data[~mask]