from os import path
import pandas as pd
import numpy as np

class WeatherSentimentData:
    def __init__(self, data_path, use_generated_data=False):
        weather_data_path = path.join(data_path, 'weather-sentiment.csv')
        if not path.exists(weather_data_path):
            return None
        df = pd.read_csv(weather_data_path)

        df = df[['is_the_category_correct_for_this_tweet', 'sentiment', 'tweet_text',
                'what_emotion_does_the_author_express_specifically_about_the_weatherconfidence']]
        df.columns = ['category_correct', 'sentiment', 'tweet_text', 'confidence']
        df = df[df.category_correct == 'Yes']
        df = df[df.sentiment != 'I can\'t tell']

        if use_generated_data:
            weather_data_path = path.join(data_path, 'weather-sentiment-gpt.csv')
            if not path.exists(weather_data_path):
                return None
            generated_df = pd.read_csv(weather_data_path)
            df = df.append(generated_df, ignore_index=True)

        self.full_data = df[['sentiment', 'tweet_text', 'confidence']]