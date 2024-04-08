import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

## Define sentiment words
love_words = ["love", "joy", "enchanting", "captivating", "beautiful", "romance", "happiness", "healing", "connection",
              "adventure", "shiny", "sweet", "nice", "light", "blue", "lightblue", "yellow", "pink", "gold", "green",
              "purple", "courage", "faith", "passion", "amazing", "new", "wonderful", "adoration", "affection",
              "devotion", "bliss", "euphoria", "fondness", "infatuation", "passionate",
              "tenderness", "warmth"]

heartbreak_words = ["upset", "wait", "hate", "dislike", "scared", "angry", "confused", "stupid", "fear", "sadness",
                    "doubts", "doubt", "hurt", "pain", "heartbreak", "dark", "tragic", "lonely", "loneliness",
                    "loathing", "devastating", "devastated", "solitary", "cry", "cold", "tears", "teardrops",
                    "turmoil", "demons", "terror", "terrors", "black", "reputation", "brown", "darkgreen", "grey",
                    "loss", "bye", "goodbye",
                    "agony", "despair", "grief", "melancholy", "sorrow", "misery", "regret", "betrayal", "abandonment",
                    "desolation"]

optimism_words = ["finally", "relief", "hope", "dreams", "wonderment", "wonder", "wonderful", "smile", "intrigue",
                  "happy", "sunshine", "bright", "uplifting", "positive", "cheerful", "optimistic", "inspiration",
                  "excitement", "vibrant",
                  "anticipation", "confidence", "faith", "gratitude", "hopeful", "positivity", "rejuvenated",
                  "resilience", "serenity", "trust"]

anger_words = ["angry", "anger", "hate", "hatred", "mad", "outraged", "resentment", "fury", "dislike", "bad", "unsafe",
               "gross",
               "loathing", "loathe", "goodbye", "liar", "lies", "lie", "rage", "bitter", "hostile", "vindictive",
               "outrageous", "provoked", "enraged", "irritated", "furious", "infuriated", "aggravation", "disgust",
               "hostility", "irritation", "outrage", "resentment", "wrath", "annoyance",
               "frustration", ]

reflection_words = ["loss", "bye", "enchanting", "demons", "terror", "terrors", "black", "reputation", "brown",
                    "darkgreen", "grey", "memories", "nostalgia", "contemplate", "introspection", "ponder",
                    "reminisce", "meditate", "introspective", "thoughtful", "reflective", "contemplation", "insight",
                    "meditation", "pensive", "rumination", "self-awareness",
                    "thought-provoking", "introspective", "philosophical", "soul-searching"]


## Define sentiment analysis function to calculate percentages
def calculate_sentiment_percentages(text):
    """
    Calculates sentiment percentages
    :param text: A text file (in this case, a csv for a particular album.
    :return: A pandas dataframe with one row per album, and  columns with percentages of words associated with that emotion
    """
    words = text.lower().split()
    total_count = len(words)

    love_count = sum(word in love_words for word in words)
    heartbreak_count = sum(word in heartbreak_words for word in words)
    optimism_count = sum(word in optimism_words for word in words)
    anger_count = sum(word in anger_words for word in words)
    reflection_count = sum(word in reflection_words for word in words)

    love_percentage = love_count / total_count * 100
    heartbreak_percentage = heartbreak_count / total_count * 100
    optimism_percentage = optimism_count / total_count * 100
    anger_percentage = anger_count / total_count * 100
    reflection_percentage = reflection_count / total_count * 100

    return love_percentage, heartbreak_percentage, optimism_percentage, anger_percentage, reflection_percentage


class LyricAnalyzer:
    def __init__(self):
        self.df = pd.DataFrame()  # initializes an empty data frame to append data to
        nltk.download('words')
        nltk.download('punkt')
        nltk.download('words')
        nltk.download('stopwords')

    def load_and_prepare_text_files(self, file_paths):
        """ Load in multiple text files representing different albums """
        dfs = []
        for file_path in file_paths:
            album_name = file_path.split("/")[-1].split(".csv")[0]  # Extract album name from file path
            df = pd.read_csv(file_path)
            df['Album'] = album_name  # Add album name as a new column
            dfs.append(df)
            print(f"CSV file '{album_name}' loaded successfully.")
        combined_df = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames into one

        # Group the data by 'Album' and concatenate the lyrics for each album
        self.df = combined_df.groupby('Album')['Lyrics'].apply(lambda x: ' '.join(x)).reset_index()

        self.df['Lyrics'] = self.df['Lyrics'].apply(self.preprocess_text)  # preprocess the lyrics



    def preprocess_text(self, text):
        """ Preprocess the texts """
        text = text.lower() # make all the text lowercase
        text = ''.join(c for c in text if c not in punctuation) # takes out punctuation
        words = word_tokenize(text) # Tokenize the text into words
        stop_words = set(stopwords.words('english')) # make sure all words are english
        words = [word for word in words if word not in stop_words] # remove stop words
        return ' '.join(words) # join all the words back into a string

    def extract_colors(self, lyrics):
        # Define a regular expression pattern to match color names
        color_pattern = re.compile(r'\b(?:red|blue|green|yellow|purple|pink|orange|black|grey|gold|brown)\b', flags=re.IGNORECASE)

        # Find all color mentions in the lyrics
        colors_mentioned = color_pattern.findall(lyrics)

        return colors_mentioned

    def plot_colors_sentiment(self):
        # Initialize a DataFrame to store colors mentioned and sentiment for each album
        album_colors_sentiment = pd.DataFrame(columns=['Album', 'Colors', 'Sentiment'])

        # Loop through each album
        for index, row in self.df.iterrows():
            # Extract colors mentioned in the lyrics
            colors_mentioned = self.extract_colors(row['Lyrics'])

            # Determine the majority sentiment for the album
            love, heartbreak, optimism, anger, reflection = calculate_sentiment_percentages(row['Lyrics'])
            majority_sentiment = max(love, heartbreak, optimism, anger, reflection, key=lambda x: x[1])[0]

            # Append album, colors, and sentiment to the DataFrame
            album_colors_sentiment = album_colors_sentiment.append({'Album': row['Album'],
                                                                    'Colors': colors_mentioned,
                                                                    'Sentiment': majority_sentiment},
                                                                   ignore_index=True)

        # Plot the colors mentioned in each album and color them according to their sentiment
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=album_colors_sentiment, x='Album', y='Colors', hue='Sentiment', palette='coolwarm')
        plt.title('Colors Mentioned in Lyrics by Album with Sentiment')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Colors Mentioned')
        plt.xlabel('Album')
        plt.legend(title='Sentiment')
        plt.show()

    def analyze_sentiment(self):
        sentiment_df = self.df.copy()  # Create a copy of the prepared DataFrame to store sentiment analysis results

        # Apply sentiment analysis to the lyrics text
        sentiment_df['Love'], sentiment_df['Heartbreak'], sentiment_df['Optimism'], sentiment_df['Anger'], sentiment_df[
            'Reflection'] = zip(
            *sentiment_df['Lyrics'].apply(calculate_sentiment_percentages))

        # Normalize percentages so each sentiment category's bar extends to the same height (representing 100%)
        sentiment_df['Total'] = sentiment_df[['Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection']].sum(axis=1)
        sentiment_df['Love'] = (sentiment_df['Love'] / sentiment_df['Total']) * 100
        sentiment_df['Heartbreak'] = (sentiment_df['Heartbreak'] / sentiment_df['Total']) * 100
        sentiment_df['Optimism'] = (sentiment_df['Optimism'] / sentiment_df['Total']) * 100
        sentiment_df['Anger'] = (sentiment_df['Anger'] / sentiment_df['Total']) * 100
        sentiment_df['Reflection'] = (sentiment_df['Reflection'] / sentiment_df['Total']) * 100
        sentiment_df['Total'] = 100
        return sentiment_df

    def plot_sentiment_distribution(self):
        df = self.analyze_sentiment() # we need the sentiment_df for this method
        barWidth = 0.85
        plt.figure(figsize=(20, 10)) # TO-DO fix figure size
        r = range(len(df))

        plt.barh(r, df['Love'], color='pink', edgecolor='white', height=barWidth, label='Love')
        plt.barh(r, df['Heartbreak'], left=df['Love'], color='blue', edgecolor='white', height=barWidth,
                 label='Heartbreak')
        plt.barh(r, df['Optimism'], left=[i + j for i, j in zip(df['Love'], df['Heartbreak'])],
                 color='lightblue',
                 edgecolor='white', height=barWidth, label='Optimism')
        plt.barh(r, df['Anger'],
                 left=[i + j + k for i, j, k in zip(df['Love'], df['Heartbreak'], df['Optimism'])],
                 color='red', edgecolor='white', height=barWidth, label='Anger')
        plt.barh(r, df['Reflection'],
                 left=[i + j + k + l for i, j, k, l in
                       zip(df['Love'], df['Heartbreak'], df['Optimism'], df['Anger'])],
                 color='limegreen', edgecolor='white', height=barWidth, label='Reflection')

        plt.ylabel('Albums', fontweight='bold')
        plt.xlabel('Sentiment Percentage', fontweight='bold')
        plt.yticks([r for r in range(len(df))], df['Album'])
        plt.title('Sentiment Percentage Distribution by Album')
        plt.legend()

        plt.gca().axes.xaxis.set_visible(False)

        plt.show()


analyzer = LyricAnalyzer()
file_paths = [
    "1989 (Taylor’s Version) [Deluxe].csv",
    "Red (Taylor’s Version).csv",
    "Midnights (3am Edition).csv",
    "evermore.csv",
    "folklore.csv",
    "Lover.csv",
    "reputation.csv",
    "Speak Now (Taylor’s Version).csv",
    "Fearless (Taylor’s Version).csv"
]
analyzer.load_and_prepare_text_files(file_paths)
analyzer.plot_colors_sentiment()

