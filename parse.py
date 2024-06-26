import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import re
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import Counter
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
import sankey as sk
import argparse

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
        # nltk.download('words')
        # nltk.download('punkt')
        # nltk.download('words')
        # nltk.download('stopwords')

    def load_text(self, file_paths):
        """
        Load in text files, process them, and add to df
        :param file_paths: list of string file paths
        :return: None
        """
        # Initialize list to store DataFrames for each album
        dfs = []
        # Iterate over each file path
        for file_path in file_paths:
            # Read lyrics from the text file
            with open(file_path, 'r') as file:
                song_lyrics = file.read()

            # Extract album name from the file name
            album_name = os.path.splitext(os.path.basename(file_path))[0]

            # Create DataFrame for the album but get rid of the word lyrics
            album_df = pd.DataFrame({'Album': [album_name.replace('_lyrics', '')], 'Lyrics': [song_lyrics]})

            # Append album DataFrame
            dfs.append(album_df)

        # Concatenate DataFrames for all albums into one DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Preprocess the lyrics
        combined_df['Lyrics'] = combined_df['Lyrics'].apply(self.preprocess_text)

        # Assign the combined DataFrame to self.df
        self.df = combined_df

    def preprocess_text(self, text):
        """
         Preprocess the texts, removing stopwords, punctuation and formatiing the words
         :param text: Text to be preprocessed (txt)
         :return: processed text (str)
        """

        # remove line at beginning of every song
        text = re.sub(r'^ContributorsTranslations.*?\n', '', text, flags=re.MULTILINE)
        # Remove text between square brackets
        text = re.sub(r'\[.*?\]', '', text)
        # Remove square brackets
        text = re.sub(r'\[.*?\]', '', text)
        # Remove the word 'embed'
        text = text.replace('Embed', '')
        text = text.replace('Lyrics', '')

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        text = text.lower()  # make all the text lowercase
        text = ''.join(c for c in text if c not in punctuation + "’")  # takes out punctuation
        words = word_tokenize(text)  # Tokenize the text into words
        stop_words = set(stopwords.words('english'))  # make sure all words are english
        words = [word for word in words if word not in stop_words]  # remove stop words
        return ' '.join(words)  # join all the words back into a string

    def statistics(self):
        """
        Create data frame that tracks word count and average word length of files
        """
        self.data = self.df.copy()

        def count_words(text):
            return len(text.split())

        # Apply the function to the 'Lyrics' column
        self.data['word_count'] = self.data['Lyrics'].apply(count_words)

        # Function to calculate average word length
        def average_word_length(text):
            words = text.split()
            word_lengths = [len(word) for word in words]
            if len(word_lengths) == 0:
                return 0
            else:
                return sum(word_lengths) / len(word_lengths)

        # Apply the function to the 'Lyrics' column
        self.data['avg_word_length'] = self.data['Lyrics'].apply(average_word_length)

        return self.data




    def _extract_colors(self, lyrics):
        """
        Finds all the colors in a string of lyrics
        :param lyrics: string of lyrics
        :return: string of colors mentioned
        """
        # Define a regular expression pattern to match color names
        color_pattern = re.compile(
            r'\b(?:red|blue|green|yellow|purple|pink|silver|lavender|orange|black|gray|gold|brown|white|burgundy|maroon|cherry|scarlet|crimson|rubies})\b',
            flags=re.IGNORECASE)

        # Find all color mentions in the lyrics
        colors_mentioned = color_pattern.findall(lyrics)

        return colors_mentioned

    def plot_colors_sentiment(self):
        """
        Plots the sentiment and color breakdown of albums into subplots
        """
        # code help from: https://stackoverflow.com/questions/58887571/plotting-2-pie-charts-side-by-side-in-matplotlib

        # First, get a dataframe with 'albums' as one column and 'colors' as another
        # where 'colors' has a list of all the colors mentioned in that album
        colors_df = self.df.copy()
        colors_df['Colors'] = colors_df['Lyrics'].apply(lambda x: self._extract_colors(x))

        # get sentiment df
        sentiment_df = self.analyze_sentiment()

        # combine the two dataframes
        merged_df = pd.merge(colors_df, sentiment_df, on=['Album', 'Lyrics'])

        # colors associated with 'Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection'
        sentiment_colors = ['pink', 'blue', 'lightblue', 'red', 'limegreen']

        # make a color map (must use hex colors since matplotlib doesn't know cherry and such)
        # Define a list of colors for the pie charts
        color_hex_map = {
            'red': '#FF0000',
            'blue': '#0000FF',
            'green': '#00FF00',
            'yellow': '#FFFF00',
            'purple': '#800080',
            'pink': '#FFC0CB',
            'silver': '#C0C0C0',
            'lavender': '#E6E6FA',
            'orange': '#FFA500',
            'black': '#000000',
            'gray': '#808080',
            'gold': '#FFD700',
            'brown': '#A52A2A',
            'white': '#FFFFFF',
            'burgundy': '#800020',
            'maroon': '#B5406C',
            'cherry': '#DE3163',
            'scarlet': '#FF2400',
            'crimson': '#DC143C',
            'rubies': '#E0115F'
        }

        num_albums = len(merged_df)
        num_cols = 2  # Number of columns in each subplot
        num_rows = (num_albums + 1) // 2  # Number of rows in the subplot grid

        fig = plt.figure(figsize=(16, 12), facecolor='none')  # Increase figsize
        outer = gridspec.GridSpec(5, 2, wspace=0.2, hspace=0.5)  # Adjust spacing

        color_legend_handles = {}
        color_legend_labels = {}
        sentiment_legend_handles = {}
        sentiment_legend_labels = {}

        for i, (album, colors, lyrics) in enumerate(zip(merged_df['Album'], merged_df['Colors'], merged_df['Lyrics'])):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

            # Create subplot for album name
            ax_album = plt.Subplot(fig, inner[:, :])
            ax_album.set_xticks([])
            ax_album.set_yticks([])
            ax_album.text(0.5, 0.5, album, ha='center', va='center', fontsize=16, fontweight='bold')
            fig.add_subplot(ax_album)

            # Create pie chart for color breakdown
            ax1 = plt.Subplot(fig, inner[0])
            color_counts = Counter(colors)
            wedges, texts, autotexts = ax1.pie(color_counts.values(), labels=None, autopct='',
                                               colors=[color_hex_map[color] for color in color_counts.keys()],
                                               wedgeprops={'edgecolor': 'black', 'linewidth': .5})
            ax1.set_title('Color Breakdown', fontsize=12, pad=5)
            ax1.set_aspect('equal')
            fig.add_subplot(ax1)

            # Create pie chart for sentiment breakdown
            ax2 = plt.Subplot(fig, inner[1])
            sentiment_row = merged_df[merged_df['Album'] == album].iloc[0]
            wedges2, texts2, autotexts2 = ax2.pie(
                sentiment_row[['Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection']], labels=None, autopct='',
                colors=sentiment_colors, wedgeprops={'edgecolor': 'black', 'linewidth': .5})
            ax2.set_title('Sentiment Breakdown', fontsize=12, pad=5)
            ax2.set_aspect('equal')
            fig.add_subplot(ax2)

            # Append handles and labels for color legend
            for color, wedge in zip(color_counts.keys(), wedges):
                if color not in color_legend_handles:
                    color_legend_handles[color] = wedge
                    color_legend_labels[color] = color

            # Append handles and labels for sentiment legend
            for sentiment, wedge in zip(['Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection'], wedges2):
                if sentiment not in sentiment_legend_handles:
                    sentiment_legend_handles[sentiment] = wedge
                    sentiment_legend_labels[sentiment] = sentiment

        # add a title
        fig.suptitle('Colors in Lyrics vs. Sentiment Breakdown', fontsize=18,
                     fontweight='bold')  # Increase fontsize for overall title
        # Create the legends
        color_legend = plt.legend(color_legend_handles.values(), color_legend_labels.values(), title='Colors',
                                  loc='lower right', bbox_to_anchor=(7.25, -.5), ncol=3)
        sentiment_legend = plt.legend(sentiment_legend_handles.values(), sentiment_legend_labels.values(),
                                      title='Sentiments', loc='lower right', bbox_to_anchor=(9.5, -.5))
        fig.add_artist(color_legend)
        fig.add_artist(sentiment_legend)

        plt.show(block=True)
        plt.close()

    def analyze_sentiment(self):
        """
        Create sentiment_df to record frequency of a sentiment category in text
        :return: sentiment_df (pandas dataframe)
        """
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
        """
        Plots the sentiment distribution by different sentiment categories
        :return: plots a bar chart of sentiment distribution
        """
        df = self.analyze_sentiment()  # we need the sentiment_df for this method
        barWidth = 0.85
        plt.figure(figsize=(16, 10))
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

    def k_counter(self, text):
        '''
        Counts the number of times a word appears in text file
        :param text:
        :return: dictionary of words and their counts from most to least common
        '''
        words = text.lower().split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_count

    # Function to count words in text
    def count_words(self, text, word_lst):
        '''
        Counts words in text file that appear in given word list
        :param text:
        :param word_lst:
        :return: develops a sankey diagram
        '''
        words = text.lower().split()
        word_count = defaultdict(int)
        [word_count.__setitem__(word, 0) for word in word_lst]
        [word_count.__setitem__(word, word_count[word] + 1) for word in words if word in word_lst]
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_count

    def wordcount_sankey(self, word_list=None, k=5):

        if word_list == []:
            for index, row in self.df.iterrows():
                album = row["Album"]
                text = row["Lyrics"]
                word_count = self.k_counter(text)
                first_k_words = [word for word, _ in word_count[:k]]
                word_list.extend(first_k_words)

        global word_dict
        sankey_data = defaultdict(dict)
        for index, row in self.df.iterrows():
            album = row["Album"]
            text = row["Lyrics"]
            word_count = self.count_words(text, word_list)
            for word, count in word_count:
                sankey_data[album][word] = count

        labels = []
        source = []
        target = []
        value = []

        for album, word_count in sankey_data.items():
            labels.append(album)
            for word, count in word_count.items():
                labels.append(word)
                source.append(labels.index(album))
                target.append(labels.index(word))
                value.append(count)

        sankey_df = pd.DataFrame({
            'Source': source,
            'Target': target,
            'Value': value
        })

        # Add labels to the DataFrame
        sankey_df['Source_Label'] = [labels[i] for i in sankey_df['Source']]
        sankey_df['Target_Label'] = [labels[i] for i in sankey_df['Target']]

        sk.make_sankey(sankey_df, 'Source_Label', 'Target_Label', 'Value', title = 'Text-to-Word Sankey')


def main():
    file_paths = [
        "1989_lyrics.txt",
        "Evermore_lyrics.txt",
        "Fearless_lyrics.txt",
        "Folklore_lyrics.txt",
        "Lover_lyrics.txt",
        "Midnights_lyrics.txt",
        "Red_lyrics.txt",
        "Reputation_lyrics.txt",
        "SpeakNow_lyrics.txt"
    ]

    # initialize lyric analyzer class and load in files
    analyzer = LyricAnalyzer()
    analyzer.load_text(file_paths)

    # define parsers for user engagement
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", help="List of Words to track in lyrics through Sankey Diagram",
                        type=list_of_strings)
    parser.add_argument("-k", "--k_val", help="Number of most common words to track for each file", type=int)
    args = parser.parse_args()
    # python parse.py -l love,fearless,wine,baby,red,blue,golden,speak,mine
    word_list = []
    if args.list:
        word_list = args.list

    k = 2
    if args.k_val:
        k = args.k_val

    # visualizations depicting the analysis
    analyzer.wordcount_sankey(word_list, k=k)
    analyzer.plot_sentiment_distribution()
    analyzer.plot_colors_sentiment()


if __name__ == "__main__":
    main()
