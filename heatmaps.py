import matplotlib
matplotlib.use('TkAgg')
from parser import data
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
df = data
print(df['Album'])

# Read the original data


# Combine all sentiment words into a single list
love_words = ["baby", "babe", "love", "joy", "enchanting", "captivating", "beautiful", "romance", "happiness", "healing", "connection",
              "adventure", "shiny", "sweet", "nice", "light", "blue", "lightblue", "yellow", "pink", "gold", "green",
              "purple", "courage", "faith", "passion", "amazing", "new", "wonderful", "adoration", "affection", "devotion", "bliss", "euphoria", "fondness", "infatuation", "passionate",
              "tenderness", "warmth"]

# Heartbreak
heartbreak_words = ["upset", "wait", "hate", "dislike", "scared", "angry", "confused", "stupid", "fear", "sadness",
                    "doubts", "doubt", "hurt", "pain", "heartbreak", "dark", "tragic", "lonely", "loneliness",
                    "loathing", "devastating", "devastated", "solitary", "cry", "cold", "tears", "teardrops",
                    "turmoil", "demons", "terror", "terrors", "black", "reputation", "brown", "darkgreen", "grey",
                    "loss", "bye", "goodbye",
                    "agony", "despair", "grief", "melancholy", "sorrow", "misery", "regret", "betrayal", "abandonment",
                    "desolation"]

# Optimism
optimism_words = ["finally", "relief", "hope", "dreams", "wonderment", "wonder", "wonderful", "smile", "intrigue",
                  "happy", "sunshine", "bright", "uplifting", "positive", "cheerful", "optimistic", "inspiration",
                  "excitement", "vibrant",
                  "anticipation", "confidence", "faith", "gratitude", "hopeful", "positivity", "rejuvenated",
                  "resilience", "serenity", "trust"]

# Anger
anger_words = ["angry", "anger", "hate", "hatred", "mad", "outraged", "resentment", "fury", "dislike", "bad", "unsafe", "gross",
               "loathing", "loathe", "goodbye", "liar", "lies", "lie", "rage", "bitter", "hostile", "vindictive",
               "outrageous", "provoked", "enraged", "irritated", "furious", "infuriated", "aggravation", "disgust", "hostility", "irritation", "outrage", "resentment", "wrath", "annoyance",
               "frustration", ]

# Reflection
reflection_words = ["loss", "bye", "enchanting", "demons", "terror", "terrors", "black", "reputation", "brown",
                    "darkgreen", "grey", "memories", "nostalgia", "contemplate", "introspection", "ponder",
                    "reminisce", "meditate", "introspective", "thoughtful", "reflective",   "contemplation", "insight", "meditation", "pensive", "rumination", "self-awareness",
                    "thought-provoking", "introspective", "philosophical", "soul-searching"]

all_sentiment_words = love_words + heartbreak_words + optimism_words + anger_words + reflection_words

# Initialize an empty list to store DataFrames for each album
album_dataframes = []

# Iterate over each album
for album in df['Album'].unique():
    prologue_text = df.loc[df['Album'] == album, 'Lyrics'].str.lower().str.split().sum()

    # Calculate word frequency
    word_freq = Counter(prologue_text)

    # Get the top 10 most common words that are also in the emotion word lists
    top_words = {word: freq for word, freq in word_freq.items() if word in all_sentiment_words}
    top_words = dict(sorted(top_words.items(), key=lambda item: item[1], reverse=True)[:10])

    # Create a DataFrame for the current album
    album_df = pd.DataFrame.from_dict(top_words, orient='index', columns=[album])
    album_df.index.name = 'Word'

    # Transpose the DataFrame to have albums as columns
    album_df = album_df.T

    # Append to the list of DataFrames
    album_dataframes.append(album_df)
    print(album)
    print(album_df)

'''
# Concatenate the list of DataFrames into a single DataFrame
top_words_df = pd.concat(album_dataframes)

# Fill NaN values with 0
top_words_df.fillna(0, inplace=True)

# Display the DataFrame
print(top_words_df)
'''