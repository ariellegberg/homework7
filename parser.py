import pandas as pd

## Define sentiment words

# Love
love_words = ["love", "joy", "enchanting", "captivating", "beautiful", "romance", "happiness", "healing", "connection",
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


## Define sentiment analysis function to calculate percentages

def calculate_sentiment_percentages(text):
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
def prepare_df(csv):
    # Read the original CSV file
    df = pd.read_csv(csv)

    # Convert lyrics to lowercase
    df['Lyrics'] = df['Lyrics'].str.lower()

    df = df[df['Album'].str.contains("Taylor’s Version", case=False)]

    # Group the data by 'Album' and concatenate the lyrics for each album
    grouped_df = df.groupby('Album')['Lyrics'].apply(lambda x: ' '.join(x)).reset_index()

    # Save the prepared DataFrame to a new CSV file
    grouped_df.to_csv('prepared_data.csv', index=False)

    # Display the first few rows of the prepared DataFrame
    return grouped_df
songs_prepared = prepare_df('songs.csv')



def df():
   # Apply sentiment analysis to the prologue text
    df = songs_prepared
    df['Love'], df['Heartbreak'], df['Optimism'], df['Anger'], df['Reflection'] = zip(
        *df['Lyrics'].apply(calculate_sentiment_percentages))

    # Normalize percentages so each sentiment category's bar extends to the same height (representing 100%)
    df['Total'] = df[['Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection']].sum(axis=1)
    df['Love'] = (df['Love'] / df['Total']) * 100
    df['Heartbreak'] = (df['Heartbreak'] / df['Total']) * 100
    df['Optimism'] = (df['Optimism'] / df['Total']) * 100
    df['Anger'] =  (df['Anger'] / df['Total']) * 100
    df['Reflection'] = (df['Reflection'] /df['Total']) * 100
    df['Total'] = 100
    return df

data = df()
print(data.columns)
