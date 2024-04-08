from collections import defaultdict
import sankey as sk
import pandas as pd

data = pd.read_csv('preprocessed_data.csv')
word_lst = ['fearless', 'think', 'love', 'someone', 'things', 'believe']


# Function to count words in text
def count_words(text, word_lst):
    words = text.lower().split()
    word_count = defaultdict(int)
    for word in word_lst:
        word_count[word] = 0
    for word in words:
        if word in word_lst:
            word_count[word] += 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count


def k_counter(text):
    words = text.lower().split()
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count


def k_words(data, k):
    all = []
    for index, row in data.iterrows():
        album = row["Album Name"]
        text = row["Prologue Text"]
        word_count = k_counter(text)
        first_k_words = [word for word, _ in word_count[:k]]
        all.extend(first_k_words)
    return (all)


def create_df(data, word_lst=word_lst):
    global word_dict
    sankey_data = defaultdict(dict)
    for index, row in data.iterrows():
        album = row["Album Name"]
        text = row["Prologue Text"]
        word_count = count_words(text, word_lst)
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

    return sankey_df


def main():
    k_list = k_words(data, 1)
    df = create_df(data, word_lst=k_list)
    sk.make_sankey(df, 'Source_Label', 'Target_Label', 'Value')


if __name__ == "__main__":
    main()
