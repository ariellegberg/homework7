import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import itertools


class PrologueAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def preprocess_text(self, text):
        text = text.lower()
        text = ''.join(c for c in text if c not in punctuation)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    def analyze_prologues(self):
        self.df['Prologue Text'] = self.df['Prologue Text'].apply(self.preprocess_text)
        self.df['Word Count'] = self.df['Prologue Text'].apply(lambda x: len(str(x).split()))
        self.df['Avg Sentence Length'] = self.df['Prologue Text'].apply(
            lambda x: len(sent_tokenize(x)) if len(sent_tokenize(x)) > 0 else 0)
        self.df['Avg Word Length'] = self.df['Prologue Text'].apply(
            lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()))

    def find_english_words(self, letter_list):
        found_words = set()
        for r in range(1, len(letter_list) + 1):
            for perm in itertools.permutations(letter_list, r):
                word = ''.join(perm).lower()
                if self.is_english_word(word):
                    found_words.add(word)
        return found_words

    def is_english_word(self, word):
        return word.lower() in words.words()

    def analyze_capital_letters(self, column_name):
        capital_letters_per_cell = []
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
        else:
            capital_letters_per_cell = self.df[column_name].apply(self.find_capital_letters)
            for i, letters in enumerate(capital_letters_per_cell):
                print(f"Cell {i + 1} in column '{column_name}': {letters}")

    def find_capital_letters(self, text):
        return [char for char in text if char.isupper()]
