import os
import pandas as pd

import os
import pandas as pd

def compile_album_lyrics_to_text(albums_path):
    """
    Compiles lyrics from individual song .txt files into a single text file per album.
    :param albums_path: Path to the directory containing album folders.
    :return: A dictionary where keys are album names and values are the compiled lyrics.
    """
    album_lyrics = {}

    for album_folder in os.listdir(albums_path):
        album_path = os.path.join(albums_path, album_folder)
        if os.path.isdir(album_path):
            album_name = album_folder
            lyrics = []

            # Compile lyrics from individual song .txt files
            for song_file in os.listdir(album_path):
                if song_file.endswith('.txt'):
                    song_path = os.path.join(album_path, song_file)
                    with open(song_path, 'r') as f:
                        lyrics.append(f.read())

            # Combine lyrics into a single text
            album_lyrics[album_name] = '\n'.join(lyrics)

    return album_lyrics



# Example usage:
albums_path = 'Albums'
lyrics_dict = compile_album_lyrics_to_text(albums_path)
print(lyrics_dict)