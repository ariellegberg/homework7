import pandas as pd

def prepare_album_csvs(csv):
    # Read the original CSV file
    df = pd.read_csv(csv)

    # Convert lyrics to lowercase
    df['Lyrics'] = df['Lyrics'].str.lower()

    # Iterate over each unique album
    for album in df['Album'].unique():
        # Filter the DataFrame for the current album
        album_df = df[df['Album'] == album]

        # Save the DataFrame to a CSV file named after the album
        album_csv_filename = f"{album}.csv"
        album_df.to_csv(album_csv_filename, index=False)

prepare_album_csvs('songs.csv')
