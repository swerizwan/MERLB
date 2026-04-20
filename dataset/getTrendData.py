from pytrends.request import TrendReq
from threading import Thread
import os
import sys
import time
import pandas as pd

# Use the same input and output for resuming work
input_file = "games.csv"
output_file = "games.csv"

# Reference query for normalization
ref_query = "book"
col_avg_search_pop = "Average Search Popularity"
col_normalized = "Normalized"

# Connect to Google Trends (authentication no longer required in the new pytrends)
pytrend = TrendReq()

# Read the input CSV file
df = pd.read_csv(input_file)

for i in range(len(df)):
    # Skip rows where data for "2016-05-01" already exists
    if "2016-05-01" in df.columns and df["2016-05-01"][i] != -1:
        continue

    game_title = df["Title"][i]

    # Replace "/" in game titles to avoid issues
    if "/" in game_title:
        game_title = game_title.replace("/", " ")

    # Use the MID if available; otherwise, use the game title
    if pd.isnull(df["Mid"][i]):
        keyword = game_title
    else:
        keyword = df["Mid"][i]

    # Combine the reference query and keyword
    q = [f"{ref_query},{keyword}"]

    # Start scraping the trend data
    while True:
        try:
            print(f"Getting trend data for {game_title}..")
            trend_payload = {'q': q, 'cat': '8', 'date': '05/2016 4m'}
            retdf = pytrend.get_historical_interest(trend_payload, return_type='dataframe')
        except KeyboardInterrupt:
            # Allow manual interruption
            print('Interrupted')
            sys.exit(0)
        except Exception as e:
            # Retry after a delay if scraping fails
            print(f"Failed to scrape data for {game_title}, retrying in 10 seconds.")
            print("Error:", e)
            time.sleep(10)
            continue
        break

    # Find the column for the game (anything except the reference query "book")
    for cols in retdf.columns:
        if cols != ref_query:
            colname = cols
            break

    # Normalize the data based on the reference query
    retdf[col_normalized] = 0.0
    for j in range(len(retdf)):
        if retdf[ref_query][j] == 0:
            retdf[col_normalized][j] = retdf[colname][j]
        else:
            retdf[col_normalized][j] = retdf[colname][j] / retdf[ref_query][j]

    # Save the individual game trend data to a CSV file
    output_path = f"output/{game_title}.csv"
    a = Thread(target=retdf.to_csv, args=(output_path,), kwargs={'encoding': 'utf-8'})
    a.start()
    a.join()

    # Reload the saved CSV and update the main dataframe
    retdf = pd.read_csv(output_path)
    for j in range(len(retdf)):
        date = retdf["date"][j]  # Change "Date" to "date" if using pytrends historical data
        if date not in df.columns:
            df[date] = -1.0
        df.loc[i, date] = retdf[col_normalized][j]

    # Calculate and store the average search popularity
    if col_avg_search_pop not in df.columns:
        df[col_avg_search_pop] = 0.0
    df.loc[i, col_avg_search_pop] = retdf[col_normalized].mean()

    # Save the updated main dataframe to the CSV
    a = Thread(target=df.to_csv, args=(output_file,), kwargs={'index': False})
    a.start()
    a.join()

    print(retdf)
    print(f"Average Search Popularity: {retdf[col_normalized].mean()}")

    i += 1
