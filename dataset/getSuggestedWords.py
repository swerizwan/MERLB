from pytrends.request import TrendReq
import json
import os
import sys
import time
import pandas as pd

input_file = "games.csv"
output_file = "games.csv"

# Load the CSV file
df = pd.read_csv(input_file)

# Dynamic Programming implementation of Levenshtein distance (Edit Distance)
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,    # Deletion
                          d[i][j - 1] + 1,    # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution
    
    return d[m][n]

# Connect to Google Trends
pytrend = TrendReq()

i = 0
for game in df["Title"]:
    # Skip already processed rows
    if pd.isnull(df["Type"][i]):
        keyword = game
        
        # Replace '/' in the title (as it can cause issues)
        if "/" in keyword:
            keyword = game.replace("/", " ")
        
        print(f"Searching for {keyword}..")
        
        # Start scraping suggestions
        while True:
            try:
                suggestions = pytrend.suggestions(keyword)
                json_list = []
                
                # Filter suggestions related to 'game' only
                for suggestion in suggestions:
                    if "game" in suggestion["type"].lower() and "develop" not in suggestion["type"].lower() and "company" not in suggestion["type"].lower():
                        json_list.append(suggestion)
                
                # Sort the suggestions by Levenshtein distance to the keyword
                json_list.sort(key=lambda x: levenshtein_distance(x["title"], keyword))
                
                if json_list:
                    # Pick the best suggestion (least edit distance)
                    suggestion = json_list[0]
                    df.loc[i, "Type"] = suggestion["type"]
                    df.loc[i, "Mid"] = suggestion["mid"]
                    df.loc[i, "New Title"] = suggestion["title"]
                    print(f"Found {suggestion['title']}, {suggestion['type']}, {suggestion['mid']}")
                    
                    # Save the updated dataframe to CSV
                    df.to_csv(output_file, index=False, encoding="utf-8")
                else:
                    print(f"No suggestions found for {keyword}.")
                
                print()
            except KeyboardInterrupt:
                print('Interrupted')
                sys.exit(0)
            except Exception as e:
                # Retry on failure after 5 seconds
                print(f"Failed to find suggestions for {keyword}. Retrying in 5 seconds.")
                print("Error:", e)
                time.sleep(5)
                continue
            
            break
    i += 1
