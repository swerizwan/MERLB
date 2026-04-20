import json
import time
import urllib.request
import csv

# Define the output CSV file where the game data will be stored
output = "games.csv"

# Twitch API URL to get the top 100 games (Twitch API limits the number of results to 100 per request)
url = "https://api.twitch.tv/kraken/games/top?limit=100"

# Open the output CSV file in write mode and create a CSV writer object
with open(output, 'w') as fp:
    a = csv.writer(fp)
    
    # Prepare the header row for the CSV file
    # 'Title' - name of the game
    # 'Popularity' - how popular the game is on Twitch
    # 'Type' - placeholder for game type (if needed in future)
    # 'Mid' - placeholder for game ID (if needed in future)
    # 'New Title' - placeholder for renaming or formatting the game title (if needed in future)
    a.writerow(['Title', 'Popularity', 'Type', 'Mid', 'New Title'])
    
    # Initialize a counter to keep track of how many games have been processed
    count = 0
    
    # Loop to continuously fetch and process the top games data from the Twitch API
    while True:
        # Make the API request and decode the response from bytes to string
        response = urllib.request.urlopen(url).read().decode()
        
        # Parse the JSON response to a Python dictionary
        data = json.loads(response)
        
        # Extract the URL for the next page of results from the '_links' section
        url = data["_links"]["next"]  # This URL provides the next set of top games
        
        # Extract the list of games from the 'top' section of the JSON data
        games = data["top"]
        
        # If there are no more games to process, exit the loop
        if len(games) == 0:
            break
        
        # Update the counter with the number of games processed in this iteration
        count += len(games)
        
        # Log the progress to the console (how many games are written so far)
        print("Writing %d rows into file. (Total=%d)" % (len(games), count))
        
        # Loop through each game in the list of 'games' from the API response
        for game in games:
            # Extract the game details (name and popularity) from each game entry
            details = game["game"]
            title = details["name"]
            popularity = details["popularity"]
            
            # Write the game title and popularity to the CSV file
            # Note: 'Type', 'Mid', and 'New Title' columns are left blank for now
            a.writerow([title, popularity])
