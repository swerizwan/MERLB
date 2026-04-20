# Import the necessary libraries from the RiotWatcher package
# LolWatcher is used to interact with Riot's API for League of Legends
# ApiError is used to handle errors that occur when making API requests
from riotwatcher import LolWatcher, ApiError
import os

# Fetch the Riot API key from environment variables
api_key = os.environ.get("RIOT_API_KEY")

# Initialize the LolWatcher object with the API key for making API requests
lol_watcher = LolWatcher(api_key)

# Define the queue type as 'RANKED_SOLO_5x5' for ranked solo/duo games
queue = 'RANKED_SOLO_5x5'

# Function to find the player with the lowest League Points (LP) in the Challenger tier for a given region
def lowest(reg: str):
    try:
        # Retrieve challenger tier players' data for the specified region and queue
        challengers = lol_watcher.league.challenger_by_queue(reg, queue)['entries']
        
        # Find the lowest League Points (LP) among all challenger players
        min_lp = min(int(x['leaguePoints']) for x in challengers)
        
        # Loop through challenger players and identify the one with the lowest LP
        for y in challengers:
            if int(y['leaguePoints']) == min_lp:
                name = y['summonerName']  # Get the summoner's name with the lowest LP
                
        # Return a formatted string indicating the player with the lowest LP in the Challenger tier
        return(f"Lowest LP in Challenger in {reg} is {name} at {min_lp} LP")

    # Handle API errors using ApiError
    except ApiError as err:
        # If the API returns a rate limit error (status code 429), inform the user to retry after the specified time
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        # If the API returns a 404 error (not found), indicate a 404 error
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise  # Raise the error for other unexpected API errors
        
# Function to get the ranked stats of a summoner by their in-game name (IGN) and region
def rank(region, ign):
    try:
        # Retrieve the account information using the summoner's IGN and region
        account_information = lol_watcher.summoner.by_name(region, ign)
        
        # Use the account ID to get the summoner's ranked stats
        ranked_stats = lol_watcher.league.by_summoner(region, account_information['id'])
        
        # Extract the tier, rank, and LP (League Points) from the ranked stats
        tier = ranked_stats[0]['tier'].capitalize()
        rank = ranked_stats[0]['rank']
        lp = ranked_stats[0]['leaguePoints']
        summonerName = ranked_stats[0]['summonerName']
        
        # Return a formatted string indicating the summoner's rank, tier, and LP
        return(f"{summonerName} is {tier} {rank} {lp}LP")
    
    # Handle API errors using ApiError
    except ApiError as err:
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise

# Function to get the runes of a summoner currently in a game, based on their IGN and region
def runes(region, ign):
    try:
        # Retrieve the account information using the summoner's IGN and region
        account_information = lol_watcher.summoner.by_name(region, ign)
        summoner_id = account_information['id']  # Get the summoner's ID
        
        # Get the current game information if the summoner is in a live game
        current_game_info = lol_watcher.spectator.by_summoner(region, summoner_id)
        
        # Get the latest version of the League of Legends data for runes
        latest = lol_watcher.data_dragon.versions_for_region(region)['n']['champion']
        static_rune_list = lol_watcher.data_dragon.runes_reforged(latest)  # Get the static rune data
        
        current_runes = []  # List to store the current runes the summoner is using
        
        # Loop through participants in the current game to find the target summoner
        for each in current_game_info['participants']:
            if each['summonerName'] == ign:
                # Extract the summoner's primary rune style and sub-style, and their perk IDs
                perk_style = each['perks']['perkStyle']
                perk_sub_style = each['perks']['perkSubStyle']
                current_perks = each['perks']['perkIds']
        
        # Loop through the static rune list to match the summoner's primary rune style and perks
        for x in static_rune_list:
            if x['id'] == perk_style:
                # Match and append the summoner's current primary runes based on the perk IDs
                for i in range(4):
                    for y in x['slots'][i]['runes']:
                        if y['id'] == current_perks[i]:
                            current_runes.append(y['name'])
        
        # Match the sub-style runes and append them to the current_runes list
        for x in static_rune_list:
            if x['id'] == perk_sub_style:
                for y in x['slots']:
                    for z in y['runes']:
                        if z['id'] == current_perks[4] or z['id'] == current_perks[5]:
                            current_runes.append(z['name'])
        
        # Check the perks in the extra runes section and append corresponding runes to the list
        if current_perks[6] == 5008:
            current_runes.append("Adaptive Force")
        elif current_perks[6] == 5005:
            current_runes.append("Attack Speed")
        elif current_perks[6] == 5007:
            current_runes.append("Scaling CDR")
            
        if current_perks[7] == 5008:
            current_runes.append("Adaptive Force")
        elif current_perks[7] == 5002:
            current_runes.append("Armor")
        elif current_perks[7] == 5003:
            current_runes.append("Magic Resist")
            
        if current_perks[8] == 5001:
            current_runes.append("Scaling Health")
        elif current_perks[8] == 5002:
            current_runes.append("Armor")
        elif current_perks[8] == 5003:
            current_runes.append("Magic Resist")
        
        # Return the current runes the summoner is using in their live game
        return(current_runes)
    
    # Handle API errors using ApiError
    except ApiError as err:
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise

# Example usage: runes("kr", "Toon Zorc")
