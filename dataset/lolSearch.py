# From https://riot-watcher.readthedocs.io/en/latest/
from riotwatcher import LolWatcher, ApiError
import os

api_key = os.environ.get("RIOT_API_KEY")

lol_watcher = LolWatcher(api_key)

queue = 'RANKED_SOLO_5x5'

def lowest(reg: str):
    try:
        challengers = lol_watcher.league.challenger_by_queue(reg, queue)['entries']
        min_lp = min(int(x['leaguePoints']) for x in challengers)
        for y in challengers:
            if int(y['leaguePoints']) == min_lp:
                name = y['summonerName']
        return(f"Lowest LP in Challenger in {reg} is {name} at {min_lp} LP")

    except ApiError as err:
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise
        
def rank(region, ign):
    try:
        account_information = lol_watcher.summoner.by_name(region, ign)
        ranked_stats = lol_watcher.league.by_summoner(region, account_information['id'])
        tier = ranked_stats[0]['tier'].capitalize()
        rank = ranked_stats[0]['rank']
        lp = ranked_stats[0]['leaguePoints']
        summonerName = ranked_stats[0]['summonerName']
        return(f"{summonerName} is {tier} {rank} {lp}LP")
    except ApiError as err:
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise

def runes(region, ign):
    try:
        account_information = lol_watcher.summoner.by_name(region, ign)
        summoner_id = account_information['id']
        current_game_info = lol_watcher.spectator.by_summoner(region, summoner_id)
        # check league's latest version
        latest = lol_watcher.data_dragon.versions_for_region(region)['n']['champion']
        static_rune_list = lol_watcher.data_dragon.runes_reforged(latest)
        current_runes = []
        
        for each in current_game_info['participants']:
            if each['summonerName'] == ign:
                perk_style = each['perks']['perkStyle']
                perk_sub_style = each['perks']['perkSubStyle']
                current_perks = each['perks']['perkIds']
        for x in static_rune_list:
            if x['id'] == perk_style:
                for i in range(4):
                    for y in x['slots'][i]['runes']:
                        if y['id'] == current_perks[i]:
                            current_runes.append(y['name'])
        for x in static_rune_list:
            if x['id'] == perk_sub_style:
                for y in x['slots']:
                    for z in y['runes']:
                        if z['id'] == current_perks[4] or z['id'] == current_perks[5]:
                            current_runes.append(z['name'])
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
        return(current_runes)
        
    except ApiError as err:
        if err.response.status_code == 429:
            return('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        elif err.response.status_code == 404:
            return('404 Error')
        else:
            raise
        
#runes("kr", "Toon Zorc")