import polars as pl
import time
import random
import os

try:
    from curl_cffi import requests
except ImportError:
    import requests

# --- CONFIGURATION ---
CURRENT_SEASON = "2025-26"  # Update this annually

DATASET_CONFIG = [
    {"entity": "player", "measure": "Usage",       "file": "PlayerStatisticsUsage.csv"},
    {"entity": "player", "measure": "Advanced",    "file": "PlayerStatisticsAdvanced.csv"},
    {"entity": "player", "measure": "Misc",        "file": "PlayerStatisticsMisc.csv"},
    {"entity": "player", "measure": "Scoring",     "file": "PlayerStatisticsScoring.csv"},
    {"entity": "team",   "measure": "Advanced",    "file": "TeamStatisticsAdvanced.csv"},
    {"entity": "team",   "measure": "Four Factors", "file": "TeamStatisticsFourFactors.csv"},
    {"entity": "team",   "measure": "Misc",        "file": "TeamStatisticsMisc.csv"},
    {"entity": "team",   "measure": "Scoring",     "file": "TeamStatisticsScoring.csv"},
]

# The Master Identity Columns (Must match what is in PlayerStatistics.csv)
PLAYER_IDENTITY = [
    "firstName", "lastName", "personId", "gameId", "gameDateTimeEst", 
    "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName", 
    "gameType", "home", "win"
]

TEAM_IDENTITY = [
    "gameId", "gameDateTimeEst", "teamCity", "teamName", "teamId", 
    "opponentTeamCity", "opponentTeamName", "opponentTeamId",
    "gameType", "home", "win"
]

# --- 1. BUILD LOOKUPS (From Local Disk) ---
def build_lookups():
    print("🏗️ Building Identity Lookups from Local CSVs...")
    
    # Check if files exist locally (Created by pipeline.py just seconds ago)
    if not os.path.exists("PlayerStatistics.csv") or not os.path.exists("TeamStatistics.csv"):
        print("❌ Critical: Standard CSVs not found in local directory.")
        return None, None

    try:
        # Load local files
        df_p = pl.read_csv("PlayerStatistics.csv", infer_schema_length=0)
        df_t = pl.read_csv("TeamStatistics.csv", infer_schema_length=0)
        
        # Standardize IDs
        df_p = df_p.with_columns(pl.col("gameId").str.replace(r"\.0$", ""))
        df_t = df_t.with_columns(pl.col("gameId").str.replace(r"\.0$", ""))
        
        # Build Lookup Maps
        player_lookup = df_p.select(PLAYER_IDENTITY).unique(subset=["gameId", "personId"])
        team_lookup = df_t.select(TEAM_IDENTITY).unique(subset=["gameId", "teamId"])
        
        print(f"   ✅ Lookups Ready. (Players: {player_lookup.height}, Teams: {team_lookup.height})")
        return player_lookup, team_lookup

    except Exception as e:
        print(f"❌ Error reading local CSVs: {e}")
        return None, None

# --- 2. API FETCHER ---
def fetch_season_data(entity, measure_type):
    url = f"https://stats.nba.com/stats/{entity}gamelogs"
    params = {
        "MeasureType": measure_type, 
        "PerMode": "Totals", 
        "LeagueID": "00", 
        "Season": CURRENT_SEASON, 
        "SeasonType": "Regular Season", 
        "DateFrom": "", "DateTo": "", "GameSegment": "", "ISTRound": "",
        "LastNGames": "0", "Location": "", "Month": "0", "OpponentTeamID": "0",
        "Outcome": "", "PORound": "0", "PaceAdjust": "N", "Period": "0",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "VsConference": "", "VsDivision": ""
    }
    
    headers = {
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true"
    }

    # Retry Loop
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1.0, 2.5))
            
            # Setup Session
            if hasattr(requests, "Session"): s = requests.Session()
            else: s = requests
            
            # Curl_cffi vs Standard Requests
            kwargs = {"timeout": 20, "headers": headers, "params": params}
            if "curl_cffi" in str(requests): kwargs["impersonate"] = "safari15_5"
            else: s.headers.update(headers)

            r = s.get(url, **kwargs)
            
            if r.status_code == 200:
                data = r.json()
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                return pl.DataFrame(rows, schema=headers, orient="row")
            elif r.status_code in [429, 500, 502, 503]:
                time.sleep(2 * (attempt + 1))
                continue
            else:
                return None
        except Exception:
            time.sleep(2)
    return None

# --- 3. MERGE LOGIC ---
def atomic_overwrite(df, filename):
    """Writes to temp file and swaps to avoid OS locking errors."""
    tmp = f"{filename}.tmp"
    try:
        df.write_csv(tmp, quote_style="necessary")
        if os.path.exists(filename): os.remove(filename)
        os.rename(tmp, filename)
    except Exception as e:
        print(f"Failed to write {filename}: {e}")

def process_dataset(raw_df, config, lookup_df):
    if raw_df is None or raw_df.height == 0: return

    # A. Rename & Clean
    rename_map = {'GAME_ID': 'gameId', 'PLAYER_ID': 'personId', 'TEAM_ID': 'teamId'}
    df = raw_df.rename({k:v for k,v in rename_map.items() if k in raw_df.columns})
    
    new_cols = {}
    for col in df.columns:
        if col.isupper() and col not in rename_map.values():
            parts = col.lower().split('_')
            camel = parts[0] + ''.join(x.title() for x in parts[1:])
            new_cols[col] = camel
    df = df.rename(new_cols)

    # Clean IDs
    if "gameId" in df.columns:
        df = df.with_columns(
            pl.col("gameId").cast(pl.Utf8).str.replace(r"\.0$", "").str.replace(r"^00", "")
        )
    for id_col in ["personId", "teamId"]:
        if id_col in df.columns:
            df = df.with_columns(pl.col(id_col).cast(pl.Utf8).str.replace(r"\.0$", ""))

    # B. Enrich (Join with Master Lookup)
    join_keys = ["gameId", "personId"] if config['entity'] == "player" else ["gameId", "teamId"]
    df_enriched = df.join(lookup_df, on=join_keys, how="left")

    # C. Select Final Columns
    identity = PLAYER_IDENTITY if config['entity'] == "player" else TEAM_IDENTITY
    present_id = [c for c in identity if c in df_enriched.columns]
    stats = sorted([c for c in df_enriched.columns if c not in present_id])
    
    # Remove junk columns (API metadata we don't want)
    stats = [c for c in stats if "Rank" not in c and c not in ["seasonYear", "AVAILABLE_FLAG"]]
    
    df_final = df_enriched.select(present_id + stats)

    # D. Merge with Local File (if exists)
    if os.path.exists(config['file']):
        try:
            # Load local file
            df_old = pl.read_csv(config['file'], infer_schema_length=0)
            
            # Anti-Join (Replace old rows for these games with new rows)
            new_game_ids = df_final.select("gameId").unique()
            df_old = df_old.join(new_game_ids, on="gameId", how="anti")
            
            # Concat
            df_merged = pl.concat([df_old, df_final], how="diagonal")
            df_merged = df_merged.sort("gameDateTimeEst", descending=True)
            
            atomic_overwrite(df_merged, config['file'])
            print(f"   ✅ Updated {config['file']}: {df_old.height} old + {df_final.height} new.")
        except:
            atomic_overwrite(df_final, config['file'])
    else:
        atomic_overwrite(df_final, config['file'])
        print(f"   ✅ Created {config['file']} ({df_final.height} rows)")

# --- 4. RUNNER ---
if __name__ == "__main__":
    print(f"\n🚀 Phase 2: Advanced Stats Update (Season: {CURRENT_SEASON})")
    
    player_lookup, team_lookup = build_lookups()
    
    if player_lookup is not None:
        for config in DATASET_CONFIG:
            print(f"   📥 Fetching {config['entity']} - {config['measure']}...")
            raw_df = fetch_season_data(config['entity'], config['measure'])
            process_dataset(raw_df, config, lookup_df=player_lookup if config['entity'] == "player" else team_lookup)
            
    print("🎉 Advanced Update Complete.")
