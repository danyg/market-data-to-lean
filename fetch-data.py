#!/usr/bin/env python3

import requests
import pandas as pd
import json
from datetime import datetime, timedelta, date
import pprint
import logging
import os
import alpaca_trade_api as tradeapi
from time import sleep, gmtime, strftime

import sys
import re

sys.path.append("/home/danyg/.secrets")
from dgsecrets import getSecret

from CustomFormatter import CustomFormatter


# logging.getLogger().setLevel(logging.DEBUG)
root_logger = logging.getLogger("fetch-data")
root_logger.setLevel(logging.DEBUG)

# create console handler with a higher log level

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())

root_logger.handlers = [ch]
# root_logger.addHandler(ch)

# Set your Alpaca API credentials
ALPACA_API_KEY = getSecret("alpaca.paper.key")
ALPACA_API_SECRET = getSecret("alpaca.paper.secret")

DATA_BASE_URL = "https://data.alpaca.markets"
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
# Base URL for Alpaca API
BASE_URL = f"{DATA_BASE_URL}/v2"
paper_url = "https://paper-api.alpaca.markets/v2"
headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}

# Define the symbol (SPY in this case) and the timeframe
symbol = "SPY"
timeframe = "1Min"  # Minute data
start = "2023-09-01T00:00:00Z"  # Start time (ISO8601 format)
end = "2023-09-01T23:59:00Z"  # End time (ISO8601 format)
yesterday = datetime.now() - timedelta(days=2)


def _check_symbol(symbol):
    logger = root_logger.getChild("check_symbol")
    try:
        # Define the endpoint URL for listing assets
        url = f"{paper_url}/assets"

        # Make a GET request to the Alpaca API
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            assets = response.json()

            # Find the asset by symbol
            asset = next((a for a in assets if a["symbol"] == symbol), None)

            if asset:
                if asset["tradable"]:
                    logger.info(f"The symbol '{symbol}' is available and tradable.")
                    return True
                else:
                    logger.info(f"The symbol '{symbol}' is available but not tradable.")
                    return False
            else:
                logger.info(f"The symbol '{symbol}' is not available.")
                return False
        else:
            logger.warning(
                f"Failed to retrieve assets. Status code: {response.status_code}"
            )
            logger.debug(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False


cache_check_symbol = {}


def check_symbol(symbol):
    global cache_check_symbol
    if symbol not in cache_check_symbol:
        cache_check_symbol[symbol] = _check_symbol(symbol)

    return cache_check_symbol[symbol]


def generate_daily_time_pairs(
    start_date: datetime, end_date: datetime, no_weekends=True
):
    # Create an empty list to hold the pairs of datetime objects
    time_pairs = []

    # Ensure that the end_date is at the end of the day (23:59:59)
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Initialize current_date to the start of the day of start_date
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Loop through each day from start_date to end_date
    while current_date <= end_date:
        # Define the start and end of the current day
        day_start = current_date
        day_end = current_date.replace(hour=23, minute=59, second=59)

        # Add the pair (day_start, day_end) to the list
        time_pairs.append((day_start, day_end))

        # Move to the next day
        # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
        if no_weekends and current_date.weekday() == 4:
            current_date += timedelta(days=3)
        else:
            current_date += timedelta(days=1)

    return time_pairs


def file_exists_and_not_empty(file_path):
    # Check if the file exists and its size is greater than 0
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def days_between_today_and(target_date: datetime):
    today = datetime.today()  # Get today's date
    delta = today - target_date
    return delta.days


def fetch_historical_data(symbol: str, start_date: str, end_date: str):
    """
    Fetch historical daily price data for a given symbol from Alpaca.
    """
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    limit = days_between_today_and(datetime.strptime(start_date, "%Y-%m-%d"))
    params = {"start": start_date, "end": end_date, "timeframe": "1Day", "limit": limit}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"Error fetching historical data: {response.text}")

    data = response.json()
    df = pd.DataFrame(data["bars"])
    df.index = pd.to_datetime(df.index)
    return df


def fetch_corporate_actions(
    symbol: str, start_date: str, end_date: str, rate_limit: int = 180
):
    """
    Fetch corporate actions (splits and dividends) for the given symbol from Alpaca in a date range.
    The API is limited to 90 days per request, so this function handles multiple requests if needed.

    Args:
        symbol (str): The stock symbol to retrieve corporate actions for.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        rate_limit (int): Maximum number of API requests per minute. Default is 180.

    Returns:
        List[dict]: A list of corporate action announcements.
    """
    logger = root_logger.getChild("fetch_corporate_actions")
    url = f"{PAPER_BASE_URL}/v2/corporate_actions/announcements"

    # Convert input strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # List to hold all corporate action data
    corporate_actions = []

    # Break the date range into 90-day intervals
    request_start = start_date
    request_end = min(request_start + timedelta(days=90), end_date)

    # Calculate time delay between requests to respect rate limit
    delay_between_requests = 60.0 / rate_limit  # in seconds

    while request_start <= end_date:
        # Format the request start and end dates
        params = {
            "symbols": symbol,
            "ca_types": "split,dividend",
            "since": request_start.strftime("%Y-%m-%d"),
            "until": request_end.strftime("%Y-%m-%d"),
        }

        # Make the API request
        logger.info(
            f'Requesting corporate_actions for {symbol} from {request_start.strftime("%Y-%m-%d")} to {request_end.strftime("%Y-%m-%d")}'
        )
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching corporate actions: {response.text}")

        # Append the results
        actions = response.json() or []
        corporate_actions.extend(actions)

        # Move to the next 90-day interval
        request_start = request_end + timedelta(days=1)
        request_end = min(request_start + timedelta(days=90), end_date)

        # Respect the rate limit by sleeping
        if request_start <= end_date:
            sleep(delay_between_requests)

    return corporate_actions


def write_factor_data(symbol: str, fromDate=datetime(2016, 1, 1)):
    raise Exception("write_factor_data is not working...")
    """
    Fetches splits and dividends from Alpaca and writes a factor file for the given symbol
    in the LEAN format.
    """
    logger = root_logger.getChild("write_factor_data")
    # Fetch historical price data
    start_date = fromDate.strftime("%Y-%m-%d")
    end_date = yesterday.strftime("%Y-%m-%d")
    prices_df = fetch_historical_data(symbol, start_date, end_date)

    if prices_df.empty:
        logger.error(f"No historical price data found for {symbol}.")
        return

    factor_cache = f"./alpaca_factor_data"
    os.makedirs(factor_cache, exist_ok=True)

    splits_file = f"{factor_cache}/{symbol}_splits.csv"
    dividends_file = f"{factor_cache}/{symbol}_dividends.csv"
    corporate_actions_file = f"{factor_cache}/{symbol}_corporate_actions.json"
    splits_df = None
    dividends_df = None

    # TODO reactivate when csv are done properly
    # if file_exists_and_not_empty(splits_file):
    #     splits_df = pd.read_csv(splits_file)
    # if file_exists_and_not_empty(dividends_file):
    #     dividends_df = pd.read_csv(dividends_file)

    def by_symbol(action):
        if "target_symbol" in action:
            return action["target_symbol"] == symbol
        return False

    if splits_df is None or dividends_df is None:
        # Fetch corporate actions (splits and dividends)
        corporate_actions = None
        if file_exists_and_not_empty(corporate_actions_file):
            corporate_actions = json.load(open(corporate_actions_file, "r"))
        else:
            corporate_actions = fetch_corporate_actions(symbol, start_date, end_date)
            json.dump(corporate_actions, open(corporate_actions_file, "w"), indent="  ")

        # Parse splits and dividends
        splits = []
        dividends = []

        logger.debug(f"corporate_actions: {len(corporate_actions)} items")
        corporate_actions = [i for i in corporate_actions if by_symbol(i)]
        logger.debug(f"corporate_actions: {len(corporate_actions)} items")

        for action in corporate_actions:
            try:
                # Handle splits
                if (
                    action["ca_type"] == "split"
                    and action["old_rate"]
                    and action["new_rate"]
                ):
                    split_ratio = float(action["new_rate"]) / float(action["old_rate"])
                    splits.append(
                        {
                            "t": pd.to_datetime(action["ex_date"]),
                            "split_ratio": split_ratio,
                        }
                    )

                # Handle dividends
                elif action["ca_type"] == "dividend" and action["cash"]:
                    dividends.append(
                        {
                            "t": pd.to_datetime(action["ex_date"]),
                            "amount": float(action["cash"]),
                        }
                    )
            except Exception as e:
                logging.error(f"Error occur when treating an action: {str(e)}")
                pprint.pp(action)
                exit(1)

        # Convert splits and dividends to DataFrames and set their indexes to dates
        splits_df = pd.DataFrame(splits).set_index("t").sort_index()
        splits_df.index = pd.to_datetime(splits_df.index)
        dividends_df = pd.DataFrame(dividends).set_index("t").sort_index()
        dividends_df.index = pd.to_datetime(dividends_df.index)

        splits_df.to_csv(splits_file)
        dividends_df.to_csv(dividends_file)
    else:
        logger.info("Reusing stored csv")
        # TODO calculate last data and fetch from there

    # TODO this calculation is wrong, is not actually calculating the factors as expected Debug!
    # Initialize price and split factors
    price_factor = 1.0
    split_factor = 1.0
    factor_rows = []

    # Track the previous factors to only write when they change
    prev_price_factor = 0
    prev_split_factor = 0

    # Iterate through the price data and adjust for splits and dividends

    factors = {}

    for idx, row in splits_df.iterrows():
        split_ratio = row["split_ratio"]
        split_factor *= split_ratio
        logger.debug(
            f"[{symbol}] splits found for: {idx} | split_factor = {split_factor}"
        )
        price = prices_df.loc[prices_df.index.asof(idx)]["c"]

        if idx not in factors:
            factors[idx] = {}
        factors[idx]["split_ratio"] = split_ratio
        factors[idx]["price"] = price

    for idx, row in dividends_df.iterrows():
        price = prices_df.loc[prices_df.index.asof(idx)]["c"]

        dividend = row["amount"]
        price_factor *= 1 - dividend / price  # 'c' is close price in Alpaca data
        logger.debug(
            f"[{symbol}] dividend found for: {idx} | price_factor = {price_factor}"
        )
        if idx not in factors:
            factors[idx] = {}
        factors[idx]["price_factor"] = price_factor
        factors[idx]["price"] = price

    # for idx, row in prices_df.iterrows():
    #     date = pd.to_datetime(row["t"]).date()  # Convert 't' (timestamp) to date

    #     logger.debug(f"[{symbol}] processing date: {date} {splits_df[date]}")

    #     # Check for splits on the date
    #     if date in splits_df_index:
    #         split_ratio = splits_df.loc[date]["split_ratio"]
    #         split_factor *= split_ratio
    #         logger.debug(
    #             f"[{symbol}] splits found for: {date} | split_factor = {split_factor}"
    #         )

    #     # Check for dividends on the date
    #     if date in dividends_df_index:
    #         dividend = dividends_df.loc[date]["amount"]
    #         price_factor *= 1 - dividend / row["c"]  # 'c' is close price in Alpaca data
    #         logger.debug(
    #             f"[{symbol}] dividend found for: {date} | price_factor = {price_factor}"
    #         )

    #     # Only write to the factor file if factors have changed
    #     if (
    #         True
    #         or price_factor != prev_price_factor
    #         or split_factor != prev_split_factor
    #     ):
    #         factor_rows.append(
    #             f"{date.strftime('%Y%m%d')},{price_factor:.6f},{split_factor:.6f},{price_factor:.6f},{split_factor:.6f}"
    #         )
    #         prev_price_factor = price_factor
    #         prev_split_factor = split_factor

    factors[pd.to_datetime(date(2050, 12, 31))] = {
        "price_factor": 1,
        "split_ratio": 1,
        "price": 0,
    }
    factors_df = pd.DataFrame.from_dict(
        factors, orient="index", columns=["price_factor", "split_ratio", "price"]
    )
    factors_df.index = pd.to_datetime(factors_df.index)

    # Create the output directory if it doesn't exist
    factor_cache = f"./lean_data/equity/usa/factor_files/"
    os.makedirs(factor_cache, exist_ok=True)

    # Write factor file
    output_path = os.path.join(factor_cache, f"{symbol.lower()}.csv")
    factors_df.to_csv(output_path, header=False, date_format="%Y%m%d")

    logger.info(f"Factor file generated for {symbol} at {output_path}")


def get_file_path(
    symbol: str, s: datetime, e: datetime, timeframe: str, no_weekends: bool
):
    clean_symbol = re.sub(r"\W", "", symbol)

    directory = (
        "./input_data" if no_weekends else "./input_data_others"
    ) + f"/{clean_symbol}"
    file_name = f"{clean_symbol}_{int(round(s.timestamp()))}_{int(round(e.timestamp()))}_{timeframe}"
    file_path = f"{directory}/{file_name}.csv"

    if not os.path.exists(directory):
        os.makedirs(directory)

    return (file_path, file_name, directory)


marketDataDownloadJobs = []
total = 0
done = 0


def enqueue_download_market_data(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe="1Min",
    no_weekends=True,
    limit_request_per_minute=180,
):
    global marketDataDownloadJobs
    global total
    global done

    datePairsFull = generate_daily_time_pairs(start, end, no_weekends)
    if len(datePairsFull) == 0:
        return

    datePairs = []
    file_path, file_name, directory = get_file_path(
        symbol, datePairsFull[0][0], datePairsFull[0][1], timeframe, no_weekends
    )
    files_in_symbol_dir = os.listdir(directory)
    for s, e in datePairsFull:
        file_path, file_name, directory = get_file_path(
            symbol, s, e, timeframe, no_weekends
        )

        banned = []
        banned_file = f"{directory}/_banned.txt"
        if file_exists_and_not_empty(banned_file):
            with open(banned_file, "r") as f:
                banned = f.read().split("\n")

        if f"{file_name}.csv" not in files_in_symbol_dir and file_name not in banned:
            datePairs.append((s, e))

    total += len(datePairs)

    marketDataDownloadJobs.append(
        (symbol, datePairs, "1Min", no_weekends, limit_request_per_minute)
    )


def convert_to_HHMMSS(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def _download_market_data(
    symbol,
    datePairs,
    timeframe="1Min",
    no_weekends=True,
    limit_request_per_minute=180,
):
    global total
    global done

    # API endpoint for getting historical bars
    logger = root_logger.getChild("download_market_data")
    url = f"{BASE_URL}/stocks/{symbol}/bars"

    for s, e in datePairs:
        file_path, file_name, directory = get_file_path(
            symbol, s, e, timeframe, no_weekends
        )

        # <Progress>
        done += 1
        remaining_minutes = (total - done) / limit_request_per_minute
        time_format = convert_to_HHMMSS(remaining_minutes * 60)
        print(
            f"\r[{done:04}/{total:04} {symbol}:{s.strftime('%Y-%m-%d')} ETA: {time_format}... {file_path}                       ",
            end="",
        )
        # </Progress>

        if file_exists_and_not_empty(file_path):
            continue

        sleep(60 / limit_request_per_minute)
        params = {
            "start": f"{s.isoformat()}Z",
            "end": f"{e.isoformat()}Z",
            "timeframe": timeframe,
        }

        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        # with open(f"./alpaca_json/{file_name}.json", "w") as file:
        #     json.dump(
        #         {"url": url, "headers": headers, "params": params, "data": data},
        #         file,
        #         indent="  ",
        #     )

        if not "bars" in data or data["bars"] is None:
            with open(f"{directory}/_banned.txt", "ab") as f:
                f.write(bytes(f"{file_name}\n", "utf8"))
            # logger.warning(f"[download_market_data] No data for day {s.isoformat()}")
            continue

        try:
            # with open(f'./alpaca_json/{file_name}.json', 'w') as file:
            #     json.dump(data, file, indent='  ')

            # Convert to a pandas DataFrame for easier manipulation
            df = pd.DataFrame(data["bars"])
            df["time"] = pd.to_datetime(df["t"], format="ISO8601")  # Convert timestamps
            df.set_index("time", inplace=True)

            # Select the needed columns (close, open, high, low, volume)
            df = df[["o", "h", "l", "c", "v"]]
            df.columns = ["open", "high", "low", "close", "volume"]

            df.to_csv(file_path)
        except Exception as err:
            logging.error(
                f'Error writing data into "{file_name}.json/csv" \n {str(err)}'
            )


def enqueue_download_from_2016(symbol, no_weekends=True):
    logger = root_logger.getChild("enqueue_download_from_2016")
    if not check_symbol(symbol):
        logger.warning(
            f"{symbol} doesn't exist or is not tradable in alpaca. Continuing..."
        )
        return

    enqueue_download_market_data(
        symbol, datetime(2016, 1, 1), datetime(2016, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2017, 1, 1), datetime(2017, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2018, 1, 1), datetime(2018, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2019, 1, 1), datetime(2019, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2020, 1, 1), datetime(2020, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2021, 1, 1), datetime(2021, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2022, 1, 1), datetime(2022, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_market_data(
        symbol, datetime(2023, 1, 1), datetime(2023, 12, 31), no_weekends=no_weekends
    )
    enqueue_download_from_2024(symbol, no_weekends)
    # write_factor_data(symbol, datetime(2016, 1, 1))


def enqueue_download_from_2024(symbol, no_weekends=True):
    logger = root_logger.getChild("enqueue_download_from_2016")
    if not check_symbol(symbol):
        logger.warning(
            f"{symbol} doesn't exist or is not tradable in alpaca. Continuing..."
        )
        return

    enqueue_download_market_data(
        symbol, datetime(2024, 1, 1), yesterday, no_weekends=no_weekends
    )


def download():
    print("")
    for args in marketDataDownloadJobs:
        _download_market_data(*args)


# Comparasion in percent using TradingView from 28 dec 2015

# 2016:  (530$) (2016: +174.34% | Volatility 0.70%  | PERF 5Y: 86% )
enqueue_download_from_2016("SPY")
# 2016:  (72$) (2016: +85.88% | Volatility 0.71%  | PERF 5Y: 24.59% )
enqueue_download_from_2016("SPLV")

enqueue_download_from_2016("KRE")  # not good
# S&P500 (60~70$) (2016: +253.21% | Volatility 0.73%  | PERF 5Y: 86% )
enqueue_download_from_2016("SPLG")

# S&P500 (90$) (2016: +253.31% | Volatility 0.79% | PERF 5Y: 119% )
enqueue_download_from_2016("SPMO")
# S&P500 (80$) (2016: +663.75% | Volatility 2.4%  | PERF 5Y: 181% )
enqueue_download_from_2016("UPRO")

# S&P500 (50$) (2016: +253.21% | Volatility 0.77%  | PERF 5Y: 104% )
enqueue_download_from_2016("SPYG")

# S&P500 (80$) (2016: +443.30% | Volatility 1.36%  | PERF 5Y: 161.96% )
enqueue_download_from_2016("SSO")

# S&P500 (65$) (2016: +156.70% | Volatility 3.04% | PERF 5Y: 67.83% )
enqueue_download_from_2016("XSMO")
# S&P500 (50$) (2016: +115.86% | Volatility 0.81% | PERF 5Y: 60.17% )
enqueue_download_from_2016("SPYV")

# S&P500 (60$) (2016: +116.47% | Volatility 1.86% | PERF 5Y: 54.12% )
enqueue_download_from_2016("IJH")

# S&P500 (520$) (2016: +127.05% | Volatility 1.17% | PERF 5Y: 106.03% )
enqueue_download_from_2016("SPGI")

enqueue_download_from_2016("AAPL")
enqueue_download_from_2016("MSFT")
enqueue_download_from_2016("NVDA")
enqueue_download_from_2016("AMD")

## NEW

enqueue_download_from_2016("QQQ")
enqueue_download_from_2016("TQQQ")
enqueue_download_from_2016("ESG")  # STOXX
enqueue_download_from_2016("HLAL")  # Japan FTSE
enqueue_download_from_2016("FTEC")  # Japan MSCI (better than argt)
enqueue_download_from_2016("ARGT")  # Japan MSCI
enqueue_download_from_2016("SMIN")  # Japan MSCI (maybe better than ARGT?)

# #

enqueue_download_from_2016("ADBE")
enqueue_download_from_2016("TSLA")
enqueue_download_from_2016("VOO")
enqueue_download_from_2016("INTC")
enqueue_download_from_2016("SHOP")  # To be used in oportunistic Algorithm

enqueue_download_from_2016("IWM")
enqueue_download_from_2016("IBM")
enqueue_download_from_2016("AMZN")
enqueue_download_from_2016("ORCL")
enqueue_download_from_2016("GOOG")
enqueue_download_from_2016("GLD")
enqueue_download_from_2016("NFLX")
enqueue_download_from_2016("V")  # Visa
enqueue_download_from_2016("MA")  # MasterCard

# enqueue_download_from_2016("BND")


# enqueue_download_market_data(
#     "BTC/USD", datetime(2024, 1, 1), yesterday, no_weekends=False
# )


# enqueue_download_from_2016("BTC/USD", no_weekends=False)
# enqueue_download_from_2016("BTC/ETH", no_weekends=False)


download()
