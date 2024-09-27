#!/usr/bin/env python3

import os
import pandas as pd
import zipfile
from datetime import datetime

# Parent directory for storing LEAN data
lean_data_dir = "./lean_data"
minute_output_dir = f"{lean_data_dir}/equity/usa/minute"
hour_output_dir = f"{lean_data_dir}/equity/usa/hour"
daily_output_dir = f"{lean_data_dir}/equity/usa/daily"


def file_exists_and_not_empty(file_path):
    # Check if the file exists and its size is greater than 0
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


# Ensure output directories exist
for directory in [minute_output_dir, hour_output_dir, daily_output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to convert time to milliseconds from midnight
def time_to_milliseconds(dt):
    # Get the time of day (hours, minutes, seconds)
    time_of_day = dt.time()
    # Calculate milliseconds from midnight
    milliseconds = (
        time_of_day.hour * 3600 + time_of_day.minute * 60 + time_of_day.second
    ) * 1000 + time_of_day.microsecond // 1000
    return milliseconds


queue = []
total = 0
done = 0


def printCurrPrice(time, symbol, val):
    global total
    global done
    print(
        f"\r[{done:4}/{total:4}] {time.strftime('%Y-%m-%d')} | {symbol} ${val/10000:5.2f}         ",
        end="",
    )


def enqueue_process_csv_to_lean(csv_file, hourly_data_all, daily_data_all):
    global queue
    global total

    try:

        file_name = os.path.basename(csv_file)
        symbol, start_timestamp, end_timestamp, _ = file_name.replace(".csv", "").split(
            "_"
        )
        symbol = symbol.lower()  # Ensure symbol is lowercase
    except Exception as e:
        print(f"ERROR: {csv_file}")
        print(e)
        exit(30)

    queue.append((csv_file, hourly_data_all, daily_data_all))
    total += 1
    return symbol


def process_queue():
    for args in queue:
        process_csv_to_lean(*args)


# Function to convert symbol's raw CSV data into LEAN format (minute, hourly, daily)
def process_csv_to_lean(csv_file, hourly_data_all, daily_data_all):
    global done
    done += 1

    # Parse symbol, start, and end timestamps from the file name
    file_name = os.path.basename(csv_file)
    symbol, start_timestamp, end_timestamp, _ = file_name.replace(".csv", "").split("_")
    symbol = symbol.lower()  # Ensure symbol is lowercase

    csv_reader = pd.read_csv(csv_file, parse_dates=["time"])

    if not (symbol in hourly_data_all):
        hourly_data_all[symbol] = []
    if not (symbol in daily_data_all):
        daily_data_all[symbol] = []

    csv_reader["time"] = pd.to_datetime(csv_reader["time"])
    csv_reader.set_index("time", inplace=True)

    # Filter data to only include the trading hours (9:30 AM to 4:00 PM)
    csv_reader = csv_reader.between_time("09:30", "16:00")

    # Group data by date to process one day at a time
    for date, daily_data in csv_reader.groupby(csv_reader.index.date):
        date_str = date.strftime("%Y%m%d")

        # convertion to deci-cent
        daily_data["open"] = (daily_data["open"] * 10000).astype(int)
        daily_data["high"] = (daily_data["high"] * 10000).astype(int)
        daily_data["low"] = (daily_data["low"] * 10000).astype(int)
        daily_data["close"] = (daily_data["close"] * 10000).astype(int)
        # other convertions
        daily_data["volume"] = (daily_data["volume"]).astype(int)
        daily_data["lean_time"] = daily_data.index.to_series().apply(
            time_to_milliseconds
        )

        df = DataFrame(
            daily_data.set_index("lean_time"),
            columns=["open", "high", "low", "close", "volume"],
        )

        ### Step 1: Save Minute Data ###
        minute_zip_filename = f"{minute_output_dir}/{symbol}/{date_str}_trade.zip"
        if not os.path.exists(f"{minute_output_dir}/{symbol}"):
            os.makedirs(f"{minute_output_dir}/{symbol}")

        with zipfile.ZipFile(minute_zip_filename, "a") as zipf:

            zipf.writestr(f"{symbol}.csv", df.to_csv(header=False))

        ### Step 2: Collect Hourly Data ###
        hourly_data = (
            daily_data.resample("h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )  # Drop incomplete hours
        hourly_data_all[symbol].append(hourly_data)

        ### Step 3: Collect Daily Data ###
        daily_data_resampled = (
            daily_data.resample("D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )  # Drop incomplete days
        daily_data_all[symbol].append(daily_data_resampled)

        printCurrPrice(
            hourly_data.index.to_series().iloc[-1],
            symbol,
            hourly_data["close"].iloc[-1],
        )

    return symbol


import pprint
from pandas import DataFrame


def process_all_csvs(data_dir):
    symbols = set()
    hourly_data_all = {}
    daily_data_all = {}
    for root, dirs, files in os.walk(data_dir):
        files.sort()
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                symbol = enqueue_process_csv_to_lean(
                    csv_file, hourly_data_all, daily_data_all
                )
                symbols.add(symbol)

    process_queue()

    for symbol in symbols:
        print(f"Processing Resolutions H, D for {symbol}...")
        saveDFtoZIPCSV(symbol, hour_output_dir, pd.concat(hourly_data_all[symbol]))
        saveDFtoZIPCSV(symbol, daily_output_dir, pd.concat(daily_data_all[symbol]))
        print(f"{symbol} Done!")


def saveDFtoZIPCSV(symbol: str, output_dir: str, df: DataFrame):
    df = df.sort_index()

    csvFile = f"{symbol}.csv"
    zip_filename = f"{output_dir}/{symbol}.zip"

    print(f"Saving {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "w") as zipf:  # Ensure single CSV file in ZIP
        zipf.writestr(csvFile, df.to_csv(header=False, date_format="%Y%m%d %H:%M"))

    print(f"{zip_filename} Done...")


# Run the script
data_dir = "./input_data"  # Your directory containing the CSV files
process_all_csvs(data_dir)
