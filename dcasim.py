#! python3

from typing import Dict, Any
from datetime import date, datetime
import time
import os
from pickle import load, dump


from requests import get


with open("api-key.txt") as fp:
    api_key = fp.readline()

api_key.strip()

#
# String constants
#
CPI_TODAY_KEY = "cpi_today"
DIVIDEND_KEY = "dividend"
CLOSE_KEY = "close"

# Get starting date for simulation, ten years ago.
start_date = date(year=date.today().year - 10, month=date.today().month, day=1)
end_date = date(year=date.today().year, month=date.today().month, day=1)


def construct_url(function="TIME_SERIES_MONTHLY_ADJUSTED", **kwargs) -> str:
    """
    Construct URL.  Every URL needs a function paramter and many functions also
    require additional parameters.

    No doubt there's a library to make this safer but I can't find one.
    """
    url = f"https://www.alphavantage.co/query?function={function}&apikey={api_key}"
    for name, value in kwargs.items():
        url += f"&{name}={value}"

    return url


def date_str_to_date(date_str: str) -> date:
    """
    Take date string of the form "YYYY-MM-DD" and return date object for first
    day of month.

    For purposes of this script, we will ignore that CPI dates are from the
    beginning of the month and most stock prices are from the end.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date(year=date_obj.year, month=date_obj.month, day=1)


def fetch_data(url: str, pickle_name: str) -> Any:
    """
    Load data, either from a cached pickle file (if it exists) or by fetching a
    URL.  We want to cache data to avoid API rate limits.

    If we fetch data, save result to a pickle file for next time.
    """
    pickle_file_path = f"./.cache/{pickle_name}.pkl"

    ONE_MONTH_SECS = 3600 * 24 * 31
    if os.path.exists(pickle_file_path):
        age = time.time() - os.path.getmtime(pickle_file_path)
    else:
        age = ONE_MONTH_SECS + 1

    if os.path.exists(pickle_file_path) and age <= ONE_MONTH_SECS:
        with open(pickle_file_path, "rb") as pkl_fp:
            data = load(pkl_fp)
    else:
        # No cached version, fetch and cache
        r = get(url, timeout=15)
        data = r.json()

        with open(pickle_file_path, "wb") as pkl_fp:
            dump(data, pkl_fp)

    return data


cpi_data = dict()


def load_cpi_data() -> None:
    """
    Load global cpi_data hash with relative price values.

    Save the first value we get as today's adjustment value.
    """

    # Get raw CPI data
    url = construct_url("CPI", interval="monthly")
    data = fetch_data(url, "cpi")

    # First element of data["data"] list is value as of today.
    #
    # Hang on to that as special entry in cpi_data hash
    cpi_data[CPI_TODAY_KEY] = float(data["data"][0]["value"])

    # Ensure today's month is included.  It's generally not because the month is
    # not over.
    cpi_data[end_date] = cpi_data[CPI_TODAY_KEY]

    #
    # CPI data has element "data" which is a list of hashes.  We want a
    # hash indexed by date with a float value.
    #
    for elt in data["data"]:
        cpi_date = date_str_to_date(elt["date"])

        if cpi_date < start_date:
            continue

        value = float(elt["value"])

        cpi_data[cpi_date] = value


def inflation_adjust(orig_date: date, value: float) -> float:
    """
    Given a value on a date, return the value in current dollars.
    """
    today_cpi = cpi_data[CPI_TODAY_KEY]

    if orig_date not in cpi_data:
        print("Could not find CPI value for {orig_date}")
        exit()

    value_cpi = cpi_data[orig_date]

    return value * today_cpi / value_cpi


def load_symbol_values(symbol: str) -> Dict[str, Dict[str, float]]:
    """
    Load closing price and dividend data for a given stock.

    Result will be a hash keyed by a YYYY-MM date string.  The value is a hash
    with "close" and "dividend" values in nominal (that is, not adjusted for
    inflation) values.
    """
    url = construct_url("TIME_SERIES_MONTHLY_ADJUSTED", symbol=symbol)
    data = fetch_data(url, symbol)

    #
    # Raw JSON data has a lot of fields we don't need.  What we do
    # need is the date, formatted as above, the adjusted close value
    # (keyed with "5. adjusted close"), and any dividend paid that
    # month, adjusted for inflation to be in current dollars.  Dividend
    # is keyed by "7. dividend amount"
    #
    prices = dict()
    for k, v in data["Monthly Adjusted Time Series"].items():
        price_date = date_str_to_date(k)
        if price_date < start_date:
            continue

        close = float(v["5. adjusted close"])
        dividend = float(v["7. dividend amount"])

        prices[price_date] = {CLOSE_KEY: close, DIVIDEND_KEY: dividend}

    return prices


load_cpi_data()
aapl_prices = load_symbol_values("AAPL")

#
# OK, now simulate buying one thousand current dollars of a
# stock each month.  Adjust stock close price to current
# dollars and buy shares.  Remember how many shares we have
# so we can compute dividends.
#
# When the stock pays a dividend, also convert to current
# dollars, multiply by number of shares, and add to
# accumulated dividends.
#
shares = 0
dividends = 0

for buy_date in sorted(aapl_prices.keys()):

    if aapl_prices[buy_date][DIVIDEND_KEY] > 0:
        per_share_div = inflation_adjust(buy_date, aapl_prices[buy_date]["dividend"])
        dividends += shares * per_share_div

    price = inflation_adjust(buy_date, aapl_prices[buy_date][CLOSE_KEY])
    shares += 1000 / price

share_value = shares * aapl_prices[end_date][CLOSE_KEY]
print(
    f"At end of simulation, bought {shares:,.0f} shares worth ${share_value:,.2f} and received ${dividends:,.2f} in dividends."
)
