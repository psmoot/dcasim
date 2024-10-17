#! python3
"""
Run historical simulations of stock transactions to see the results of dollar
cost averaging.

Raw stock price and CPI data comes from Alphavantage.  At some point we could
plug in other data sources.

This script tries to work in current dollars.  We take historical values and
adjust them to current dollars using a CPI deflator.  I'll use the term
"nominal" to mean the value is the actual value at the date of the transaction.
"""
from enum import Enum
from typing import Dict, Any
from datetime import date, datetime
import time
import os
from pickle import load, dump

import argparse
from requests import get
from tabulate import tabulate


with open("api-key.txt") as fp:
    api_key = fp.readline()

api_key.strip()


class StockPrice:
    """
    Represensation of a stock price.  We save the date, high, low, and close for
    the stock price over an interval (typically a month).

    Prices can be retrieved in either current or nominal values.  Values are
    stored in nominal values and adjusted for inflation to get current values.

    We also store any dividends paid in the interval.
    """

    class PriceAdjustment(Enum):
        CURRENT = 1
        NOMINAL = 2

    class Price(Enum):
        OPEN = 1
        CLOSE = 2
        HIGH = 3
        LOW = 4
        DIVIDEND = 5
        ADJUSTED_CLOSE = 6

    def __init__(self, price_date: date, data: Dict):
        """
        Create instance from JSON Alphavantage blob.

        date is the date the data represents.  We need this to inflate and
        deflate values.

        Extract fields from JSON blob and store as members.
        """
        self.date = price_date

        self.prices = dict()
        self.prices[StockPrice.Price.OPEN] = float(data["1. open"])
        self.prices[StockPrice.Price.HIGH] = float(data["2. high"])
        self.prices[StockPrice.Price.LOW] = float(data["3. low"])
        self.prices[StockPrice.Price.CLOSE] = float(data["4. close"])
        self.prices[StockPrice.Price.ADJUSTED_CLOSE] = float(data["5. adjusted close"])
        self.prices[StockPrice.Price.DIVIDEND] = float(data["7. dividend amount"])

    def get_price(
        self, which: Price, inflation: PriceAdjustment = PriceAdjustment.CURRENT
    ):
        """
        Get the open/high/low/close price of a stock in an interval, in either
        nominal or current dollars.
        """
        price = self.prices[which]
        if inflation == StockPrice.PriceAdjustment.CURRENT:
            price = cpi_data.inflate(self.date, price)

        return price


class Inflation:
    """
    Manage data about inflation.  Convert between nominal and current dollars.
    """

    def __init__(self):
        self.cpi_data = dict()
        self.today = date(year=date.today().year, month=date.today().month, day=1)

    def load_data(self, start_date: date) -> None:
        """
        Fetch data from source, parse, and fill in cpi_data dict.
        """

        # Get raw CPI data
        url = construct_url("CPI", interval="monthly")
        data = fetch_data(url, "cpi")

        # First element of data["data"] list typically is the start of the
        # previous month.  Save that as today's CPI value so we have a value for
        # the current, partial month.
        self.cpi_data[self.today] = float(data["data"][0]["value"])

        #
        # CPI data has element "data" which is a list of hashes.  We want a
        # hash indexed by date with a float value.
        #
        for elt in data["data"]:
            cpi_date = date_str_to_date(elt["date"])

            if cpi_date < start_date:
                continue

            self.cpi_data[cpi_date] = float(elt["value"])

    def inflate(self, past_date: date, past_value: float) -> float:
        """
        Given a nominal value on a date, return the value in current dollars.
        """
        today_cpi = self.cpi_data[self.today]

        if past_date not in self.cpi_data:
            print("Could not find CPI value for {orig_date}")
            exit()

        value_cpi = self.cpi_data[past_date]

        return past_value * today_cpi / value_cpi

    def deflate(self, past_date: date, current_value: float) -> float:
        """
        Given a value in current dollars, deflate it to what it would have been on
        some past date.
        """
        today_cpi = self.cpi_data[self.today]

        if past_date not in self.cpi_data:
            print("Could not find CPI value for {past_date}")
            exit()

        past_cpi = self.cpi_data[past_date]

        return current_value * past_cpi / today_cpi


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

        if "Error Message" in data:
            raise ValueError(
                f"Error fetching data for {pickle_name}, {data['Error Message']}"
            )

        with open(pickle_file_path, "wb") as pkl_fp:
            dump(data, pkl_fp)

    return data


parser = argparse.ArgumentParser(
    description="Simulate buying stocks with dollar cost averaging"
)
parser.add_argument(
    "--duration", "-d", type=int, default=10, help="Number of years to simulate"
)

# Main program begins here
parser.add_argument(
    "--symbol",
    "-s",
    type=str,
    dest="symbols",
    action="extend",
    nargs="+",
    required=True,
    help="Stock symbol to simulate",
)

parser.add_argument("--verbose", "-v", action="count", default=0)

parser.add_argument(
    "--skill",
    "-S",
    choices=["best", "worst", "close"],
    help="How skillful to pick prices: the best, worst, or closing price for the period",
)
args = parser.parse_args()

# Get starting date for simulation, ten years ago.
start_date = date(
    year=date.today().year - args.duration, month=date.today().month, day=1
)
end_date = date(year=date.today().year, month=date.today().month, day=1)

cpi_data = Inflation()
cpi_data.load_data(start_date)


def load_stock_values(ticker_symbol: str) -> Dict[date, StockPrice]:
    """
    Load prices and dividend data for a given stock.

    Result will be a hash keyed by a date.  The date will be the first day of an
    interval, typically a month.
    """
    url = construct_url("TIME_SERIES_MONTHLY_ADJUSTED", symbol=ticker_symbol)
    data = fetch_data(url, ticker_symbol)

    if "Monthly Adjusted Time Series" not in data:
        print(f"Did not fetch data")
        raise RuntimeError(f"Did not fetch data for {ticker_symbol}, {data}")

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

        prices[price_date] = StockPrice(price_date, v)

    return prices


share_prices = dict()
for symbol in args.symbols:
    share_prices[symbol] = load_stock_values(symbol.upper())

# We'll use tabulate to print results from this list of lists.
output = []

#
# Now simulate buying one thousand current dollars of a stock each month.
# Adjust stock close price to current dollars and buy shares.  Remember how many
# shares we have so we can compute dividends.
#
# When the stock pays a dividend, also convert to current dollars, multiply by
# number of shares, and add to accumulated dividends.
#
for symbol in args.symbols:
    shares = 0
    dividends = 0
    cost_basis = 0

    for buy_date in sorted(share_prices[symbol].keys()):

        dividend = share_prices[symbol][buy_date].get_price(StockPrice.Price.DIVIDEND)
        dividends += shares * dividend

        if args.skill == "close":
            share_price = share_prices[symbol][buy_date].get_price(
                StockPrice.Price.ADJUSTED_CLOSE
            )
        else:
            # Figure out ratio of close to adjusted close, apply that ratio to
            # either the high or low price to estimate the adjusted high or low.
            # This isn't entirely accurate w.r.t. dividends.
            ratio = share_prices[symbol][buy_date].get_price(
                StockPrice.Price.ADJUSTED_CLOSE
            ) / share_prices[symbol][buy_date].get_price(StockPrice.Price.CLOSE)
            if args.skill == "best":
                buy_price = share_prices[symbol][buy_date].get_price(
                    StockPrice.Price.LOW
                )
            else:
                buy_price = share_prices[symbol][buy_date].get_price(
                    StockPrice.Price.HIGH
                )
            share_price = buy_price * ratio

        new_shares = 1000 / share_price
        shares += new_shares

        new_basis = cpi_data.deflate(buy_date, 1000)
        cost_basis += new_basis

        if args.verbose > 0:
            print(
                f"On {buy_date} bought {new_shares:,.2f} for ${new_basis:,.2f} at ${share_price:,.2f} per share."
            )

    # Need first and last date we could have bought shares, which might not be
    # start_date and end_date
    first_buy_date = min(share_prices[symbol].keys())
    last_buy_date = max(share_prices[symbol].keys())

    share_value = shares * share_prices[symbol][last_buy_date].get_price(
        StockPrice.Price.CLOSE
    )

    gain = share_value + dividends - cost_basis

    output.append(
        [
            symbol.upper(),
            first_buy_date,
            last_buy_date,
            shares,
            cost_basis,
            share_value,
            dividends,
            gain,
        ]
    )

print(f"At end of simulation from {start_date} to {end_date}")
print(
    tabulate(
        output,
        headers=[
            "Stock",
            "From",
            "To",
            "Shares",
            "Basis",
            "Present value",
            "Dividends",
            "Gain",
        ],
        floatfmt=",.2f",
    )
)
