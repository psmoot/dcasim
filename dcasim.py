#! python3
"""Simulations of stock transactions to see the results of dollar cost averaging.

Raw stock price and CPI data comes from Alphavantage.  At some point we could
plug in other data sources.

This script tries to work in current dollars.  We take historical values and
adjust them to current dollars using a CPI deflator.  I'll use the term
"nominal" to mean the value is the actual value at the date of the transaction.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from pickle import dump, load
from typing import Any, Dict, List
from collections import namedtuple

from requests import get
from tabulate import tabulate

logger = logging.getLogger(__name__)


class StockPrice:
    """Representation of a stock price.

    We save the date, high, low, and close for the stock price over an interval
    (typically a month).

    Prices can be retrieved in either current or nominal values.  Values are
    stored in nominal values and adjusted for inflation to get current values.

    We also store any dividends paid in the interval.
    """

    class PriceAdjustment(Enum):
        """How to adjust prices for inflation."""

        CURRENT = 1
        NOMINAL = 2

    class Price(Enum):
        """What price to use for a given date."""

        OPEN = 1
        CLOSE = 2
        HIGH = 3
        LOW = 4
        DIVIDEND = 5
        ADJUSTED_CLOSE = 6

    def __init__(self, price_date: date, data: dict[str, str]) -> None:
        """Create instance from JSON Alphavantage blob.

        date is the date the data represents.  We need this to inflate and
        deflate values.

        Extract fields from JSON blob and store as members.
        """
        self.date = price_date

        self.prices = {}
        self.prices[StockPrice.Price.OPEN] = float(data["1. open"])
        self.prices[StockPrice.Price.HIGH] = float(data["2. high"])
        self.prices[StockPrice.Price.LOW] = float(data["3. low"])
        self.prices[StockPrice.Price.CLOSE] = float(data["4. close"])
        self.prices[StockPrice.Price.ADJUSTED_CLOSE] = float(data["5. adjusted close"])
        self.prices[StockPrice.Price.DIVIDEND] = float(data["7. dividend amount"])

    def get_price(
        self, which: Price, inflation: PriceAdjustment = PriceAdjustment.CURRENT
    ) -> float:
        """Get the open/high/low/close price of a stock.

        All prices are relative to an interval, typically a month.  Values can
        be returned in either nominal or current dollars.
        """
        price = self.prices[which]
        if inflation == StockPrice.PriceAdjustment.CURRENT:
            price = cpi_data.inflate(self.date, price)

        return price

    def transaction_price(self) -> float:
        """Compute price to buy or sell based on skill level.

        If you're the most skillfull, use the lowest buying price or highest
        sell price.

        If you're least skillfull, use the reverse.

        If you're lazy, just use the close price.
        """
        if args.skill == "close":
            share_price = self.get_price(StockPrice.Price.ADJUSTED_CLOSE)
        else:
            # Figure out ratio of close to adjusted close, apply that ratio to
            # either the high or low price to estimate the adjusted high or low.
            # This isn't entirely accurate w.r.t. dividends.
            ratio = self.get_price(StockPrice.Price.ADJUSTED_CLOSE) / self.get_price(
                StockPrice.Price.CLOSE
            )

            # If we're a skilled buyer or unskilled seller, use the lowest price.
            #
            # If we're a skilled seller or unskilled buyer, use the highest price.
            if (args.skill == "best" and args.action == "buy") or (
                args.skill == "worst" and args.action == "sell"
            ):
                price = self.get_price(StockPrice.Price.LOW)
            else:
                price = self.get_price(StockPrice.Price.HIGH)

            share_price = price * ratio

        return share_price


class Inflation:
    """Manage data about inflation.  Convert between nominal and current dollars."""

    def __init__(self) -> None:
        """Create new inflation entry for given time."""
        self.cpi_data = {}

    def get_edge_dates(self) -> tuple[date, date]:
        """Return the earliest and latest dates for which we have CPI data."""
        return self.start_date, self.end_date

    def load_data(self, start_date: date) -> None:
        """Fetch data from source, parse, and fill in cpi_data dict."""
        # Get raw CPI data
        url = construct_url("CPI", interval="monthly")
        data = fetch_data(url, "cpi")

        #
        # CPI data has element "data" which is a list of hashes.  We want a
        # hash indexed by date with a float value.
        #
        for elt in data["data"]:
            cpi_date = date_str_to_date(elt["date"])

            if cpi_date < start_date:
                continue

            self.cpi_data[cpi_date] = float(elt["value"])

        self.start_date = min(self.cpi_data.keys())
        self.end_date = max(self.cpi_data.keys())

    def inflate(self, past_date: date, past_value: float) -> float:
        """Given a nominal value on a date, return the value in current dollars."""
        today_cpi = self.cpi_data[self.end_date]

        if past_date not in self.cpi_data:
            logger.error(f"Could not find CPI value for {past_date}")
            sys.exit(1)

        value_cpi = self.cpi_data[past_date]

        return past_value * today_cpi / value_cpi

    def deflate(self, past_date: date, current_value: float) -> float:
        """Deflate current dollars to nominal dollars on past date."""
        today_cpi = self.cpi_data[self.end_date]

        if past_date not in self.cpi_data:
            logger.error(f"Could not find CPI value for {past_date}")
            sys.exit(1)

        past_cpi = self.cpi_data[past_date]

        return current_value * past_cpi / today_cpi


def construct_url(
    function: str = "TIME_SERIES_MONTHLY_ADJUSTED", **kwargs: dict[str, Any]
) -> str:
    """Construct URL of data source.

    Every URL needs a function parameter and many functions also require
    additional parameters.

    No doubt there's a library to make this safer but I can't find one.
    """
    url = f"https://www.alphavantage.co/query?function={function}"
    for name, value in kwargs.items():
        url += f"&{name}={value}"

    return url


def date_str_to_date(date_str: str) -> date:
    """Convert date string of the form "YYYY-MM-DD" to date object.

    Returned dates are always for the first day of the month.  For purposes of
    this script, we will ignore that CPI dates are from the beginning of the
    month and most stock prices are from the end.  It's close enough.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
    return date(year=date_obj.year, month=date_obj.month, day=1)


ONE_MONTH_SECS = 3600 * 24 * 31


# Remember if we hit the daily API limit.  If we have, only load cached data or fail.
at_api_limit = False


def fetch_data(url: str, pickle_name: str) -> dict[str, str]:
    """Load data for a symbol.

    Load from cached pickle file, if available and recent enough, to avoid API
    rate limits.

    If we fetch data, save result to a pickle file for next time.
    """
    pickle_file_path = Path(f"./.cache/{pickle_name}.pkl")
    global at_api_limit
    age_limit = ONE_MONTH_SECS * 3

    if pickle_file_path.exists():  # noqa: SIM108
        age = time.time() - pickle_file_path.stat().st_mtime
    else:
        age = age_limit + 1

    if pickle_file_path.exists() and age <= age_limit:
        logger.info(f"Loading data for {pickle_name} from {pickle_file_path}.")
        with pickle_file_path.open(mode="rb") as pkl_fp:
            data = load(pkl_fp)  # noqa: S301
    elif at_api_limit:
        logger.info(
            f"Not loading {pickle_name}, no cached data and at API limit for the day."
        )
        return None
    else:
        logger.info(f"Fetching data for {pickle_name} from {url}.")
        # No cached version, fetch and cache
        with Path("api-key.txt").open() as fp:
            api_key = fp.readline()

        api_key.strip()
        url += f"&apikey={api_key}"

        logger.debug(f"Fetching data from {url}")
        r = get(url, timeout=15)
        data = r.json()
        logger.debug(f"Results are {data}")

        if "Error Message" in data:
            msg = f"Error fetching data for {pickle_name}, {data['Error Message']}"
            logger.error(msg)
            return None

        # API rate limit warnings might be an element named "Note" or
        # "Information" depending on whether this is the first or second time
        # today we've hit the limit.
        for err_tag in "Information", "Note":
            if err_tag in data and "Our standard API rate limit" in data[err_tag]:
                logger.warning(
                    f"Exceeded rate limit fetching data for {pickle_name}, {data[err_tag]}."
                )
                at_api_limit = True
                return None

        with pickle_file_path.open(mode="wb") as pkl_fp:
            logger.debug(f"Saving results from query: {data}")
            dump(data, pkl_fp)

    return data


def load_stock_values(ticker_symbol: str) -> Dict[date, StockPrice]:
    """Load prices and dividend data for a given stock.

    Result will be a hash keyed by a date.  The date will be the first day of an
    interval, typically a month.
    """
    url = construct_url("TIME_SERIES_MONTHLY_ADJUSTED", symbol=ticker_symbol)
    data = fetch_data(url, ticker_symbol)

    # If data failed to load, just return None
    if data is None:
        return None

    if "Monthly Adjusted Time Series" not in data:
        logger.error(f"Did not fetch data for {ticker_symbol}")
        raise RuntimeError(f"Did not fetch data for {ticker_symbol}, {data}")

    #
    # Raw JSON data has a lot of fields we don't need.  What we do
    # need is the date, formatted as above, the adjusted close value
    # (keyed with "5. adjusted close"), and any dividend paid that
    # month, adjusted for inflation to be in current dollars.  Dividend
    # is keyed by "7. dividend amount"
    #
    prices = {}
    for k, v in data["Monthly Adjusted Time Series"].items():
        price_date = date_str_to_date(k)
        if price_date < start_date:
            continue

        prices[price_date] = StockPrice(price_date, v)

    # Check whether any prices were within the simulation time range.
    if len(prices) == 0:
        logger.warning(f"No sales data for {ticker_symbol} in simulation time range.")
        return None

    return prices


SimResult = namedtuple(
    "SimResult",
    [
        "symbol",
        "first_buy_date",
        "last_buy_date",
        "shares",
        "basis",
        "end_share_value",
        "dividends",
        "gain",
        "gain_pct",
    ],
    defaults=[
        None,
        None,
        None,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
)


def simulate(share_prices: dict[date, StockPrice]) -> SimResult:
    """Simulate buying or selling stock for all stocks in data set.

    All calculations done in current dollars, except computing cost basis.
    """
    # We'll use tabulate to print results from this list of lists.
    output = []

    for s in share_prices:
        if args.action == "buy":
            results = simulate_buying_stock(s, share_prices[s])
        elif args.action == "sell" and args.shares:
            results = simulate_selling_by_shares(s, share_prices[s])
        elif args.action == "sell" and args.dollars is not None:
            results = simulate_selling_constant_dollars(s, share_prices[s])
        else:
            # Can't happen because parser won't allow it.
            logger.error("Don't understand {args.action} action.")
            sys.exit(1)

        output.append(results)

    # Sort output by total gain.
    output.sort(key=lambda row: row.gain_pct, reverse=True)

    # Compute summary row if there's more than one stock.
    if len(args.symbols) > 1:
        summary_list = ["Total", start_date, end_date]
        for i in range(3, len(output[0])):
            summary_list.append(sum([row[i] for row in output]))

        # It would be nice to use named fields here but a namedtuple has to be
        # created all at once, it's immutable.
        summary_list[8] = summary_list[7] / summary_list[4] * 100
        output.append(SimResult._make(summary_list))

    return output


def simulate_buying_stock(s: str, prices: dict[date, StockPrice]) -> list(SimResult):
    """Simulate buying one stock.  Return list of results from the purchase.

    Buy one thousand current dollars of a stock each month. Adjust stock close
    price to current dollars and buy shares.  Remember how many shares we have
    so we can compute dividends.

    When the stock pays a dividend, also convert to current dollars, multiply by
    number of shares, and add to accumulated dividends.
    """
    logger.info(f"Simulating buying {s}.")

    shares = 0.0
    dividends = 0.0
    cost_basis = 0.0

    # Need first and last date we could have bought shares, which might not be
    # start_date and end_date.  We need data for both CPI and stock prices for
    # all dates.
    cpi_start, cpi_end = cpi_data.get_edge_dates()
    first_buy_date = max(cpi_start, min(prices.keys()))
    last_buy_date = min(cpi_end, max(prices.keys()))

    for buy_date in sorted(prices.keys()):
        if buy_date < cpi_start or buy_date > cpi_end:
            continue

        dividend = prices[buy_date].get_price(StockPrice.Price.DIVIDEND)
        dividends += shares * dividend

        share_price = prices[buy_date].transaction_price()

        new_shares = 1000 / share_price
        shares += new_shares

        new_basis = cpi_data.deflate(buy_date, 1000)
        cost_basis += new_basis

        logger.info(
            f"On {buy_date} bought {new_shares:,.2f} of {s} for ${new_basis:,.2f} at ${share_price:,.2f} per share.",
        )

    # TODO(psmoot): handle situation where we don't buy up to current price and
    # there's inflation or a split between the last buy date and today.
    end_share_value = shares * prices[last_buy_date].get_price(StockPrice.Price.CLOSE)

    gain = end_share_value + dividends - cost_basis

    return SimResult(
        symbol=s.upper(),
        first_buy_date=first_buy_date,
        last_buy_date=last_buy_date,
        shares=shares,
        basis=cost_basis,
        end_share_value=end_share_value,
        dividends=dividends,
        gain=gain,
        gain_pct=gain / cost_basis * 100,
    )


def simulate_selling_by_shares(s: str, prices: dict[date, StockPrice]) -> list:
    """Simulate selling one stock, selling same number of shares each time.

    Return list of results from the purchase.

    Assume we start with $100,000 in the given stock.  Compute starting shares
    and number of sell periods.  Sell an equal number of shares each period.

    When the stock pays a dividend, convert that to current dollars, multiply
    per-share dividend by number of shares we held at that point, and add to
    accumulated dividends.
    """
    dividends = 0.0
    proceeds = 0.0

    # Need first and last date we could have bought shares, which might not be
    # start_date and end_date.  We need both stock and CPI data for all dates.
    cpi_start, cpi_end = cpi_data.get_edge_dates()

    first_sell_date = max(cpi_start, min(prices.keys()))
    last_sell_date = min(cpi_end, max(prices.keys()))

    num_sales = len(prices)
    basis = 100000.0
    start_shares = basis / prices[first_sell_date].get_price(
        StockPrice.Price.ADJUSTED_CLOSE
    )
    shares_to_sell = start_shares / num_sales

    shares = start_shares
    for sell_date in sorted(prices.keys()):
        if sell_date < cpi_start or sell_date > cpi_end:
            continue

        dividend = prices[sell_date].get_price(StockPrice.Price.DIVIDEND)
        dividends += shares * dividend

        share_price = prices[sell_date].transaction_price()

        sale_proceeds = share_price * shares_to_sell
        proceeds += sale_proceeds
        shares -= shares_to_sell

        logger.info(
            f"On {sell_date} sold {shares_to_sell:,.2f} of {s} for ${sale_proceeds:,.2f} at ${share_price:,.2f} per share."
        )

    return SimResult(
        symbol=s.upper(),
        first_buy_date=first_sell_date,
        last_buy_date=last_sell_date,
        shares=start_shares,
        basis=basis,
        end_share_value=0,
        dividends=dividends,
        gain=proceeds,
        gain_pct=proceeds / basis * 100,
    )


def simulate_selling_constant_dollars(s: str, prices: dict[date, StockPrice]) -> list:
    """Simulate selling one stock, selling constant dollars.

    Return list of results from the purchase.

    Assume we start with $100,000 in the given stock.  Sell the same number of
    dollars of stock until we either run out of months or shares.
    """
    dividends = 0.0
    proceeds = 0.0

    # Need first and last date we could have bought shares, which might not be
    # start_date and end_date.  We need both stock and CPI data for all dates.
    cpi_start, cpi_end = cpi_data.get_edge_dates()

    first_sell_date = max(cpi_start, min(prices.keys()))
    last_sell_date = min(cpi_end, max(prices.keys()))

    basis = 100000.0
    start_shares = basis / prices[first_sell_date].get_price(
        StockPrice.Price.ADJUSTED_CLOSE
    )
    shares = start_shares

    for sell_date in sorted(prices.keys()):
        if sell_date < cpi_start or sell_date > cpi_end:
            continue

        dividend = prices[sell_date].get_price(StockPrice.Price.DIVIDEND)
        dividends += shares * dividend

        # Compute how many shares to sell.  If we have N periods left, sell
        # 1/Nth of our shares.
        share_price = prices[sell_date].transaction_price()
        shares_to_sell = min(shares, args.dollars / share_price)

        sale_proceeds = share_price * shares_to_sell
        proceeds += sale_proceeds
        shares -= shares_to_sell

        logger.info(
            f"On {sell_date} sold {shares_to_sell:,.2f} ({shares:,.2f} remaining) of {s} for ${sale_proceeds:,.2f} at ${share_price:,.2f} per share."
        )

        if shares <= 0.0:
            last_sell_date = sell_date
            break

    return SimResult(
        symbol=s.upper(),
        first_buy_date=first_sell_date,
        last_buy_date=last_sell_date,
        shares=start_shares,
        basis=100000.0,
        end_share_value=shares * prices[last_sell_date].transaction_price(),
        dividends=dividends,
        gain=proceeds,
        gain_pct=0,
    )


def parse_args() -> None:
    """Parse command line arguments.  Leave results in global args variable."""
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
        "--action",
        "-a",
        choices=["buy", "sell"],
        help="Action to simulate: buying with DCA, selling constant number of shares, selling constant number of dollars",
        default="buy",
    )

    parser.add_argument(
        "--shares",
        help="Sell fixed number of shares each month",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--dollars",
        help="Dollars to sell each month (assuming starting value of $100,000)",
        type=int,
    )

    parser.add_argument(
        "--skill",
        "-S",
        choices=["best", "worst", "close"],
        help="How skillful to pick prices: the best, worst, or closing price for the period",
    )

    global args
    args = parser.parse_args()

    if args.action == "sell":
        if not args.shares and args.dollars is None:
            logger.error(f"Must specify either --shares or --dollars when selling.")
            sys.exit(1)
        elif args.shares and args.dollars is not None:
            logger.error("Cannot sell both shares and dollars.")
            sys.exit(1)

    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)


def initialize_globals() -> None:
    """Set start_date and end_date variables based on --duration argument.

    Load global cpi_data variable.
    """
    # Set globals start_date and end_date, the beginning and end of simulation.
    global start_date
    start_date = date(
        year=date.today().year - args.duration, month=date.today().month, day=1
    )

    global cpi_data
    cpi_data = Inflation()
    cpi_data.load_data(start_date)

    global end_date
    end_date = date(year=date.today().year, month=date.today().month, day=1)


def main() -> None:
    """Main program for simulator."""  # noqa: D401
    parse_args()

    initialize_globals()

    share_prices = {}
    for symbol in args.symbols:
        try:
            prices = load_stock_values(symbol.upper())
            if prices is not None:
                share_prices[symbol] = prices
        except ValueError:  # noqa: PERF203
            logger.warning("Symbol {symbol} isn't a valid stock symbol, skipping.")
            continue

    if len(share_prices) == 0:
        logger.warning("Did not load any share prices.")
        return

    output = simulate(share_prices)
    print(f"At end of {args.action} simulation from {start_date} to {end_date}")
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
                "Gain (%)",
            ],
            floatfmt=",.2f",
        )
    )


if __name__ == "__main__":
    logging.basicConfig()
    main()
