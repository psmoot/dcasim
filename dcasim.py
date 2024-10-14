#! python3

import requests
from json import dumps

with open("api-key.txt") as fp:
    api_key = fp.readline()

api_key.strip()


def get_url(function="TIME_SERIES_MONTHLY_ADJUSTED", **kwargs) -> str:
    """
    Construct URL.  Every URL needs a function paramter and many functions also
    require additional parameters.

    No doubt there's a library to make this safer but I can't find one.
    """
    url = f"https://www.alphavantage.co/query?function={function}&apikey={api_key}"
    for name, value in kwargs.items():
        url += f"&{name}={value}"

    return url


# Get CPI data for last 20 years
url = get_url("CPI", interval="monthly")
r = requests.get(url)
data = r.json()
print("CPI data")
print(dumps(data, indent=4))

stock_symbol = "AAPL"

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={stock_symbol}&apikey={api_key}"
url = get_url("TIME_SERIES_MONTHLY_ADJUSTED", symbol=stock_symbol)
r = requests.get(url)
data = r.json()

print(dumps(data, indent=4))
