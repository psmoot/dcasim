Simulate buying stock using dollar cost averaging (DCA).

We get historical stock prices from https://alphavantage.co.  They have daily,
weekly, and monthly "adjusted" stock prices.  These prices reflect splits and
discount old prices based on dividends (that is, if the company pays a dividend,
its older stock prices gets reduced by the same amount).

Alphavantage also has Consumer Price Index (CPI) data.  We want to adjust for
inflation.

The simulation assumes you start 10 years ago and buy one dollar of any given
stock once a month. The amount you buy adjusts up for inflation.  At the end,
you'll have a bunch of shares and some cash dividends (also adjusted for
inflation plus a 1% rate of return).

Eventually we'll want a corresponding sell-side simulation.  My theory is that
to dollar cost average on the sell side, one should sell an equal number of
shares every month, not an equal number of dollars.

To run this program, get an API key from Alphavantage and put in a file named
"api-key.txt" in the root directory of this project.  The key "demo" seems to
work because that's what all the demo scripts use.