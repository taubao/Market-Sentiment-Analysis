import datetime as dt
import yfinance as yf
import requests
from nltk.sentiment import SentimentIntensityAnalyzer


### 1. get recent close price data
def get_close_prices(ticker, months=3):
    """
    get close price from yahoo finance using yfinance
    return only close price
    """
    end_date = dt.date.today()
    # use 30 days * month for simple estimate
    start_date = end_date - dt.timedelta(days=months * 30)

    # download price data
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    # if no data, return none
    if data.empty:
        return None

    # return only close price column
    return data["Close"]


### 2. check trend based on price change
def classify_trend(close_series):
    """
    look at first price and last price
    check if rising / falling / flat
    and return percent change
    """
    if close_series is None or len(close_series) < 2:
        return "unknown", 0.0

    first_price = float(close_series.iloc[0])
    last_price = float(close_series.iloc[-1])

    # percent change
    pct_change = (last_price - first_price) / first_price * 100

    # simple rule to decide trend
    if pct_change > 5:
        label = "rising"
    elif pct_change < -5:
        label = "falling"
    else:
        label = "stagnant"

    return label, pct_change


### 3. get news titles from yahoo finance
def get_yahoo_news_titles(ticker, limit=5):
    """
    get news titles from yahoo finance search api
    return a list of titles
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    data = r.json()

    items = data.get("news", [])
    titles = []

    # normal for loop to get titles
    for item in items:
        title = item.get("title")
        if title:
            titles.append(title)

        # stop when hit the limit
        if len(titles) == limit:
            break

    return titles


### 4. run vader sentiment on news titles
sia = SentimentIntensityAnalyzer()

def vader_score_for_titles(titles):
    """
    take a list of titles
    return average sentiment score and label
    """
    if not titles:
        return 0.0, "Neutral"

    scores = []

    # get score for each title
    for title in titles:
        s = sia.polarity_scores(title)
        scores.append(s["compound"])

    # simple for loop to add up scores
    total = 0.0
    for i in range(len(scores)):
        total += scores[i]

    avg_score = total / len(scores)

    # simple rules
    if avg_score > 0.2:
        label = "Positive"
    elif avg_score < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return avg_score, label


### 5. run full analysis for one stock
def analyze_one_stock(ticker):
    """
    get 3-month price trend
    get news sentiment
    return both in a dict
    """
    ticker = ticker.upper().strip()

    # price trend check
    close_prices = get_close_prices(ticker, months=3)
    trend_label, pct_change = classify_trend(close_prices)

    # news sentiment check
    titles = get_yahoo_news_titles(ticker, limit=5)
    avg_sent, sent_label = vader_score_for_titles(titles)

    result = {
        "ticker": ticker,
        "trend_label": trend_label,
        "pct_change": pct_change,
        "avg_sentiment": avg_sent,
        "sentiment_label": sent_label,
        "titles": titles,
    }

    return result


### 6. main program to handle many tickers
def main():
    print("welcome to grace's stock analyzer")
    user_input = input("enter stock symbols separated by commas (example: aapl, tsla): ")

    # split user input
    raw_tickers = user_input.split(",")

    # clean spaces and ignore empty
    tickers = []
    for t in raw_tickers:
        symbol = t.strip().upper()
        if symbol:
            tickers.append(symbol)

    all_results = []

    # run one by one
    for ticker in tickers:
        print("\n-----------------------------------")
        print(f"analyzing {ticker} ...")
        stock_result = analyze_one_stock(ticker)
        all_results.append(stock_result)

        trend = stock_result["trend_label"]
        pct = stock_result["pct_change"]
        sent_label = stock_result["sentiment_label"]
        sent_score = stock_result["avg_sentiment"]

        print(f"price trend (3 months): {trend} ({pct:.2f}% change)")
        print(f"news mood: {sent_label} (avg score = {sent_score:.3f})")

        # short summary
        if trend == "rising":
            t_msg = "price is going up"
        elif trend == "falling":
            t_msg = "price is going down"
        elif trend == "stagnant":
            t_msg = "price is mostly flat"
        else:
            t_msg = "not enough price data"

        if sent_label == "Positive":
            s_msg = "news mood is good"
        elif sent_label == "Negative":
            s_msg = "news mood is bad"
        else:
            s_msg = "news mood is neutral"

        print("\nsummary:")
        print(t_msg, s_msg)

    return all_results


if __name__ == "__main__":
    main()

