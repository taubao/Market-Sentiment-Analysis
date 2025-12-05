import os
import re
import json
import datetime as dt
import requests
import yfinance as yf
from newspaper import Article
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
    nltk.data.find("sentiment/vader_lexicon.zip")


# --------------------------------------------------------------
# Load API keys
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# HuggingFace pipeline for sentiment
hf_sentiment = pipeline("sentiment-analysis")

# Prepare VADER analyzer
_vader = SentimentIntensityAnalyzer()


# --------------------------------------------------------------
# Price helpers
def get_close_prices(ticker, months=3):
    """
    Get recent close prices from Yahoo Finance using yfinance.

    Returns a pandas Series of close prices or None if data is empty.
    """
    end_date = dt.date.today()
    # Simple estimate: 30 days per month
    start_date = end_date - dt.timedelta(days=months * 30)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
    )

    if data.empty:
        return None

    return data["Close"]


def classify_trend(close_series):
    """
    Look at first and last price.
    Decide if the stock is Rising / Falling / Stagnant.
    Returns (label, percent_change).
    """
    if close_series is None or len(close_series) < 2:
        return "Unknown", 0.0

    first_price = float(close_series.iloc[0])
    last_price = float(close_series.iloc[-1])

    pct_change = (last_price - first_price) / first_price * 100

    if pct_change > 5:
        label = "Rising"
    elif pct_change < -5:
        label = "Falling"
    else:
        label = "Stagnant"

    return label, pct_change


# --------------------------------------------------------------
# Yahoo Finance news
def get_yahoo_news(ticker, limit=5):
    """
    Given a stock ticker (like 'AAPL' or 'TSLA'), fetch the most recent
    news headlines from Yahoo Finance. Returns a list of dicts
    with keys 'title' and 'url'.
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"

    # Yahoo sometimes blocks non-browser requests; User-Agent helps
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    data = r.json()

    items = data.get("news", [])
    cleaned = []

    for item in items:
        title = item.get("title")
        link = item.get("link")

        if title and link:
            cleaned.append({"title": title, "url": link})

        if len(cleaned) == limit:
            break

    return cleaned


# --------------------------------------------------------------
# Article text extraction
def get_article_text(url):
    """
    Try to download and parse an article given its URL.
    Returns the article text or None if extraction fails.
    """
    try:
        article = Article(url, language="en")
        article.download()
        article.parse()

        if len(article.text) < 50:
            return None

        return article.text
    except Exception:
        return None


# --------------------------------------------------------------
# Hugging Face sentiment
def hf_score(text, chunk_size=400):
    """
    Run HuggingFace sentiment on the full text by splitting into chunks.

    Returns a dict:
        {
            "label": "Positive" | "Negative" | "Neutral",
            "score": abs(average_numeric),
            "numeric": average_numeric  # between -1 and 1
        }
    """
    try:
        if not text:
            return {"label": "Neutral", "score": 0.0, "numeric": 0.0}

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        numeric_scores = []

        for ch in chunks:
            result = hf_sentiment(ch)[0]
            label = result["label"]
            score = result["score"]

            numeric = score if label == "Positive" else -score
            numeric_scores.append(numeric)

        avg_numeric = sum(numeric_scores) / len(numeric_scores)

        if avg_numeric > 0.2:
            final_label = "Positive"
        elif avg_numeric < -0.2:
            final_label = "Negative"
        else:
            final_label = "Neutral"

        return {
            "label": final_label,
            "score": abs(avg_numeric),
            "numeric": avg_numeric,
        }
    except Exception:
        return {"label": "Neutral", "score": 0.0, "numeric": 0.0}


# --------------------------------------------------------------
# OpenAI sentiment
def openai_score(text):
    """
    Use OpenAI to classify sentiment and return (sentiment_label, numeric_score).

    Score is between -1 and 1.
    """
    prompt = (
        "You are a financial news sentiment classifier. "
        "Classify the sentiment of this text as Positive, Neutral, or Negative. "
        "Give a numeric score between -1 and 1, where -1 is very negative and 1 is very positive. "
        "Respond ONLY in valid JSON with keys 'sentiment' and 'score'."
    )

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    # keep under a safe size
                    "content": text[:4000] if text else "",
                },
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.output_text)
        sentiment = data.get("sentiment", "Neutral")
        score = float(data.get("score", 0.0))

        # Hard clamp just in case
        if score > 1:
            score = 1.0
        if score < -1:
            score = -1.0

        return sentiment, score

    except Exception:
        return "Neutral", 0.0


# --------------------------------------------------------------
# NLTK VADER sentiment
def nltk_vader_score(text):
    """
    Run NLTK VADER on article text.
    Returns dict with label, score (abs), and numeric compound value.
    """
    try:
        snippet = text or ""
        if not snippet.strip():
            return {"label": "Neutral", "score": 0.0, "numeric": 0.0}

        scores = _vader.polarity_scores(snippet)
        compound = scores.get("compound", 0.0)

        if compound > 0.2:
            label = "Positive"
        elif compound < -0.2:
            label = "Negative"
        else:
            label = "Neutral"

        return {"label": label, "score": abs(compound), "numeric": compound}
    except Exception:
        return {"label": "Neutral", "score": 0.0, "numeric": 0.0}


# --------------------------------------------------------------
# Combine model scores for a single article
def combine_article_sentiment(hf, openai_num, vader):
    """
    Combine HF, OpenAI, and VADER numeric scores for one article.
    Returns (label, average_score).
    """
    avg = (hf["numeric"] + openai_num + vader["numeric"]) / 3

    if avg > 0.2:
        label = "Positive"
    elif avg < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return label, avg


# --------------------------------------------------------------
# Analyze news for a stock
def analyze_stock(ticker):
    """
    Retrieve news for a ticker and run three sentiment models on each article.

    Returns a list of dicts, each containing:
        title, url, hf, openai_sentiment, openai_num,
        nltk_vader, combined_label, combined_score
    """
    articles = get_yahoo_news(ticker)

    if not articles:
        print("No news found for this stock.")
        return []

    results = []

    for item in articles:
        title = item["title"]
        url = item["url"]

        text = get_article_text(url)
        if not text:
            print("Could not extract full article text. Using title only.")
            text = title

        hf_result = hf_score(text)
        openai_raw, openai_num = openai_score(text)
        nltk_result = nltk_vader_score(text)

        combined_label, combined_num = combine_article_sentiment(
            hf_result, openai_num, nltk_result
        )

        results.append(
            {
                "title": title,
                "url": url,
                "hf": hf_result,
                "openai_sentiment": openai_raw,
                "openai_num": openai_num,
                "nltk_vader": nltk_result,
                "combined_label": combined_label,
                "combined_score": combined_num,
            }
        )

    return results


# --------------------------------------------------------------
# Overall sentiment across all articles
def compute_overall_sentiment(results):
    """
    Average HF + OpenAI + VADER scores across all articles.

    Returns (label, average_score).
    """
    if not results:
        return "Neutral", 0.0

    hf_sum = sum(r["hf"]["numeric"] for r in results)
    ai_sum = sum(r["openai_num"] for r in results)
    nv_sum = sum(r["nltk_vader"]["numeric"] for r in results)

    avg = (hf_sum + ai_sum + nv_sum) / (3 * len(results))

    if avg > 0.2:
        label = "Positive"
    elif avg < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return label, avg


# --------------------------------------------------------------
# Stock metadata
def get_stock_name(ticker):
    """
    Look up a human readable company name from Yahoo Finance.
    Falls back to the ticker if nothing is found.
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers).json()
    quotes = r.get("quotes", [])

    if not quotes:
        return ticker

    return quotes[0].get("shortname", ticker)


# --------------------------------------------------------------
# Main
def main():
    """
    Simple command line entry point for testing without Flask.
    """
    ticker = input("Enter a stock symbol (e.g., AAPL, TSLA): ").upper().strip()

    close_prices = get_close_prices(ticker, months=3)
    trend_label, pct_change = classify_trend(close_prices)

    results = analyze_stock(ticker)
    stock_name = get_stock_name(ticker)

    print("---------------------------------------")
    print(f"Stock: {stock_name} ({ticker})")
    print(f"Price trend (3 months): {trend_label} ({pct_change:.2f}%)")
    print("---------------------------------------")

    for entry in results:
        print("Title:", entry["title"])
        print("URL:", entry["url"])

        hf_label = entry["hf"]["label"].capitalize()
        hf_num = round(entry["hf"]["numeric"], 3)
        print(f"HuggingFace: {hf_label} ({hf_num})")

        ai_raw = entry["openai_sentiment"]
        ai_num = round(entry["openai_num"], 3)
        print(f"OpenAI: {ai_raw} ({ai_num})")

        nv = entry["nltk_vader"]
        nv_label = nv["label"].capitalize()
        nv_num = round(nv["numeric"], 3)
        print(f"NLTK VADER: {nv_label} ({nv_num})")

        combined = entry["combined_label"]
        combined_score = round(entry["combined_score"], 3)
        print(f"Combined: {combined} ({combined_score})")

        print("---------------------------------------")

    if results:
        label, avg = compute_overall_sentiment(results)
        print(f"Overall sentiment for {ticker}: {label} ({avg:.3f})")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
