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

# NLTK VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available (safe call)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    try:
        nltk.download('vader_lexicon')
    except Exception:
        pass

# --------------------------------------------------------------
# Load API keys
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

hf_sentiment = pipeline("sentiment-analysis")

# Prepare VADER analyzer
_vader = SentimentIntensityAnalyzer()


### 1. get recent close price data (from first part; kept)
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


### 2. check trend based on price change (from first part; kept)
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
        label = "Rising"
    elif pct_change < -5:
        label = "Falling"
    else:
        label = "Stagnant"

    return label, pct_change


# --------------------------------------------------------------
# Load free Yahoo Finance News API
def get_yahoo_news(ticker, limit=5):
    """
    Given a stock ticker (like 'AAPL' or 'TSLA'), fetch the most recent
    news headlines from Yahoo Finance. Return a list of dictionaries
    with the article title and URL.
    """

    # Unofficial Yahoo Finance search endpoint.
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"

    # AI: Yahoo sometimes blocks requests that don't look like real browsers, adding a User-Agent makes Yahoo accept the request
    headers = {"User-Agent": "Mozilla/5.0"}

    # Send the get request to Yahoo Finance and store the response as a dictionary
    r = requests.get(url, headers=headers)
    data = r.json()

    # Store Yahoo's news section, which has multiple articles, in a list
    items = data.get("news", [])

    cleaned = []

    # Loop through each returned news item
    for item in items:
        title = item.get("title")    # article headline
        link = item.get("link")      # link to the full article

        # We only keep items that have both a title and a link
        if title and link:
            cleaned.append({"title": title, "url": link})

        # Stop once we've collected the desired number of articles
        if len(cleaned) == limit:
            break

    return cleaned


# --------------------------------------------------------------
# Extract full article
def get_article_text(url):

    # Download and parse the article using the newspaper library
    try:
        article = Article(url, language='en')
        article.download()
        article.parse()

        # Reject extremely short texts, since that most likely means the extraction failed
        if len(article.text) < 50:
            return None

        # Return the extracted article text
        return article.text

    # If an error occurs, return None
    except:
        return None
    

# --------------------------------------------------------------
# Hugging Face Sentiment Analysis
def hf_score(text):
    try:
        # AI: Run the model on the first 500 characters, since there is a token limit
        result = hf_sentiment(text[:500])[0]
        label = result["label"]
        score = result["score"]

        # Convert the label into a numeric sentiment value
        numeric = score if label == "POSITIVE" else -score

        # Return both the raw label/score and the numeric score
        return {"label": label, "score": score, "numeric": numeric}

    # If the model fails, return a neutral default
    except:
        return {"label": "NEUTRAL", "score": 0.0, "numeric": 0.0}


# --------------------------------------------------------------
# OpenAI Sentiment Analysis
def extract_numeric_from_openai(text):
    """
    Extract a numeric sentiment score (between -1 and 1) from the OpenAI response.
    """

    # Look for any number inside the response text
    match = re.search(r"(-?\d+\.\d+)|(-?\d+)", text)

    # If we found a number and it falls within the score range, treat it as the score
    if match:
        num = float(match.group())
        if -1.0 <= num <= 1.0:
            return num

    # If no suitable number was found, return a neutral value
    return 0.0


def openai_score(text):
    # Build the prompt asking OpenAI to classify sentiment and provide a numeric score
    prompt = (
        "Classify the sentiment of this text as Positive, Neutral, or Negative. "
        "Also provide a numeric score between -1 and 1.\n"
        "Respond ONLY in this JSON format exactly:\n"
        "{\"sentiment\": \"Positive|Neutral|Negative\", \"score\": <number between -1 and 1>}.\n\n"
        + text
    )

    try:
        # Send the request to the OpenAI API and get the model's response text
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        raw = response.output_text.strip()

        # Try to load JSON safely
        try:
            data = json.loads(raw)
            sentiment = data.get("sentiment", "Neutral")
            score = float(data.get("score", 0.0))
        except:
            sentiment = "Neutral"
            score = 0.0

        return sentiment, score

    # If the API call fails, return a default neutral result
    except:
        return "OpenAI sentiment unavailable.", 0.0


# --------------------------------------------------------------
# NLTK VADER Sentiment Analysis on news articles (full article)
def nltk_vader_score(text):
    """
    Conducts sentiment analysis using NLTK VADER on news article text (not just title).
    Mimics the style of hf_score: returns dict with label/score/numeric.
    """
    try:
        snippet = (text or "")[:2000]
        if not snippet.strip():
            return {"label": "NEUTRAL", "score": 0.0, "numeric": 0.0}

        s = _vader.polarity_scores(snippet)
        compound = s.get("compound", 0.0)

        if compound > 0.2:
            label = "POSITIVE"
        elif compound < -0.2:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {"label": label, "score": abs(compound), "numeric": compound}
    except Exception:
        return {"label": "NEUTRAL", "score": 0.0, "numeric": 0.0}


# --------------------------------------------------------------
# Run Sentiment Analysis on Stock Articles
def analyze_stock(ticker):
    # Retrieve recent Yahoo Finance news items for the given ticker
    articles = get_yahoo_news(ticker)

    # If no articles were found, return an empty list
    if not articles:
        print("No news found for this stock.")
        return []

    results = []

    # Loop through each news article returned by Yahoo
    for item in articles:
        title = item["title"]
        url = item["url"]

        # Try to extract the full article text, use title if needed
        text = get_article_text(url)
        if not text:
            print("Could not extract full article text. Using title only.\n")
            text = title

        # Run three sentiment systems on the text
        hf_result = hf_score(text)
        openai_raw, openai_num = openai_score(text)
        nltk_result = nltk_vader_score(text)

        # Store all results for this article in a dictionary
        results.append({
            "title": title,
            "url": url,
            "hf": hf_result,
            "openai_sentiment": openai_raw,
            "openai_num": openai_num,
            "nltk_vader": nltk_result
        })

    # Return the list of sentiment results
    return results


# --------------------------------------------------------------
# Compute Average Sentiment ( average across 3 models)
def compute_overall_sentiment(results):
    # Sum the numeric sentiment scores from Hugging Face
    hf_sum = sum(r["hf"]["numeric"] for r in results)

    # Sum the numeric sentiment scores from OpenAI
    ai_sum = sum(r["openai_num"] for r in results)

    # Sum the numeric sentiment scores from NLTK VADER
    nv_sum = sum(r["nltk_vader"]["numeric"] for r in results)

    # Average the three models' totals across all articles
    # Avoid division by zero; results is guaranteed non-empty here by caller checks
    avg = (hf_sum + ai_sum + nv_sum) / (3 * len(results))

    # Decide the overall label based on the averaged score
    if avg > 0.2:
        label = "Positive"
    elif avg < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    # Return both the label and the numeric score
    return label, avg


# --------------------------------------------------------------
# Extract Stock Name
def get_stock_name(ticker):
    # Query Yahoo Finance search endpoint for metadata
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers).json()

    # Yahoo puts company info under "quotes"
    quotes = r.get("quotes", [])

    # Return the ticker if no name found
    if not quotes:
        return ticker

    # Return the short company name if available
    return quotes[0].get("shortname", ticker)


# --------------------------------------------------------------
# Main program
def main():
    # Ask the user for a stock ticker and standardize formatting
    ticker = input("Enter a stock symbol (e.g., AAPL, TSLA): ").upper().strip()

    # 1) Price trend from close prices
    close_prices = get_close_prices(ticker, months=3)
    trend_label, pct_change = classify_trend(close_prices)

    # 2) Fetch news and run all sentiment analysis steps
    results = analyze_stock(ticker)

    # Print stock name and ticker
    stock_name = get_stock_name(ticker)
    print("---------------------------------------")
    print(f"Stock: {stock_name} ({ticker})")
    print(f"Price trend (3 months): {trend_label} ({pct_change:.2f}%)")
    print("---------------------------------------")

    # Print each article’s individual sentiment results
    for entry in results:
        print("Title:", entry["title"])
        print("URL:", entry["url"])

        # Hugging Face
        hf_label = entry["hf"]["label"].capitalize()
        hf_num = round(entry["hf"]["numeric"], 3)
        print(f"Hugging Face Sentiment: {hf_label} ({hf_num})")

        # OpenAI
        ai_raw = entry["openai_sentiment"]
        ai_num = round(entry["openai_num"], 3)
        print(f"OpenAI Sentiment: {ai_raw} ({ai_num})")

        # NLTK VADER on article text
        nv = entry["nltk_vader"]
        nv_label = nv["label"].capitalize()
        nv_num = round(nv["numeric"], 3)
        print(f"NLTK VADER: {nv_label} ({nv_num})")

        print("---------------------------------------")

    # Compute the combined final sentiment across all articles (3-model average)
    if results:
        label, avg = compute_overall_sentiment(results)
        print(f"Overall Sentiment (HF + OpenAI + NLTK) for {ticker}: {label} ({avg:.3f})")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()