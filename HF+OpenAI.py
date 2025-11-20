import os
import requests
import re
import json
from newspaper import Article
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------------------
# Load API keys
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

hf_sentiment = pipeline("sentiment-analysis")


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

        # Run both sentiment systems on the text
        hf_result = hf_score(text)
        openai_raw, openai_num = openai_score(text)

        # Store all results for this article in a dictionary
        results.append({
            "title": title,
            "url": url,
            "text_sample": text[:250],
            "hf": hf_result,
            "openai_sentiment": openai_raw,
            "openai_num": openai_num,
        })

    # Return the list of sentiment results
    return results


# --------------------------------------------------------------
# Compute Average Sentiment
def compute_overall_sentiment(results):
    # Sum the numeric sentiment scores from Hugging Face
    hf_sum = sum(r["hf"]["numeric"] for r in results)

    # Sum the numeric sentiment scores from OpenAI
    ai_sum = sum(r["openai_num"] for r in results)

    # Average the two models' totals across all articles
    avg = (hf_sum + ai_sum) / (2 * len(results))

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

    # Fetch news and run all sentiment analysis steps
    results = analyze_stock(ticker)

    # Print stock name and ticker
    stock_name = get_stock_name(ticker)
    print("---------------------------------------")
    print(f"Stock: {stock_name} ({ticker})")
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

        print("---------------------------------------")

    # Compute the combined final sentiment across all articles
    label, avg = compute_overall_sentiment(results)

    # Print the final overall sentiment summary
    print(f"Overall Sentiment for {ticker}: {label} ({avg:.3f})")


if __name__ == "__main__":
    main()
