from flask import Flask, render_template, request
from analysis import (
    get_close_prices,
    classify_trend,
    analyze_stock,
    get_stock_name,
    compute_overall_sentiment,
)

app = Flask(__name__)


@app.route("/")
def index():
    """
    Home page with a simple form where the user can enter a ticker.
    """
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Handle the form submission, run the analysis, and show results.
    """
    ticker = request.form.get("ticker", "").upper().strip()

    if not ticker:
        return render_template("error.html", message="Please enter a stock symbol.")

    # 1) Price trend
    close_prices = get_close_prices(ticker, months=3)
    trend_label, pct_change = classify_trend(close_prices)

    # 2) News + sentiment
    results = analyze_stock(ticker)

    # If no news, show a friendly message
    if not results:
        stock_name = get_stock_name(ticker)
        return render_template(
            "result.html",
            ticker=ticker,
            stock_name=stock_name,
            trend_label=trend_label,
            pct_change=pct_change,
            articles=[],
            overall_label="Neutral",
            overall_score=0.0,
            no_news=True,
        )

    # 3) Overall sentiment
    overall_label, overall_score = compute_overall_sentiment(results)

    stock_name = get_stock_name(ticker)

    return render_template(
        "result.html",
        ticker=ticker,
        stock_name=stock_name,
        trend_label=trend_label,
        pct_change=pct_change,
        articles=results,
        overall_label=overall_label,
        overall_score=overall_score,
        no_news=False,
    )


if __name__ == "__main__":
    app.run(debug=True)
