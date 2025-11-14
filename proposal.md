# Project Proposal

## The Big Idea

Our project focuses on combining stock trends with news sentiment to give users a quick picture of how a company is doing. The output will show users two key pieces of information: stock price trend and community sentiment.  

First, we will use the Yahoo Finance API to get the closing prices of each stock from the last 3 months to see if the price is rising, falling, or stagnant. Then, we will use three sentiment models on recent news articles on the stocks to determine the community’s sentiment on the stocks. Using this, the program will generate a short summary for each stock, such as: “This stock is currently rising. Community sentiment is positive, which may suggest a good short-term outlook.” 


Minimum Viable Product: 

- Let the user enter one or more stock tickers on a web app.
- Classify each stock’s trend (rising / falling / stagnant).
- Calculate an average sentiment score for the stocks 
- Display a one-to-two sentence summary. 

## Learning Objectives

Our main goal is to learn how to use Python for real-world problem solving. We want to learn how to work with data from online sources and apply concepts from class in a practical way. This includes designing and building a tool that is easy for users to understand and interact with. In particular, we want more experience collecting information using APIs and developing a website using Python. 

## Implementation Plan 

We will use the Yahoo Finance API with the yfinance library to get stock price data and news articles. We plan to use pandas so we can use a data frame to organize the data. For example, calculating changes in stock price, and matching each news article to the correct stock. 

For sentiment analysis, we will use three tools: NLTK VADER, a Hugging Face model, and the OpenAI API. Since each tool measures tone differently, they may yield varying outcomes. By comparing the three and taking the average scores, we can get a more balanced view of overall sentiment. 

To see how the stocks are performing, we will use their current and historical prices and assess if there is a rising, falling, or stagnant trend over a certain period of time. Then, we will make a short summary for each stock, including its current condition and community sentiment. 

Our project will be presented as a web app, where the user can input the stocks they wish to analyze. This will be done using Flask. If time allows, we also plan to add an LLM bot that can respond to user questions about the stocks. 

## Project Schedule
#### 1. Phase One: Stock trend Analysis and Sentiment Analysis
   - 11/22: Draft check
   - 11/23: Finish

#### 2. Phase Two: Synthesize Work and Create Final Product
   - 11/24: Meet to calculate average sentiment scores
   - 12/01: Finish/meet to combine stock trend and sentiment analysis into presentable web app 

#### 3. Phrase Three: Additional Features
  - 12/01 to 12/05: Explore other add-ons, such as an LLM question bot or displaying the stock price trends in a graph
  - Implement them if they are functional 

## Collaboration Plan 

#### Independent tasks: 

- Grace gathers and analyzes stock price info + conducts sentiment analysis using NLTK model

- Tauria conducts sentiment analysis using Hugging Face + OpenAI models

#### Collaborative tasks: 

- Calculate average sentiment 
- Combine individual work to make summary output
- Create web app

We decided to split the tasks like this so we can work independently, and one person’s work is not dependent on the other person finishing their part. This is also more efficient than only working together, since we have different availabilities. However, we’ll are also planning to practice our teamwork by working together during the last week to assemble our final product. 

## Risks and Limitations: 

Some tickers don’t have many recent articles on Yahoo Finance, and the API can change or get rate-limited. 

- We will cache data and show when news is missing. 

Different tools give different results. VADER, Hugging Face, and OpenAI may disagree because they use different methods. 

- We’ll compare them and use an average of their scores in our analysis. 

Sentiment does not always match stock movement, so a positive tone doesn’t necessarily mean the price will rise. 

- We will state that our results are descriptions, not predictions. 

## Additional Course Content 

We would benefit from more in-depth lessons on:
- Working with APIs 
- Web app development 
- Creating data visualizations 