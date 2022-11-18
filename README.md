Tracks and analyzes inflation sentiment on twitter using machine learning. [Work in progress].

Objectives of the project:
- to detect and extract tweets about inflation with sncrape library
- to analyze inflation sentiment using Naive Bayes classifier

Prerequisites:
- Python 3.8.1
- Pandas
- snscrape

Files:
- twitter_scraping.py: detects and saves inflation related tweets into Pandas DataFrame.
- two_classes.py: Naive Bayes classifier which uses two classes: negative and positive. Model only for comparison.
- three_classes.py: NBC which uses three classes: negative, positive and neutral.
- three_classes_para.py: includes min_counts parameter
- dataset: Sentiment140 (http://help.sentiment140.com/for-students)
