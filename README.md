# MSDS-Practicum2-NLP-Disaster-Tweets
Predict which Tweets are about real disasters and which ones are not
Project Summary:

NLP is a rapid growing field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language. Twitter is one of the powerful communication medium. Twitter sentiment analysis is one of my favorite project, which I have done in this Data science masters course (MSDS600). Twitter has become an important communication channel, especially in times of emergency. One of the advantages of smartphone is, it enables the people to announce the emergency they’re observing in the real-time. Because of these the different agencies like disaster management, news agencies are continuously monitoring twitter for getting the up to date information.
The objective of this project is to Explore and analyze the data using python, libraries and create some visualizations along with the model implementation and predict whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
Data and variables:

In this particular Kaggle competition, challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. We’ll have access to a dataset of 10,000 tweets that were hand classified.
Kaggle competition link: https://www.kaggle.com/c/nlp-getting-started 
Columns
•	id - a unique identifier for each tweet
•	text - the text of the tweet
•	location - the location the tweet was sent from (may be blank)
•	keyword - a particular keyword from the tweet (may be blank)
•	target - this denotes whether a tweet is about a real disaster (1) or not (0)
Data Cleaning Methods:

When we usually deal with language modelling, or natural language processing, we are more concerned about the cleaning process in order to avoid errors or incorrect result. Some of the preprocessing task I have done for this proect is removing url, emojis. Html tags, punctuations, spell corrections, stopwords etc
Analysis Methods/ Visualizations:

For exploring the data, I have Created maps/plots using plotly library. Created tweet length plots for top and bottom 20 tweets, retrieved top 20 tweets by word count and bottom 20 tweets by word count as well. Created tweet stopword analysis plots using plotly library,  pulled out top 20 tweets by stopwords ans bottom 20 tweets by stopwords as well. Created plots for finding out the most used punctuation marks and space words.
Created ngran by using nltk library. The essential concepts in text mining is n-grams, which are a set of co-occurring or continuous sequence of n items from a sequence of large text or sentence
Machine Learning Models:

I’m implemented BERT model to do tokenization, classification and prediction with using transformers. (Bidirectional Encoder Representations from Transformers is a technique for NLP pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. Google is leveraging BERT to better understand user searches. Wikipedia). For optimization and model comparison I’m have implimented Roberta model, Facebook AI open-sourced a new deep-learning natural-language processing (NLP) model, robustly-optimized BERT approach (RoBERTa). Based on Google's BERT pre-training model, RoBERTa includes additional pre-training improvements that achieve state-of-the-art results on several benchmarks.
Results/Conclusions:

Overall accuracy depending up on the optimization, training and data cleaning you do on the data. In this project I have implemented two approaches one with BERT PyTorch and other one with RoBERTa TensorFlow. RoBERTa TensorFlow with keras is easier to build and use in real time on the other hand PyTorch give you more flexibility and options, but we need to write more python code to get the result.
References:

•	https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/
•	https://towardsdatascience.com/natural-language-processing-nlp-top-10-applications-to-know-b2c80bd428cb
•	https://www.researchgate.net/figure/Example-for-tweet-text-preprocessing_fig2_322713146
•	https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e
•	https://blog.camelot-group.com/2019/03/exploratory-data-analysis-an-important-step-in-data-science/
•	https://huggingface.co/transformers/main_classes/tokenizer.html
•	https://huggingface.co/transformers/main_classes/optimizer_schedules.html
•	https://huggingface.co/transformers/model_doc/roberta.html?fbclid=IwAR0q4NiE-b1C8v4-Eusq6QhL3heYpfZa4XJXfpy0rlZkrqn819JNE-g78h4
•	https://analyticsindiamag.com/bert-classifier-with-tensorflow-2-0/
•	https://www.infoq.com/news/2019/09/facebook-roberta-nlp/

