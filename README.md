# NLP Analysis into what makes a Yelp Restaurant Review Useful

The goal of this project is to gain insight into what makes a yelp restaurant review useful based off of the text.
Given that there are many different flavors of useful reviews, we're going to try and extract the commonalities that
fruitful reviews share in hopes of eventually creating a model that predicts whether or not a user's review will be useful.

# Notebook Summaries

# 01 - Data Collection and Extraction

In this I read in, clean, and filter the data. The Reviews date all the way back to 2012, and there may be some confounding factors between how many useful votes a review has gotten and how old it is, so I filter out older reviews. Additionally, since the end goal is an interface that will tell how useful is, and obviously that conflicts with time so ideally I want to tell how useful a review will be over a fixed period of time. Restaurant reviews are the largest subset of the data, so I narrow our data to reviews that fall into the restaurant category. The final dataframe consisting of reviews contains just over 3 million reviews.

# 02 - Exploratory Data Analysis

First and foremost, I create a couple of different dependent variables to decide which is the best to use for our predicted variable. Between plain old useful votes, log useful, and binary labeling (useful vs not useful), log useful proves to show the best shape of distributions, account for the many outliers, and correlate highest with our NLP features later on. 

Next, I used the readability library from pypi to generate 35 NLP features whose categories are readability metrics, sentence info, word usage, and sentence beginnings. The most pertinent features prove to be those related to the sentence info category, namely the number of characters, sentences, paragraphs, word types, and the number of unique words used in a review divided by the total number of words used in a review (type_token_ratio). Using our previously created categories for the target variable (not useful, useful, very useful), I used visualization techniques to see that there is a huge difference in mean character count for these categories. More specifically, reviews marked as not useful had a mean character count of ~350, useful had a mean of ~500, and very useful had a character count of ~850. From this we can understand that the longer a review is, the more likely it is that it's useful, which makes a lot of sense intuitively. 

I also use NLTK's vader sentiment analysis in order to extrapolate user attitude from each review. From this I was able to realize that in general, more useful reviews tend to be more negative. The reasoning behind this can be couple with the previous finding that longer reviews are additionally more useful, as longer reviews tend to be more negative corresponding to a user venting about their bad experience. When a user has a positive experience, they are much more likely to leave a short, thoughtless review that doesn't contain much information. I also found that there is a high positive correlation between the number of different words (wordtype) a user uses and how useful their review is. 

Finally, In order to prepare for the next notebook, the ML prototype, I implemented PCA for dimensionality reduction and analyze the features that it utilizes to get a better understanding of what the most telling features are. PCA was able to use 5 features to capture 80% of the variance from the original 40 features. I used visualization through scatterplots, barplots, and histograms to recognize what the first two PCA features were using from the original dataset to capture the variance. The first principal component exploits the sentiment columns I created, including number of stars, positive, negative, neutral, and compound (pos + neg + neu) ratings of a review. The second principal component focused on the technical NLP features including type token ratio, number of characters, number of wordtypes, and number of syllables in a given review, among many others. 

In summary, I was able to extract and engineer many different aspects of what makes a review useful. The ideal useful review looks something like a longer, critical review that has a dense vocabulary. 

# 03 - Machine Learning Prototype

In notebook, the goal was to create several models to predict approximately how useful a review will be. Because log of the number of useful votes (log_useful) had the highest correlation with our engineered NLP features, I decided that this would be the best target variable to predict. Unfortunately, since this makes it a regression problem, the interpretability of our results goes down (as opposed to if it were classification). Nonetheless, I was still able to create several successful models and interpret the results. 

I used 6-fold cross validation, along with PCA feature selection to determine the best matrix for predicability. The models that I used were linear regression as a baseline, and eventually a more complex random forest regressor. The most important features for our model proved to be those discovered in the EDA section of this project, namely the sentence info characters. For a more in depth analysis of the model and its utility take a look at the notebook, where I use many different metrics to evaluate its effectiveness and give a brief explanation of most of these metrics if you're not familiar. 


