# NLP Analysis into what makes a Yelp Restaurant Review Useful

The goal of this project is to gain insight into what makes an online review useful. Given that there are many different styles of writing, we're going to try and extract the commonalities that the most useful reviews share in hopes of eventually creating a model that predicts how useful a review will be. I deployed a web application containing my model that can be accessed here:

[ec2-54-244-57-203.us-west-2.compute.amazonaws.com]

### Tools used: Python, HTML, CSS, Docker, AWS EC2, sklearn, matplotlib/seaborn, flask, numpy, pandas, nltk, spacy, statsmodels, linear regression, randomforest, PCA, VIF

# File / Notebook Summaries

# 01 - Data Collection and Extraction

In this I read in, clean, and filter the data. The Reviews date all the way back to 2012, and there may be some confounding factors between how many useful votes a review has gotten and how old it is, so I filter out older reviews. Additionally, since the end goal is an interface that will tell how useful is, and obviously that conflicts with time so ideally I want to tell how useful a review will be over a fixed period of time. Restaurant reviews are the largest subset of the data, so I narrow our data to reviews that fall into the restaurant category. The final dataframe consisting of reviews contains just over 3 million rows.

# 02 - Exploratory Data Analysis

First and foremost, I create a couple of different dependent variables to decide which is the best to use for our predicted variable. Between plain old useful votes, log useful, and binary labeling (useful vs not useful), log useful proves to show the best shape of distributions, account for the many outliers, and correlate highest with our NLP features later on. 

Next, I used the readability library from pypi to generate 35 NLP features whose categories are readability metrics, sentence info, word usage, and sentence beginnings. The most pertinent features prove to be those related to the sentence info category, namely the number of characters, sentences, paragraphs, word types, and the number of unique words used in a review divided by the total number of words used in a review (type_token_ratio). Using our previously created categories for the target variable (not useful, useful, very useful), I used visualization techniques to see that there is a huge difference in mean character count for these categories. More specifically, reviews marked as not useful had a mean character count of ~350, useful had a mean of ~500, and very useful had a character count of ~850. From this we can understand that the longer a review is, the more likely it is that it's useful, which makes a lot of sense intuitively. 

I also use NLTK's vader sentiment analysis in order to extrapolate user attitude from each review. From this I was able to realize that in general, more useful reviews tend to be more negative. The reasoning behind this can be couple with the previous finding that longer reviews are additionally more useful, as longer reviews tend to be more negative corresponding to a user venting about their bad experience. When a user has a positive experience, they are much more likely to leave a short, thoughtless review that doesn't contain much information. I also found that there is a high positive correlation between the number of different words (wordtype) a user uses and how useful their review is. 

Finally, In order to prepare for the next notebook, the ML prototype, I implemented PCA for dimensionality reduction and analyze the features that it utilizes to get a better understanding of what the most telling features are. PCA was able to use 5 features to capture 80% of the variance from the original 40 features. I used visualization through scatterplots, barplots, and histograms to recognize what the first two PCA features were using from the original dataset to capture the variance. The first principal component exploits the sentiment columns I created, including number of stars, positive, negative, neutral, and compound (pos + neg + neu) ratings of a review. The second principal component focused on the technical NLP features including type token ratio, number of characters, number of wordtypes, and number of syllables in a given review, among many others. 

In summary, I was able to extract and engineer many different aspects of what makes a review useful. The ideal useful review looks something like a longer, critical review that has a dense vocabulary. 

# 03 - Machine Learning Prototype

In notebook, the goal was to create several models to predict approximately how useful a review will be. Because log of the number of useful votes (log_useful) had the highest correlation with our engineered NLP features, I decided that this would be the best target variable to predict. Unfortunately, since this makes it a regression problem, the interpretability of our results goes down (as opposed to if it were classification). Nonetheless, I still modeled the data with several variants of the original feature matrix and interpretted the results visually. 

I used 6-fold cross validation, along with PCA feature selection to determine the best matrix for predicability. In order to set up PCA, I initialized a baseline linear regression model using two simple features that I engineered in the previous notebook which provided valuable insight. From there, I normalized the data and ran linear regression on the entire matrix, and examined the weights of the coefficients for each feature to gain insight into how each feature played a role in the final prediction. 

Since the goal of PCA is to extract your features containing the most variance, it is an important step to remove features/columns with high multicollinearity. You may have features with high variance, but if they're describing the same thing, then PCA may hone in on these but ultimately won't tell you very much and you'll end up losing information in the process. In order to account for this, I calculated the variance inflation factor (VIF) for each feature. Essentially, this builds off of linear regression by quantifying the severity of multicollinearity conducted in regression analysis. I dropped 10 columnns, or about 1/4 of the original matrix, and from there ran my cross validation, where I iteratively started with just one principal component for the PCA matrix, and built all the way up to 20, where the training error and subsequent validation error bottomed out around. 

Finally, I was left with three main data matrices, the original (40 features), the VIF reduced matrix (28 features), and the PCA matrix (20 features). 

From there I moved onto a more advanced, non-linear tree-based model of random forest regression. After gridsearch hyperparameter tuning and optimization, I ended up with six different models in total, three linear regression and three random forest, where for each of the two subsets, one model is trained on on of the three data matrices. Random forest allowed me to visualize which features were most pertinent for each data matrix.

In order to evaluate performance from each model, I looked at the $r^2$ coefficient, RMSE training and test error, and residual plots. In general, the results seemed to suggest that the more complex the model, the better its performance. 


# endtoend.py / tests.py

Production style code conglomeration of the three notebooks. This code runs independently from the command line and is well tested from tests.py, which tests every function in endtoend.py, and have several test cases for each function, covering edge cases. There is useful logging and documentation at critical points throughout the file, as well as each function having a docstring description. This file represents the entire ETL end-to-end pipeline.

# App.py / templates / static / Dockerfile

This series of files represent the back and front end of the web application. I pickled my transformations and models and used flask to create a simple user interface for my API. templates and static contain html code for the front end interaction and display of the user interface. Then, in order to allow anyone else to spin up my application without installing the right tools or libraries, I containerized my flask application with Docker. Finally, I hosted the docker container on an AWS ec2 instance for the public to use. 
