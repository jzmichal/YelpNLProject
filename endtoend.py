import pandas as pd
import re
import spacy
import nltk
import readability
import pickle
import syntok.segmenter as segmenter
import numpy as np
from spacy_cld import LanguageDetector
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from spacy.lang.en import English
from pandas.io.json import json_normalize
from sklearn.ensemble import RandomForestRegressor

#Read in Reviews and write to a dataframe
reviews=[]
review_reader = pd.read_json('yelp_dataset/review.json', lines=True, chunksize=100000)
most_recent = pd.Timestamp('2018-11-11 20:55:31')
begin = True
for chunk in review_reader:
    chunk = chunk[(chunk.useful >= 0) & (chunk.useful < 200)]
    chunk = chunk[chunk["text"].apply(lambda t: len(t) > 5)]
    chunk.dropna(inplace = True)
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['year'] = pd.DatetimeIndex(chunk['date']).year
    chunk['nb_days'] = chunk.date.apply(lambda d: (most_recent - d) / np.timedelta64(1, 'D'))
    chunk['nb_days'] = chunk['nb_days'].astype(int)
    chunk = chunk[chunk.nb_days >= 30]
    chunk = chunk[chunk.year >= 2014]
    chunk["log_useful"] = np.log(chunk.useful+1)
    chunk.stars = chunk.stars.astype(int)
    chunk.text = chunk.text.astype(str)
    reviews.append(chunk)

review_df = pd.concat(reviews)
review_df.drop(columns = ["cool", "date", "funny", "review_id",
                "user_id", "year", "nb_days", "business_id"], inplace = True)


#filter out reviews that aren't in english
nlp = spacy.load('en')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)
def only_English(text):
    try:
        doc = nlp(text)
    except:
        return False
    if 'en' in doc._.language_scores.keys():
        return doc._.language_scores['en'] > .9
    return False

review_df = review_df[review_df.text.apply(only_English)]



#Pre-Process text by removing stop words, punctuation, lowercase all the text,
#tokenizing the text (split it into words), and lemmatizing
wpt = nltk.WordPunctTokenizer()
stop_words = stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
def get_tag(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN

def clean_review(review):
    # lower case and remove special characters\whitespaces
    review = re.sub(r'[^a-zA-Z\s]', '', review, re.I|re.A)
    review = review.lower()
    review = review.strip()

    # tokenize document
    tokens = wpt.tokenize(review)
    # filter stopwords out of document
    review = [token for token in tokens if token not in stop_words]

    #get POS tags for the review
    pos_tags = pos_tag(review)

    # lemmatize review
    review = [WordNetLemmatizer().lemmatize(t[0], get_tag(t[1])) for t in pos_tags]

    # re-create document from filtered tokens
    review = ' '.join(review)
    return review

review_df["cleaned_review"] = review_df["text"].apply(lambda x: clean_review(x))


def rename_cols(key, d):
    """Add more annotation to column names for better understanding of the context of the NLP Features"""
    new_dict = {}
    for nested_key in d[key].keys():
        new_dict[key + " " + nested_key] = d[key][nested_key]
    return pd.Series(new_dict)
def unpack(text):
    """The readability API returns a nested dictionary of dictionaries, where each key in the original dictionary
    corresponds to a feature category e.g. sentence info, readability metric. This unpacks it and adds the new rows
    to the dataframe"""
    tokenized = '\n\n'.join(
     '\n'.join(' '.join(token.value for token in sentence)
        for sentence in paragraph)
     for paragraph in segmenter.analyze(text))
    nested_feature_dict = readability.getmeasures(tokenized, lang = 'en')
    row = []
    for k in nested_feature_dict.keys():
        row_fragment = rename_cols(k, nested_feature_dict)
        row.append(row_fragment)
    return pd.concat(row)

#Add the readability features to our dataframe
review_df = review_df.merge(review_df.text.apply(unpack), left_index = True, right_index = True)

#apply NLTK's SentimentIntensityAnalyzer, which gives us negative, positive, neutral,
#and compound (all of these added up together) rating of each review's text
sid = SentimentIntensityAnalyzer()
sentiments = review_df["cleaned_review"].apply(lambda x: sid.polarity_scores(x))
sentiments_df = json_normalize(sentiments)

#merge new feature dataframe with original review dataframe
review_df.reset_index(inplace = True, drop = True)
sentiments_df.reset_index(inplace = True, drop = True)
review_df = pd.concat((review_df, sentiments_df), axis = 1)

review_df.reset_index(inplace = True)
review_df.drop(columns = ["index", "cleaned_review", "text"], inplace = True)

X,y = review_df.drop(columns = ["log_useful"]), review_df.log_useful
#Scale data so that each feature follows a normal distribution,
#with mean 0 and standard deviation 1
X_standarized = preprocessing.scale(X)
X_standarized = pd.DataFrame(X_standarized, columns = X.columns)
remove_cols = []
def removeInflatedCols(df):
    """
    Remove columns/features with a Variance Inflation Factor of > 100.
    """
    X = add_constant(df)
    VIF_series = [[variance_inflation_factor(X_standarized.values, i),
    X_standarized.columns[i]] for i in range(X_standarized.shape[1])]
    for row in VIF_series:
        print(row[0])
        if row[0] > 100 and row[1] != "const":
            remove_cols.append(row[1])
            df.drop(columns = row[1], inplace = True)
    return df

X_standarized = removeInflatedCols(X_standarized)

#Create PCA instance to transform our matrix to PCA version
pca = PCA(n_components=20)

#Create PCA column names
pca_cols = []
for i in range(20):
    pca_cols.append("PCA_{}".format(i))

X_pca = pca.fit_transform(X_standarized)
X_pca = pd.DataFrame(X_pca, columns = pca_cols)

#Run GridSearch to find the optimal hyperparameters
n_estimators = [10,50,100, 200]
max_features = ['auto', 'sqrt']
max_depth = [5,10,15, None]
min_samples_split = [5,10,15]
min_samples_leaf = [4,8]
bootstrap = [True, False]
# Create the random grid
param_grid = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split, 'bootstrap': bootstrap, 'max_features': max_features,
              'max_depth': max_depth}


rf = GridSearchCV(RandomForestRegressor(random_state=13), param_grid)
rf.fit(X_pca, y.values.ravel())
new_model = RandomForestRegressor(**rf.best_params_)
new_model.fit(X_pca, y.values.ravel())

rf_model = pickle.dumps(new_model)
pca_model = pickle.dumps(pca)
