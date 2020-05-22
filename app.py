
import readability
import nltk
import joblib
import pandas as pd
import numpy as np
from readability import getmeasures
from flask import Flask, request, render_template
import syntok.segmenter as segmenter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
nltk.download('vader_lexicon')

app = Flask(__name__)
scaler = joblib.load(open('scaler.pkl', 'rb'))
pca_transformer = joblib.load(open('pca_transformer.pkl', 'rb'))
rf_model = joblib.load(open('rf_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Works only for a single sample
    if request.method == 'POST':
        if not request.form.getlist('review'):
            return render_template('home.html', predicted = 'You need to enter a review for the restaurant')
        if not request.form.getlist('star'):
            return render_template('home.html', predicted = 'You need to give a number of stars for the restaurant')
        text = request.form.getlist('review')[0]
        stars = request.form.getlist('star')[0]

        tokenized = '\n\n'.join('\n'.join(' '.join(token.value
                        for token in sentence) for sentence in paragraph)
                            for paragraph in segmenter.analyze(text))
        nested_feature_dict = readability.getmeasures(tokenized, lang = 'en')
        new_cols = {"stars": int(stars)}
        for k in nested_feature_dict.keys():
            new_dict = {}
            for nested_key in nested_feature_dict[k].keys():
                new_cols[k + " " + nested_key] = nested_feature_dict[k][nested_key]
        df = pd.DataFrame(new_cols, index = [0])
        remove_cols = ['readability grades Kincaid', 'readability grades ARI',
        'readability grades FleschReadingEase', 'sentence info characters_per_word',
        'sentence info syll_per_word', 'sentence info words_per_sentence', 'sentence info characters',
         'sentence info syllables', 'sentence info long_words']
        df.drop(columns = remove_cols, inplace = True)

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

        sid = SentimentIntensityAnalyzer()
        sentiments = sid.polarity_scores(text)
        sentiments_df = json_normalize(sentiments)
        df["compound"] = sentiments_df["compound"]

        df_scaled = scaler.transform(df)
        df_transformed = pca_transformer.transform(df_scaled)
        prediction = rf_model.predict(df_transformed)  # runs globally loaded model on the data
        print(prediction)
        return render_template('home.html', predicted = round(np.exp(prediction[0]), 2))
    return render_template('home.html', predicted = 'Error')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
