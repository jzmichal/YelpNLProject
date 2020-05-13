from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)
pca_transformer = pickle.load(open('pca_model.pkl', 'rb'))
rf_model = pickle.load(open('yelp_model.pkl', 'rb'))

@app.route("/")
def home():
    return "Hello! This is the main page <h1>HELLO<h1>"


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        tokenized = '\n\n'.join('\n'.join(' '.join(token.value
                        for token in sentence) for sentence in paragraph)
                            for paragraph in segmenter.analyze(text))
        nested_feature_dict = readability.getmeasures(tokenized, lang = 'en')
        rows = []
        for k in nested_feature_dict.keys():
            new_dict = {}
            for nested_key in d[k].keys():
                new_dict[k + " " + nested_key] = d[k][nested_key]
            new_cols = pd.Series(new_dict)
            rows.append(new_cols)
        df = pd.concat(rows)
        remove_cols = ['readability grades Kincaid', 'readability grades ARI',
        'readability grades FleschReadingEase', 'sentence info characters_per_word',
        'sentence info syll_per_word', 'sentence info words_per_sentence', 'sentence info characters',
         'sentence info syllables', 'sentence info words', 'sentence info long_words', 'neg', 'neu',
         'pos']
        df.drop(columns = remove_cols, inplace = True)
        prediction = model.predict(df)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
