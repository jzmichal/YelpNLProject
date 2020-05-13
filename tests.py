import unittest
from endtoend import only_English
from endtoend import unpack
from endtoend import rename_cols
from endtoend import clean_review
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy_cld import LanguageDetector
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import re
import pandas as pd
import readability
import syntok.segmenter as segmenter

class TestSum(unittest.TestCase):

    def test_onlyEnglish(self):
        s_0 = "slabiky, ale liší se podle významu"
        s_1 = "how about this one : 通 asfަ"
        s_2 = "now try this one with english words only anddd some typoooos"
        self.assertEqual(only_English(s_0), False, "Should be False")
        self.assertEqual(only_English(s_1), False, "Should be False")
        self.assertEqual(only_English(s_2), True, "Should be True")

    def test_clean(self):
        s_0 = """excuse me, and i might drink a little more than I should tonight.
                Don't care what they say, let's do it tonight."""
        s_1 = """Food was good except service was pitiful.
                Told me I wasn't allowed to take leftovers home and if I left any food on my plate,
                I'd be charged extra, because apparently it's a waste"""

        s_0_cleaned = 'excuse might drink little tonight dont care say let tonight'
        s_1_cleaned = """food good except service pitiful tell wasnt allow take leftover
                        home leave food plate id charge extra apparently waste"""
        self.assertEqual(clean_review(s_0, s_0_cleaned, "Should be cleaned like this"))
        self.assertEqual(clean_review(s_1, s_1_cleaned, "Should be cleaned like this"))

    def test_renameCols(self):
        nested_dict = { 'dictA': {'key_1': 'value_1'},
                'dictB': {'key_2': 'value_2', 'key_3': 'value_3'}}
        srs_1 = pd.series({'dictA key_1': 'value_1'})
        srs_2 = pd.series({'dictB key_2': 'value_2', 'dictB key_3': 'value_3'})
        self.assertEqual(rename_cols("dictA", nested_dict), srs_1, "Test with just one key value pair")
        self.assertEqual(rename_cols("dictB", nested_dict), srs_2, "Test with multiple key value pairs")

    def test_removeInflatedCols(self):
        df_0 = pd.DataFrame(
            {'a': [1, 1, 2, 3, 4],
             'b': [2, 2, 3, 2, 1],
             'c': [4, 6, 7, 8, 9],
             'd': [4, 3, 4, 5, 4]})
        self.assertEqual(removeInflatedCols(df_0), df_0, "Don't remove anything")

        df_1 = pd.DataFrame(
            {'a': [50, 0, -50, 1000, -400],
             'b': [60, 1, -45, 1050, -390],
             'c': [4, 6, 7, 8, 9],
             'd': [4, 3, 4, 5, 4]})
        df_1_cleaned = pd.DataFrame(
            {'c': [4, 6, 7, 8, 9],
             'd': [4, 3, 4, 5, 4]})
        self.assertEqual(removeInflatedCols(df_1), df_1_cleaned, "Remove first two columns")
