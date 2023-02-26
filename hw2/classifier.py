from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import sys, string

words = ['a b c','b','c','d','e','f','g']

nltk.download('stopwords')

def data_preprocessing(authorlist):
    f = open(authorlist, "r")
    file_names = f.readlines()
    clean_words = {}
    for name in file_names:
        individual_words = []
        ftemp = open(name[:-1], 'r', encoding='utf-8')
        temp_lines = ftemp.readlines()
        for line in temp_lines:
            individual_words = individual_words + line.split()
        # Three lines below are from https://machinelearningmastery.com/clean-text-machine-learning-python/
        # The lines filter out punctuation, make the string lowercase, then remove all non alphanumeric characters
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in individual_words]
        lower = [w.lower() for w in stripped]
        alpha = [w for w in lower if w.isalpha()]
        # The following line will remove stopwords
        sw = set(stopwords.words('english'))
        clean = [w for w in alpha if not w in sw]
        clean_words[name] = clean 
        ftemp.close()
    return clean_words

print(data_preprocessing('authorlist.txt'))
