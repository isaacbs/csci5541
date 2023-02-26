from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.lm import MLE
import sys, string

words = ['a b c','b','c','d','e','f','g']


def data_preprocessing(authorlist):
    f = open(authorlist, "r")
    file_names = f.readlines()
    clean_words = {}
    for name in file_names:
        sentences = []
        ftemp = open(name[:-1], 'r', encoding='utf-8')
        temp_lines = ftemp.readlines()
        real_lines = ""
        for line in temp_lines:
            real_lines += line
        sentences = real_lines.split('.')

        table = str.maketrans('', '', string.punctuation)
        stripped = []
        clean_sentences = []
        # padded = []
        for sentence in sentences[:-1]:
            stripped = sentence.translate(table)
            lower = stripped.lower()
            individual_sentence = lower.split()
            
            clean_sentence = []
            
            sw = set(stopwords.words('english'))
            for w in individual_sentence:
                alpha = "" 
                for l in w:
                    if l.isalpha():
                        alpha += l
                if alpha not in sw and alpha != "":
                    clean_sentence += [alpha]
                
            clean_sentences += [clean_sentence]
        # padded = list(flatten(pad_both_ends(s, n=2) for s in clean_sentences))
        train, vocab = padded_everygram_pipeline(2, clean_sentences)
        ftemp.close()
        clean_words[name[:-1]] = (train, vocab)
    return clean_words

def train(name, model):
    lm = MLE(2)
    lm.fit(model[0], model[1])
    print(lm.vocab)


processed = data_preprocessing('authorlist.txt')
for key, value in processed.items():
    train(key, value)
# print(processed)