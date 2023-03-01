import nltk
from nltk.corpus import stopwords
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import bigrams
from nltk.util import everygrams

import string, sys


def data_preprocessing(name, test):
    # Declare necessary variables 
    clean_words = {}
    
    sentences = []
    ftemp = open(name[:-1], 'r', encoding='utf-8')
    temp_lines = ftemp.readlines()
    real_lines = ""
    for line in temp_lines:
        real_lines += line
    sentences = real_lines.split('.')

    # Clean the data 
    table = str.maketrans('', '', string.punctuation)
    stripped = []
    clean_sentences = []
    for sentence in sentences[:-1]:
        stripped = sentence.translate(table)
        lower = stripped.lower()
        individual_sentence = lower.split()

        # Remove non-alphabetic strings (i.e. punctuation) from the data
        clean_sentence = []
        for word in individual_sentence:
            alpha = "" 
            for index in word:
                if index.isalpha():
                    alpha += index
            if alpha != "":
                clean_sentence += [alpha]
        clean_sentences += [clean_sentence]

    ftemp.close()

    # If test flag is present, then we want to train model on all data, if not, we need to break data into a training and dev set.
    if test:
        train, vocab = padded_everygram_pipeline(3, clean_sentences)
        return [(train, vocab), []]
    else:
        # Break data into train and dev set
        clean_length = len(clean_sentences)//10
        train_set = clean_sentences[clean_length+1:]

        #train model on train set
        train, vocab = padded_everygram_pipeline(3, train_set)

        #separate dev set and create everygrams based on each sentence
        dev_set = clean_sentences[:clean_length]
        every = []
        for sentence in dev_set:
            padded_trigrams = list(pad_both_ends(sentence, n=3))
            every += [list(everygrams(padded_trigrams, max_len=3))]
        return [(train, vocab), every]


def test_data_processing(name):
    # Declare necessary variables 

    ftemp = open(name, 'r', encoding='utf-8')
    temp_lines = ftemp.readlines()
    table = str.maketrans('', '', string.punctuation)
    clean_sentences = []
    every = []

    for line in temp_lines:
        stripped = []

        # Clean the data 
        stripped = line.translate(table)
        lower = stripped.lower()
        individual_sentence = lower.split()

        # Remove non-alphabetic strings (i.e. punctuation) from the data
        clean_sentence = []
        for word in individual_sentence:
            alpha = "" 
            for index in word:
                if index.isalpha():
                    alpha += index
            if alpha != "":
                clean_sentence += [alpha]
        clean_sentences += [clean_sentence]
    
    for sentence in clean_sentences:
        padded_trigrams = list(pad_both_ends(sentence, n=3))
        every += [list(everygrams(padded_trigrams, max_len=3))]

    ftemp.close()
    return every 

# Our data has been processed and cleaned, now we train our model
def model_training(model, order):
    lm = MLE(order)

    train = model[0]
    vocab = model[1]
    lm.fit(train, vocab)

    return lm

def use_model(model, sequence):
    print(model.perplexity(sequence))

# Check for command line arguments. If there is more than just an authorlist present, then check if the second argument is the test flag. If it is, train the model fully on all of the data in the authorfile and output the classification result for each line in the testfile.
if len(sys.argv) == 2:
    f = open(sys.argv[1], "r")
    file_names = f.readlines()
    print("List of files we will be reading in: ", file_names)

    models = {}
    dev_data = {}
    for text in file_names:
        data_package = data_preprocessing(text, False)
        models[text[:-1]] = model_training(data_package[0], 3)
        dev_data[text[:-1]] = data_package[1]
    print(dev_data)



# Open the file indicating which text files to read in


# Read in each file and process it's data





'''
print("\nTraining the model...")
model_list = []
sentence_data = []

for key, value in processed_data.items():
    print(key)
    print(value[0])
    model_list += [model_training(key, value, 3)]
    sentence_data += value[2]
#     print(sentence_data)


'''