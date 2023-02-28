import nltk
from nltk.corpus import stopwords
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE

import string


def data_preprocessing(authorlist):
    # Declare necessary variables 
    clean_words = {}

    # Open the file indicating which text files to read in
    f = open(authorlist, "r")
    file_names = f.readlines()
    print("List of files we will be reading in: ", file_names)

    # Read in each file and process it's data
    for name in file_names:
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
        padded = []
        for sentence in sentences[:-1]:
            stripped = sentence.translate(table)
            lower = stripped.lower()
            individual_sentence = lower.split()
            
            # Remove non-alphabetic strings (i.e. punctuation) from the data
            clean_sentence = []
            stop_words = set(stopwords.words('english'))
            for word in individual_sentence:
                alpha = "" 
                for index in word:
                    if index.isalpha():
                        alpha += index
                if alpha != "":
                    clean_sentence += [alpha]
            clean_sentences += [clean_sentence]

        clean_sentences = list(flatten(pad_both_ends(sent, n=2) for sent in clean_sentences))
        train, vocab = padded_everygram_pipeline(2, clean_sentences)

        ftemp.close()
        clean_words[name[:-1]] = (train, vocab, clean_sentences)
    
    return clean_words


print("\nProcessing the data...")
processed_data = data_preprocessing('authorlist.txt')


# Our data has been processed and cleaned, now we train our model
def model_training(name, model):
    lm = MLE(2)

    train = model[0]
    vocab = model[1]
    lm.fit(train, vocab)

    return lm


print("\nTraining the model...")
model_list = []
sentence_data = []

for key, value in processed_data.items():
    print(key)
    model_list += [model_training(key, value)]
    sentence_data += [value[2]]


def use_model(model, sequence):
    print(model.perplexity(sequence))
    

print("\nUsing the trained model...")
use_model(model_list[0], [sentence_data[0], sentence_data[1]])