import re
import string
from pickle import dump
from unicodedata import normalize
import numpy as np



# tokenize text

def load_doc(textfile):

    # open the file
    file = open(textfile, mode = 'rt', encoding='utf-8')

    # read text file
    text = file.read()

    # close the file
    file.close()

    return text

# function to create english-german pairs from text

def pairing(text):

    lines = text.strip().split('\n')

    pairs = [line.split('\t') for line in lines]

    return pairs

# clean and normalize data

def clean_pairs(lines):

    cleaned = []

    # creates re object for all excluding all characters?
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    table = str.maketrans('', '', string.punctuation)


    for pair in lines:
        clean_pair = []

        for line in pair:
            line = normalize('NFD', line).encode(encoding = 'ascii', errors = 'ignore')

            line = line.decode('UTF-8')


            line = line.split()

            # convert to lower case
            line = [word.lower() for word in line]

            # remove punctuation characters
            line = [word.translate(table) for word in line]

            # remove non-printable characters
            line = [re_print.sub('', w) for w in line]

            # remove tokens with number in it
            line = [word for word in line if word.isalpha()]

            # store as string
            clean_pair.append(' '.join(line))

        cleaned.append(clean_pair)

    return np.array(cleaned)


# save a list of clean sentences to a file
def save_to_file(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print(f'Saved: {filename}')

# load data set
filename = 'deu-eng.txt'
doc = load_doc(filename)

# turn in to pairs
pairs = pairing(doc)

# clean sentences
clean_pairs_array = clean_pairs(pairs)

# save clean pairs
save_to_file(clean_pairs_array, 'english-german.pkl')

# spot check
for i in range(100):
 print(f'[{clean_pairs_array[i,0]}] => [{clean_pairs_array[i,1]}]')







