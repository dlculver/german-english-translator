from pickle import dump, load
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu



# hyperparameters
n_sentences = 20000
train_size = round(n_sentences*0.8)
#test_size = 4000


# ancillary functions

# load clean data set
def load_clean_dataset(filename):
	return load(open(filename, 'rb'))

# save a list of cleaned sentences
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print(f'Saved: {filename}')

# function to create tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# find max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
# returns np array of sentences, which have been converted into integer sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences
	X = pad_sequences(X, maxlen = length, padding = 'post')
	return X

# function to one hot encode target sequences
def encode_target(sequences, vocab_size):
	ylist = []
	#print(sequences.shape)
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes = vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	#print(y.shape)
	y = y.reshape((sequences.shape[0], sequences.shape[1], vocab_size))
	return y

# function to define model that will be trained
def define_model(source_vocab, target_vocab, source_timesteps, target_timesteps, n_units):
	model = Sequential()
	# add in TextVectorization layer
	model.add(Embedding(source_vocab, n_units, input_length=source_timesteps, mask_zero=True))
	model.add(Bidirectional(LSTM(n_units)))
	model.add(RepeatVector(target_timesteps))
	model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
	model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
	return model

# the following functions will be for obtaining the translations from the model

# to turn integers into words via the tokenizer
def integer_to_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# converts a prediction to a string
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose = 0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = []
	for i in integers:
		word = integer_to_word(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = [], []
	for i, source in enumerate(sources):
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, tokenizer, source)
		raw_target, raw_source = raw_dataset[i]
		if i < 10:
			print(f'source = [{raw_source}], target = [{raw_target}], prediction = [{translation}]')
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))




# # uncomment the following to create training and test sets.
# # load data set
# raw_dataset = load_clean_dataset('english-german.pkl')
#
# # pick out shortest n_sentences examples, think about how to incorporate longer sentences later
# dataset = raw_dataset[:n_sentences, :]
#
# # shuffle data set
# # we need to shuffle to make sure that different length sentences are uniformly distributed between the train and test data
# np.random.shuffle(dataset)
#
# # split into train, test
# train, test = dataset[:train_size], dataset[train_size:]
#
# # save data
# save_clean_data(dataset, 'english-german-both.pkl')
# save_clean_data(train, 'english-german-train.pkl')
# save_clean_data(test, 'english-german-test.pkl')

# load saved data
dataset = load_clean_dataset('english-german-both.pkl')
train = load_clean_dataset('english-german-train.pkl')
test = load_clean_dataset('english-german-test.pkl')
print(dataset[0])
# prepare English tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print(f'English vocab size: {eng_vocab_size}')
print(f'English max sentence length: {eng_length}')

# vorbereiten Deutscher tokenizer
deu_tokenizer = create_tokenizer((dataset[:, 1]))
deu_vocab_size = len(deu_tokenizer.word_index) + 1
deu_length = max_length(dataset[:, 1])
print(f'German vocab size: {deu_vocab_size}')
print(f'German max sentence length: {deu_length}')

# prepare training and test sets for the model
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_target(trainY, eng_vocab_size)

testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:,0])
testY = encode_target(testY, eng_vocab_size)

# # uncomment below if you want to define and fit model
# # define model
# model = define_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
#
# # summarize model
# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)
#
# # fit model
# filename = 'model.h5'
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(trainX, trainY, epochs=30, batch_size=100, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

# load model
model = load_model('model.h5')
# test on some training sequences
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)

# example = testX[4]
# print(example.shape)
# example = example.reshape((1, example.shape[0]))
# print(example.shape)
# prediction = model.predict(example, verbose = 0)
# print(prediction.shape)
# print(prediction[0][4])

# # list words from most probable to least probable
# vector = prediction[0][3]
# print(vector.shape)
# integer_list = []
# word_list = []
# for i in range(50):
# 	idx = np.argmax(vector)
# 	integer_list.append(idx)
# 	word = integer_to_word(idx, eng_tokenizer)
# 	word_list.append(word)
# 	vector[idx] = 0
#
# print(word_list)

