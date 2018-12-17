import pandas as pd
import numpy as py
import jieba
import gensim
#from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import sys
from util import *
from keras.models import Sequential
from keras.layers import Dense, GRU, GRUCell,LSTM,Bidirectional,Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from preprocess import *
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def data_process(train_x_csv,train_y_csv,embed_dim,maxlen,dict_path):
	lines,train_y = read_train_data(train_x_csv,train_y_csv,dict_path)
	train_x = padLines(lines,maxlen)
	w2v = word2vec.Word2Vec.load("w2v.model")
	transformByWord2vec(train_x,w2v,embed_dim)
	
	return train_x,train_y

def RNN_model(train_x,train_y,batch_size,epochs,embed_dim,max_len):
	#model
	model = Sequential()
	model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, input_shape=(max_len, embed_dim)))
	model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))
	#model.add(Bidirectional(LSTM(512, activation='tanh', use_bias=True, dropout=0.5, recurrent_dropout=0.5, return_sequences=True),input_shape=(max_len, embed_dim)))
	#model.add(Bidirectional(LSTM(512, activation='tanh', use_bias=True, dropout=0.5, recurrent_dropout=0.5)))
	model.add(Dense(512, activation='selu'))
	model.add(Dense(512, activation='selu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	model.summary()
	
	save = ModelCheckpoint('./rnn.h5', 
		monitor='val_acc', 
		verbose=1,
		save_best_only = True, 
		save_weights_only=False,
		mode='auto', 
		period=1) # save improved model only
	learning_rate_function = ReduceLROnPlateau(
		monitor='val_acc', 
		patience=4, 
		verbose=1,
		factor=0.5, 
		min_lr=0.000005)
		
	model_result=model.fit(train_x, train_y, 
			batch_size=batch_size, 
			epochs=epochs, 
			validation_split=0.1,
			callbacks=[learning_rate_function, save])

	model.save('rnn.h5')
	plt.plot(model_result.history['acc'])
	plt.plot(model_result.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_result.history['loss'])
	plt.plot(model_result.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def DNN_model(train_x,train_y,batch_size,epochs,embed_dim,max_len):
	#model
	model = Sequential()
	model.add(Flatten(input_shape=(max_len,embed_dim)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(512,activation='relu'))
	model.add(Dense(128,activation='relu'))
	model.add(Dense(64,activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	model.summary()
	
	save = ModelCheckpoint('./dnn.h5', 
		monitor='val_acc', 
		verbose=1,
		save_best_only = True, 
		save_weights_only=False,
		mode='auto', 
		period=1) # save improved model only
	learning_rate_function = ReduceLROnPlateau(
		monitor='val_acc', 
		patience=4, 
		verbose=1,
		factor=0.5, 
		min_lr=0.000005)
		
	model_result=model.fit(train_x, train_y, 
			batch_size=batch_size, 
			epochs=epochs, 
			validation_split=0.1,
			callbacks=[learning_rate_function, save])

	plt.plot(model_result.history['acc'])
	plt.plot(model_result.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_result.history['loss'])
	plt.plot(model_result.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
if __name__ == "__main__":
	train_x_csv = sys.argv[1]
	train_y_csv = sys.argv[2]
	test_csv	= sys.argv[3]
	dict_path	= sys.argv[4]
	
	segmentation(train_x_csv,test_csv,dict_path)
	
	#train
	embed_dim = 256
	train(embed_dim)
	
	#parameter
	batch_size = 512
	epochs = 100
	max_len		= 50
	#limit GPU usage
	#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth=True
	#sess = tf.Session(config=config)
	
	train_x,train_y = data_process(train_x_csv,train_y_csv,embed_dim,max_len,dict_path)
	train_x = np.array(train_x)
	
	print("train_x shape:",train_x.shape)
	
	RNN_model(train_x,train_y,batch_size,epochs,embed_dim,max_len)
	
	