from keras.models import load_model
import sys
import pandas as pd
import numpy as np
import jieba
from util import *
#from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec


def data_process(test_csv,embed_dim,maxlen,dict_path):
	lines = read_test_data(test_csv,dict_path)
	test_x = padLines(lines,maxlen)
	w2v = word2vec.Word2Vec.load("w2v.model")
	transformByWord2vec(test_x,w2v,embed_dim)
	
	return test_x
	

if __name__ == "__main__":
	
	test_csv = sys.argv[1]
	dict	= sys.argv[2]
	ans_csv = sys.argv[3]
	
	embed_dim = 256
	maxlen = 50
	
	test = data_process(test_csv,embed_dim,maxlen,dict)
	test = np.array(test)

	
	model = load_model("rnn.h5")
	
	print("Loaded model from disk")
	
	y_pre = model.predict_classes(test)

	
	
	print("Y_PRE shape",y_pre.shape)
	
	with open(ans_csv, "w") as f:
		f.write("id,label\n")
		for i in range(len(y_pre)-1):
			f.write(str(i) + "," + str(int(y_pre[i])) + "\n")
		f.write(str(len(y_pre)-1) + "," + str(int(y_pre[len(y_pre)-1])))