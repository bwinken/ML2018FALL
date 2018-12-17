import pandas as pd
import numpy as np
import jieba

def read_train_data(train_x,train_y,dict_path,label=True):
	print(" Reading Training Data ...")
	
	lines = []
	
	with open(train_x,'r',encoding='utf_8')as f:
		head = f.readline()
		train = f.readlines()
	train = [seg.split(',')[1] for seg in train]
	
	#using library
	jieba.set_dictionary(dict_path)
	
	for line in train:
		#seg_list = list(jieba.cut_for_search(line))
		seg_list = list(jieba.cut(line,cut_all=False))
		lines.append(seg_list)
		
	if label==True:
		labels = pd.read_csv(train_y)
		labels = labels[['label']].values
		
	if label==True:
		return lines,labels
	
	return lines
	
def read_test_data(test_x,dict_path):
	print(" Reading Testing Data ...")
	
	lines = []
	
	with open(test_x,'r',encoding='utf_8')as f:
		head = f.readline()
		test = f.readlines()
	test = [seg.split(',')[1] for seg in test]
	
	#using library
	jieba.set_dictionary(dict_path)
	
	for line in test:
		#seg_list = list(jieba.cut_for_search(line))
		seg_list = list(jieba.cut(line,cut_all=False))
		lines.append(seg_list)

	return lines
	
def padLines(lines,maxlen):
	maxlinelen = 0
	for i, s in enumerate(lines):
		maxlinelen = max(len(s), maxlinelen)
	maxlinelen = max(maxlinelen, maxlen)
	for i, s in enumerate(lines):
		lines[i] = (['_r'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
	return lines
	
def transformByWord2vec(lines,w2v,embed_dim):
	for i, s in enumerate(lines):
		for j, w in enumerate(s):
			if w in w2v.wv:
				lines[i][j] = w2v.wv[w]
			else:
				lines[i][j] = [0] * embed_dim
				
