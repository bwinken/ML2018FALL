import pandas as pd
import numpy as py
import jieba
import sys
from gensim.models import word2vec
	
def segmentation(train_x_csv,test_x_csv,dict_path):
	segmentation = open("seg.txt", "w", encoding = "utf-8")
	
	with open(train_x_csv) as f:
		head = f.readline()
		train = f.readlines()
	
	with open(test_x_csv) as f:
		head = f.readline()
		test = f.readlines()
		
	#split the first column out
	train = [seg.split(',')[1] for seg in train]
	test = [seg.split(',')[1] for seg in test]
	
	#using library
	jieba.set_dictionary(dict_path)
	
	#start segment and combine into segmentation.txt
	for sentence in train:
		sentence = sentence.strip("\n")
		#pos = jieba.cut_for_search(sentence)
		pos = jieba.cut(sentence,cut_all=False)
		for term in pos:
			segmentation.write(term + " ")
	for sentence in test:
		sentence = sentence.strip("\n")
		#pos = jieba.cut_for_search(sentence)
		pos = jieba.cut(sentence,cut_all=False)
		for term in pos:
			segmentation.write(term + " ")
			
def segmentation_test(train_x_csv,test_x_csv):
	segmentation = open("seg_test.txt", "w", encoding = "utf-8")
	
	with open(train_x_csv) as f:
		head = f.readline()
		train = f.readlines()
	
	with open(test_x_csv) as f:
		head = f.readline()
		test = f.readlines()
		
	#split the first column out
	train = [seg.split(',')[1] for seg in train]
	test = [seg.split(',')[1] for seg in test]
	
	#start segment and combine into segmentation.txt

	for sentence in train:
		sentence = sentence.strip("\n")
		for term in sentence:
			segmentation.write(term + " ")
	for sentence in test:
		sentence = sentence.strip("\n")
		for term in sentence:
			segmentation.write(term + " ")	
			
def train(embed_dim):
	print("Training... ...")
	# Load file
	sentence = word2vec.Text8Corpus("seg.txt")
	# Setting degree and Produce Model(Train)
	model = word2vec.Word2Vec(sentence, size = embed_dim, window = 10, min_count = 5, workers = 4, sg = 1)
	# Save model 
	model.save("w2v.model")
	print("model 已儲存完畢")
		
if __name__ == "__main__":
	train_x_csv = sys.argv[1]
	test_csv	= sys.argv[2]
	dict_path	= sys.argv[3]
	
	print("TRAIN:",train_x_csv)
	print("TEST:",test_csv)
	print("DICT:",dict_path)
	
	#segmentation
	segmentation(train_x_csv,test_csv,dict_path)
	
	#train
	embed_dim = 256
	train(embed_dim)
	
	#wv = word2vec.Word2Vec.load("w2v.model")

	