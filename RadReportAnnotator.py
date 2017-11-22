"""
RadReportAnnotator
Authors: jrzech, eko

This is a library of methods for automatically inferring labels for a corpus or radiological documents given a set of manually-labeled data.

"""

#usual imports for data science
import numpy as np
import pandas as pd
import sys
import os
import math
from tqdm import tqdm

#sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#NLP imports
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

#gensim for word embedding featurization
import gensim
from collections import namedtuple

#misc
import glob
import os.path
import multiprocessing
import random


def join_montage_files(data_dir,NAME_UNID_REPORTS, NAME_TEXT_REPORTS):
	"""
 	Joins several montage files in excel format into a single pandas dataframe
 	Args:
  		data_dir: a filepath pointing to a directory containing montage files in excel format
  		NAME_UNID_REPORTS: column name of unique id / accession id in reports xlsx
  		NAME_TEXT_REPORTS: column name of report text in reports xlsx
 	Returns:
  		df_data: a pandas dataframe containing texts from montage
 	"""
	print("building pre-corpus")
	datafiles = os.listdir(data_dir)
	df_data = pd.read_excel(os.path.join(data_dir,datafiles[0]))
	datafiles.remove(datafiles[0])
	for subcorpus in datafiles:
		df_data = df_data.append(pd.read_excel(os.path.join(data_dir,subcorpus)))
	print('pre-corpus built')

	df_data.rename(columns={NAME_UNID_REPORTS:'Accession Number',NAME_TEXT_REPORTS:'Report Text'},inplace=True)
	return df_data

def preprocess_data(df_data, en_stop, stem_words=True):
	"""
	Takes a dataframe of montage files and a list of stop words and returns a list of lda_inputs. The lda_inputs
	list consists of sublists of stemmed unigrams.
	Args:
		df_data: a dataframe of joined montage files.
		en_stop: a list of english stop_words from the stop_words library
		stem_words: argument indicating whether or not to stem words
	Returns:
		lda_inputs: a list of lists of stemmed words from each text within the montage dataframe
	"""
	if(stem_words==False):
		print("NOTE - NOT STEMMING")
	p_stemmer = PorterStemmer()
	processed_reports = []
	accession_index=[]

	print("preprocessing reports")
	for i in tqdm(range(0,df_data.shape[0])):

		tokenizer = RegexpTokenizer(r'\w+')
		process = df_data['Report Text'].iloc[i]

		process = str(process)
		process = process + "..." # add a period, sometimes it's missing at end
		process = process.lower()

		z = len(process)
		k = 0
		#remove line breaks
		process=process.replace("^M", " ") # 
		process=process.replace("\n", " ") # 
		process=process.replace("\r", " ") #   
		process=process.replace("_", " ") #   
		process=process.replace("-", " ") #   
		process=process.replace(",", " , ") # 
		process=process.replace("  ", " ") # 
		process=process.replace("  ", " ") # 
		process=process.replace("  ", " ") # 
		process=process.replace("  ", " ") # 
		process=process.replace("  ", " ") # 

		process = re.sub(r'\d+', '',process)
		process=process.replace(".", " SENTENCEEND ")  # create end characters

		process_tokenized = tokenizer.tokenize(process)
		process_stopped = [i for i in process_tokenized if not i in en_stop]
        
		if(stem_words==True):        
			process_stemmed = [p_stemmer.stem(i) for i in process_stopped]
		else:
			process_stemmed = process_stopped

		processed_reports.append(process_stemmed)
		#include n grams in lda_input
	return processed_reports

def remove_infrequent_tokens(processed_reports,freq_threshold,labeled_indices):
	"""
	Takes a list of processed_preports and removes infrequent tokens (defined as occurring < freq_threshold times) from them.
	Args:
		processed_reports: list of lists of stemmed words after initial processing, where each entry corresponds to a report
		freq_threshold: count threshold, remove words occuring < freq_threshold time from corpus. note - considers only unlabeled corpus, not labeled corpus, to avoid peeking into labeled data. 
		labeled_indices: indices of processed_reports that are labeled reports - these are excluded from frequency calculations.
	Returns:
		process_reports_postcountfilter: list of lists of stemmed words after initial processing, where each entry corresponds to a report, after low frequency words have been removed
	"""
	word_count = common_stems(processed_reports, labeled_indices)
	d = dict((k,v) for k, v in word_count.items() if v >= freq_threshold)
	process_reports_postcountfilter=[[] for x in range(0,len(processed_reports))]
	for i in range(0,len(processed_reports)):
		for token in processed_reports[i]:
			if token in d: 
				process_reports_postcountfilter[i].append(token)
	return process_reports_postcountfilter

def create_ngrams(processed_reports, labeled_indices, N_GRAM_SIZES, freq_threshold):
	"""
	Takes a a processed_reports list, specified n_gram size list, and a frequency threshold at which
	to eliminate tokens with < frequency of appearance returns creates n_grams as well as removes ngrams that signify end of sentence
	Args:
		processed_reports: a list of text lists of stemmed unigrams ready for conversion into n-gram text lists
		labeled_indices: exclude these from calculcation of n-gram cutoff
		N_GRAM_SIZES: a list of ints specifying the n-gram sizes to include in the texts of the future corpus
		freq_threshold: the frequency threshold for n-gram inclusions. N-grams that occur with frequency < threshold will be removed from corpus
	Returns:
		processed_outputs_clean: a list of text lists of n-grams that are ready to be processed into a corpus	
	"""
	processed_outputs = []
	print("creating n-grams")
	for report in tqdm(processed_reports[:]):
		new_report = []
		end=len(report)
		#CREATES 4-grams - for all n-grams, we don't allow "no" to be in middle of n-gram, don't allow sentenceend token to be in n-gram
		if 4 in N_GRAM_SIZES:
			for i in range (0,end-3):
				if (report[i+1] != "no" and report[i+2] != "no" and report[i+3] != "no" and report[i].lower()!= "sentenceend" and report[i+1].lower()!= "sentenceend" and report[i+2].lower()!= "sentenceend" and report[i+3]!= "sentenceend"): #no only at beginning
					new_report.append(report[i] +"_" +report[i+1] + "_" + report[i+2] + "_" + report[i+3]) 
		#CREATES 3-grams
		if 3 in N_GRAM_SIZES:
			for i in range (0,end-2):
				if (report[i+1] != "no" and report[i+2] != "no" and report[i].lower()!= "sentenceend" and report[i+1].lower()!= "sentenceend" and report[i+2].lower()!= "sentenceend"): #no only at beginning
					new_report.append(report[i] +"_" +report[i+1] + "_" + report[i+2])
		#CREATES 2-grams
		if 2 in N_GRAM_SIZES:
			for i in range (0,end-1):
				if (report[i+1] != "no" and report[i].lower()!= "sentenceend" and report[i+1].lower()!= "sentenceend"): #no only at beginning
					new_report.append(report[i] +"_" +report[i+1])
		#CREATES unigrams
		if 1 in N_GRAM_SIZES:
			for i in range (0,end):
				if(report[i].lower()!= "sentenceend" and report[i]!= "no"): # we take no out as a unigram in bow
					new_report.append(report[i])
		processed_outputs.append(new_report)

	#remove low freq tokens
	word_count = common_stems(processed_outputs, labeled_indices)
	print("number of unique n-grams:", len(word_count)) 
	d = dict((k,v) for k, v in word_count.items() if v >= freq_threshold)
	print("number of unique n-grams after filtering out low frequency tokens:", len(d))

	#remove tokens that occurred infrequently from processed_outputs --> processed_outputs_clean
	processed_outputs_clean=[[] for x in range(0,len(processed_outputs))]
	for i in range(0,len(processed_outputs)):
		for token in processed_outputs[i]:
			if token in d: 
				processed_outputs_clean[i].append(token)
	return processed_outputs_clean

def get_labeled_indices(df_data,validation_file,TRAIN_INDEX_OVERRIDE):
	"""
	Returns numerical indices of reports in df_data for which we have labeled data in validation_file; will set labeled reports as unlabeled if in TRAIN_INDEX_OVERRIDE
	Args:
		df_data: dataframe containing report text and accession ids
		validation_file: dataframe containing accession ids and labels
		TRAIN_INDEX_OVERRIDE: list of numerical indices to treat as unlabeled; necessary to train d2v model if all your data is labeled as it uses exclusively unlabeled data to train to avoid peeking into labeled data
	Returns:
		return_indices: indices we treat as labeled
	"""
	validation = pd.read_excel(validation_file)
	validation.set_index('Accession Number')
	validation_cases=validation['Accession Number'].tolist()
	all_indices = df_data['Accession Number'].tolist()
	return_indices=[]
	for i in all_indices:
		if i in validation_cases and i not in TRAIN_INDEX_OVERRIDE: # if something is manually overrided to be in train, don't put it in test
			return_indices.append(True)
		else:
			return_indices.append(False)
	return return_indices

def common_stems(ngram_list, exclude_indices):
	"""
	Takes a list of ngrams, ngram_list, and returns the most frequently appearing stems as a dict item of word:word_count pairs
	is flagged to write output to memory.
	Args:
		ngram_list: list of all n_grams
		exclude_indices:rows to ignore when doing count (labeled data)
	Returns:
		word_count: dict of ngram:ngram_count pairs
	"""
	word_count={}
	i=0
	excluded=0	
	for entry in ngram_list:
		if exclude_indices[i]==False:
			for word in entry:
				if word not in word_count:
					#add word with entry 1
					word_count[word] = 1
				else:
					#increment entry by 1
					word_count[word]=word_count[word]+1
		else:
			excluded=excluded+1  
		i=i+1
	d = dict((k,v) for k, v in word_count.items())

	return word_count


def build_train_test_corpus(df_data, ngram_list, labeled_filepath,TRAIN_INDEX_OVERRIDE):
	"""
	Takes the master corpus, the ngram_list, and a filepath pointing to a labeled spreadsheet
	and builds a labeled_corpus consisting of labelled data and an unlabeled_corpus
	of non-labelled data
	Args:
		df_data: a dataframe consisting of the original set of excel files with report text and accession id
		ngram_list: list of all n-grams in corpus
		labeled_filepath: path to file containing accession ids and labels
		TRAIN_INDEX_OVERRIDE: indices to treat as unlabeled data regardless of presence of labels.
	Returns:
		train_corpus: a corpus consisting of unlabelled texts that will be used for model construction
		test_corpus: a corpus consisting of labelled held-out texts that will be used for model validation
		dictionary: a dictionary compromised of the LDA input n-grams
		labeled_indices: the indices for the validation files
	"""
	dictionary = gensim.corpora.Dictionary(ngram_list)
	corpus = [dictionary.doc2bow(input) for input in ngram_list]
	if(not labeled_filepath is None):
		outcomes = pd.read_excel(labeled_filepath)
		outcomes.set_index('Accession Number')
		labeled_cases=outcomes['Accession Number'].tolist()
	else:
		labeled_cases=[]
	labeled_indices = []
	not_labeled_indices = []
	train_data_lda = np.ones(df_data.shape[0],dtype=bool)
	num_removed=0
	for i in range(0,df_data.shape[0]):
		if df_data['Accession Number'].iloc[i] in labeled_cases and df_data['Accession Number'].iloc[i] not in TRAIN_INDEX_OVERRIDE:
			train_data_lda[i]=False
			labeled_indices.append(i)
			num_removed += 1
		else:
			not_labeled_indices.append(i)
	unlabeled_corpus = [corpus[i] for i in not_labeled_indices]
	labeled_corpus = [corpus[i] for i in labeled_indices]

	return corpus, unlabeled_corpus, labeled_corpus, dictionary, labeled_indices


def build_d2v_corpora(df_data,d2v_inputs,labeled_indices):
	"""
	Build corpora in format for doc2vec gensim implementation
	Args:
		df_data: a dataframe consisting of the original set of excel files with report text and accession id
		d2v_inputs: list of lists of tokens, where each entry in d2v_inputs corresponds to a report
		labeled_indices: indices of labeled reports (and those we treat as labeled due to TRAIN_INDEX_OVERRIDE)
	Returns:
		unlabeled_corpus: a corpus consisting of unlabelled texts that will be used for feature construction
		labeled_corpus: a corpus consisting of labelled held-out texts that will be used for Lasso regression training
		total_unlabeled_words: count of total words in unlabeled corpus
	"""

	SentimentDocument = namedtuple('SentimentDocument', 'words tags')
	unlabeled_docs = [] 
	labeled_docs = []  
	total_unlabeled_words=0
	i=0
	for line in d2v_inputs:
		words = line # [x for x in line if x != 'END']
		tags = '' + str(df_data['Accession Number'].iloc[i])
		if(i in labeled_indices):
			labeled_docs.append(SentimentDocument(words,tags))
		else:
			unlabeled_docs.append(SentimentDocument(words,tags))
			total_unlabeled_words+=len(words)
		i+=1

	print('%d unlabeled reports for featurization, %d labeled reports for modeling' % (len(unlabeled_docs), len(labeled_docs)))
	return unlabeled_docs, labeled_docs, total_unlabeled_words


def train_d2v(unlabeled_docs, labeled_docs, D2V_EPOCH, DIM_DOC2VEC, W2V_DM, W2V_WINDOW, total_unlabeled_words):
	"""
	Train doc2vec/word2vec model.

	Args:
		unlabeled_docs: unlabeled corpus
		labeled_docs: labeled corpus
		D2V_EPOCHS: number of epochs to train d2v model; 20 has worked well in our experiments; parameter for gensim doc2vec
		DIM_DOC2VEC: dimensionality of embedding vectors, we explored values 50-800; parameter for gensim doc2vec
		W2V_DM: 1 is PV-DM, otherwise PV-DBOW; parameter for gensim doc2vec
		W2V_WINDOW: number of words window to use  in doc2vec model; parameter for gensim doc2vec
		total_unlabeled_words: total words in unlabeled corpus; argument for gensim doc2vec

	Returns:
		d2vmodel: trained doc2vec model.
	"""

	cores = multiprocessing.cpu_count()
	assert gensim.models.doc2vec.FAST_VERSION > -1, "speed up"
	print("started doc2vec training")
	d2vmodel = gensim.models.Doc2Vec(dm=W2V_DM, size=DIM_DOC2VEC, window=W2V_WINDOW, negative=5, hs=0, min_count=2, workers=cores)
	d2vmodel.build_vocab(unlabeled_docs + labeled_docs)  
	d2vmodel.train(unlabeled_docs, total_words=total_unlabeled_words, epochs=D2V_EPOCH)
	return d2vmodel

def calc_auc(predictor_matrix,eligible_outcomes_aligned, all_outcomes_aligned,N_LABELS, pred_type, header,ASSIGNFOLD_USING_ROW=False):
	"""
	Train Lasso models using 60% of labeled data with generated features and labels; calculate AUC, accuracy, 
	confusion matrix for each label on remaining 40% of labeled data.

	Args:
		
		predictor_matrix: numpy matrix of features available to use as input to Lasso logistic regression
		eligible_outcomes_aligned: dataframe of labels we are predicting
		all_outcomes_aligned: dataframe of all labels, including those we excluded due to infrequent positive/negative occurences - we use it for accession id
		N_LABELS: total number of labels we are predicting
		pred_type: label indicating what variables went into predictor_matrix
		results_dir: directory to which to save results
		header: header for predictor matrix
		ASSIGNFOLD_USING_ROW: normally 60/40 split done randomly, you can fix it to use first 60% of rows if you need replicability 
							   but be wary of introducing distortion into train/test set with dates, etc.: recommend randomly sorting
							   rows in excel beforehand if you opt for this.

	Returns:

		lasso_models: list of all trained lasso logistic regression models from sklearn, where index corresponds to relative index in columns of eligible_outcomes_aligend
	"""

	if predictor_matrix.shape[1]!=len(header):
		print("predictor_matrix.shape[1]="+str(predictor_matrix.shape[1]))
		print("len(header)"+str(len(header)))
		raise ValueError("predictor_matrix shape doesn't match header, investigate")
	all_coef = pd.concat([ pd.DataFrame(header)], axis = 1)	
	
	lasso_models={}
	model_types = ["Lasso"]

	r = list(range(eligible_outcomes_aligned.shape[0]))
	random.shuffle(r)
	
	if(ASSIGNFOLD_USING_ROW):
		assignfold = pd.DataFrame(data=list(range(eligible_outcomes_aligned.shape[0])), columns=['train'])
	else:
		assignfold = pd.DataFrame(data=r, columns=['train'])

	cutoff = np.floor(0.6*eligible_outcomes_aligned.shape[0])
	
	train=assignfold['train']<cutoff
	test=assignfold['train']>=cutoff
	
	N_TRAIN=eligible_outcomes_aligned.ix[train,:].shape[0]
	N_HELDOUT=eligible_outcomes_aligned.ix[test,:].shape[0]
	print("n_train in modeling="+str(N_TRAIN))
	print("n_test in modeling="+str(N_HELDOUT))
	
	confusion = pd.DataFrame(data=np.zeros(shape=(eligible_outcomes_aligned.shape[1]*len(model_types),6),dtype=np.int),columns=['Label (with calcs on held out 40 pct)','AUC','TP','FP','TN','FN'])

	resultrow=0
	for i in range(0,N_LABELS):
		PROCEED=True;
		#need to make sure we don't have an invalid setting -- ie, a train[x] set of labels that is uniform, else Lasso regression fails
		if(len(set(eligible_outcomes_aligned.ix[train,i].tolist())))==1:
			PROCEED=False;
			raise ValueError ("fed label to lasso regression with no variation - cannot compute - investigate")
 
		if(PROCEED):
			
			for model_type in model_types:
				if(model_type=="Lasso"):
					parameters = { "penalty": ['l1'], 
								   "C": [64,32,16,8,4,2,1,0.5,0.25,0.1,0.05,0.025,0.01,0.005]
								 }
					
					cv = StratifiedKFold(n_splits=5)
					grid_search = GridSearchCV(LogisticRegression(), param_grid=parameters, scoring='neg_log_loss', cv=cv)
					grid_search.fit(predictor_matrix[train,:],np.array(eligible_outcomes_aligned.ix[train,i]))				
					best_parameters0 = grid_search.best_estimator_.get_params()
					
					model0 = LogisticRegression(**best_parameters0)					

				model0.fit(predictor_matrix[np.array(train),:],eligible_outcomes_aligned.ix[train,i])
				pred0=model0.predict_proba(predictor_matrix[np.array(test),:])[:,1]
				coef = pd.concat([ pd.DataFrame(header),pd.DataFrame(np.transpose(model0.coef_))], axis = 1)	
				df0 = pd.DataFrame({'predict':pred0,'target':eligible_outcomes_aligned.ix[test,i], 'label':all_outcomes_aligned['Accession Number'][test]})
							  
				calc_auc=roc_auc_score(np.array(df0['target']),np.array(df0['predict']))
				if(i%10==0):
					print("i="+str(i))
				save_name=str(list(eligible_outcomes_aligned.columns.values)[i])

				target_predicted=''.join(e for e in save_name if e.isalnum())

				#confusion: outcome TP TN FP FN
				thresh = np.mean(df0['target'])
				FP=0
				FN=0
				TP=0
				TN=0
				for j in df0.index:
					cpred=df0.ix[j][1]
					ctarget = df0.ix[j][2]

					if cpred>=thresh and ctarget==1:
						TP+=1
					if cpred<thresh and ctarget==1:
						FN+=1
					if cpred>=thresh and ctarget==0:
						FP+=1
					if cpred<thresh and ctarget==0:
						TN+=1
						
				#save results		
				confusion.iloc[resultrow,0]=list(eligible_outcomes_aligned.columns.values)[i]
				confusion.iloc[resultrow,1]=calc_auc
				confusion.iloc[resultrow,2]=TP
				confusion.iloc[resultrow,3]=FP
				confusion.iloc[resultrow,4]=TN
				confusion.iloc[resultrow,5]=FN

				#let's rebuild model using all data before we save it to use for prediction;
				model0 = LogisticRegression(**best_parameters0)	
				model0.fit(predictor_matrix,eligible_outcomes_aligned.ix[:,i])
				lasso_models[i]=model0

				resultrow+=1
		
	return lasso_models, confusion


def generate_labeled_data_features(labeled_file,
					   labeled_indices,
					   DIM_DOC2VEC,
					   df_data,
					   processed_reports,
					   DO_PARAGRAPH_VECTOR,
					   DO_WORD2VEC,
					   dictionary,
					   corpus,
					   d2vmodel,
					   d2v_inputs):
	"""
	Generate numerical features to be used in Lasso logistic regressions using text data for labeled reports.
	Note: output reorganizes indices in order to align 

	Args:

		labeled_file: path to file with labels and accession ids
		labeled_indices: indices of labeled data (or data we treat as labeled because of TRAIN_INDEX_OVERRIDE)
		DIM_DOC2VEC: embedding dimensionality of doc2vec
		df_data: dataframe containing original reports and accession ids
		processed_reports: list of list of words, each entry in original list corresponding to a report
		DO_PARAGRAPH_VECTOR: use paragraph vector features?
		DO_WORD2VEC: use average word embedding features? 
		dictionary: a dictionary compromised of the LDA input n-grams
		corpus: corpus with both unlabeled and labeled data, list of lists
		d2vmodel: trained doc2vec model object 
		d2v_inputs: reports processed into d2v input format

	Returns:

		bow_matrix: numpy matrix with indicator bow features (1 if word present, 0 else), each row corresponds to a report
		pv_matrix: numpy matrix with paragraph vector embedding features, each row corresponds to a report
		w2v_matrix: numpy matrix with average word embedding features, each row corresponds to a report
		accid_list: give index into original corpus of each case; for quick debugging and spot-checking of cases
		orig_text: original text; for quick debugging and spot-checking of cases
		orig_input:original processed_report; for quick debugging and spot-checking of cases

	"""

	# #generate weight and feature matrix for held out labeled data
	outcomes = pd.read_excel(labeled_file)
	outcomes.set_index('Accession Number')
	bow_matrix = np.zeros(shape=(len(labeled_indices),len(dictionary)),dtype=np.int)

	pv_matrix = np.zeros(shape=(len(labeled_indices),DIM_DOC2VEC),dtype=np.float64)
	w2v_matrix = np.zeros(shape=(len(labeled_indices),DIM_DOC2VEC),dtype=np.float64)
	accid_list = []
	orig_text=[]
	orig_input=[]

	j=0
	print("generating features")
	for i in tqdm(labeled_indices): 
		if df_data['Accession Number'].iloc[i] not in list(outcomes['Accession Number']):
			raise Exception(" df_data i @ " + str(i) +" = " +str(df_data['Accession Number'].iloc[i])+ " not in set of held out cases, examine" )
		accid_list.append(df_data['Accession Number'].iloc[i])
		orig_text.append(df_data['Report Text'].iloc[i])
		orig_input.append(processed_reports[i])

		#fill feature columns - if ngram shows up in the document, mark it as 1, else leave as 0
		for k in range(0,len(corpus[i])):
			bow_matrix[j][corpus[i][k][0]]=1	
		
		if(DO_PARAGRAPH_VECTOR):
			vect = d2vmodel.infer_vector(d2v_inputs[i],alpha=0.01, steps=50)
	
			for k in range(0,len(vect)):
				pv_matrix[j,k]=vect[k]
		
		if(DO_WORD2VEC):
		
			#we want to use vectors based on word average:
			temp_avg =np.zeros(shape=(DIM_DOC2VEC),dtype=np.float64)
			m_avg=0
			real_words=0
			for k in range(0,len(d2v_inputs[i])):
				
				#ignore special end character, otherwise proceed
				if(d2v_inputs[i][k].lower()!="sentenceend"):
					real_words+=0
					try:
						#if it can't find the word, zero it out
						weight_avg = 1					
						temp_avg = np.add(temp_avg,weight_avg*d2vmodel[d2v_inputs[i][k]])					
						m_avg +=weight_avg
					except:
						pass # do nothing
			
			if(real_words>0): temp_avg = np.divide(temp_avg,m_avg) #if vector was empty, just leave it zero

			for k in range(0,DIM_DOC2VEC):
				w2v_matrix[j,k]=temp_avg[k]		
		
		j+=1
	return bow_matrix, pv_matrix,w2v_matrix,accid_list,orig_text,orig_input

def generate_wholeset_features(DIM_DOC2VEC,
					  processed_reports,
					   DO_PARAGRAPH_VECTOR,
					   DO_WORD2VEC,
					   dictionary,
					   corpus,
					   d2vmodel,
					   d2v_inputs):
	"""
	Generate numerical features to be used in Lasso logistic regressions using text data for all reports (labeled and unlabeled)

	Args:

		DIM_DOC2VEC: embedding dimensionality of doc2vec
		processed_reports: list of list of words, each entry in original list corresponding to a report
		DO_PARAGRAPH_VECTOR: use paragraph vector features?
		DO_WORD2VEC: use average word embedding features? 
		dictionary: a dictionary compromised of the LDA input n-grams
		corpus: corpus with both unlabeled and labeled data, list of lists
		d2vmodel: trained doc2vec model object 
		d2v_inputs: reports processed into d2v input format

	Returns:

		bow_matrix: numpy matrix with indicator bow features (1 if word present, 0 else), each row corresponds to a report
		pv_matrix: numpy matrix with paragraph vector embedding features, each row corresponds to a report
		w2v_matrix: numpy matrix with average word embedding features, each row corresponds to a report
	"""
	
	bow_matrix = np.zeros(shape=(len(corpus),len(dictionary)),dtype=np.int)
	pv_matrix = np.zeros(shape=(len(corpus),DIM_DOC2VEC),dtype=np.float64)
	w2v_matrix = np.zeros(shape=(len(corpus),DIM_DOC2VEC),dtype=np.float64)

	j=0
	for i in tqdm(range(0,len(corpus))): 

		#fill feature columns - if ngram shows up in the document, mark it as 1, else leave as 0
		for k in range(0,len(corpus[i])):
			bow_matrix[j][corpus[i][k][0]]=1	
		
		if(DO_PARAGRAPH_VECTOR):
			vect = d2vmodel.infer_vector(d2v_inputs[i],alpha=0.01, steps=50)
	
			for k in range(0,len(vect)):
				pv_matrix[j,k]=vect[k]
		
		if(DO_WORD2VEC):
		
			#we want to use vectors based on word average:
			temp_avg =np.zeros(shape=(DIM_DOC2VEC),dtype=np.float64)
			m_avg=0
			real_words=0
			for k in range(0,len(d2v_inputs[i])):
				
				#ignore special end character, otherwise proceed
				if(d2v_inputs[i][k].lower()!="sentenceend"):
					real_words+=0
					try:
						#if it can't find the word, zero it out
						weight_avg = 1					
						temp_avg = np.add(temp_avg,weight_avg*d2vmodel[d2v_inputs[i][k]])					
						m_avg +=weight_avg
					except:
						pass # do nothing
			
			if(real_words>0): temp_avg = np.divide(temp_avg,m_avg) #if vector was empty, just leave it zero

			for k in range(0,DIM_DOC2VEC):
				w2v_matrix[j,k]=temp_avg[k]		
		
		j+=1
	return bow_matrix,pv_matrix,w2v_matrix


def generate_outcomes(labeled_file,accid_list,N_THRESH_OUTCOMES):
	"""
	Generate dataframe of labels to be used in Lasso logistic regressions

	Args:
		labeled_file: path to file with labels and accession ids
		accid_list: list of accession ids of each row in the labeled data that are also present in exported reports; 
				 	needed to eliminate labeled reports for which we have no text (mistranscribed accession IDs, etc.)
		N_THRESH_OUTCOMES: eliminate outcomes that don't have this many positive / negative examples

	Returns:

		eligible_outcomes_aligned: dataframe of labels eligible for prediction
		all_outcomes_aligned: dataframe of all labels
		N_LABELS: total number of labels we predict
		outcome_header_list: list of headers corresponding to each label
	"""

	outcomes = pd.read_excel(labeled_file)
	outcomes.set_index('Accession Number')
	outcomes_aligned2 = pd.DataFrame(data=accid_list, index=accid_list, columns=['Accession Number'])
	all_outcomes_aligned = pd.merge(outcomes_aligned2, outcomes, sort=False)

	#modify outcome matrix to only include outcomes with n_thresh_outcomes +/- observations

	outcome_remove=[]
	N_LABELS=all_outcomes_aligned.shape[1]
	print("total labels:"+str(N_LABELS))
	for i in range(0,N_LABELS):
		check=sum(all_outcomes_aligned.iloc[:,i])

		if(check<N_THRESH_OUTCOMES):
			outcome_remove.append(i)
		elif(check>((all_outcomes_aligned.shape)[0]-N_THRESH_OUTCOMES)):
			outcome_remove.append(i)
		elif(math.isnan(check)):
			outcome_remove.append(i)

	eligible_outcomes_aligned=all_outcomes_aligned.drop(all_outcomes_aligned.columns[outcome_remove],axis=1)

	N_LABELS=eligible_outcomes_aligned.shape[1]
	print("labels eligible for inference:"+str(N_LABELS))

	outcome_header_list=list(eligible_outcomes_aligned)
	outcome_header_list=[x.replace(",",".") for x in outcome_header_list]
	outcome_header_list=",".join(outcome_header_list)
	
	return eligible_outcomes_aligned,all_outcomes_aligned, N_LABELS, outcome_header_list


def write_silver_standard_labels(corpus,
								N_LABELS,
								eligible_outcomes_aligned,
								DIM_DOC2VEC,
								processed_reports,
								DO_BOW,
								DO_PARAGRAPH_VECTOR,
								DO_WORD2VEC,
								dictionary,
								d2vmodel,
								d2v_inputs, 
								lasso_models,
								accid_list, 
								labeled_indices,
								df_data, 
								SILVER_THRESHOLD):
	"""
	Generate inferred labels using trained Lasso regression models; override with hand-labeled data when available.

	Args:

		corpus: list of lists of tokens, each entry in original list corresponds to report
		N_LABELS: total labels we predict
		eligible_outcomes_aligned: dataframe of eligible labels for prediction
		DIM_DOC2VEC: embedding dimensionality of average word embedding features
		processed_reports: corpus of processed reports
		DO_BOW: include bag of words features?
		DO_PARAGRAPH_VECTOR: include paragraph vector features?
		DO_WORD2VEC: include average word embedding features?
		dictionary: dictionary mapping word to integer representation
		d2vmodel: trained doc2vec feature
		d2v_inputs: reports processed into doc2vec format
		lasso_models: list of saved Lasso logistic regression models, each index corresponds to a corresponding column in eligible_outcomes_aligned
		accid_list: list of accession ids of each row in the labeled data that are also present in exported reports
		labeled_indices: indices of labeled data (or data we treat as labeled because of TRAIN_INDEX_OVERRIDE)
		df_data: dataframe containing original reports and accession ids
		SILVER_THRESHOLD: "mean" or "fiftypct", defines threshold for converting probabilities to binary labels (mean of label vs. 50%).  
		                  note that in either case it will be overridden with true labels when available
 
	Returns:

		pred_outcome_df: dataframe containing accession ids and inferred labels

	"""
		
	pred_outcome_matrix_binary = np.zeros(shape=(len(corpus),N_LABELS),dtype=np.int)
	pred_outcome_matrix_proba = np.zeros(shape=(len(corpus),N_LABELS),dtype=np.float16)

	#we classify as true/false based on mean of predictor - note dependence on self.SILVER_THRESHOLD
	if(SILVER_THRESHOLD=="mean"):
		class_thresh = eligible_outcomes_aligned.mean(axis=0)
	elif(SILVER_THRESHOLD=="fiftypct"):
		class_thresh = [0.5]*eligible_outcomes_aligned.shape[1]

	for x in range(0,len(corpus),2000):
		#generate features for whole dataset so we can return inferred labels for deep learning on images themselves
		whole_bow_matrix,whole_pv_matrix,whole_w2v_matrix=generate_wholeset_features(
					  DIM_DOC2VEC,
					  processed_reports[x:x+2000],
					   DO_PARAGRAPH_VECTOR,
					   DO_WORD2VEC,dictionary,corpus[x:x+2000],d2vmodel,d2v_inputs[x:x+2000])
		
		#use everything available for prediction - done in chunks to avoid memory issues
		#whole_combined_matrix=np.hstack((whole_w2v_matrix,whole_bow_matrix,whole_pv_matrix))
		if(DO_BOW and DO_WORD2VEC and DO_PARAGRAPH_VECTOR): whole_combined_matrix=np.hstack((whole_bow_matrix,whole_w2v_matrix,whole_pv_matrix))

		if(DO_BOW and DO_WORD2VEC and not DO_PARAGRAPH_VECTOR): whole_combined_matrix=np.hstack((whole_bow_matrix,whole_w2v_matrix))
		if(DO_BOW and not DO_WORD2VEC and DO_PARAGRAPH_VECTOR): whole_combined_matrix=np.hstack((whole_bow_matrix,whole_pv_matrix))
		if(not DO_BOW and DO_WORD2VEC and DO_PARAGRAPH_VECTOR): whole_combined_matrix=np.hstack((whole_w2v_matrix,whole_pv_matrix))

		if(DO_BOW and not DO_WORD2VEC and not DO_PARAGRAPH_VECTOR): whole_combined_matrix=whole_bow_matrix
		if(not DO_BOW and DO_WORD2VEC and not DO_PARAGRAPH_VECTOR): whole_combined_matrix=whole_w2v_matrix
		if(not DO_BOW and not DO_WORD2VEC and DO_PARAGRAPH_VECTOR): whole_combined_matrix=whole_pv_matrix

		for i in range(0,N_LABELS):
			pred_proba=lasso_models[i].predict_proba(whole_combined_matrix)[:,1]
			pred_binary = (pred_proba > class_thresh[i]).astype(int)
			pred_outcome_matrix_proba[x:x+2000,i]=pred_proba
			pred_outcome_matrix_binary[x:x+2000,i]=pred_binary

	#generate list of accession #s for export
	accession_list = []
	for i in range(0,len(corpus)):
		accession_list.append(df_data['Accession Number'].iloc[i])

	pred_outcome_proba_df = pd.DataFrame(pred_outcome_matrix_proba, index = accession_list, columns = list(eligible_outcomes_aligned.columns.values) )
	pred_outcome_binary_df = pd.DataFrame(pred_outcome_matrix_binary, index = accession_list, columns = list(eligible_outcomes_aligned.columns.values) )
		
	#get accuracy by column

	outcome_lookup ={}
	for i in range(0,len(accid_list)):
		outcome_lookup[accid_list[i]]=i

	errors = np.zeros(shape=(N_LABELS,1),dtype=np.int)
	denom = np.zeros(shape=(N_LABELS,1),dtype=np.int)
	tp = np.zeros(shape=(N_LABELS,1),dtype=np.int)
	fp = np.zeros(shape=(N_LABELS,1),dtype=np.int)
	tn = np.zeros(shape=(N_LABELS,1),dtype=np.int)
	fn = np.zeros(shape=(N_LABELS,1),dtype=np.int)

	for i in range(0,len(corpus)):
		if i in labeled_indices: # need to evaluate
			#grab accession #
			accno = df_data['Accession Number'].iloc[i]			
			
			for k in range(0,N_LABELS):

				#does our predicted value match the true one? if not, record discrepancy  
				if(eligible_outcomes_aligned.ix[outcome_lookup[accno],k]!=pred_outcome_binary_df.iloc[i,k]):
					errors[k]+=1
				denom[k]+=1

				#set probabilistic predictions to labeled ones regardless
				pred_outcome_proba_df.iloc[i,k]=eligible_outcomes_aligned.ix[outcome_lookup[accno],k]

				#if disagreement btw pred and hand-labeled data, use hand labeled
				if(eligible_outcomes_aligned.ix[outcome_lookup[accno],k]!=pred_outcome_binary_df.iloc[i,k]):
					pred_outcome_binary_df.iloc[i,k]=eligible_outcomes_aligned.ix[outcome_lookup[accno],k]

	#print('classifier accuracy by label on all labeled data including that used to train it (process integrity check)')
	#print(str(1-(errors/denom)))

	return pred_outcome_binary_df,pred_outcome_proba_df
		
		
def give_stop_words():
	"""
	Returns list of stop words.

	Arguments:

		None

	Returns:
	
		stop_words: a list of stop words
	"""
	stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
 "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him',
 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's",
 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's",
 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
 "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while',
 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
 'yourself', 'yourselves',"dr","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

	return stop_words


class RadReportAnnotator(object):
	
	def __init__(self, report_dir_path, validation_file_path):
		"""
		Initialize RadReportAnnotator class 

		Args: 

			report_dir_path: FOLDER where reports are located in montage xls. Expects columns titled "Accession Number" and "Report Text"; can specify alternate labels in define_config()
			validation_file_path: FILE with human-labeled reports file. Expects column titled "Accession Number" as first column, every subsequent column will be interpreted as a label to be predicted.

		Returns:

			Nothing

		"""

		#USER MODIFIABLE SETTINGS - USE define_config() TO SET
		self.DO_BOW=None
		self.DO_WORD2VEC=None
		self.DO_PARAGRAPH_VECTOR=None
		self.DO_SILVER_STANDARD=None   
		self.STEM_WORDS=None
		self.N_GRAM_SIZES = None
		self.DIM_DOC2VEC = None
		self.N_THRESH_CORPUS=None
		self.N_THRESH_OUTCOMES=None
		self.TRAIN_INDEX_OVERRIDE = None
		self.SILVER_THRESHOLD=None
		self.NAME_UNID_LABELED_FILE = None
		self.NAME_UNID_REPORTS= None
		self.NAME_TEXT_REPORTS= None		


		#SETTINGS YOU WILL LIKELY WITH TO LEAVE AS IS, BUT CAN MODIFY IF NEEDED
		self.D2V_EPOCH = 20 # 20 works well, # of epochs to train D2V for
		self.W2V_DM = 1 # 1 is PV-DM, otherwise PV-DBOW
		self.W2V_WINDOW = 5 #we can try 3,5,7
		self.data_dir = report_dir_path #"Base directory for raw reports
		self.validation_file =  validation_file_path #"File containing report annotations")
		self.ASSIGNFOLD_USING_ROW=False # normally in lasso regression modeling 60% train / 40% test splits are done randomly. you can do them by row if you need consistency across runs


		#MENTIONING CLASS OBJECTS USED INTERNALLY LATER
		self.df_data=None
		self.processed_reports=None
		self.labeled_indices=None
		self.d2v_inputs=None
		self.ngram_reports =None 
		self.corpus = None 
		self.train_corpus =  None  
		self.test_corpus  = None 
		self.dictionary = None 
		self.labeled_indices = None
		self.train_docs = None # w2v
		self.test_docs = None
		self.d2vmodel = None
		self.bow_matrix = None
		self.combined = None
		self.pv_matrix = None
		self.w2v_matrix = None
		self.accid_list = None
		self.orig_text = None
		self.orig_input = None
		self.eligible_outcomes_aligned = None
		self.all_outcomes_aligned = None
		self.N_LABELS = None
		self.outcome_header_list = None
		self.lasso_models = None
		self.inferred_binary_labels = None
		self.inferred_proba_labels = None
		self.headers = None
		self.accuracy = None


	def define_config(self, DO_BOW=True, DO_WORD2VEC=False, DO_PARAGRAPH_VECTOR=False,DO_SILVER_STANDARD=True,STEM_WORDS=True,N_GRAM_SIZES=[1],DIM_DOC2VEC=200,N_THRESH_CORPUS=1,N_THRESH_OUTCOMES=1,TRAIN_INDEX_OVERRIDE=[], SILVER_THRESHOLD="mean", NAME_UNID_REPORTS="Accession Number",NAME_TEXT_REPORTS="Report Text"):
		"""
		Sets parameters for RadReportAnnotator.

		Args:

			DO_BOW: True to use indicator bag of words-based features (1 if word present in doc, 0 if not). 
			DO_WORD2VEC: True to use word2vec-based average word embedding fatures. 
			DO_PARAGRAPH_VECTOR: True to use word2vec-based paragraph vector embedding fatures. 
			DO_SILVER_STANDARD: True to infer labels for unlabeled reports.
			STEM_WORDS: True to stem words for BOW analysis; words are unstemmed in doc2vec analysis
			N_GRAM_SIZES: Which set of n-grams to use in BOW analysis: [1] = 1 grams only, [3] = 3 grams only, [1,2,3] = 1, 2, and 3- grams.
			DIM_DOC2VEC: Dimensionality of doc2vec manifold; recommend value in 50 to 400
			N_THRESH_CORPUS: ignore any n-grams that appear fewer than N times in the entire corpus
			N_THRESH_OUTCOMES: do not train models for labels that don't have at least this many positive and negative examples. 
			TRAIN_INDEX_OVERRIDE: list of accession numbers we force to be treated as unlabeled data even though they are labeled (ie, these will *not* be used in Lasso regressions). May be used if all of your reports are labeled, as some unlabeled reports are required for d2v training.
			SILVER_THRESHOLD: how to threshold probability predictions in infer_labels to get binary labels. 
			                  can be ["mean","mostlikely"]
			                  mean sets any predicted probability greater than population mean to 1, else 0; e.g., prediction 0.10 in a label with average 0.05 is set to 1
			                  mostlikely sets any predicted probability >50% to 1, otherwise 0
			                  both settings have issues, and class imbalance is a major issue in training convolutional nets.
			                  we recommend using probabilities if your model can accomodate it. 
 			NAME_UNID_REPORTS: column name of accession number / unique report id in the read-in *reports* file. provided for convenience as there may be many report files.
			NAME_TEXT_REPORTS: column name of report text in the read-in reports file. provided for convenience as there may be many report files.
		Returns:

			Nothing

		"""

		self.DO_BOW=DO_BOW #generate results for bag of words approach?
		self.DO_WORD2VEC=DO_WORD2VEC #generate resultes (tfidf and avg weight) for word2vec approach?
		self.DO_PARAGRAPH_VECTOR=DO_PARAGRAPH_VECTOR #generate results for paragraph vector approach?
		self.DO_SILVER_STANDARD=DO_SILVER_STANDARD	#generate silver standard labels?
		self.STEM_WORDS=STEM_WORDS #should we stem words for BOW, LDA analysis? (we never stem words or doc2vec/w2v analysis, see below)
		if not N_GRAM_SIZES in ([1],[2],[3],[1,2],[1,3],[1,2,3]):
			raise ValueError('Invalid N_GRAM_SIZES argument:'+str(N_GRAM_SIZES)+", please review documentation for proper format (e.g., [1])")
		self.N_GRAM_SIZES = N_GRAM_SIZES  # how many n-grams to use in BOW, LDA analyses? [1] = 1 grams only, [3] = 3 grams only, [1,2,3] = 1, 2, and 3- grams.
		self.DIM_DOC2VEC = DIM_DOC2VEC #dimensionality of doc2vec manifold
		self.N_THRESH_CORPUS=N_THRESH_CORPUS # delete any n-grams that appear fewer than N times in the entire corpus
		self.N_THRESH_OUTCOMES=N_THRESH_OUTCOMES # delete any predictors that don't have at least N-many positive and negative examples
		self.TRAIN_INDEX_OVERRIDE = TRAIN_INDEX_OVERRIDE # define a list of indices you want to force to be included as unlabeled data even though they are labeled (ie, these will *not* be used for predictions). Some unlabeled reports are required for d2v training."""
		self.SILVER_THRESHOLD=SILVER_THRESHOLD
		self.NAME_UNID_REPORTS = NAME_UNID_REPORTS  
		self.NAME_TEXT_REPORTS = NAME_TEXT_REPORTS

		if(self.DO_BOW==False and self.DO_WORD2VEC==False and self.DO_PARAGRAPH_VECTOR==False): raise ValueError("DO_BOW and DO_WORD2VEC and DO_PARAGRAPH_VECTOR cannot both be false")

	def build_corpus(self):
		"""
		Builds corpus of reports and and generates numerical features from reports for later analysis.
		Please run define_config() beforehand.

		Arguments:

			None

		Returns:

			None
		"""

		#assemble dataframe of reports
		self.df_data = join_montage_files(self.data_dir, self.NAME_UNID_REPORTS, self.NAME_TEXT_REPORTS) # build dataframe with all the report text

		#get list of stop words
		en_stop = give_stop_words()

		# preprocess report text, get list with length (# reports) and text after first round of processing. 
		# if curious to see how it works, look at processed_reports[0] to see first report.
		self.processed_reports = preprocess_data(self.df_data, en_stop, stem_words=True) 

		#determine which indices should be used for 
		self.labeled_indices = get_labeled_indices(self.df_data,self.validation_file,self.TRAIN_INDEX_OVERRIDE)

		#build n-grams of desired size, takes a list of sizes and frequency threshold as inputs
		self.ngram_reports = create_ngrams(
		self.processed_reports,
		self.labeled_indices,
		N_GRAM_SIZES=self.N_GRAM_SIZES,
		freq_threshold=self.N_THRESH_CORPUS) #now we create n-grams

		# generate inputs for doc2vec/word2vec model 
		# can see example report - d2v_inputs[0]		
		self.d2v_inputs= remove_infrequent_tokens(self.processed_reports,self.N_THRESH_CORPUS,self.labeled_indices) 

		#assemble train/test corpora and a word dict. 
		self.corpus, self.train_corpus, self.test_corpus, self.dictionary, self.labeled_indices = build_train_test_corpus(
			self.df_data,
			self.ngram_reports,
			self.validation_file,
			self.TRAIN_INDEX_OVERRIDE)

		#train doc2vec/word2vec if indicated:
		if(self.DO_WORD2VEC or self.DO_PARAGRAPH_VECTOR):
			self.train_docs, self.test_docs, self.total_train_words = build_d2v_corpora(self.df_data,self.d2v_inputs,self.labeled_indices)
			self.d2vmodel=train_d2v(self.train_docs, self.test_docs, self.D2V_EPOCH, self.DIM_DOC2VEC, self.W2V_DM, self.W2V_WINDOW, self.total_train_words)

	def infer_labels(self):
		"""
		Infers labels for unlabeled documents.
		Please run build_corpus() beforehand.

		Arguments:

			None

		Returns:

			self.inferred_labels: dataframe containing inferred labels
		"""

		#get the numerical features of text we need to train models for labels
		self.bow_matrix, self.pv_matrix,self.w2v_matrix,self.accid_list,self.orig_text,self.orig_input=generate_labeled_data_features(
							   self.validation_file,
							   self.labeled_indices,
							   self.DIM_DOC2VEC,
							   self.df_data,
							   self.processed_reports,
							   self.DO_PARAGRAPH_VECTOR,
							   self.DO_WORD2VEC, 
							   self.dictionary,
							   self.corpus,
							   self.d2vmodel,
							   self.d2v_inputs)

		#get and process labels for reports
		self.eligible_outcomes_aligned,self.all_outcomes_aligned, self.N_LABELS, self.outcome_header_list = generate_outcomes(
			self.validation_file,
			self.accid_list,
			self.N_THRESH_OUTCOMES)

		#to generate silver standard labels -- use whatever features are generated (word2vec average word embeddings, bow features, paragraph vector matrix)
		if(self.DO_BOW and self.DO_WORD2VEC and self.DO_PARAGRAPH_VECTOR): self.combined=np.hstack((self.bow_matrix,self.w2v_matrix,self.pv_matrix))

		if(self.DO_BOW and self.DO_WORD2VEC and not self.DO_PARAGRAPH_VECTOR): self.combined=np.hstack((self.bow_matrix,self.w2v_matrix))
		if(self.DO_BOW and not self.DO_WORD2VEC and self.DO_PARAGRAPH_VECTOR): self.combined=np.hstack((self.bow_matrix,self.pv_matrix))
		if(not self.DO_BOW and self.DO_WORD2VEC and self.DO_PARAGRAPH_VECTOR): self.combined=np.hstack((self.w2v_matrix,self.pv_matrix))

		if(self.DO_BOW and not self.DO_WORD2VEC and not self.DO_PARAGRAPH_VECTOR): self.combined=self.bow_matrix
		if(not self.DO_BOW and self.DO_WORD2VEC and not self.DO_PARAGRAPH_VECTOR): self.combined=self.w2v_matrix
		if(not self.DO_BOW and not self.DO_WORD2VEC and self. DO_PARAGRAPH_VECTOR): self.combined=self.pv_matrix		

		#create header for combined predictor matrix so we can interpret coefficients
		self.headers=[]
		if(self.DO_BOW): 				self.headers=self.headers + [self.dictionary[i] for i in self.dictionary]
		if(self.DO_WORD2VEC): 			self.headers=self.headers + ["W2V"+str(i) for i in range(0,self.DIM_DOC2VEC)]
		if(self.DO_PARAGRAPH_VECTOR): 	self.headers=self.headers + ["PV"+str(i) for i in range(0,self.DIM_DOC2VEC)]


		pred_type = "combined" # a label for results
		print("dimensionality of predictor matrix:"+str(self.combined.shape))

		#run lasso regressions
		self.lasso_models, self.accuracy = calc_auc(self.combined,self.eligible_outcomes_aligned,self.all_outcomes_aligned,  self.N_LABELS, pred_type, self.headers,self.ASSIGNFOLD_USING_ROW)

		#infer labels	
		self.inferred_binary_labels, self.inferred_proba_labels = write_silver_standard_labels(self.corpus,
			self.N_LABELS,
			self.eligible_outcomes_aligned,
			self.DIM_DOC2VEC,
			self.processed_reports,
			self.DO_BOW,
			self.DO_PARAGRAPH_VECTOR,
			self.DO_WORD2VEC,
			self.dictionary,
			self.d2vmodel,
			self.d2v_inputs,
			self.lasso_models,
			self.accid_list, 
			self.labeled_indices,
			self.df_data,
			self.SILVER_THRESHOLD)
		return self.inferred_binary_labels, self.inferred_proba_labels
	
if __name__ == "__main__":
	
	CTHAnnotator = RadReportAnnotator(report_dir_path="C:\\Users\\jrzec\\Desktop\\clean_nlp_code\\pseudodata\\reports",
									  validation_file_path="C:\\Users\\jrzec\\Desktop\\clean_nlp_code\\pseudodata\\labels\\labeled_reports.xlsx")

	#set arguments here - examples 
	CTHAnnotator.define_config(DO_BOW=True,
		DO_WORD2VEC=True,
		DO_PARAGRAPH_VECTOR=False,
		TRAIN_INDEX_OVERRIDE=range(0,1000), 
		N_GRAM_SIZES=[1,2,3],
		SILVER_THRESHOLD="fiftypct",
		NAME_UNID_REPORTS = "ACCID", 
		NAME_TEXT_REPORTS ="REPORT")

	#build corpus and train models
	CTHAnnotator.build_corpus()

	#infer labels
	binary_labels, proba_labels = CTHAnnotator.infer_labels()


	#examine quality of predictions
	print("\naccuracy:")
	print(CTHAnnotator.accuracy)

	#see predictions
	print("\nprobabilistic predictions:")
	print(proba_labels.head())
	print("\nbinary predictions:")
	print(binary_labels.head())

	print("\ninference finished")




