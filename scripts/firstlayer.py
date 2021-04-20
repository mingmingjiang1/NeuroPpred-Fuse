import os
from itertools import *
import numpy as np
import pickle
from utils.Read_fasta import readFasta
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from collections import Counter
import math
import re
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import argparse

dir_name = os.path.dirname(os.getcwd())


file_train_pos = dir_name + r"\dataset\Pos_train_fasta.txt"
file_train_neg = dir_name + r"\dataset\Neg_train_fasta.txt"
file_test_pos = dir_name + r"\dataset\Pos_test_fasta.txt"
file_test_neg = dir_name + r"\dataset\Neg_test_fasta.txt"


letters = list('ACDEFGHIKLMNPQRSTVWY')
Amino_acids = ['A','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
'R', 'S','T','V', 'W','Y']

Amino_acids_map = {'A':0,'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
'R':14, 'S':15,'T':16,'V':17, 'W':18,'Y':19}


Amino_acids_ = list(product(Amino_acids,Amino_acids))
Amino_acids_ = [i[0]+i[1] for i in Amino_acids_]
"""GGAP"""
def GGAP(seqs):
	seqs_ = []
	for seq in seqs:
		GGAP_feature = []
		num = 0
		for i in range(len(seq)-3):
			GGAP_feature.append((seq[i]+seq[i+3]))
			
		seqs_.append([GGAP_feature.count(i)/(len(seq)-3) for i in Amino_acids_])
			
		
	return seqs_

"""AAC"""
def AAC(seqs):
	seqs_ = []
	for seq in seqs:
		#for i in seq:
		AAC_feature = []
		for i in Amino_acids:
			AAC_feature.append(seq.count(i)/len(seq))
			#AAC_feature.count(i)/len(seq)
		seqs_.append(AAC_feature)
	return seqs_

"""DPC"""
def DPC(seqs):
	seqs_ = []
	for seq in seqs:
		DPC_feature = []
		for i in range(0,len(seq)-1):
			DPC_feature.append(seq[i:i+2])
		seqs_.append([DPC_feature.count(i)/len(DPC_feature) for i in Amino_acids_])
	return seqs_
	
	
"""CTD"""
group1 = {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY',
		  'polarizability': 'GASDT', 'charge': 'KR', 'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'}
group2 = {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS',
		  'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV', 'secondarystruct': 'VIYCWFT',
		  'solventaccess': 'RKQEND'}
group3 = {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED',
		  'polarizability': 'KMHFRYW', 'charge': 'DE', 'secondarystruct': 'GNPSD', 'solventaccess': 'MSPTHY'}
groups = [group1, group2, group3]
propertys = ('hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge', 'secondarystruct',
			 'solventaccess')

def Count_C(sequence1, sequence2):
	sum = 0
	for aa in sequence1:
		sum = sum + sequence2.count(aa)
	return sum


def Count_D(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]
	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence))
					break
		if myCount == 0:
			code.append(0)
	return code


def CTD(seqs):
	encodings = []
	for seq in seqs:
		code = []
		code2 = []
		CTDD1 = []
		CTDD2 = []
		CTDD3 = []
		aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
		for p in propertys:
			c1 = Count_C(group1[p], seq) / len(seq)
			c2 = Count_C(group2[p], seq) / len(seq)
			c3 = 1 - c1 - c2
			code = code + [c1, c2, c3]

			c1221, c1331, c2332 = 0, 0, 0
			for pair in aaPair:
				if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
					c1221 = c1221 + 1
					continue
				if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
					c1331 = c1331 + 1
					continue
				if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
					c2332 = c2332 + 1
			code2 = code2 + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
			CTDD1 = CTDD1 + [value / float(len(seq)) for value in Count_D(group1[p], seq)]
			CTDD2 = CTDD2 + [value / float(len(seq)) for value in Count_D(group2[p], seq)]
			CTDD3 = CTDD3 + [value / float(len(seq)) for value in Count_D(group3[p], seq)]
		encodings.append(code + code2 + CTDD1 + CTDD2 + CTDD3)
	return encodings   

"""ASDC"""

def ASDC(seqs):
	
	seqs_ = []
	
	for seq in seqs:
		ASDC_feature = []
		skip = 0
		for i in range(len(seq)):
		   
			ASDC_feature.extend(Skip(seq,skip)) 
			skip+=1
		seqs_.append([ASDC_feature.count(i)/len(ASDC_feature) for i in Amino_acids_])
	return seqs_
  
  
def Skip(seq,skip):
	
	element = []
	for i in range(len(seq)-skip-1):
		element.append(seq[i]+seq[i+skip+1])
	return element
		 
			
"""PSAAC"""
def PSAAC(seqs):
	seqs_ = []
	PSAAC_profile_forward = []
	PSAAC_profile_backward = []
	forward_seq = []
	backward_seq = []
	i = 0
	for seq in seqs:
		forward_seq.append(list(seq[:5]))
		backward_seq.append(list(seq[-5:]))
		
	for position in range(5):
		PSAAC_profile_forward.append([list(np.array(forward_seq)[:,position]).count(amino)/len(seqs) for amino in Amino_acids])
	
	for position in range(5):
		PSAAC_profile_backward.append([list(np.array(backward_seq)[:,position]).count(amino)/len(seqs) for amino in Amino_acids])
	
	
	for seq in forward_seq:
		num = 0
		new_seq = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		for amino in seq:
			index_ = Amino_acids.index(amino)
			new_seq[index_] = np.array(PSAAC_profile_forward)[num,index_]
			num+=1
			
		seqs_.append(new_seq)
	
	for seq in backward_seq:
		num = 0
		new_seq = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		for amino in seq:
			index_ = Amino_acids.index(amino)
			new_seq[index_] = np.array(PSAAC_profile_backward)[num,index_]
			num+=1

		seqs_[i].extend(new_seq)
		i+=1
	return seqs_


def AAE_1(fastas):
	length = float(len(fastas))
	amino_acids = dict.fromkeys(letters, 0)
	encodings = []
	for AA in amino_acids:
		hits = [a.start() for a in list(re.finditer(AA, fastas))]
		p_prev = 0
		p_next = 1
		sum = 0
		while p_next < len(hits):
			distance = (hits[p_next] - hits[p_prev]) / length
			sum += distance * math.log(distance, 2)
			p_prev = p_next
			p_next += 1
		amino_acids[AA] = -sum
		encodings.append(amino_acids[AA])
	return encodings


def AAE(seq):
	encodings = []
	for fastas in seq:
		fastas_NT5 = "%s" % fastas[:5]
		fastas_CT5 = "%s" % fastas[-5:]
		encodings_full = AAE_1(fastas)
		encodings_CT5 = AAE_1(fastas_CT5)
		encodings_NT5 = AAE_1(fastas_NT5)
		encodings.append(encodings_full + encodings_NT5 + encodings_CT5)
	return encodings


def AAI_1(fastas):
	encodings = []
	fileAAindex1 = open(r'utils/AAindex_1.txt')
	fileAAindex2 = open(r'utils/AAindex_2.txt')
	records1 = fileAAindex1.readlines()[1:]
	records2 = fileAAindex2.readlines()[1:]
	AAindex1 = []
	AAindex2 = []
	for i in records1:
		AAindex1.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
	for i in records2:
		AAindex2.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
	index = {}
	for i in range(len(letters)):
		index[letters[i]] = i
	fastas_len = len(fastas)
	for i in range(len(AAindex1)):
		total = 0
		for j in range(fastas_len):
			temp = AAindex1[i][index[fastas[j]]]
			total = total + float(temp)
		encodings.append(total / fastas_len)
	for i in range(len(AAindex2)):
		total = 0
		for j in range(fastas_len):
			temp = AAindex2[i][index[fastas[j]]]
			total = total + float(temp)
		encodings.append(total)
	return encodings


def AAI(seqs):
	encodings = []
	for fastas in seqs:
		fastas_NT5 = "%s" % fastas[:5]
		fastas_CT5 = "%s" % fastas[-5:]

		encodings_full = AAI_1(fastas)
		encodings_CT5 = AAI_1(fastas_CT5)
		encodings_NT5 = AAI_1(fastas_NT5)
		encodings.append(encodings_full + encodings_NT5 + encodings_CT5)
	return encodings


def pre_process(filename_pos, filename_neg = None):
	seq_pos = readFasta(filename_pos)
	if(filename_neg == None):
		seqs_all = np.array(seq_pos)
		return seqs_all
	seq_neg = readFasta(filename_neg)
	seqs_all = np.concatenate((np.array(seq_pos),np.array(seq_neg)), axis=0)
	seqs_all_Y = np.concatenate((np.ones((np.array(seq_pos).shape[0]), dtype='int'), np.zeros((np.array(seq_neg).shape[0]))), axis=0)
	return seqs_all, seqs_all_Y


"""shuffle"""
def shuffle_dataset(seqs,seqs_y):
	np.random.seed(0)
	permutation = np.random.permutation(seqs.shape[0])
	seqs = seqs[permutation]
	seqs_y = seqs_y[permutation]
	return seqs,seqs_y


def Mergefeature(mode = None, seqs_all = None, seqs_all_Y = None, seqs_all_test = None, seqs_all_Y_test = None):
	if(mode == 'test'):
		X_train_psaac = X_train
		X_train_psaac_tmp = np.array(PSAAC(X_train))
		X_train_ggap = np.array(GGAP(X_train))
		X_train_asdc = np.array(ASDC(X_train))
		X_train_aac = np.array(AAC(X_train))
		X_train_dpc = np.array(DPC(X_train))
		X_train_ctd = np.array(CTD(X_train))
		X_test_new = np.concatenate((X_test_ggap, X_test_asdc, X_test_aac,X_test_dpc,X_test_ctd, X_test_aae, X_test_aai, X_test_psaac_tmp),axis=1)
		return X_test_new
		
		
	feature_list = []
	X_train,Y_train = shuffle_dataset(seqs_all,seqs_all_Y)
	X_test,Y_test = shuffle_dataset(seqs_all_test,seqs_all_Y_test)
	X_all = np.concatenate((X_train,X_test),axis=0)
	Y_all = np.concatenate((Y_train,Y_test),axis=0)
	X_train_psaac = X_train
	X_train_psaac_tmp = np.array(PSAAC(X_train))
	X_train_ggap = np.array(GGAP(X_train))
	X_train_asdc = np.array(ASDC(X_train))
	X_train_aac = np.array(AAC(X_train))
	X_train_dpc = np.array(DPC(X_train))
	X_train_ctd = np.array(CTD(X_train))
	X_train_aae = np.array(AAE(X_train))
	X_train_aai = np.array(AAI(X_train))


	X_test_psaac_tmp = np.array(PSAAC(X_test))
	X_test_ggap = np.array(GGAP(X_test))
	X_test_asdc = np.array(ASDC(X_test))
	X_test_aac = np.array(AAC(X_test))
	X_test_dpc = np.array(DPC(X_test))
	X_test_ctd = np.array(CTD(X_test))
	X_test_aae = np.array(AAE(X_test))
	X_test_aai = np.array(AAI(X_test))


	X_train_new = np.concatenate((X_train_ggap, X_train_asdc, X_train_aac, X_train_dpc, X_train_ctd), axis=1)
	X_test_new = np.concatenate((X_test_ggap, X_test_asdc, X_test_aac,X_test_dpc,X_test_ctd, X_test_psaac_tmp),axis=1)

	X_train_merge = np.concatenate((X_train_ggap, X_train_asdc, X_train_aac, X_train_dpc, X_train_ctd, X_train_psaac_tmp), axis=1)
	
	return X_train, Y_train, X_test, Y_test, X_train_new, X_train_merge, X_test_new, X_train_psaac, X_train_ggap, X_train_asdc, X_train_dpc, X_train_ctd, X_train_aac


def cv(X,Y,i):
	avg_score_valid = {}
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  
	scores_train = []
	scores_test = []
	#clf = SVC(kernel='rbf')  
	#clf = RandomForestClassifier()
	#clf = KNeighborsClassifier()
	#clf = GaussianNB()
	clf = GradientBoostingClassifier()
	print("Notation: ========================================>This is %s features:" % (i))
			
			
	for train_index, test_index in kf.split(X,Y):

		kf_x_train = X[train_index]
		kf_y_train = Y[train_index]
		kf_x_test = X[test_index]
		kf_y_test = Y[test_index]
	   
		if(i=="X_train_psaac"):
			kf_x_train = np.array(PSAAC(kf_x_train))
			kf_x_test = np.array(PSAAC(kf_x_test))
		if(i=="X_train"):
			kf_x_train = np.array(PSAAC(kf_x_train))
			kf_x_test = np.array(PSAAC(kf_x_test))
			kf_x_train = np.concatenate((X_train_new[train_index],kf_x_train),axis=1)
			kf_x_test = np.concatenate((X_train_new[test_index],kf_x_test), axis=1)
			
		clf.fit(kf_x_train,kf_y_train)		
		scores_train.append(clf.score(kf_x_train,kf_y_train))
		scores_test.append(clf.score(kf_x_test,kf_y_test))
	avg_score_train = sum(scores_train)/10
	avg_score_test = sum(scores_test)/10
	print("This is validation score: %s" % (avg_score_test))
	print("\n")
			
	return avg_score_valid



def SingleFeature():
	feature_list = ["X_train_psaac","X_train_ggap","X_train_asdc","X_train_dpc","X_train_ctd","X_train_aac","X_train_aae","X_train_aai","X_train"]
	for i in feature_list:
		result = cv(eval(i),Y_train,i)


seqs_all, seqs_all_Y = pre_process(file_train_pos, file_train_neg)
seqs_all_test, seqs_all_Y_test = pre_process(file_test_pos, file_test_neg)
X_train, Y_train, X_test, Y_test, X_train_new, X_train_merge, X_test_new, X_train_psaac, X_train_ggap, X_train_asdc, X_train_dpc, X_train_ctd, X_train_aac = Mergefeature(mode = 'train', seqs_all = seqs_all, seqs_all_Y = seqs_all_Y, seqs_all_test = seqs_all_test, seqs_all_Y_test = seqs_all_Y_test)





if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", help="train or test")
	parser.add_argument("--filename", help="the file name you want to test")
	args = parser.parse_args()
	if(args.mode == 'train'):
		seqs_all, seqs_all_Y = pre_process(file_train_pos, file_train_neg)
		seqs_all_test, seqs_all_Y_test = pre_process(file_test_pos, file_test_neg)
		X_train, Y_train, X_test, Y_test, X_train_new, X_train_merge, X_test_new, X_train_psaac,X_train_ggap, X_train_asdc, X_train_dpc, X_train_ctd, X_train_aac = Mergefeature(args.mode, seqs_all, seqs_all_Y, seqs_all_test, seqs_all_Y_test)
		SingleFeature()
	elif(args.mode == 'test'):
		seqs_all = pre_process(args.filename)
		X_test_new = Mergefeature(args.mode, seqs_all)
		with open("resultof_1layer", "w") as f:
			f.readlines(X_test_new) 
	












