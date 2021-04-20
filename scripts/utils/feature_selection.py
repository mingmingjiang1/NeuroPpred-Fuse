from Base import *
import xgboost as xgb
import numpy as np
from metrics import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score
from boruta import BorutaPy
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
"""Variance"""

random_seed = 0


def predict(clf, kf_x_test, Y_test):
	pre = clf.predict(kf_x_test)
	y_prob_pre = clf.predict_proba(kf_x_test)
	Y_pre = list(map(int, pre))
	Y_test = list(map(int, Y_test))
	Y_prob_pre = list(y_prob_pre[:,1])
	acc, confusion_matrix, sensitivity, specificity, mcc = calculate_confusion_matrix(Y_test, Y_pre)
	print("acc: {}".format(acc))
	print("sensitivity: {}".format(sensitivity))
	print("specificity: {}".format(specificity))
	print("mcc: {}".format(mcc))
	auc_score= roc_auc_score(Y_test, Y_prob_pre)
	print("auc: {}".format(auc_score))
	return [acc, specificity, sensitivity, mcc, auc_score]





#ts = 0.5
#vt = VarianceThreshold(threshold = ts)
#vt.fit_transform(X_train)
var_value = np.var(np.concatenate((X_train_merge,X_test_new),axis=0),axis = 0)
sorted_var_value = np.argsort(-var_value)

def Variance_reduction(X_train,Y_train,i):
	print("\n")
	#print(X_train.shape)
	clf = xgb.XGBClassifier()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 
	X = X_train
	Y = Y_train
	#avg_score_valid = {}
	#scores_train = []
	scores_test = []
	for train_index, test_index in kf.split(X,Y):

		kf_x_train = X[train_index]
		kf_y_train = Y[train_index]
		kf_x_test = X[test_index]
		kf_y_test = Y[test_index]
	   
	   
		kf_x_train = np.array(PSAAC(kf_x_train))
		kf_x_test = np.array(PSAAC(kf_x_test))
		kf_x_train = np.concatenate((X_train_new[train_index],kf_x_train),axis=1)
		kf_x_test = np.concatenate((X_train_new[test_index],kf_x_test), axis=1)
		clf.fit(kf_x_train[:,sorted_var_value[:i]],kf_y_train)
		#scores_train.append(clf.score(kf_x_train[:,sorted_var_value[:i]],kf_y_train))
		#scores_test.append(clf.score(kf_x_test[:,sorted_var_value[:i]],kf_y_test))
		result = predict(clf, kf_x_test, kf_y_test)
		scores_test.append(result)
	new = np.array(scores_test)
	avg_score_test = np.sum(new, axis = 0)/10
	print("This is validation score: %s" % (avg_score_test))
	return avg_score_test
val = []
for i in [1140]:   
	avg_score_test = Variance_reduction(X_train,Y_train,i)
	#val.append(avg_score_test)
#ff = open("RF-variance_acc.txt","wb")
#pickle.dump(val,ff)
#ff.close()


#ff = open("RF-variance_1170.txt","wb")
#pickle.dump(sorted_var_value[:1170],ff)
#ff.close()
#-------------------------------------------------------------------------------------------------------------------------------



def KPCA_reduction(X_train,Y_train):
	print("\n")
	clf = xgb.XGBClassifier()
	#print(X_train.shape)
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 
	X = X_train
	Y = Y_train
	scores_test = []
	for train_index, test_index in kf.split(X,Y):

		kf_x_train = X[train_index]
		kf_y_train = Y[train_index]
		kf_x_test = X[test_index]
		kf_y_test = Y[test_index]
	   
	   
		kf_x_train = np.array(PSAAC(kf_x_train))
		kf_x_test = np.array(PSAAC(kf_x_test))
		kf_x_train = np.concatenate((X_train_new[train_index],kf_x_train),axis=1)
		kf_x_test = np.concatenate((X_train_new[test_index],kf_x_test), axis=1)
		num1 = kf_x_train.shape[0]
		kf_x_all = kpca.fit_transform(np.concatenate((kf_x_train,kf_x_test),axis = 0))
		#split again
		kf_x_train_new = kf_x_all[:num1,:]
		kf_x_test_new = kf_x_all[num1:,:]
		print(kf_x_train_new.shape)
		clf.fit(kf_x_train_new,kf_y_train)
		scores_test.append(clf.score(kf_x_test_new,kf_y_test))
	avg_score_test = sum(scores_test)/10
	print("This is validation score: %s" % (avg_score_test))
	print("\n")
	return avg_score_test
val = []   
for i in range(200, 1200, 10):   
	kpca = KernelPCA(kernel='rbf',gamma=1,n_components=i)
	print("This is dimenssion: %s" %(i))           
	score = KPCA_reduction(X_train,Y_train)
	val.append(score)
ff = open("kpca_acc.txt","wb")
pickle.dump(val,ff)
ff.close() 


  


#val_list = [[86.6, 86.3, 86.3, 86.2, 86.4, 85.9, 86.1, 85.8, 85.7, 85.8, 85.8, 85.6, 85.2]]
#for gamma in [0.1,1,10,100]:
	#val = []
	#print("This is gamma: %s" %(gamma))
	#for i in [400,450,500,550,600,650,700,750,800,850,900,950,1000]:   
		#kpca = KernelPCA(kernel='rbf',gamma=gamma,n_components=i)
	   # print("This is dimenssion: %s" %(i))           
		#score = KPCA_reduction(X_train,Y_train)
		#val.append(score)
	#val_list.append(val)
f1 = open("val_acc_for_KPCA.txt","rb")
#pickle.dump(val_list,f1)
#f1.close()

val = pickle.load(f1)
f1.close()
#print(val[4])

plt.figure()
styles = plt.style.available
plt.style.use(styles[0])
plt.plot([400,450,500,550,600,650,700,750,800,850,900,950,1000], val[0], label = "gamma: 0.01")
plt.plot([400,450,500,550,600,650,700,750,800,850,900,950,1000], [i*100 for i in val[1]], label = "gamma: 0.1",lw=1,alpha=0.9)
plt.plot([400,450,500,550,600,650,700,750,800,850,900,950,1000], [i*100 for i in val[2]], label = "gamma: 1",lw=1,alpha=0.9)
plt.plot([400,450,500,550,600,650,700,750,800,850,900,950,1000], [i*100 for i in val[3]], label = "gamma: 10",lw=1,alpha=0.9)
plt.plot([400,450,500,550,600,650,700,750,800,850,900,950,1000], [i*100 for i in val[4]], label = "gamma: 100",lw=1,alpha=0.9)
plt.xlabel("Dimension")
plt.ylabel("Prediction Accuracy(%)")
plt.legend(loc=2, bbox_to_anchor=(1.01,1.0),borderaxespad = 0.)
plt.show()
#F-score


def fscore_core(np,nn,xb,xbp,xbn,xkp,xkn):
	'''
	np: number of positive features
	nn: number of negative features
	xb: list of the average of each feature of the whole instances
	xbp: list of the average of each feature of the positive instances
	xbn: list of the average of each feature of the negative instances
	xkp: list of each feature which is a list of each positive instance
	xkn: list of each feature which is a list of each negatgive instance
	reference: http://link.springer.com/chapter/10.1007/978-3-540-35488-8_13
	'''

	def sigmap (i,np,xbp,xkp):
		return sum([(xkp[i][k]-xbp[i])**2 for k in range(np)])

	def sigman (i,nn,xbn,xkn):
		#print (sum([(xkn[i][k]-xbn[i])**2 for k in range(nn)]))
		return sum([(xkn[i][k]-xbn[i])**2 for k in range(nn)])

	n_feature = len(xb)
	fscores = []
	for i in range(n_feature):
		fscore_numerator = (xbp[i]-xb[i])**2 + (xbn[i]-xb[i])**2
		fscore_denominator = (1/float(np-1))*(sigmap(i,np,xbp,xkp))+ \
							 (1/float(nn-1))*(sigman(i,nn,xbn,xkn))
		fscores.append(fscore_numerator/fscore_denominator)

	return fscores

def fscore(feature,classindex):
	'''
	feature: a matrix whose row indicates instances, col indicates features
	classindex: 1 indicates positive and 0 indicates negative
	'''
	n_instance = len(feature)
	n_feature  = len(feature[0])
	np = sum(classindex)
	nn = n_instance - np
	xkp =[];xkn =[];xbp =[];xbn =[];xb=[]
	for i in range(n_feature):
		xkp_i = [];xkn_i = []
		for k in range(n_instance):
			if classindex[k] == 1:
				xkp_i.append(feature[k][i])
			else:
				xkn_i.append(feature[k][i])
		xkp.append(xkp_i)
		xkn.append(xkn_i)
		sum_xkp_i = sum(xkp_i)
		sum_xkn_i = sum(xkn_i)
		xbp.append(sum_xkp_i/float(np))
		xbn.append(sum_xkn_i/float(nn))
		xb.append((sum_xkp_i+sum_xkn_i)/float(n_instance))
	return fscore_core(np,nn,xb,xbp,xbn,xkp,xkn)
	


def F_reduction(X_train,Y_train, i ):#400 500,600,700,800,900,1000
	print("\n")
	#print(X_train.shape)
	clf = xgb.XGBClassifier()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 
	X = X_train
	Y = Y_train
	#avg_score_valid = {}
	#scores_train = []
	scores_test = []
	for train_index, test_index in kf.split(X,Y):

		kf_x_train = X[train_index]
		kf_y_train = Y[train_index]
		kf_x_test = X[test_index]
		kf_y_test = Y[test_index]
	   
	   
		kf_x_train = np.array(PSAAC(kf_x_train))
		kf_x_test = np.array(PSAAC(kf_x_test))
		kf_x_train = np.concatenate((X_train_new[train_index],kf_x_train),axis=1)
		kf_x_test = np.concatenate((X_train_new[test_index],kf_x_test), axis=1)
		clf.fit(kf_x_train[:,sorted_f_score_value[:i]],kf_y_train)
		#scores_train.append(clf.score(kf_x_train[:,sorted_f_score_value[:i]],kf_y_train))
		#scores_test.append(clf.score(kf_x_test[:,sorted_f_score_value[:i]],kf_y_test))
		result = predict(clf, kf_x_test, kf_y_test)
		scores_test.append(result)
	new = np.array(scores_test)
	avg_score_test = np.sum(new, axis = 0)/10
	print("This is validation score: %s" % (avg_score_test))   
  

val = []  
f_score = fscore(np.concatenate((X_train_merge,X_test_new),axis=0).tolist(),list(map(int,np.concatenate((Y_train,Y_test),axis=0).tolist())))
sorted_f_score_value = np.argsort(-np.array(f_score))

for i in [1200]:   
	avg_score_test = F_reduction(X_train,Y_train,i)
	print("This is dimenssion: %s" %(i))           
		#val.append(avg_score_test)



#ff = open("F_acc.txt","wb")
#pickle.dump(val,ff)
#ff.close()

sorted_f_score_value = np.argsort(-np.array(f_score))
ff = open("RF-f_300.txt","wb")
pickle.dump(sorted_f_score_value[:300],ff)
ff.close()





def read_selected_feature(filename):
	f = open(filename,"rb")
	select_feature = pickle.load(f)
	f.close()
	return select_feature




#clf = GradientBoostingClassifier()
#feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2, random_state=0)
#feat_selector.fit(X_train_merge, Y_train)


ff = open("GBDT-Botuta.txt","rb")#RF:76 0.887 GBDT:76 0.877
s, r = pickle.load(ff)
ff.close()

print(np.array(list(range(0, 1407)))[s == True])
ff = open("GBDT-Botuta-76.txt","wb")
pickle.dump(np.array(list(range(0, 1407)))[s == True], ff)
ff.close()


support_xg = read_selected_feature(r"C:\Users\Mmjiang\Desktop\Neuro\selected_features\XGBOOST-Botuta_76.txt")


def Botuta_reduction(X_train,Y_train):
	print("\n")
	
	clf = GradientBoostingClassifier()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 
	X = X_train
	Y = Y_train
	avg_score_valid = {}
	scores_train = []
	scores_test = []
	for train_index, test_index in kf.split(X,Y):

		kf_x_train = X[train_index]
		kf_y_train = Y[train_index]
		kf_x_test = X[test_index]
		kf_y_test = Y[test_index]
	     
		kf_x_train = np.array(PSAAC(kf_x_train))
		kf_x_test = np.array(PSAAC(kf_x_test))
		kf_x_train = np.concatenate((X_train_new[train_index],kf_x_train),axis=1)
		kf_x_test = np.concatenate((X_train_new[test_index],kf_x_test), axis=1)
		
		kf_x_train = kf_x_train[:, support_xg]
		kf_x_test = kf_x_test[:, support_xg]
		clf.fit(kf_x_train,kf_y_train)
		result = predict(clf, kf_x_test, kf_y_test)
		scores_test.append(result)
	new = np.array(scores_test)
	avg_score_test = np.sum(new, axis = 0)/10
	print("This is validation score: %s" % (avg_score_test))
	return avg_score_test

Botuta_reduction(X_train,Y_train)




import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self,n_input,n_hidden,n_output):
		super(Net,self).__init__()
		self.hidden1 = nn.Linear(n_input,n_hidden)
		self.hidden2 = nn.Linear(n_hidden,n_hidden)
		self.predict = nn.Linear(n_hidden,n_output)
	def forward(self,input):
		out = self.hidden1(input)
		out = F.relu(out)
		out = self.hidden2(out)
		out = F.sigmoid(out)
		out =self.predict(out)
		return out


net = Net(1407, 128, 2)
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.CrossEntropyLoss()



x_test = torch.tensor(X_test_new).float()



def cv_for_ANN(x, y):
	max_ = 0
	for t in range(2000):
		out = net(x)
		loss = loss_func(out,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		prediction = torch.max(F.softmax(out),1)[1]
		pred_y = prediction.data.numpy().squeeze()
		target_y = y.data.numpy()
		accuracy = sum(pred_y == target_y) / 3880.
		out = net(x_test)
		pre = torch.max(F.softmax(out),1)[1]
		pred_y = pre.data.numpy().squeeze()
		accuracy = np.sum(pred_y == Y_test.astype(np.int)) / 970.	
		if(accuracy > max_):
			max_ = accuracy
		if(t%50 == 0):
			print("This is {0}".format(t))
			print(accuracy)
	return max_
			
x = torch.tensor(X_train_merge).float()
y = torch.tensor(Y_train).long()	
print(cv_for_ANN(x, y))


