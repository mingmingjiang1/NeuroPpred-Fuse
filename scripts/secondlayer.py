"""second layer: rf, xgboost, mGBDT,  based on this three classifier and two feataure selections"""
random_seed = 0
from firstlayer import *
from utils import metrics
from utils import RFE
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
import os.path as osp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

dir_name = os.path.dirname(os.getcwd())

des_dir = dir_name + r"\pngs"
#sys.path.insert(0, "lib") 
feature_selection_xg = ["chosen_features", "support_", "f_score", "selected_variance"]
labels = ["Relief", "Boruta", "F-score", "Variance"]
names = ["RF", "XGBoost", "GBDT"]
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed) 
colors = np.array(["red","blue"])


def read_selected_feature(filename):
	f = open(filename,"rb")
	select_feature = pickle.load(f)
	f.close()
	return select_feature
	


"""Relief"""
chosen_features = read_selected_feature(dir_name + r"\scripts\utils\relief_710.txt")


"""Boruta"""
support_ = read_selected_feature(dir_name + r"\scripts\utils\Botuta_76.txt")


"""Variance"""
selected_variance = read_selected_feature(dir_name + r"\scripts\utils\variance_1140.txt")

"""F-score"""
f_score = read_selected_feature(dir_name + r"\scripts\utils\f_1200.txt")


train_score = []

def cv_for_rf(mode, fs, x, y, fs_str, x1, y1):
	i = 0
	score = []
	index = np.array([False]*3880)
	logits_train = np.empty((3880, 2))
	logits_test = np.empty((10, x1.shape[0], 2))
	for train_index, test_index in kf.split(x,y):
		clf = RandomForestClassifier()
		index[train_index] = True
		train_set, train_set_y = x[index], y[index]#训练集索引对应的原始序列
		test_set, test_set_y = x[~index], y[~index]#验证集索引对应的原始序列
		kf_x_train = np.array(PSAAC(train_set))#编码
		kf_x_test = np.array(PSAAC(test_set))#编码
		x_train = np.concatenate((X_train_new[index], kf_x_train),axis=1)#合并训练集特征
		x_test = np.concatenate((X_train_new[~index], kf_x_test), axis=1)#合并验证集特征
		
		clf.fit(x_train[:, fs], train_set_y)#训练
		logits_train[~index] = clf.predict_proba(x_test[:, fs])#在验证集上做预测
		score.append(clf.score(x_test[:, fs], test_set_y))
		logits_test[i] = clf.predict_proba(x1[:, fs])#在测试集上做预测
		if(mode == 'train'):
			print("=================================This is test result on Random forest:")
			print(clf.score(x1[:, fs], y1))
		index[train_index] = False
		i += 1
	#with open ("rf_{}.txt".format(fs_str), 'wb') as f: 
		#pickle.dump([logits_train, logits_test], f) 
	s = sum(score)/10
	train_score.append(s)
	return logits_train, logits_test


def cv_for_xgboost(mode, fs, x, y, fs_str, x1, y1):
	i = 0
	score = []
	index = np.array([False]*3880)
	logits_train = np.empty((3880, 2))
	logits_test = np.empty((10, x1.shape[0], 2))	
	for train_index, test_index in kf.split(x,y):
		clf = xgb.XGBClassifier()
		index[train_index] = True
		train_set, train_set_y = x[index], y[index]#训练集索引对应的原始序列
		test_set, test_set_y = x[~index], y[~index]#验证集索引对应的原始序列
		kf_x_train = np.array(PSAAC(train_set))#编码
		kf_x_test = np.array(PSAAC(test_set))#编码
		x_train = np.concatenate((X_train_new[index], kf_x_train),axis=1)#合并训练集特征
		x_test = np.concatenate((X_train_new[~index], kf_x_test), axis=1)#合并验证集特征
			
		clf.fit(x_train[:,fs],train_set_y)#训练	
		logits_train[~index] = clf.predict_proba(x_test[:,fs])#在验证集上做预测
		score.append(clf.score(x_test[:, fs], test_set_y))
		logits_test[i] = clf.predict_proba(x1[:,fs])#在测试集上做预测	
		if(mode == 'train'):
			print("=================================This is test result on XGBoost:")
			print(clf.score(x1[:, fs], y1))
		index[train_index] = False
		i += 1
	#with open ("xgboost_{}.txt".format(fs_str), 'wb') as f: 
		#pickle.dump([logits_train, logits_test], f)
	s = sum(score)/10 
	train_score.append(s)
	return logits_train, logits_test



def cv_for_gbdt(mode, fs, x, y, fs_str, x1, y1):
	i = 0
	score = []
	index = np.array([False]*3880)
	logits_train = np.empty((3880, 2))
	logits_test = np.empty((10, x1.shape[0], 2))	
	for train_index, test_index in kf.split(x,y):
		clf = GradientBoostingClassifier()
		index[train_index] = True
		train_set, train_set_y = x[index], y[index]#训练集索引对应的原始序列
		test_set, test_set_y = x[~index], y[~index]#验证集索引对应的原始序列
		kf_x_train = np.array(PSAAC(train_set))#编码
		kf_x_test = np.array(PSAAC(test_set))#编码
		x_train = np.concatenate((X_train_new[index], kf_x_train),axis=1)#合并训练集特征
		x_test = np.concatenate((X_train_new[~index], kf_x_test), axis=1)#合并验证集特征
		
		clf.fit(x_train[:,fs],train_set_y)#训练	
		logits_train[~index] = clf.predict_proba(x_test[:,fs])#在验证集上做预测
		score.append(clf.score(x_test[:, fs], test_set_y))
		logits_test[i] = clf.predict_proba(x1[:,fs])#在测试集上做预测	
		if(mode == 'train'):
			print("=================================This is test result on GBDT:")
			print(clf.score(x1[:,fs], y1))
		index[train_index] = False
		i += 1
	#with open ("gbdt_{}.txt".format(fs_str), 'wb') as f: 
		#pickle.dump([logits_train, logits_test], f) 
	s = sum(score)/10
	train_score.append(s)
	return logits_train, logits_test


def first_layer(mode, clf, x, y, res, x1, y1):
	if(clf == "rf"):
		for i in feature_selection_xg:
			if("rf" in res):
				res["rf"].append(cv_for_rf(mode, eval(i), X_train, Y_train, i, x1, y1))
			else:
				res["rf"] = [cv_for_rf(mode, eval(i), X_train, Y_train, i, x1, y1)]
	if(clf == "xgboost"):
		for i in feature_selection_xg:
			if("xgboost" in res):
				res["xgboost"].append(cv_for_xgboost(mode, eval(i), X_train, Y_train, i, x1, y1))
			else:
				res["xgboost"] = [cv_for_xgboost(mode, eval(i), X_train, Y_train, i, x1, y1)]
	if(clf == "gbdt"):
		for i in feature_selection_xg:
			if("gbdt" in res):
				res["gbdt"].append(cv_for_gbdt(mode, eval(i), X_train, Y_train, i, x1, y1))
			else:
				res["gbdt"] = [cv_for_gbdt(mode, eval(i), X_train, Y_train, i, x1, y1)]
	return res
			
		
#实际一共有6个分类器	
def second_layer(mode, y, x1, y1 = None):
	model=LogisticRegression()
	train = np.empty(shape=[3880, 0])
	test = np.empty(shape=[10, x1.shape[0], 0])
	res = {}
	for clf in ["rf", "gbdt", "xgboost"]:
		res = first_layer(mode, clf, X_train, Y_train, res, x1, y1)
	#with open ("first_layer_4_feature.txt", 'wb') as f: 
		#pickle.dump(res, f)
	for key, values in res.items():
		for i in values:#遍历每种特征选择方式,实际是遍历列表，i是元组，每次是添加一个 970*1,#元组第一个值是训练值，第二个值是预测值
			train = np.append(train, i[0], axis = 1)#每个都是3880*2
			test = np.append(test, i[1], axis = 2)#每个都是10*970*2
	model.fit(X=train,y=y)
	print(model.score(np.sum(test, axis = 0)/10, Y_test))#查看测试集准确率
	Y_pre = model.predict(np.sum(test, axis = 0)/10)
	Y_pre = list(map(int, Y_pre))
	Y_prob_pre = model.predict_proba(np.sum(test, axis = 0)/10)#查看测试集预测概率
	y_0=list(Y_prob_pre[:,1])
	return Y_pre, y_0
	
Y_test = Y_test.astype(np.int64)
Y_train = Y_train.astype(np.int64)

def predict(mode, y, x1, y1 = None):
	pre, y_prob_pre = second_layer(mode, y, x1, y1)
	acc, confusion_matrix, sensitivity, specificity, mcc = metrics.calculate_confusion_matrix(Y_test, pre)
	print("acc: {}".format(acc))
	print("sensitivity: {}".format(sensitivity))
	print("specificity: {}".format(specificity))
	print("mcc: {}".format(mcc))
	print(confusion_matrix)
	auc_score= roc_auc_score(Y_test, y_prob_pre)
	print("auc: {}".format(auc_score))
	print("score on training set:", sum(train_score)/12)
	fpr,tpr,threshold = roc_curve(Y_test, y_prob_pre)
	f = open("roc_ourmodel","wb")
	pickle.dump([fpr,tpr,threshold],f)
	f.close()
	return y_prob_pre




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", help="train or test")
	parser.add_argument("--filename", help="the file name of first layer output")
	args = parser.parse_args()
	if(args.mode == 'train'):
		predict('train', Y_train, X_test_new, Y_test)
	if(args.mode == 'test'):
		predict('train', Y_train, args.filename, None)




"""


	
#3-4: 0.905
#3-2: 0.903
#2-2: 0.903
#2-4: 0.906

def plot(clf, x, y, n, x1, y1):
	res = {}
	res = first_layer(clf, x, y, res, x1, y1)
	m = 0
	for i in range(len(feature_selection)):
		save_path = osp.join(des_dir, "pred_{0}_{1}.svg".format(clf, i))
		print(res[clf][i][0].shape)
		plot2d(res[clf][i][0], color = colors[Y_train], save_path = save_path, clf = names[n], fs = labels[m])		
		m += 1
plot("xgboost", X_train, Y_train, 0, X_test_new, Y_test)	



score = []
unique_id = []
unique_score = []


#特征重要性排序, 4种编码，3种分类器都需要特征排序
def important_feature(clf, fs, x, y):
	if(clf=="xgboost"):
		clf = xgb.XGBClassifier()
		clf.fit(x[:, fs], y)
		sorted_idx = clf.feature_importances_.argsort()
		print(np.array(fs)[sorted_idx[-10:]])
		print(clf.feature_importances_[sorted_idx[-10:]])
		fig = plt.figure(figsize=(6,6))
		plt.barh([str(i) for i in np.array(fs)[sorted_idx[-10:]]], clf.feature_importances_[sorted_idx[-10:]], color = "#E64B35B2", label = "RF-Variance")
		plt.xlabel("Feature importance score", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.ylabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.xticks(fontproperties = 'Times New Roman', size = 14)
		plt.yticks(fontproperties = 'Times New Roman', size = 14)
		plt.legend(loc= 0)
		plt.savefig("rf-variance.svg", dpi = 200)
		#plt.show()
		
		
		
		#score.extend(np.array(fs)[sorted_idx[-10:]])	
		#for i in range(len(sorted_idx[-10:])):
			#if(np.array(fs)[sorted_idx[-10:][i]] not in unique_id):
				#unique_id.append(np.array(fs)[sorted_idx[-10:][i]])
				#unique_score.append(clf.feature_importances_[sorted_idx][-10:][i])
		#label = [str(i) for i in np.array(fs)[sorted_idx[-10:]]]
	if(clf=="rf"):
		clf = RandomForestClassifier()
		clf.fit(x[:, fs], y)
		sorted_idx = clf.feature_importances_.argsort()
		#score.extend(np.array(fs)[sorted_idx[-10:]])
		fig = plt.figure(figsize=(6,6))
		plt.barh([str(i) for i in np.array(fs)[sorted_idx[-10:]]], clf.feature_importances_[sorted_idx[-10:]], color = "#E64B35B2", label = "RF-Variance")
		plt.xlabel("Feature importance score", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.ylabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.xticks(fontproperties = 'Times New Roman', size = 14)
		plt.yticks(fontproperties = 'Times New Roman', size = 14)
		plt.legend(loc= 0)
		plt.savefig("rf-variance.svg", dpi = 200)		
		
		
		#for i in range(len(sorted_idx[-10:])):
			#if(np.array(fs)[sorted_idx[-10:][i]] not in unique_id):
				#unique_id.append(np.array(fs)[sorted_idx[-10:][i]])
				#unique_score.append(clf.feature_importances_[sorted_idx][-10:][i])
		#label = [str(i) for i in np.array(fs)[sorted_idx[-10:]]]
	if(clf=="gbdt"):
		clf = GradientBoostingClassifier()
		clf.fit(x[:, fs], y)
		sorted_idx = clf.feature_importances_.argsort()
		#score.extend(np.array(fs)[sorted_idx[-10:]])
		fig = plt.figure(figsize=(6,6))
		plt.barh([str(i) for i in np.array(fs)[sorted_idx[-10:]]], clf.feature_importances_[sorted_idx[-10:]], color = "#E64B35B2", label = "GBDT-Variance")
		plt.xlabel("Feature importance score", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.ylabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 16})
		plt.xticks(fontproperties = 'Times New Roman', size = 14)
		plt.yticks(fontproperties = 'Times New Roman', size = 14)
		plt.legend(loc= 0)
		plt.savefig("GBDT-variance.svg", dpi = 200)	
		#for i in range(len(sorted_idx[-10:])):
			#if(np.array(fs)[sorted_idx[-10:][i]] not in unique_id):
				#unique_id.append(np.array(fs)[sorted_idx[-10:][i]])
				#unique_score.append(clf.feature_importances_[sorted_idx][-10:][i])
		#label = [str(i) for i in np.array(fs)[sorted_idx[-10:]]]
	
	return unique_id, unique_score, score

#, "selected_variance"
#important_feature("gbdt", eval("f_score"), X_train_merge, Y_train)




for clf in ["xgboost", "rf"]:
	for i in feature_selection:
		important_feature(clf, eval(i), X_train_merge, Y_train)





unique_id = np.array(unique_id)
unique_score = np.array(unique_score)



idx = np.argsort(np.array(unique_score))
print(idx)
unique_id = unique_id[idx]
label = [str(i) for i in unique_id]	


fig = plt.figure(figsize=(15,8))
plt.barh(label, unique_score[idx], color = "peru")
plt.xlabel("Feature importance score", fontdict={'family' : 'Times New Roman', 'size' : 16})
plt.ylabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 16})
#plt.tick_params(axis='both',which='major',labelsize=7)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.savefig("1.svg", dpi = 200)

	

	

score = set(score)#去除重复

res = [0] * 6
for j in score:
	if(j in range(0,400)): res[0] += 1
	if(j in range(400,800)): res[1] += 1
	if(j in range(800,820)): res[2] += 1
	if(j in range(820, 1220)):  res[3] += 1
	if(j in range(1220,1367)): res[4] += 1
	if(j in range(1367,1407)): res[5] += 1



res[0] = (res[0]/ 400)
res[1] = (res[1] / 400)
res[2] = (res[2]/ 20)
res[3] = (res[3] / 400)
res[4] = (res[4] / 147)
res[5] = (res[5] / 40)
		

fig = plt.figure(figsize=(15,8))
label = ["GGAP", "ASDC", "AAC", "DPC", "CTD", "PSAAC"]
plt.bar(x = label, height = res, color = "peru")
plt.xlabel("Dimension of Selected Feature", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("Percentage of Feature Category", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.savefig("2.svg",dpi= 200)
#plt.savefig("rf-xg.jpg")



for i in ["xgboost", "rf", "gbdt"]:
	score = []
	unique_id = []
	unique_score = []
	for i in feature_selection:
		important_feature(clf, eval(i), X_train_merge, Y_train, score, unique_id, unique_score)
"""	





	
	




