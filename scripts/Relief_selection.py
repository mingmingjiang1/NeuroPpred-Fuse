# coding=utf-8
from utils import RFE
from Baseclassifier import *
import xgboost as xgb
import numpy as np
from utils.metrics import *
from sklearn.metrics import roc_curve, auc, roc_auc_score


"""relif for svm and cv for svm                 """



"""
clf1 = xgb.XGBClassifier()
clf1.fit(np.concatenate((X_train_merge,X_train_psaac),axis=1),Y_train)

print(clf1.score(np.concatenate((X_train_merge,X_train_psaac),axis=1),Y_train))
print(clf1.score(np.concatenate((X_test_merge,X_test_psaac),axis=1),Y_test))


clf2.fit(np.concatenate((X_train_merge,X_train_psaac),axis=1)[:,chosen_features],Y_train)
print(clf2.score(np.concatenate((X_train_merge,X_train_psaac),axis=1)[:,chosen_features],Y_train))
print(clf2.score(np.concatenate((X_test_merge,X_test_psaac),axis=1)[:,chosen_features],Y_test))
"""


def predict(clf, kf_x_test, Y_test):
	pre = clf.predict(kf_x_test[:,chosen_features])
	y_prob_pre = clf.predict_proba(kf_x_test[:,chosen_features])
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

#clf = SVC()
#print(clf)
#clf.fit(np.concatenate((X_train_merge,X_train_psaac),axis=1)[:,chosen_features],Y_train)
#print(clf.score(np.concatenate((X_train_merge,X_train_psaac),axis=1)[:,chosen_features],Y_train))
#print(clf.score(np.concatenate((X_test_merge,X_test_psaac),axis=1)[:,chosen_features],Y_test))
#---------------------------------------------------------------------------------
def read_selected_feature(filename):
	f = open(filename,"rb")
	select_feature = pickle.load(f)
	f.close()
	return select_feature
chosen_features = read_selected_feature(r"utils\relief_710.txt")



def Relif_selection(X_train,Y_train):
    print("\n")
    clf = xgb.XGBClassifier()
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


        clf.fit(kf_x_train[:,chosen_features],kf_y_train)
        #scores_test.append(clf.score(kf_x_test[:,chosen_features],kf_y_test))
        result = predict(clf, kf_x_test, kf_y_test)
        scores_test.append(result)
    new = np.array(scores_test)
    avg_score_test = np.sum(new, axis = 0)/10
    print("This is validation score: %s" % (avg_score_test))
    return avg_score_test
   


"""


#val = []  
#for i in [710]:   
        #relif= Relif(5000,i)
        #_,chosen_features = relif.fit_transform(np.concatenate((X_train_merge,X_test_new),axis=0),np.concatenate((Y_train,Y_test),axis=0))
avg_score_test = Relif_selection(X_train,Y_train)
        #print("This is dimenssion: %s" %(i))           
        #val.append(avg_score_test)    
#ff = open("RF-relief_acc.txt","wb")
#pickle.dump(val,ff)
#ff.close()


    
relif= Relif(5000, 770)
_, chosen_features = relif.fit_transform(np.concatenate((X_train_merge,X_test_new),axis=0),np.concatenate((Y_train,Y_test),axis=0))
ff = open("RF-relief_770.txt","wb")
pickle.dump(chosen_features, ff)
ff.close()


"""


	














