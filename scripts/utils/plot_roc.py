import pickle 
import matplotlib.pyplot as plt

f1 = open("roc_GNB","rb")
s1 = pickle.load(f1)
f1.close()


f2 = open("roc_KNN","rb")
s2 = pickle.load(f2)
f2.close()



f3 = open("roc_SVM","rb")
s3 = pickle.load(f3)
f3.close()


f4 = open("roc_kf","rb")
s4 = pickle.load(f4)
f4.close()


f5 = open("roc_GBDT","rb")
s5 = pickle.load(f5)
f5.close()


f6 = open("roc_xgboost","rb")
s6 = pickle.load(f6)
f6.close()

f7 = open("roc_ourmodel","rb")
s7 = pickle.load(f7)
f7.close()

"""
xg = [0.808,0.807,0.862,0.840,0.876,0.843,0.898]
svm = [0.809,0.807,0.840,0.839,0.810,0.812,0.640]
rf = [0.827,0.818,0.861,0.846,0.853,0.835,0.885]
knn = [0.782,0.713,0.773,0.744,0.784,0.788,0.626]
gbdt = [0.794,0.784,0.841,0.816,0.841,0.812,0.875]
nb = [0.753,0.728,0.760,0.750,0.658,0.735,0.741]



a = ["PSAAC", "GGAP", "ASDC", "DPC", "CTD", "AAC", "Merged"]



x_14 = list(range(len(a)))
x_15 = [i+0.1 for i in x_14]
x_16 = [i+0.2 for i in x_14]
t = [i+0.25 for i in x_14]
x_17 = [i+0.3 for i in x_14]
x_18 = [i+0.4 for i in x_14]
x_19 = [i+0.5 for i in x_14]

fig = plt.figure(figsize=(15,8))
ax = plt.gca()
plt.bar(range(len(a)), xg, width = 0.1, label = "XGBoost")
plt.bar(x_15, svm, width = 0.1, label = "SVM")
plt.bar(x_16, rf, width = 0.1, label = "RF")
plt.bar(x_17, knn, width = 0.1, label = "KNN")
plt.bar(x_18, gbdt, width = 0.1, label = "GBDT")
plt.bar(x_19, nb, width = 0.1, label = "NB")
plt.ylabel("Accuracy", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("Feature descriptors", fontdict={'family' : 'Times New Roman', 'size'   : 16})
ax.set_yticklabels("%.2f" %i for i in [0,0.25,0.5,0.75,1.0]) 
plt.xticks(t, a, fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.legend(bbox_to_anchor = (1.009,1.1), loc = 1, ncol=1, prop={'family' : 'Times New Roman', 'size'   : 16})
plt.savefig("class.svg")

"""
styles = plt.style.available
plt.style.use(styles[7])

plt.figure(dpi = 200)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], color='navy', lw=2, alpha=0.9,linestyle='--')
plt.plot(s1[0],s1[1],label="NB",lw=1,alpha=0.9)
plt.plot(s2[0],s2[1],label="KNN",lw=1,alpha=0.9)
plt.plot(s3[0],s3[1],label="SVM",lw=1,alpha=0.9)
plt.plot(s4[0],s4[1],label="RF",lw=1,alpha=0.9)
plt.plot(s5[0],s5[1],label="GBDT",lw=1,alpha=0.9)
plt.plot(s6[0],s6[1],label="XGBoost",lw=1,alpha=0.9)
plt.plot(s7[0],s7[1],label="Our model",lw=1,alpha=0.9)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.legend(bbox_to_anchor = (1.007,1.08), loc = 1, ncol=7, prop={'family' : 'Times New Roman', 'size'   : 6})
plt.xlabel("FPR", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("TPR", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.savefig("roc.svg")

