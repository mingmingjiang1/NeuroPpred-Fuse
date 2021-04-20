from Base import *
import xgboost as xgb
import numpy as np
from metrics import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score
from matplotlib_venn import venn3






#styles = plt.style.available
#plt.style.use(styles[0])

def plot():
	f = open("kpca_acc.txt","rb")#RF:320
	val1 = pickle.load(f)
	f.close()
	f = open("relief_acc.txt","rb")
	val2 = pickle.load(f)
	f.close()
	f = open("F_acc.txt","rb")
	val3 = pickle.load(f)
	f.close()
	f = open("variance_acc.txt","rb")
	val4 = pickle.load(f)
	f.close()
	print(max(val2))
	print(val3.index(max(val3)))
	print(val4.index(max(val4)))
	
	"""
	fig = plt.figure(figsize = figsize)
	ax = plt.gca()
	#if(feature_selection == "KPCA"):
	plt.plot(list(range(200, 1200, 10)), [i*100 for i in val1], label = "kpca",lw=1,alpha=0.9, color = 'red')
	plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 16})
	plt.ylabel("Prediction Accuracy", fontdict={'family' : 'Times New Roman', 'size'   : 16})
	#plt.legend(loc=0, borderaxespad = 0., prop = font1)
		#plt.savefig("kpca.jpg")
	#if(feature_selection == "Relief"):
	plt.plot(list(range(200, 1200, 10)), [i*100 for i in val2], label = "Relief", lw=1, alpha=0.9, color = 'blue')
		#plt.xlabel("Dimension", font)
		#plt.ylabel("Prediction Accuracy(%)", font)
		#plt.legend(loc=0, borderaxespad = 0., prop = font1)	
		#plt.show()
		#plt.savefig("relief.jpg")
	#if(feature_selection == "F-score"):
	plt.plot(list(range(200, 1200, 10)), [i*100 for i in val3], label = "F-score", lw=1, alpha=0.9, color = 'green')
		#plt.xlabel("Dimension", font)
		#plt.ylabel("Prediction Accuracy(%)", font)
		#plt.legend(loc=0, borderaxespad = 0., prop = font1)
		#plt.show()
		#plt.savefig("fscore.jpg")			
	#if(feature_selection == "Variance"):
	plt.plot(list(range(200, 1200, 10)), [i*100 for i in val4], label = "Variance", lw=1, alpha=0.9, color = 'black')
		#plt.xlabel("Dimension", font)
		#plt.ylabel("Prediction Accuracy(%)", font)
		#plt.legend(loc=0, borderaxespad = 0., prop = font1)	
		#plt.show()	
	#plt.legend(loc=0, borderaxespad = 0., prop = font1)
	plt.yticks(fontproperties = 'Times New Roman', size = 14)
	plt.xticks(fontproperties = 'Times New Roman', size = 14)
	ax.set_yticklabels("%.2f" %i for i in [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91]) 
	plt.legend(bbox_to_anchor = (1.007,1.08), loc = 1, ncol=4, prop={'family' : 'Times New Roman', 'size'   : 16})
	
	plt.savefig("RF-4-feature selection.svg", dpi = 200)
	print(list(range(200, 1200, 10))[val1.index(max(val1))])
	print(list(range(200, 1200, 10))[val2.index(max(val2))])
	print(list(range(200, 1200, 10))[val3.index(max(val3))])
	print(max(val3))
	print(list(range(200, 1200, 10))[val4.index(max(val4))])
	"""


plot()	
	
	
#plot("relief_acc.txt", "Relief")
#plot("kpca_acc.txt", "KPCA")
#plot("F_acc.txt", "F-score")
#plot("variance_acc.txt", "Variance")
"""
def read_selected_feature(filename):
	f = open(filename,"rb")
	select_feature = pickle.load(f)
	f.close()
	return select_feature

f1_score = read_selected_feature("f_1200.txt")
variance = read_selected_feature("variance_1140.txt")
boruta = read_selected_feature("Botuta_76.txt")
relief = read_selected_feature("relief_710.txt")




#plt.figure(figsize = figsize)
fig = plt.figure(figsize=(15,8), dpi = 200)


#import venn
#labels = venn.get_labels([set(f1_score), set(variance), set(relief), set(boruta)], fill = ['number', 'logic', 'percent'])
#fig, ax = venn.venn4(labels, names = ("F-score", "Variance-based", "Relief", "Boruta"), dpi = 96, fontsize = 14)
#plt.style.use('seaborn-whitegrid')
#ax.set_axis_on()
#venn.draw_text(fig, ax, x = 0.25, y = 0.2, text = "number:logic(percent)", fontsize=12, ha='center', va = 'center')
plt.figure()
venn3(subsets = [set(f1_score), set(boruta), set(relief)], set_labels = ("F-score", "Boruta", "Relief"))
plt.savefig("fbr.svg",dpi=200)
plt.figure()
venn3(subsets = [set(f1_score), set(variance), set(relief)], set_labels = ("F-score", "Variance-based", "Relief"))
plt.savefig("fvr.svg",dpi=200)
plt.figure()
venn3(subsets = [set(f1_score), set(variance), set(boruta)], set_labels = ("F-score", "Variance-based", "Boruta"))
plt.savefig("fvb.svg",dpi=200)
plt.figure()
venn3(subsets = [set(variance), set(boruta), set(relief)], set_labels = ("Variance-based", "Boruta", "Relief"))
plt.savefig("vbr.svg",dpi=200)




def read_selected_feature(filename):
	f = open(filename,"rb")
	select_feature = pickle.load(f)
	f.close()
	return select_feature

f1_score = read_selected_feature("f_1200.txt")
variance = read_selected_feature("variance_1140.txt")
boruta = read_selected_feature("Botuta_76.txt")
relief = read_selected_feature("relief_710.txt")

f = ["f1_score", "variance", "boruta", "relief"]

res = {}


for i in f:
	res[i] = [0] * 6
	for j in range(len(eval(i))):
		if(eval(i)[j] in range(0,400)): res[i][0] += 1
		if(eval(i)[j] in range(400,800)): res[i][1] += 1
		if(eval(i)[j] in range(800,820)): res[i][2] += 1
		if(eval(i)[j] in range(820, 1220)):  res[i][3] += 1
		if(eval(i)[j] in range(1220,1367)): res[i][4] += 1
		if(eval(i)[j] in range(1367,1407)): res[i][5] += 1
for i in f:
	res[i][0] = np.array(res[i][0]) / 400
	res[i][1] = np.array(res[i][1]) / 400
	res[i][2] = np.array(res[i][2]) / 20
	res[i][3] = np.array(res[i][3]) / 400
	res[i][4] = np.array(res[i][4]) / 147
	res[i][5] = np.array(res[i][5]) / 40



fig = plt.figure(figsize=(15,8), dpi = 200)

fix, ax = plt.subplots(2, 2, sharex = True, sharey = True)
label = ["GGAP", "ASDC", "AAC", "DPC", "CTD", "PSAAC"]
x = range(len(label))
ax[0][0].bar(x = label, height = res["f1_score"], width = 0.3, label = "F-score", color = 'darkorange')
ax[0][0].legend(bbox_to_anchor = (1.02,1.15), loc = 1, ncol=4, prop={'family' : 'Times New Roman', 'size'   : 8})
ax[0][1].bar(x = label, height = res["variance"], width = 0.3, label = "Variance-based", color = 'darkorange')
ax[0][1].legend(bbox_to_anchor = (1.02,1.15), loc = 1, ncol=4, prop={'family' : 'Times New Roman', 'size'   : 8})
ax[1][0].bar(x = label, height = res["boruta"], width = 0.3, label = "Boruta", color = 'darkorange')
ax[1][0].legend(bbox_to_anchor = (1.02,1.15), loc = 1, ncol=4, prop={'family' : 'Times New Roman', 'size'   : 8})
ax[1][1].bar(x = label, height = res["relief"],width = 0.3, label = "Relief", color = 'darkorange')
ax[1][1].legend(bbox_to_anchor = (1.02,1.15), loc = 1, ncol=4, prop={'family' : 'Times New Roman', 'size'   : 8})

ax[1][0].set_xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 10})
ax[1][0].set_xticklabels(label, fontproperties = 'Times New Roman', size = 9)
#ax[1][0].tick_params(axis='both',which='major',labelsize=5)
ax[0][0].set_ylabel("Percentage of Feature Category", fontdict={'family' : 'Times New Roman', 'size'   : 10})
ax[0][0].set_yticklabels("%.2f" %i for i in [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


ax[1][1].set_xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size'   : 10})
ax[1][1].set_xticklabels(label, fontproperties = 'Times New Roman', size = 9)
#ax[1][1].tick_params(axis='both',which='major',labelsize=5)
ax[1][0].set_ylabel("Percentage of Feature Category", fontdict={'family' : 'Times New Roman', 'size'   : 10})
ax[1][0].set_yticklabels("%.2f" %i for i in [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
#plt.show()
plt.savefig("new.svg", dpi = 200)
"""
