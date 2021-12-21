import scipy.io as sio
import time
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

	# Load data
	dataset = 'cora'
	data = sio.loadmat('{}.mat'.format(dataset))
	if(dataset == 'large_cora'):
		X=data['X']	
		A = data['G']			
		gnd = data['labels']
		gnd = gnd[0, :]
	else:
		X = data['fea']
		A = data['W']	
		gnd = data['gnd']	
		gnd = gnd.T
		gnd = gnd - 1
		gnd = gnd[0, :]

	# Store some variables
	XtX= X.dot(X.T)
	XXt= X.T.dot(X)
	N = X.shape[0]
	k = len(np.unique(gnd))
	I = np.eye(N)
	I2 = np.eye(X.shape[1])
	if sp.issparse(X):
	    X = X.todense()

	# Normalize A
	A = A + I
	D = np.sum(A,axis=1)
	D = np.power(D,-0.5)
	D[np.isinf(D)] = 0
	D = np.diagflat(D)
	A = D.dot(A).dot(D)

	# Get filter G
	Ls = I - A
	G = I - 0.5*Ls

	# Get the Polynomials of A
	A2 = A.dot(A)
	A3 = A2.dot(A)
	A4 = A3.dot(A)
	A5 = A4.dot(A)



	# Set f(A)
	A_ = A+A2

	# Set the order of filter
	G_ = G
	kk = 1

	acc_list = []
	nmi_list = []
	f1_list = []
	nowa = []
	nowk = []
	best_acc = []
	best_nmi = []
	best_f1 = []
	best_a = []
	best_k = []

	# Set the list of alpha
	list_a = [1e-4,1e-2,1,10,100]


	print("f(A)=A+A2")

	# Set the range of filter order k 
	while(kk <= 5):

		#compute
	    X_bar = G_.dot(X)
	    XtX_bar = X_bar.dot(X_bar.T)
	    XXt_bar = X_bar.T.dot(X_bar)
	    tmp_acc = []
	    tmp_nmi = []
	    tmp_f1 = []
	    tmp_a = []

	    for a in list_a:
	        tmp = np.linalg.inv(I2 + XXt_bar/a)
	        tmp = X_bar.dot(tmp).dot((X_bar.T))
	        tmp = I/a -tmp/(a*a)
	        S = tmp.dot(a * A_ + XtX_bar)  
	        C = 0.5 * (np.fabs(S) + np.fabs(S.T))
	        print("a={}".format(a), "k={}".format(kk))
	        u, s, v = sp.linalg.svds(C, k=k, which='LM')

	        kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
	        predict_labels = kmeans.predict(u)

	        # 几个metric

	        cm = clustering_metrics(gnd, predict_labels)
	        ac, nm, f1 = cm.evaluationClusterModelFromLabel()

	        print(
	            'acc_mean: {}'.format(ac),
	            'nmi_mean: {}'.format(nm),
	            'f1_mean: {}'.format(f1),
	            'max_element :{}'.format(np.max(A_)),
	            '\n' * 2)
	        acc_list.append(ac)
	        nmi_list.append(nm)
	        f1_list.append(f1)
	        nowa.append(a)
	        nowk.append(kk)

	        tmp_acc.append(ac)
	        tmp_nmi.append(nm)
	        tmp_f1.append(f1)
	        tmp_a.append(a)
	        
	        
	#         a = a + 50
	    nxia = np.argmax(tmp_acc)
	    best_acc.append(tmp_acc[nxia])
	    best_nmi.append(tmp_nmi[nxia])
	    best_f1.append(tmp_f1[nxia])
	    best_a.append(tmp_a[nxia])
	    best_k.append(kk)
	    kk += 1
	    G_ = G_.dot(G)
	
	# all of the results
	for i in range(np.shape(acc_list)[0]):
	    print("a = {:>.6f}".format(nowa[i]),
	          "k={:>.6f}".format(nowk[i]),
	          "ac = {:>.6f}".format(acc_list[i]),
	          "nmi = {:>.6f}".format(nmi_list[i]),
	          "f1 = {:>.6f}".format(f1_list[i]))
	# the best results for each k
	for i in range(np.shape(best_acc)[0]):
	    print("for k={:>.6f}".format(best_k[i]),
	            "the best a = {:>.6f}".format(best_a[i]),
	          "ac = {:>.6f}".format(best_acc[i]),
	          "nmi = {:>.6f}".format(best_nmi[i]),
	          "f1 = {:>.6f}".format(best_f1[i]))
	    
	# the best result of all experiment
	xia = np.argmax(acc_list)
	print("the best state:")
	print("a = {:>.6f}".format(nowa[xia]),
	          "k={:>.6f}".format(nowk[xia]),
	          "ac = {:>.6f}".format(acc_list[xia]),
	          "nmi = {:>.6f}".format(nmi_list[xia]),
	          "f1 = {:>.6f}".format(f1_list[xia]))