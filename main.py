import numpy as np
import scipy.io as sio
from mielm2 import MIELM

def normMinMax(X):
	norm_X = np.matrix(np.zeros(X.shape))
	min_val = np.min(X,0)
	max_val = np.max(X,0)
	norm_X = (X - min_val)/(max_val - min_val) * 2 -1
	return norm_X

def normSigmoid(X):
	return (1 - np.exp(-X)) / (1 + np.exp(-X))

def savetxt(name, bag_ids, features, labels):
	y = np.concatenate((bag_ids.T, features.T, labels.T)).T
	np.savetxt('results/'+name+'.csv', y, delimiter=',', newline='\n')
	print(name +' saved')
	# np.savetxt(name+'.csv', y, fmt='%d' + '%.16f '*230 + '%d %d', delimiter=',', newline='\n')

#path datasets
path = '/media/prabowo/Drive/Data/Documents/University/Final Test/Datasets/MIL Datasets/'
datasets = (('elephant_100x100_matlab','elephant'),('fox_100x100_matlab', 'fox'), ('tiger_100x100_matlab', 'tiger'), ('musk1norm_matlab','musk1norm'))
dataset = datasets[0]

#load elephant dataset
temp_dataset = sio.loadmat(path+dataset[0]+'.mat')

labels = temp_dataset['labels'].toarray().T
bag_ids = temp_dataset['bag_ids'].T
features =  temp_dataset['features'].toarray()

#savetxt
savetxt(dataset[0], bag_ids, features, labels)

#preprocessing
#feature selection 
zero_column = np.append(np.nonzero(np.logical_not(np.sum(np.logical_not(features==0),0))) , np.nonzero(np.logical_not(np.max(features,0) - np.min(features,0))))
zero_column = np.sort(zero_column)
features = np.delete(features, zero_column, 1)

#savetxt
savetxt(dataset[0]+'_feature_selected', bag_ids, features, labels)

#normalization
# features = normalization(features)
# features = normMinMax(features)
features = normSigmoid(features)

#savetxt
savetxt(dataset[0]+'_normalized', bag_ids, features, labels)

num_of_bag = np.unique(bag_ids).shape[0]
num_of_instance = features.shape[0]

#encoding target of the bags
temp_labels = labels
labels = np.zeros((temp_labels.shape[0], (np.unique(temp_labels).shape[0])))
for i in range(labels.shape[0]):
	for j in range(np.unique(temp_labels).shape[0]):
		if temp_labels[i] == np.unique(temp_labels)[j]:
			break
	labels[i,j] = 1
labels = np.logical_not(labels) * 2 - 1

#grouping instances into bag
temp_bags = np.zeros((num_of_bag, 3), dtype=object)
for i in range(num_of_bag):
	temp_bags[i, 0] = features[(bag_ids == np.unique(bag_ids)[i]).nonzero()[0],:] 				#instances
	temp_bags[i, 1] = bag_ids[(bag_ids == np.unique(bag_ids)[i]).nonzero()[0],:]				#bag id
	temp_bags[i, 2] = np.mean(labels[(bag_ids == np.unique(bag_ids)[i]).nonzero()[0],:],0)		#bag label


#shuffle data per class
positif_bags = temp_bags[(np.nonzero(np.array(([lbl for lbl in temp_bags[:,2]]))[:,1] == 1)[0])]
negatif_bags = temp_bags[(np.nonzero(np.logical_not(np.array(([lbl for lbl in temp_bags[:,2]]))[:,1] == 1))[0])]

np.random.shuffle(positif_bags)
np.random.shuffle(negatif_bags)

temp_bags = np.concatenate((positif_bags, negatif_bags))

savetxt(dataset[0]+'_shuffle', np.concatenate((temp_bags[:,1])), np.concatenate((temp_bags[:,0])), labels)

#=================================
mode = ('kmean_mielm', 'mielm')
num_of_fold = 10
num_of_bag_per_category = (50,70,90,100)
num_of_sample = 30
final_result = np.zeros((len(num_of_bag_per_category), 7)) #[ num_of_bag_per_category, acurate K-Means MI-ELM, learning time K-Means MI-ELM, testing time K-Means MI-ELM, acurate MI-ELM, learning time MI-ELM, testing time MI-ELM]


#config
M = 1000
C = 2**5	#regularization coeficient
r = 7		#power level
c = 26		#num of cluster

for n, m in enumerate(num_of_bag_per_category):
	final_result[n, 0] = m*2 
	bags = np.concatenate((temp_bags[0:m], temp_bags[100:100+m]))
	dataset_name = dataset[1]+str(m)+'x'+str(m)
	mielm = MIELM(dataset_name, num_of_fold, bags, M, C, r, c)
	
	print('Dataset Info')
	print('======================================')
	print('Dataset: ', dataset_name)
	print('Num of Bag: ', bags.shape[0])
	print('Num of Instance: ', np.concatenate((bags[:,1])).shape[0])
	print("======================================\n")

	print("Kmean MI ELM")
	#mielm = MIELM(dataset, num_of_fold, bags, M, C, r, c, mode[0])
	final_result[n, 1:4] = mielm.learn(num_of_sample, mode[0])

	print("MI ELM")
	#mielm = MIELM(dataset, num_of_fold, bags, M, C, r, c, mode[1])
	final_result[n, 4:7] = mielm.learn(num_of_sample, mode[1])

# print(final_result)
np.savetxt('results/'+dataset_name+' final_result.csv', final_result, delimiter=',', newline='\n')
print('Finish')

#=================================



# temp_bags = np.concatenate((temp_bags[0:100], temp_bags[100:11]))

# print('Dataset: ', dataset)
# print('Num of Bag: ', temp_bags.shape[0])
# print('Num of Instance: ', np.concatenate((temp_bags[:,1])).shape[0])
# print()

# for M in range(5,50, 5):
# MIELM Original
# # for i in (100, 1000):
# M = 10 #1000	#hidden node
# C = 2**5	#regularization coeficient
# r = 7		#power level
# c = None		#num of cluster

# print('MIELM')
# print('Num of Hidden Node:', M)
# print('Regularization Coeficient:', C)
# print('Power Level:', r)
# mielm = MIELM(num_of_fold, temp_bags, M, C, r, c, mode[1])


# # 	# MIELM with k-means
# # for i in (10, 50, 100, 1000):
# # for j in (10, 20, 26):
# M = 10 #35	#hidden node
# C = 2**5 #2**5	#regularization coeficient
# r = None		#power level
# c = 26 #26	#num of cluster

# print('Kmean MIELM')
# print('Num of Hidden Node:', M)
# print('Regularization Coeficient:', C)
# print('Num of Cluster:', c)

# mielm = MIELM(num_of_fold, temp_bags, M, C, r, c, mode[0])
# print()
# # print()