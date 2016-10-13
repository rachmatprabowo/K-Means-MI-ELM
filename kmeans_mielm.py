import numpy as np
import time
from prettytable import PrettyTable
import sys as Sys

class MIELM:

	def __init__(self, dataset_name, num_of_group, dataset, num_of_hidden_node, regularization_coeficient, power_level, num_of_cluster):
		self.dataset_name = dataset_name
		self.num_of_group = num_of_group

		self.num_of_class =  np.unique(dataset[0,2]).shape[0]
		self.num_of_bag = dataset.shape[0]
		self.num_of_bag_per_class = int(self.num_of_bag / self.num_of_class)

		self.num_bag_per_group = int(self.num_of_bag / num_of_group)
		self.num_bag_per_class_of_group = int(self.num_bag_per_group / self.num_of_class)

		self.num_of_hidden_node = num_of_hidden_node # M
		self.C = regularization_coeficient
		self.r = power_level

		# if mode == 'kmean_mielm':
		self.num_of_cluster = num_of_cluster #Lc
		self.centeroids = np.matrix(np.zeros((self.num_of_cluster, self.num_of_hidden_node)))

		#create table
		# self.table = PrettyTable(['Validasi Ke-', 'Akurasi (%) Latih', 'Akurasi Uji(%)', 'Waktu Latih (s)', 'Waktu Validasi (s)'])
		self.table = PrettyTable(['Validasi Ke-', 'Akurasi Uji(%)', 'Waktu Latih (s)', 'Waktu Validasi (s)'])
		self.result_table = np.zeros((0,4), dtype=object)
		
		#reshape dataset to be sparse by class
		dataset = dataset.reshape(self.num_of_class, self.num_of_bag_per_class, 3)

		#build folds for crossing validation
		self.folds = self.buildGroups(num_of_group, dataset)
		self.num_of_features = self.folds[0,0,0,0].shape[1]
		self.weight = np.matrix(np.random.rand(self.num_of_features, self.num_of_hidden_node)) * 2-1
		self.bias = np.matrix(np.random.rand(1, num_of_hidden_node)) * 2-1

		# self.crossValidation(num_of_group, schedule, mode)

		#create schedule for crossing validation
		self.schedule = self.crossingGroups(num_of_group)

	def learn(self, num_of_sample, mode):

		samples = np.zeros((num_of_sample, 3))
		for i in range(num_of_sample):
			self.printProgress(i, num_of_sample, barLength=30)
			#cross validation
			samples[i,:] = self.crossValidation(i, self.num_of_group, self.schedule, mode)
		
		self.savetxt(self.dataset_name+'_samples_'+mode, samples)
		# print('{0:.4f},'.format(self.mean_of_result[0]), '{0:.3f},'.format(self.mean_of_result[1]), '{0:.3f}'.format(self.mean_of_result[2]))
		# result = np.concatenate((np.mean(samples,0), np.sqrt(np.sum(np.power(samples,2),0)) / samples.shape[0]-1))
		return np.mean(samples,0)

	def buildGroups(self, num_of_group, dataset):
		groups = np.zeros((num_of_group, self.num_of_class, self.num_bag_per_class_of_group, 3), dtype=object) # (k, m, Lmk)

		for k in range(num_of_group):
			for m in range(self.num_of_class):
				for n  in range(self.num_bag_per_class_of_group):
					groups[k, m, n] = dataset[m, k * self.num_bag_per_class_of_group + n]
		return groups

	def crossingGroups(self, k):
		schedule = np.array(np.zeros((k,2)), dtype=object)
		for i in range(k):
			schedule[i,0] = np.concatenate((
					np.array(range(0, k-1-i), dtype=int),
					np.array(range(k-i, k), dtype=int)
				))
			schedule[i,1] = k-1-i
		return schedule

	def crossValidation(self, num_sample, num_of_group, schedule, mode):
		#result for acurate, learning time, and testing time
		cv_result = np.zeros((num_of_group, 3))

		for k in range(num_of_group):

			self.printProgress(k, num_of_group, barLength=30)

			training_set = self.folds[schedule[k][0]].reshape((num_of_group-1) * self.num_of_class * self.num_bag_per_class_of_group, 3)
			testing_set = self.folds[schedule[k][1]].reshape(self.num_of_class * self.num_bag_per_class_of_group, 3)
			# learning_time = 99999
			# testing_time = 99999

			
			if mode == 'mielm':
				#MIELM
				#-------------------
				#training
				start = time.time() # + 
				Betha = self.trainingMIELM(training_set[:,0], training_set[:,1], training_set[:,2])
				end = time.time() # +
				learning_time = end-start # + for time measurement
				
				Y_train = self.estimateMIELM(training_set[:,0], training_set[:,1], Betha)
				T_train = np.concatenate((training_set[:,2])).reshape(self.num_of_training_bag, self.num_of_class)

				E_train = np.zeros((T_train.shape[0]))
				for i in range(T_train.shape[0]):
					if Y_train[i].argmax() != T_train[i].argmax(): E_train[i] = 1

				#testing
				start = time.time() # + 
				Y_test = self.estimateMIELM(testing_set[:,0], testing_set[:,1], Betha)
				end = time.time() # +

				T_test = np.concatenate((testing_set[:,2])).reshape(self.num_of_testing_bag, self.num_of_class)
				testing_time = end-start # + for time measurement

				E_test = np.zeros((T_test.shape[0]))
				for i in range(T_test.shape[0]):
					if Y_test[i].argmax() != T_test[i].argmax(): E_test[i] = 1

				# print('[MIELM].', 100* (1-np.mean(E)), '%') # -

			elif mode == 'kmean_mielm':
				#Kmean MIELM
				#-----------------

				#training
				start = time.time() # + 
				# Betha = self.trainingKmeanMIELM(training_set[:,0], training_set[:,1], training_set[:,2])
				Betha = self.trainingKmeanMIELM2(training_set[:,0], training_set[:,1], training_set[:,2])
				end = time.time() # +
				learning_time = end-start # + for time measurement
				
				Y_train = self.estimateKmeanMIELM(training_set[:,0], training_set[:,1], Betha)
				T_train = np.concatenate((training_set[:,2])).reshape(self.num_of_training_bag, self.num_of_class)

				E_train = np.zeros((T_train.shape[0]))
				for i in range(T_train.shape[0]):
					if Y_train[i].argmax() != T_train[i].argmax(): E_train[i] = 1

				#testing
				start = time.time() # + 
				Y_test = self.estimateKmeanMIELM(testing_set[:,0], testing_set[:,1], Betha)
				end = time.time() # +

				T_test = np.concatenate((testing_set[:,2])).reshape(self.num_of_testing_bag, self.num_of_class)
				testing_time = end-start # + for time measurement

				E_test = np.zeros((T_test.shape[0]))
				for i in range(T_test.shape[0]):
					if Y_test[i].argmax() != T_test[i].argmax(): E_test[i] = 1

			cv_result[k,:] = np.array(((1-np.mean(E_test))*100, learning_time, testing_time))
			#- row = np.array((k+1, '{0:.2f}'.format(100* (1-np.mean(E_test))), '{0:.3f}'.format(learning_time), '{0:.3f}'.format(testing_time)), dtype=object)
			# row = np.array((k+1, '{0:.2f}'.format(100* (1-np.mean(E_train))), '{0:.2f}'.format(100* (1-np.mean(E_test))), '{0:.3f}'.format(learning_time), '{0:.3f}'.format(testing_time)), dtype=object)
			#- self.table.add_row(row)
			# self.result_table = np.insert(self.result_table, self.result_table.shape[0], row, 0)
			# print('[validasi ke-',k+1,'].', 'Akurasi: {0:.4f}%'.format(100* (1-np.mean(E))), "t train: {0:.3f}".format(learning_time),' ',"t test: {0:.3f}".format(testing_time) ) # +
			# print(k+1,',', '{0:.4f},'.format(100* (1-np.mean(E_test))), '{0:.3f},'.format(learning_time), '{0:.3f}'.format(testing_time))
		#- print(self.table)
		# np.savetxt('result'+str(datetime.datetime.now())+'.csv', self.result_table, fmt='%i '+ '%2.f ' + '%3.f ' *2, delimiter=',', newline='\n')
		# print(mean_of_result)
		
		#save result of cross validation
		self.savetxt(self.dataset_name+'_cv_result_'+mode+'_'+str(num_sample+1), cv_result)
		
		#return mean of cv result 		
		return (np.mean(cv_result,0))

	def trainingMIELM(self, training_set, bag_ids, training_label_set):
		#init
		self.num_of_training_bag = training_set.shape[0]
		self.mid_feature_per_bag = np.zeros((self.num_of_training_bag, self.num_of_hidden_node)) #center feature of each training bag (L*M)
		self.sigma_square = np.zeros((1, self.num_of_training_bag)) #bag feature matrix G using Gaussian Similarity function
		X = np.concatenate((training_set)) #(X* Num of Feature)
		
		# X = self.normalization(X) #normalization dataset
		temp_H = X * self.weight + self.bias 
		temp_H = (1 - np.exp(-temp_H) / (1 + np.exp(-temp_H))) # -
		# temp_H = 1 / ( 1 + np.exp(-temp_H)) # +

		H = self.regroupingInstanceToEachBag(temp_H, self.num_of_training_bag, bag_ids)

		G = np.zeros((self.num_of_training_bag, self.num_of_training_bag))

		for i in range(self.num_of_training_bag):
			self.mid_feature_per_bag[i,:] = np.mean(H[i],0)
			sim = np.power((temp_H - self.mid_feature_per_bag[i]),2) # squared euclidean distance ( ||x-y||^2^2 => d2(p,q) = (p1-q1)^2 + (p2-q2)^2 +...+ (pn-qn)^2 )
			self.sigma_square[0,i] = np.mean(np.sum(sim,1)) #variance
			temp_G = np.exp(-(np.sum(sim,1) * self.r / (4 * self.sigma_square[0,i]))) #.reshape(self.self.num_of_training_bag, int(X.shape[0]/self.self.num_of_training_bag))
			temp_G = self.regroupingInstanceToEachBag(temp_G, self.num_of_training_bag, bag_ids) # L*1

			for j in range(self.num_of_training_bag):
				G[i,j] = np.mean(temp_G[j], 0)

		T = np.asmatrix(np.concatenate((training_label_set)).reshape(self.num_of_training_bag, self.num_of_class)).T
		G = np.asmatrix(G)

		Betha = G.T * np.linalg.inv(np.eye(self.num_of_training_bag)/self.C + G * G.T) * T.T
		return Betha

	def trainingKmeanMIELM(self, training_set, bag_ids, training_label_set, max_iteration=50):
		#init
		self.num_of_training_bag = training_set.shape[0]
		self.mid_feature_per_bag = np.zeros((self.num_of_training_bag, self.num_of_hidden_node)) #center feature of each training bag (L*M)
		self.sigma_square = np.zeros((1, self.num_of_training_bag)) #bag feature matrix G using Gaussian Similarity function
		X = np.concatenate((training_set)) #(X* Num of Feature)
		
		# X = self.normalization(X) #normalization dataset
		temp_H = X * self.weight + self.bias 
		temp_H = (1 - np.exp(-temp_H) / (1 + np.exp(-temp_H))) # -
		# temp_H = 1 / ( 1 + np.exp(-temp_H)) # +

		H = self.regroupingInstanceToEachBag(temp_H, self.num_of_training_bag, bag_ids)

		#calucate mean each bag of all bag
		for i in range(self.num_of_training_bag):
			self.mid_feature_per_bag[i,:] = np.mean(H[i],0)

		#propose algorithm#
		#choose data (random) to be the first centeroid of cluster 
		centeroids = self.mid_feature_per_bag[np.random.randint(self.num_of_training_bag, size=(self.num_of_cluster))]

		for i in range(max_iteration):
			#create cluster_data and temp_cluster_data
			bool_cluster_data = np.zeros((self.num_of_training_bag, self.num_of_cluster))

			#calculate distance the data to the centeroid with euclidean distance
			distance = np.zeros((self.num_of_training_bag, self.num_of_cluster))
			for n in range(self.num_of_cluster):
				distance[:,n] = np.sqrt(np.sum(np.power(self.mid_feature_per_bag - centeroids[n],2),1)).reshape(self.num_of_training_bag)

			#clustering the data to the nearest centeroid
			for n in range(self.num_of_training_bag):
				bool_cluster_data[n,distance[n].argmin()] = 1

			#calculate new centeroid
			new_centeroids = (np.asmatrix(self.mid_feature_per_bag).T * np.asmatrix(bool_cluster_data)).T
			for n in range(self.num_of_cluster):
				m = np.count_nonzero(bool_cluster_data[:,n]) #num of data in the cluster n
				if m != 0: new_centeroids[n,:] = new_centeroids[n,:] / m

			if int(np.mean(new_centeroids == centeroids)):
				self.centeroids = centeroids
				break
			centeroids = new_centeroids

		#calculate matrix G with euclidean distance of the mean each bag of all bag with the centeroid
		G = np.zeros((self.num_of_training_bag, self.num_of_cluster)) #(L, K)
		for n in range(self.num_of_cluster):
			G[:,n] = np.sqrt(np.sum(np.power(self.mid_feature_per_bag - self.centeroids[n],2),1)).reshape(self.num_of_training_bag)
		#end of propose algorithm#

		T = np.asmatrix(np.concatenate((training_label_set)).reshape(self.num_of_training_bag, self.num_of_class)).T
		G = np.asmatrix(G)

		Betha = G.T * np.linalg.inv(np.eye(len(G))/self.C + G * G.T) * T.T
		return Betha

	def estimateMIELM(self, testing_set, bag_ids, Betha):
		#init
		self.num_of_testing_bag = testing_set.shape[0]
		X = np.concatenate((testing_set)) #(X* Num of Feature)
		
		# X = self.normalization(X) #normalization dataset
		temp_H = X * self.weight + self.bias 
		temp_H = (1 - np.exp(-temp_H) / (1 + np.exp(-temp_H))) # -
		# temp_H = 1 / ( 1 + np.exp(-temp_H)) # +
		H = self.regroupingInstanceToEachBag(temp_H, self.num_of_testing_bag, bag_ids)

		G = np.zeros((self.num_of_testing_bag, self.num_of_training_bag))

		for i in range(self.num_of_training_bag):
			sim = np.power((temp_H - self.mid_feature_per_bag[i]),2) # squared euclidean distance ( ||x-y||^2^2 => d2(p,q) = (p1-q1)^2 + (p2-q2)^2 +...+ (pn-qn)^2 )
			temp_G = np.exp(-(np.sum(sim,1) * self.r / (4 * self.sigma_square[0,i]))) #.reshape(self.self.num_of_testing_bag, int(X.shape[0]/self.self.num_of_training_bag))
			temp_G = self.regroupingInstanceToEachBag(temp_G, self.num_of_testing_bag, bag_ids) # L*1

			for j in range(self.num_of_testing_bag):
				G[j,i] = np.mean(temp_G[j], 0)

		G = np.asmatrix(G)
		Y = G * Betha		
		return Y
	
	def estimateKmeanMIELM(self, testing_set, bag_ids, Betha):
		#init
		self.num_of_testing_bag = testing_set.shape[0]
		X = np.concatenate((testing_set)) #(X* Num of Feature)
		
		# X = self.normalization(X) #normalization dataset
		temp_H = X * self.weight + self.bias 
		temp_H = (1 - np.exp(-temp_H) / (1 + np.exp(-temp_H))) # -
		# temp_H = 1 / ( 1 + np.exp(-temp_H)) # +
		H = self.regroupingInstanceToEachBag(temp_H, self.num_of_testing_bag, bag_ids)

		G = np.zeros((self.num_of_testing_bag, self.num_of_training_bag))

		#calucate mean each testing bag of all testing bag 
		mid_feature_per_bag = np.zeros((self.num_of_testing_bag, self.num_of_hidden_node)) #center feature of each training bag (Lt*M)
		for i in range(self.num_of_testing_bag):
			mid_feature_per_bag[i,:] = np.mean(H[i],0)

		#calculate matrix G with euclidean distance of the mean each bag of all bag with the centeroid
		G = np.zeros((self.num_of_testing_bag, self.num_of_cluster))
		for n in range(self.num_of_cluster):
			G[:,n] = np.sqrt(np.sum(np.power(mid_feature_per_bag - self.centeroids[n],2),1)).reshape(self.num_of_testing_bag)
			
		G = np.asmatrix(G)
		Y = G * Betha
		return Y

	def trainingKmeanMIELM2(self, training_set, bag_ids, training_label_set, max_iteration=50):
		#init
		self.num_of_training_bag = training_set.shape[0]
		self.mid_feature_per_bag = np.zeros((self.num_of_training_bag, self.num_of_hidden_node)) #center feature of each training bag (L*M)
		self.sigma_square = np.zeros((1, self.num_of_training_bag)) #bag feature matrix G using Gaussian Similarity function
		X = np.concatenate((training_set)) #(X* Num of Feature)
		
		# X = self.normalization(X) #normalization dataset
		temp_H = X * self.weight + self.bias 
		temp_H = (1 - np.exp(-temp_H) / (1 + np.exp(-temp_H))) # -
		# temp_H = 1 / ( 1 + np.exp(-temp_H)) # +

		H = self.regroupingInstanceToEachBag(temp_H, self.num_of_training_bag, bag_ids)

		#calucate mean each bag of all bag
		for i in range(self.num_of_training_bag):
			self.mid_feature_per_bag[i,:] = np.mean(H[i],0)

		#propose algorithm#
		#create boolean cluster data set
		bool_cluster_data = np.zeros((self.num_of_training_bag, self.num_of_cluster))
		
		#alocation data randomly to boolean cluster data set
		rand_indices = np.random.randint(self.num_of_cluster, size=(self.mid_feature_per_bag.shape[0]))
		for i in range(self.num_of_cluster):
			bool_cluster_data[np.nonzero(rand_indices == i)[0], i] = 1

		centeroids = np.zeros((1, self.num_of_cluster))

		for i in range(max_iteration):
			#calculate tmp centeroid
			tmp_centeroids = (np.asmatrix(self.mid_feature_per_bag).T * np.asmatrix(bool_cluster_data)).T
			for n in range(self.num_of_cluster):
				m = np.count_nonzero(bool_cluster_data[:,n]) #num of data in the cluster n
				if m != 0: tmp_centeroids[n,:] = tmp_centeroids[n,:] / m

			if int(np.mean(tmp_centeroids == centeroids)):
				self.centeroids = centeroids
				break
			centeroids = tmp_centeroids
			
			#clear bool_cluster_data
			bool_cluster_data = bool_cluster_data * 0

			#calculate distance the data to the centeroid with euclidean distance
			distance = np.zeros((self.num_of_training_bag, self.num_of_cluster))
			for n in range(self.num_of_cluster):
				distance[:,n] = np.sqrt(np.sum(np.power(self.mid_feature_per_bag - centeroids[n],2),1)).reshape(self.num_of_training_bag)

			#clustering the data to the nearest centeroid
			for n in range(self.num_of_training_bag):
				bool_cluster_data[n,distance[n].argmin()] = 1

		#calculate matrix G with euclidean distance of the mean each bag of all bag with the centeroid
		G = np.zeros((self.num_of_training_bag, self.num_of_cluster)) #(L, K)
		for n in range(self.num_of_cluster):
			G[:,n] = np.sqrt(np.sum(np.power(self.mid_feature_per_bag - self.centeroids[n],2),1)).reshape(self.num_of_training_bag)
		#end of propose algorithm#

		T = np.asmatrix(np.concatenate((training_label_set)).reshape(self.num_of_training_bag, self.num_of_class)).T
		G = np.asmatrix(G)

		#calculate output weight
		Betha = G.T * np.linalg.inv(np.eye(len(G))/self.C + G * G.T) * T.T
		return Betha

	def regroupingInstanceToEachBag(self, instances, num_of_bag, bag_ids):
		Bags = np.zeros((num_of_bag), dtype=object)
		#regrouping instance to each bag
		for i in range(num_of_bag):
			Bags[i] = (instances[(np.concatenate((bag_ids)) == bag_ids[i][0]).nonzero()[0],:])
		return Bags

	def printProgress (self, iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100): # +
		filledLength    = int(round(barLength * iteration / float(total)))
		percents        = round(100.00 * (iteration / float(total)), decimals)
		bar             = '#' * filledLength + '-' * (barLength - filledLength)
		Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix))
		Sys.stdout.flush()
		if iteration == total:
			Sys.stdout.flush()
			print()

	def savetxt(self, name, performance):
		np.savetxt('results/'+name+'.csv', performance, delimiter=',', newline='\n')
		print(name +' saved')
