import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class Model:
	def __init__(self, filepath):
		self.data = pandas.read_csv(filepath)
		self.traindata = 0
		self.testdata = 0
		self.successes = self.data.loc[data['state']=='successful']
		self.fails = self.data.loc[data['state']=='failed']
		self.targetTrain = 0
		self.targetTest = 0

		#We want to remove all rows that contain anything in Row 14, 
		#manually titled 'exists', as this implies that the data is messed up
		badrows = self.data[self.data['exists'].notnull()]
		self.data = pd.concat([self.data, badrows, badrows]).drop_duplicates(keep=False)

		#We want to remove the rows that don't contain a type datetime in them. They have int 
		#instead. We can use iterrows() even though it is slow, as I wrote to the csv file already
		#With the fixed information. 

		indicesToDelete = []
		indicesToKeep = []

		for index, row in data.iterrows():
			try:
				int(row['deadline'])
				indicesToDelete.append(row[index])
			except:
				indicesToKeep.append(index)
		self.data = self.data.drop(indicesToDelete, axis=0)

		#Convert to datetime object
		self.data = self.data['launched'].apply(convertToTime) # TODO: Add helper function
		self.data = self.data['deadline'].apply(convertToTime) # TODO: Add helper function

		#Convert numbers to floats
		self.data["goal"] = self.data['goal'].astype('float')
		self.data["pledged"] = self.data['pledged'].astype('float')

		#Make a new column for Time Difference - one of the features I will test
		self.data['timeDifference'] = (self.data['deadline'] - data['launched'])
		self.data['timeDifference'] = self.data.timeDifference.dt.total_seconds()

		#Want to convert success/fail to boolean. Set any other state to 2
		self.data['state'] = self.data['state'].apply(convertToBool)
		self.data = self.data.drop(self.data[self.data.state == 2].index)

		#Want to convert datetime to month in new column
		self.data['month_launched'] = (self.data.launched.dt.month)
		self.data['month_deadline'] = (self.data.deadline.dt.month)

		#Based off of the graphs generated, I found that the month 

		#One Hot Encoding all the discrete categories
		columns = ['country', 'main_category','currency',  'category']
		self.data = self.genEncoder(self.data, columns)

		#Not doing sentiment analysis or using word2vec/gloVe, as I believe that it is not very applicable due to the specificity of the titles. 
		#Instead, I am vectorizing the title through its length. 
		self.data['titleLength'] = self.data['name'].apply(lambda x: len(str(x)))

		#Need to scale the data, as most of the algorithms used below need scaled data for good results.
		#Don't scale state however, as this is our target
		for i in self.data.keys():
			if self.data[i].dtype == 'float64' and i!='ID' and i!='state':
				self.data[i] = scale(self.data[i])


	def genEncoder(self, data, columns):
		for i in columns:
			le_column = LabelEncoder()
			ohe_column = OneHotEncoder()

			data[i+'Encoded'] = le_column.fit_transform(data[i].fillna('0'))

			columnBinaryArray = ohe_column.fit_transform(data[i+'Encoded'].values.reshape(-1,1)).toarray()
			categoryOneHot = pd.DataFrame(columnBinaryArray, columns = [i+ '_' + str(int(k)) for k in range(columnBinaryArray.shape[1])])

			data = pd.concat([data, categoryOneHot], axis=1)

		return data

	def genTrainTestData(self):
		#Delete all unncessary, unvectorized columns
		for i in data.keys():
			if not data[i].dtype =='float':
			data = data.drop(i, axis=1)
		#count = data['state'].value_counts() #Done to check how many success/fails. 168221/113081 ratio found.
		#It is important to check this as we need to make sure our data is balanced.
		#A ratio of 1.5 is acceptable, especially given our data and after consulting with internet resources
		
		features = data.values[:, :data.columns.get_loc("state")]
		target = data.values[:, data.columns.get_loc("state")]

		successfailratio = 10

		while successfailratio > 1.5:
			self.featureTrain, self.featureTest, self.targetTrain, self.targetTest = train_test_split(featuers,target, test_size=0.2)
			counts = self.train['state'].value_counts().tolist()
			successfailratio = counts[0]/counts[1]

		#Remove the NaN values from each array. TODO: Replace it with the means of the columns
		self.featureTrain = np.nan_to_num(self.featureTrain)
		self.featureTest = np.nan_to_num(self.featureTest)
		self.targetTrain = np.nan_to_num(self.targetTrain)
		self.targetTrain = np.nan_to_num(self.targetTrain)

		return 1

	#Generate graphs for the unintuitive features to validate them
	def genBarGraph(self, column):
		#Edit a little to generate non-ratio graphs
		labels = self.fails[column].value_counts().keys().tolist()
		countsfails = self.fails[column].value_counts().tolist()
		countssuccesses = self.successes[column].value_counts().tolist()

		ratios = []
		for i in range(len(labels)):
			ratios.append(countsfails[i]/countssuccesses[i])

		index = np.arange(len(labels))
		plt.xticks(index, labels, fontsize=5, rotation=30)
		plt.xlabel(column, fontsize=5)
		plt.ylabel('Ratio', fontsize=5)
		plt.title('Success/Fail ratio for each' + column)
		plt.bar(index, ratios)
		plt.show()

		return 1

	#Generate Gaussian Naive Bayes Classifier as baseline classifier to compare other models with. Has a 65% accuracy
	def gaussianNaiveBayes(self):
		model = GaussianNB()
		model.fit(self.featureTrain, self.targetTrain)

		prediction = model.predict(self.targetTest)
		pred_accuracy = classification_report(self.targetTest, prediction)

		return pred_accuracy

	#Generate Decision Tree Classifier and tune the parameters using GridSearchCV to get the optimal random_state parameter
	def decisionTreeClassifier(self):
		#Use the roc-auc scoring 
		scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
		clf = GridSearchCV(DecisionTreeClassifier(random_state=15), param_grid={'min_samples_split': range(2, 253, 10)}, scoring=scoring, cv=5, refit='AUC', return_train_score=True)
		
		clf.fit(train, target_train)
		optimalSplit = clf.best_params_

		#Gives an optimalSplit of 242

		model = DecisionTreeClassifier(random_state=clf.best_params_)
		model.fit(self.featureTrain, self.targetTrain)

		prediction = model.predict(self.targetTest)
		pred_accuracy = classification_report(self.targetTest, prediction)

		return pred_accuracy

	def logisticRegression(self):
		model = LogisticRegression()

		model.fit(self.featureTrain, self.targetTrain)

		prediction = model.predict(self.targetTest)
		pred_accuracy = classification_report(self.targetTest, prediction)

		return pred_accuracy
	
	def supportVectorMachine(self):
		model = SVC(kernel='linear', C=1E10)
		
		model.fit(self.featureTrain, self.targetTrain)

		prediction = model.predict(self.targetTest)
		pred_accuracy = classification_report(self.targetTest, prediction)

		return pred_accuracy

	def Predict(self, x):
		for i in self.data.keys():
			if not i in x.keys():
				x[i] = self.data[i].mean()

		#
		prediction = self.model.predict(x)
		pred_accuracy = accuracy_score(prediction, x, normalize=True)

		return accuracy_score





def convertToTime(date):
	dateupdate = datetime.strptime(date, '%Y-%m-%d %H:%M')
	return dateupdate

def convertToBool(item):
	if item=='failed':
		return 0
	if item=='successful':
		return 1
	else:
		return 2











