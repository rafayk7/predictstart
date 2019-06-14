import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D



def convertToTime(date):
	dateupdate = datetime.strptime(date, '%Y-%m-%d %H:%M')
	return dateupdate

def func2(item):
	if not '-' in item:
		return float('nan')
	else:
		return item

data = pd.read_csv('ks-projects-201612.csv', encoding='cp1252')
data.fillna('0')



badrows = data[data['exists'].notnull()]

data = pd.concat([data, badrows, badrows]).drop_duplicates(keep=False)
print(data['name'].dtype)

# indicesToDelete = []
# indicesToKeep = []

# for index, row in data.iterrows():
# 	try:
# 		int(row['deadline'])
# 		indicesToDelete.append(row[index])
# 	except:
# 		indicesToKeep.append(index)

def convertToBool(item):
	if item=='failed':
		return 0
	if item=='successful':
		return 1
	else:
		return 2

data["goal"] = data['goal'].astype('float')
data["pledged"] = data['pledged'].astype('float')

train, test = train_test_split(data.fillna('0'), test_size=0.2)

print(train['state'].value_counts())
print(len(train))
print(len(test))

data['state'] = data['state'].apply(convertToBool)
print(data['state'].dtype)
print(data['state'].head())

# print(data.shape)
data = data.drop(data[data.state == 2].index)

#print(data['state'].head())

# print(data.dtypes)

# print(data.keys())

data['deadline'] = data['deadline'].apply(convertToTime)

data['launched'] = data['launched'].apply(convertToTime)

data['timeDifference'] = (data['deadline'] - data['launched'])

data['timeDifference'] = data.timeDifference.dt.total_seconds()

def genEncoder(data, columns):
	for i in columns:
		le_column = LabelEncoder()
		ohe_column = OneHotEncoder()

		data[i+'Encoded'] = le_column.fit_transform(data[i].fillna('0'))

		columnBinaryArray = ohe_column.fit_transform(data[i+'Encoded'].values.reshape(-1,1)).toarray()
		categoryOneHot = pd.DataFrame(columnBinaryArray, columns = [i+ '_' + str(int(k)) for k in range(columnBinaryArray.shape[1])])

		data = pd.concat([data, categoryOneHot], axis=1)

		del le_column
		del ohe_column

	return data


data['month_launched'] = (data.launched.dt.month)
data['month_deadline'] = (data.deadline.dt.month)

successes = data.loc[data['state']==1]
fails = data.loc[data['state']==0]

# print(successes)
# print(fails)
# labels = fails['month_deadline'].value_counts().keys().tolist()
# countsfails = fails['month_deadline'].value_counts().tolist()
# countssuccesses = successes['month_deadline'].value_counts().tolist()

# ratios = []
# for i in range(len(labels)):
# 	ratios.append(countsfails[i]/countssuccesses[i])


# index = np.arange(len(labels))
# plt.xticks(index, labels, fontsize=5, rotation=30)
# plt.xlabel('Month Deadline', fontsize=5)
# plt.ylabel('Month Deadline', fontsize=5)
# plt.title('Success/Fail ratio for each Month of Deadline')
# plt.bar(index, ratios)
# plt.show()

columns = ['country', 'main_category','currency',  'category']

data = genEncoder(data, columns)
# print(data.dtypes)
for i in data.keys():
	print(i)
	if data[i].dtype == 'float64' and i!='ID' and i!='state':
		data[i] = scale(data[i])

print(data['goal'])


data['titleLength'] = data['name'].apply(lambda x: len(str(x)))
for i in data.keys():
	print(i)
	if data[i].dtype == 'float64' and i!='ID' and i!='state':
		data[i] = scale(data[i])
print(data.shape)

for i in data.keys():
	if not data[i].dtype =='float':
		data = data.drop(i, axis=1)

print(data.shape)

features = data.values[:, :data.columns.get_loc("state")]
target = data.values[:, data.columns.get_loc("state")]

train, test, target_train, target_test = train_test_split(features, target, test_size=0.2)

train = np.nan_to_num(train)
test = np.nan_to_num(test)
target_test = np.nan_to_num(target_test)
target_train = np.nan_to_num(target_train)
print(target_train)

# def PolynomialRegression(degree=2, **kwargs):
#     return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# indices = np.linspace(-0.1, 1.1, 64747)[:, None]

# for i in [1,3,5,7]:
# 	model_y = PolynomialRegression(i).fit(train, target_train).predict(test)
# 	plt.plot(indices.ravel(), model_y, label='i={0}'.format(i))
# 	# clf = GridSearchCV(DecisionTreeClassifier(random_state=15), param_grid={'min_samples_split': range(2, 253, 10)}, scoring=scoring, cv=5, refit='AUC', return_train_score=True)

# clf.fit(train, target_train)

model = DecisionTreeClassifier(class_weight=None, max_depth=5,
            max_features=None, max_leaf_nodes=8, min_samples_leaf=5,
            min_samples_split=10, min_weight_fraction_leaf=0.0, splitter='random')

depths = np.linspace(1,30, 30)

training_results = []
testing_results  = []

#x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)


# for depth in depths:
# 	model = DecisionTreeClassifier(max_depth=depth)
# 	model.fit(train, target_train)

# 	predictionTrain = model.predict(train)

# 	falsePositiveRate, truePositiveRate, thresholds = roc_curve(target_train, predictionTrain, pos_label=2)

# 	auc = auc(falsePositiveRate, truePositiveRate)

# 	training_results.append(auc)

	# predictionTest = model.predict(test)

	# falsePositiveRate, truePositiveRate, thresholds = roc_curve(test, predictionTest)

	# auc = auc(falsePositiveRate, truePositiveRate)

	# testing_results.append(auc)

# line1, = plt.plot(depths, training_results, 'b', label='Train AUC')
# # line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel('max features')
# plt.show()

model.fit(train, target_train)

prediction = model.predict(test)

# print(pred_accuracy)
pred_accuracy = model.score(test, target_test)
print(pred_accuracy)

# # print(clf.best_params_)

# model = SVC(kernel='linear', C=1E10)
# model.fit(train, target_train)

# target_pred = model.predict(test)
# print(accuracy_score(train, target_pred, normalize = True))

# clf = RandomForestRegressor(200)
# clf.fit(train, target_train)

# target_pred = clf.predict(test)

# print(accuracy_score(target_test, target_pred, normalize = True))


# I would never get 









# print(train['state'].value_counts())
# print(len(train))
# print(len(test))

# print(data['titleLength'].head())
#print(data['month_launched'].head())
#print(data.dtypes)

# def callEncoder(data, columns):
# 	for i in columns:
# 		dagenEncoder(data, i)

# print(data.keys())
#data = genEncoder(data, columns)



# print(data.keys())
# le_category = LabelEncoder()
# ohe_category = OneHotEncoder()
# data['countryEncoded'] = le_category.fit_transform(data['country'])
# #print(data['countryEncoded'].head())

# categoryBinaryArray = ohe_category.fit_transform(data['countryEncoded'].values.reshape(-1,1)).toarray()
# categoryOneHot = pd.DataFrame(categoryBinaryArray, columns = ["Country_"+str(int(i)) for i in range(categoryBinaryArray.shape[1])])
# data = pd.concat([data, categoryOneHot], axis=1)
# #print(data['Country_2'].head())


# #print(data.keys())


# le_category = LabelEncoder()
# ohe_category = OneHotEncoder()
# data['categoryEncoded'] = le_category.fit_transform(data['category'])
# #print(data['categoryEncoded'].head())

# categoryBinaryArray = ohe_category.fit_transform(data['categoryEncoded'].values.reshape(-1,1)).toarray()

# categoryOneHot = pd.DataFrame(categoryBinaryArray, columns = ["Category_"+str(int(i)) for i in range(categoryBinaryArray.shape[1])])
# data = pd.concat([data, categoryOneHot], axis=1)

# #print(data['Category_2'].head())
# le_mainCategory = LabelEncoder()
# le_currency = LabelEncoder()
# le_country = LabelEncoder()

