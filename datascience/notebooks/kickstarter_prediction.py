#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'datascience'))
	print(os.getcwd())
except:
	pass

#%%
#Make necessary imports

#Utils
import numpy as np
import pandas as pd
import random
#Graphs
import seaborn as sns
import matplotlib.pyplot as plt
#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Models and selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier
#I/O for model
import pickle


#%%
#Loading Data in
data = pd.read_csv('../input/ks-projects-201801.csv')

#%% [markdown]
# **Looking at basic datatypes and unique values of each column**

#%%
def load_and_explore(data):
    print("############# PREVIEW ########################")
    print(data.head())
    print("############# DATA TYPES ########################")
    print(data.info())
    print("############# NO. OF UNIQUE VALS ########################")
    print(data.nunique())

load_and_explore(data)

#%% [markdown]
# This shows us that the launched and deadline columns, which should be datetypes are columns and so should be converted. The state column should be converted to a boolean value, and needs to be dealt with as there should only be two states. Let's deal with this now.

#%%
print(data["state"].value_counts())
def convert_to_bool(item):
    if item=="failed":
        return 0
    elif item=="successful":
        return 1
    else:
        return 2

data["state"] = data["state"].apply(convert_to_bool)
print(data["state"].value_counts())

#%% [markdown]
# We see that 46986 columns are neither successful or failed, and are cancelced, still live, suspended or undefined. So we delete these. Now, set the id column as id, and start generating some new features. First, we will look at the date. From the launched and deadline columns, the natural features to consider are:
# 1. Duration
# 2. Month it was launched in
# 3. Month of Deadline
# 4. Quarter it was launched in
# 5. Quarter of Deadline
# 
# The year doesn't make much sense to consider as we will be making predictions on live projects and so the year doesn't make sense. 

#%%
#Delete rows that are not 0,1 in state
data = data[data.state !=2 ]

#Set ID
data = data.set_index('ID')
print(data.head())
print(data.shape)


#%%
#Create new features from launched, deadline

#First, convert to datetime
data["launched"] = pd.to_datetime(data["launched"], format="%Y-%m-%d %H:%M:%S")
data["deadline"] = pd.to_datetime(data["deadline"], format="%Y-%m-%d")

#Create new features

#Duration in days
data["duration"] = (data["deadline"] - data["launched"]).dt.days

#Quarter, month of launched and deadline date
data["launch_month"] = data["launched"].dt.month
data["launch_quarter"] = data["launched"].dt.quarter
data["deadline_month"] = data["deadline"].dt.month
data["deadline_quarter"] = data["deadline"].dt.quarter

#The launch hour may also have an impact, as it may affect when/if it goes viral
data["launch_hour"] = data["launched"].dt.hour
data[['launched','deadline', 'duration','launch_month', 'launch_quarter', 'deadline_month', 'deadline_quarter']].head()

#%% [markdown]
# Let's create new features from the title now. Performing sentiment analysis or other NLP techniques doesn't make much sense as each title is quite different as it describes the specific project, and may have an impact on making the title vectorization the same as the category. Therefore, I will stick to basic title vectorization techniques.

#%%
#Create new features from title

#Length of title
data["title_length"] = data["name"].apply(lambda x: len(str(x)))

#Number of words
data["title_words"] = data["name"].apply(lambda x: len(str(x).split(' ')))

#Number of symbols
data['title_symbols'] = data["name"].apply(lambda x: str(x).count('!') + str(x).count('?'))

data[['title_length', 'title_words', 'title_symbols']].head()

#%% [markdown]
# Now finally, some additional features will be created based on the backers/pledged. Make any inferences from the pledged column is risky because of the fact that we don't know when the data was scraped, so I am not sure if considering that column heavily is a smart idea. I will only look at two metrics: amount pledged per backer.

#%%
#Create new pledged feature

#New column for if there are any backers
data["backers_exist"] = np.where(data["backers"]>0, "True", "False")

#Make a mask for rows that contain backers
mask_backers_exist = (data["backers"]>0)

#Enter 0 for where backers don't exist, and the pledged per backer for where they do
data['pledged_per_backer'] = 0
data.loc[mask_backers_exist, 'pledged_per_backer'] = data["pledged"] / data["backers"]

#Review new metric
print(data[["pledged", "backers", "pledged_per_backer"]].head())
print("MAX VALS")
print(data[["pledged", "backers","pledged_per_backer"]].max())

#%% [markdown]
# 
#%% [markdown]
# Now we must find those columns with NaN values, and deal with them accordingly. 

#%%
#Find columns with NaN values
data.isna().any()[lambda x:x]
data.isna().sum()

#%% [markdown]
# We see that the name, usd pledged columns have NaN values. Therefore, we must deal with this on a case by case basis. Let's look at the name column's null values.

#%%
data[data["name"].isnull()]

#%% [markdown]
# As we can see, since there are only 4, we can replace the features created from the title for these with the means. Let's look at the USD pledged column.

#%%
data[data["usd pledged"].isnull()]

#%% [markdown]
# From here, we see that the columns that are null for usd pledged have a weird country value as well. We do know the currency, and so getting the country shouldn't be difficult. From this, we can replace the country column and the usd pledged column by making an exchange rate dictionary. So, let's do this. First, we find all the currencies. Then, find the country from the currency, and then the usd pledged from the currency.

#%%
#Get Currencies and Countries
print(data["currency"].value_counts())
print(data["country"].value_counts())


#%%
#Make dictionary mapping currency to country code
curr_to_country  = {
    "USD": ["US"],
    "GBP": ["GB"],
    "EUR": ["DE", "FR", "IT", "NL","ES", "IE", "BE", "AT", "LU"], #Denmark, France, Italy, Netherlands, Spain, Ireland, Belgium, Austria, Luxembourg
    "CAD": ["CA"],
    "AUD": ["AU"],
    "SEK": ["SE"],
    "MXN": ["MX"],
    "NZD": ["NZ"],
    "DKK": ["DK"],
    "CHF": ["CH"],
    "NOK": ["NO"],
    "HKD": ["HK"],
    "SGD": ["SG"],
    "JPY": ["JP"]
}

#Find all rows with bad country names
mask_bad_countries = (data["country"]=='N,0"')

#Randomly get country from the EUR array, because we can not know which country it was
data["good_country"] = data["currency"].apply(lambda x: random.choice(curr_to_country[x]))

#Replace the bad countries with the fixed country. Don't do for all as that will lose truth due to the EUR
data.loc[mask_bad_countries, "country"] = data["good_country"]

data["country"].value_counts()

#%% [markdown]
# We can see that the faulty data has been removed

#%%
#Now find USD pledged based on currency
curr_usd_exchange_rate = {
    "USD": 1.0,
    "CAD": 0.75,
    "MXN": 0.052,
    "SGD": 0.73,
    "EUR": 1.12,
    "AUD": 0.69,
    "CHF": 1.00,
    "DKK": 0.15,
    "GBP": 1.26,
    "HKD": 0.13,
    "JPY": 0.0092,
    "NOK": 0.11,
    "NZD": 0.65,
    "SEK": 0.11
}

#Get exchange rate
data["exchange_rate"] = data["currency"].apply(lambda x: curr_usd_exchange_rate[x])

#Get the good value of usd pledged
data["usd_pledged_new"] = data["pledged"] * data["exchange_rate"]

#Replace all bad values with the good value
data.loc[mask_bad_countries, "usd pledged"] = data["usd_pledged_new"]
data.isna().sum()

#%% [markdown]
# Now, all the NaN values for the country and the usd pledged have been fixed! Now, we can just drop the NaN rows with the names, as they are only 4 rows like this.

#%%
data = data.dropna()
data.isna().sum()

#%% [markdown]
# Now that all of the NaN and faulty values have been dealt with, we can start filtering the important features. To start, we will do some manual analysis here. We will start by looking at the successes/fails of the date features to see if there is much variation.

#%%
#Function to generate labels, success/fail ratios for columns
def gen_bar_chart_data(columns):
    labels = []
    ratios = []
    for col in date_visualize_columns:
        label = success_rows[col].value_counts().keys().tolist()
        count_fails = fail_rows[col].value_counts().tolist()
        count_successes = success_rows[col].value_counts().tolist()

        ratio = []
        for i in range(len(label)):
            ratio.append(count_fails[i]/count_successes[i])

        labels.append(label)
        ratios.append(ratio)
    
    return labels, ratios


#%%
#Generate the data - launched, launched hour, deadline, duration, launched quarter, deadline quarter
date_visualize_columns = ["launch_hour", "duration", "launch_quarter", "deadline_quarter"]

plt.style.use('fivethirtyeight')

#Fails, successes
fail_rows = data.loc[data['state']==0]
success_rows = data.loc[data['state']==1]

#Labels, ratios
labels, ratios = gen_bar_chart_data(date_visualize_columns)
      
#Plot
fig1, (graph1, graph2) = plt.subplots(2)
fig2, (graph3, graph4) = plt.subplots(2)

graph1.bar(labels[0], ratios[0])
graph1.set_title(date_visualize_columns[0])

graph2.bar(labels[1], ratios[1])
graph2.set_title(date_visualize_columns[1])

graph3.bar(labels[2], ratios[2])
graph3.set_title(date_visualize_columns[2])

graph4.bar(labels[3], ratios[3])
graph4.set_title(date_visualize_columns[3])

fig1.suptitle("Date columns")
fig2.suptitle("Quarter Data")
fig1.show()

#%% [markdown]
# These graphs show that there is significant variation in the success/fail ratios in the launch hour and duration, but not as much in the launch and deadline quarter. Next we will look at similar graphs for the title features we generated. 

#%%
date_visualize_columns = ["title_length", "title_words", "title_symbols"]

labels, ratios = gen_bar_chart_data(date_visualize_columns)

fig1, (graph1, graph2) = plt.subplots(2)
fig2, (graph3) = plt.subplots(1)

graph1.bar(labels[0], ratios[0])
graph1.set_title(date_visualize_columns[0])

graph2.bar(labels[1], ratios[1])
graph2.set_title(date_visualize_columns[1])
fig1.suptitle("Title Columns")

graph3.bar(labels[2], ratios[2])
graph3.set_title(date_visualize_columns[2])

#%% [markdown]
# As we can see, while there is no clear correlation, all of these seem to be of importance. Next, we will find correlations between variables and plot a heatmap to see where important correlations are. This will give us a preliminary overview of important relationships in our dataset. 

#%%
#Get top correlations
corr_matrix = data.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(20,20))

#Plot heatmap
graph = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#%% [markdown]
# We see a good correlation between the backers and the amount pledged, which furthers the importance of creating the feature that we created. Additionally, there is a blob of high correlations around the launch/deadline month/quarter and the features of the title, but this is pretty artificial as it is simply because they point to the same thing. For this reason, we will choose not to keep the deadline/launch month, but will keep the quarter as it might still be important. Additionally, looking at the state, we see that there is a slightly higher correlation in the title features, so we will make sure not to remove those. Finally, we will use a chi squared test to find the top 10 most important features in the dataset. Before we do that, however, we must one-hot encode our dataset. First, lets see what dtypes we have. 

#%%
object_data = data.select_dtypes(include=['object'])
object_data.keys()

#%% [markdown]
# We see the columns that are not a numerical type. We won't be using the name, backers_exist or good_country columns as features were already extracted from them (name) or were used for the understanding of other data. Therefore, we need to one hot encode the following columns. First, we will look at the values in them and see if they are appropriate for future purposes.

#%%
cols_one_hot_encode = ['category', 'main_category', 'currency','country', 'launch_quarter', 'deadline_quarter', 'launch_hour']
for col in cols_one_hot_encode:
    print(data[col].value_counts())

#%% [markdown]
# From this, we see that some values in the category and main_category columns contains an ampersand (&) while some contain spaces and some contain hyphens (-). These need to be removed. 

#%%
data_encoded = pd.get_dummies(data, prefix=cols_one_hot_encode, columns=cols_one_hot_encode)
data_encoded.shape

#%% [markdown]
# Now, we must finalize our dataframe by removing the columns that we don't want analyzed in our model. These are 
# 1. Name, as we have already extracted important features from it. 
# 2. Launch month, as we did not see any importance in our manual analysis.
# 3. Deadline month, as we did not see any importance in our manual analysis.
# 4. Backers Exist,
# 5. Good Country,
# 6. Exchange Rate, 
# 7. USD pledged New, as these were for our preprocessing.
# 8. Launched, as it is a timestamp.
# 9. Deadline, as it is a timestamp.
# 10. Pledged,
# 11. usd pledged,
# 12. usd pledged real, as they introduce data leakage into our dataset.
# 13. backers, to prevent data leakage.

#%%
data_analyze = data_encoded.drop(['launched','pledged','usd pledged','backers', 'usd_pledged_real', 'deadline','name', 'launch_month', 'deadline_month', 'backers_exist', 'good_country', 'exchange_rate', 'usd_pledged_new'], axis=1)
#We will first need the mean and standard deviation of all of our columns so that we can use it to make predictions later. 
mean_std = data_analyze.agg([np.mean, np.std])

#Write to file
mean_std.to_csv('statistics.csv')
mean_std.head()


#%%
#Check dtypes and shape of dataframe now
data_analyze.dtypes

#%% [markdown]
# As we can see, everything is an integer float or datetime now, and so is good to go. It is now 256 columns wide and we only removed 4 rows. We will now create two arrays: one with the names of all features and one with the names of the target. Then, we will normalize the features of our dataset.

#%%
#Remove state from independent columns
features = list(data_analyze)
features.remove('state')

#Not sure if OHE columns should be normalized, so I will do it for now and then experiment later to see which gives better results
data_analyze_scaled = pd.DataFrame(preprocessing.normalize(data_analyze[features]))
data_analyze_scaled.columns = features

data_analyze_scaled.index = data_analyze.index

# # #Set same index column
# data_analyze_scaled['index'] = data_analyze_scaled.index
# data_analyze['index'] = data_analyze.index

# # #Add target to normalized dataframe
# data_analyze_scaled = data_analyze_scaled.merge(data_analyze[['index', 'state']], left_on='index', right_on='index')
# data_analyze_scaled = data_analyze_scaled.set_index('index')


#%%
data_analyze_scaled.head()


#%%
#Get the target array
target = data_analyze['state']

#Show final data before splitting
print(target[:5])
data_analyze_scaled.head()

#%% [markdown]
# Now we need to split this into a training and testing set. However, the dataset has an imbalance in the number of successes/fails, therefore we need to ensure that our training set does not have this imbalance as well. A maximum success/fail of 1.5 has been chosen, and we will now split our data to ensure it falls within this ratio.
# 
# 

#%%
#Loop until a good ratio < 1.5 is found
success_fail_ratio = 10
iterated = 0

while success_fail_ratio > 1.5:
    features_train, features_test, target_train, target_test = train_test_split(
        data_analyze_scaled,
        target, 
        random_state=42,
        test_size = 0.2)
    
    counts = target_train.value_counts().tolist()
    success_fail_ratio = counts[0]/counts[1]
    iterated+=1

print("ITERATED %d times to get a ratio of %f" % (iterated, success_fail_ratio))

#Error check data leakage
if 'state' in list(features_train) or 'state' in list(features_test) or 'state' in list(data_analyze_scaled_x):
    print("PROBLEM")
else:
    print("FINE")
    
print("Training shape (%d, %d)" % (features_train.shape), (target_train.shape))
print("Testing shape (%d, %d)" % (features_test.shape), (target_test.shape))

#%% [markdown]
# Model Creation 
#%% [markdown]
# Create a baseline Gaussian Bayes Classifier to get a minimum to base other models off of.

#%%
base_model = GaussianNB()

#Fit
base_model_fit = base_model.fit(features_train, target_train)

#Predict
pred = base_model.predict(features_test)
accuracy = classification_report(pred, target_test)

print('\n Percentage accuracy')
print(accuracy)

#%% [markdown]
# A logistic regression model is next!

#%%
logreg = LogisticRegression()

#Fit
logreg.fit(features_train, target_train)

#Predict
pred = logreg.predict(features_test)
accuracy = classification_report(pred, target_test)

print('\n Percentage accuracy')
print(accuracy)

#%% [markdown]
# Finally, LGBM Models are popular for classification tasks like this, so we will utilize them.

#%%
#LGBM Classifier. Get slightly tuned parameters with plug and play testing 
lgbm_class = LGBMClassifier(
        n_estimators=300,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=10,
        reg_alpha=.1,
        reg_lambda=.05,
        min_split_gain=.005
    )

lgbm_class.fit(features_train, 
        target_train,
        eval_set= [(features_train, target_train), (features_test, target_test)], 
        eval_metric='auc', 
        verbose=0, 
        early_stopping_rounds=30
       )

pred = lgbm_class.predict(features_test)

print('\n Percentage accuracy')
print(classification_report(pred, target_test))

#%% [markdown]
# Now, since LGBM performed the best (as expected), train it on all of the data. I won't be able to see the accuracy this time. 

#%%
final_model = LGBMClassifier(
        n_estimators=300,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=10,
        reg_alpha=.1,
        reg_lambda=.05,
        min_split_gain=.005
    )

#Fit
final_model.fit(data_analyze_scaled, 
        target,
        eval_set= [(features_train, target_train), (features_test, target_test)], 
        eval_metric='auc', 
        verbose=0, 
        early_stopping_rounds=30
       )

#%% [markdown]
# Get Feature Importances

#%%
#Get most important features from the LGBM Model
feature_importances = sorted(zip(final_model.feature_importances_, features), reverse=True)

#Construct dataframe to save
feature_importances_df = pd.DataFrame(feat_importances)
feature_importances_df.columns = ['importance', 'feature']
feature_importances_df.index = feature_importances_df.feature
feature_importances_df.drop('feature', 1, inplace=True)

#Show dataframe
print(feature_importances_df.head())

#Save the file
feature_importances_df.to_csv('feature_importances.csv')
feature_importances_df

#Save the model both .sav and .pkl
pickle.dump(final_model, open('finalmodel.sav', 'wb'))
pickle.dump(final_model, open('finalmodel.pkl', 'wb'))

#%% [markdown]
# And we are done :) Now natural next steps include
# 1. Turning this from a classification problem into a regression problem, with the % goal achieved column as the target.
# 2. Using GridSearchCV to tune hyperparameters.

