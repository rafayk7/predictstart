import pickle
import pandas as pd
import json
import random
import numpy as np
import shap
import operator

class Predictor:
    def __init__(self, modelpath):
        self.model = pickle.load(open(modelpath, 'rb'))
        self.notTooAcc = False
        self.explainer = shap.TreeExplainer(self.model)
        self.features = []

    def test(self):
        print(self.model)

    def getCurrencyRates(self):
        currencyRates = []
        with open('./config.json', 'r') as f:
            arr = json.load(f)
            return arr

    def preprocess(self, jsonInput):
        # Read saved dataframe with unique values
        baseDF = pd.read_csv('./datascience/data/unique_value_df.csv', index_col=0)

        cols_one_hot_encode = ['category', 'main_category', 'currency', 'country', 'launch_quarter', 'deadline_quarter',
                               'launch_hour']

        currencyRates = self.getCurrencyRates()

        # Initial Data Entry from Scrape
        dataObj = {
            'name': jsonInput['title'],
            'category': jsonInput['category'],
            'main_category': jsonInput['main_category'],
            'currency': jsonInput['currency'],
            'deadline': jsonInput['deadline_date'],
            'goal': jsonInput['goal'],
            'launched': jsonInput['launched_date'],
            'pledged': jsonInput['pledged'],
            'state': 0,
            'backers': jsonInput['backers'],
            'country': jsonInput['country'],
            'usd pledged': jsonInput['usd_pledged'],
            'usd_pledged_real': jsonInput['usd_pledged'],
            'usd_goal_real': jsonInput['goal'] * currencyRates[jsonInput['currency']]
        }

        # Convert to Dataframe, with proper column order
        data = pd.DataFrame(dataObj, columns=baseDF.columns, index=[1000])

        # Convert to datetime
        data["launched"] = pd.to_datetime(data["launched"], format="%Y-%m-%d %H:%M:%S")
        data["deadline"] = pd.to_datetime(data["deadline"], format="%Y-%m-%d")

        # Get Duration
        data["duration"] = (data["deadline"] - data["launched"]).dt.days

        # Quarter, month of launched and deadline date
        data["launch_month"] = data["launched"].dt.month
        data["launch_quarter"] = data["launched"].dt.quarter
        data["deadline_month"] = data["deadline"].dt.month
        data["deadline_quarter"] = data["deadline"].dt.quarter

        # Random hour, as we don't know
        data['launch_hour'] = random.randint(0, 23)

        # Title Features

        # Length of title
        data["title_length"] = data["name"].apply(lambda x: len(str(x)))

        # Number of words
        data["title_words"] = data["name"].apply(lambda x: len(str(x).split(' ')))

        # Number of symbols
        data['title_symbols'] = data["name"].apply(lambda x: str(x).count('!') + str(x).count('?'))

        # New column for if there are any backers
        data["backers_exist"] = np.where(data["backers"] > 0, "True", "False")

        # Make a mask for rows that contain backers
        mask_backers_exist = (data["backers"] > 0)

        # Enter 0 for where backers don't exist, and the pledged per backer for where they do
        data['pledged_per_backer'] = 0
        data.loc[mask_backers_exist, 'pledged_per_backer'] = data["pledged"] / data["backers"]

        # Get all unique values in col
        unique_vals = {}
        for column in cols_one_hot_encode:
            unique_vals[column] = baseDF[column].unique().tolist()

        vaz = []
        # Get column exists/not exists
        for column, vals in unique_vals.items():
            x = data[column].isin(vals).any()
            vaz.append(x)

        # Filter cols_one_hot_encode to get bad ones
        fillNaOn = []
        for col, b in zip(cols_one_hot_encode, vaz):
            if (not b):
                fillNaOn.append(col)

        # Replace bad ones with random of good ones
        for col in fillNaOn:
            data[col] = random.choice(unique_vals[col])
            self.notTooAcc = True

        # Add new value
        baseDF = baseDF.append(data)

        # Fix OHE names
        baseDF['category'].apply(self.replace_ampersand).apply(self.replace_hyphen).apply(self.remove_extraspace).apply(
            self.replace_space)
        baseDF['main_category'].apply(self.replace_ampersand).apply(self.replace_hyphen).apply(
            self.remove_extraspace).apply(self.replace_space)

        # Notify lack of accuracy to due NaN value
        if len(data.isna().any()[lambda x: x].tolist()) > 0:
            self.notTooAcc = True

        # Fill all NaN with mean first, then 0
        baseDF.fillna(baseDF.mean(), inplace=True)
        baseDF.fillna(0, inplace=True)

        # #One Hot Encode
        baseDF = pd.get_dummies(baseDF, prefix=cols_one_hot_encode, columns=cols_one_hot_encode)
        baseDF = baseDF.drop(
            ['launched', 'state', 'pledged', 'usd pledged', 'backers', 'usd_pledged_real', 'deadline', 'name',
             'launch_month', 'deadline_month', 'backers_exist', 'good_country', 'exchange_rate', 'usd_pledged_new'],
            axis=1)

        # Normalize
        normalizer = pickle.load(open('./datascience/data/normalizer.pkl', 'rb'))
        norm = pd.DataFrame(normalizer.transform(baseDF))

        # Pretty
        norm.columns = baseDF.columns
        norm.index = baseDF.index

        self.features = norm.columns.tolist()
        return norm.loc[1000, :]

    # Functions to fix the categories/main categories names
    def replace_ampersand(self, val):
        if isinstance(val, str):
            return val.replace('&', 'and')
        else:
            return val

    def replace_hyphen(self, val):
        if isinstance(val, str):
            return val.replace('-', '_')
        else:
            return val

    def remove_extraspace(self, val):
        if isinstance(val, str):
            return val.strip()
        else:
            return val

    def replace_space(self, val):
        if isinstance(val, str):
            return val.replace(' ', '_')
        else:
            return val

    def get_top_k_features(self, k, imp_vals, max_min):
        # Get most important features on both ends, +/-
        if max_min == 0:
            return np.argpartition(imp_vals, k)[:k]
        elif max_min == 1:
            return np.argpartition(imp_vals, -k)[-k:]

    def predict(self, jsonObj):
        # Get shap values
        x = self.preprocess(jsonObj).values.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x)

        imp_vals = shap_vals[0, :]

        # Get prediction
        label = self.model.predict(x).tolist()

        # So multiple of one kind isn't sent
        main_cat_done = False
        cat_done = False
        launch_hour_done = False
        launch_quarter_done = False

        # No. of features to get at beginning
        k = 5

        # Dictionary for storing results to send
        topN = {}

        # If success, only look at max. If fail, only look at min
        while len(topN) <=5:
            if label[0] == 1:
                max_indices = self.get_top_k_features(k, imp_vals, label[0])
                for i in range(0, k, 1):
                    feat_name = self.features[max_indices[i]]
                    imp_val = imp_vals[max_indices[i]]

                    if feat_name.startswith('main_category'):
                        if main_cat_done:
                            continue
                        else:
                            main_cat_done = True
                            topN[feat_name] = imp_val

                    elif feat_name.startswith('category'):
                        if cat_done:
                            continue
                        else:
                            cat_done = True
                            topN[feat_name] = imp_val

                    elif feat_name.startswith('launch_hour'):
                        if launch_hour_done:
                            continue
                        else:
                            launch_hour_done = True
                            topN[feat_name] = imp_val

                    elif feat_name.startswith('launch_quarter'):
                        if launch_quarter_done:
                            continue
                        else:
                            launch_quarter_done = True
                            topN[feat_name] = imp_val

                    else:
                        topN[feat_name] = imp_val

            elif label[0] == 0:
                min_indices = self.get_top_k_features(k, imp_vals, label[0])

                for i in range(0, k, 1):
                    feat_name = self.features[min_indices[i]]
                    imp_val = imp_vals[min_indices[i]]

                    if feat_name.startswith('main_category'):
                        if main_cat_done:
                            continue
                        else:
                            main_cat_done = True
                            topN[feat_name] = imp_val
                        continue

                    elif feat_name.startswith('category'):
                        if cat_done:
                            continue
                        else:
                            cat_done = True
                            topN[feat_name] = imp_val
                        continue
                    elif feat_name.startswith('launch_hour_done'):
                        if launch_hour_done:
                            continue
                        else:
                            launch_hour_done = True
                            topN[feat_name] = imp_val
                        continue
                    else:
                        topN[feat_name] = imp_val
            k = k+5

        # topN = sorted(topN.items(), key=operator.itemgetter(1))

        for key, value in topN.items():
            print("%s with imp %f" % (key, value))

        return label, self.notTooAcc, topN
