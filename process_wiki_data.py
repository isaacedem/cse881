__author__ = 'idorgbefu'

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

def read_files():
    #I hard-coded file names
    article_art = pd.read_json(open('./article.art'))
    article_rev = pd.read_json(open('./article.rev'))
    article_rev['is_talk'] = 0

    talk_art = pd.read_json(open('./talk.art'))
    talk_rev = pd.read_json(open('./talk.rev'))
    talk_rev['is_talk'] = 1

    revs = pd.concat([talk_rev, article_rev])
    arts = article_art.set_index('title').join(talk_art.set_index('title'), rsuffix='_talk').reset_index()

    if 'anon' not in revs.columns:
        revs['anon'] = np.nan
    if 'minor' not in revs.columns:
        revs['minor'] = np.nan
    if 'suppressed' not in revs.columns:
        revs['suppressed'] = np.nan
    if 'userhidden' not in revs.columns:
        revs['userhidden'] = np.nan

    revs.loc[~revs.anon.isnull(), 'anon'] = 1
    revs.loc[~revs.minor.isnull(), 'minor'] = 1
    revs.loc[~revs.suppressed.isnull(), 'suppressed'] = 1
    revs.loc[~revs.userhidden.isnull(), 'userhidden'] = 1

    revs.loc[revs.anon.isnull(), 'anon'] = 0
    revs.loc[revs.minor.isnull(), 'minor'] = 0
    revs.loc[revs.suppressed.isnull(), 'suppressed'] = 0
    revs.loc[revs.userhidden.isnull(), 'userhidden'] = 0

    return [arts, revs]

def compute_features(titles, revs):
    #percent of article editors that are talk editors
    #mean, median edits per editor

    features = titles.copy()

    features['num_revisions'] = (revs.groupby('title').size())[0]
    features['num_talk_revisions'] = (revs.groupby('title').is_talk.sum().fillna(0))[0]
    features['num_art_revisions'] = (features['num_revisions'] - features['num_talk_revisions'])[0]

    features['percent_revisions_talk'] = (features.num_talk_revisions / features.num_revisions)[0]
    features['percent_revisions_art'] = (features.num_art_revisions / features.num_revisions)[0]
    features['num_talkr_per_num_artr'] = (features.num_talk_revisions / features.num_art_revisions)[0]

    features['num_minor'] = (revs.groupby('title').minor.sum())[0]
    features['num_talk_minor'] = (revs[revs.is_talk == 1].groupby('title').minor.sum())[0]
    features['num_talk_minor'].fillna(0, inplace=True)
    features['num_art_minor'] = (revs[revs.is_talk == 0].groupby('title').minor.sum())[0]
    features['num_art_minor'].fillna(0, inplace=True)

    features['percent_minor'] = (features.num_minor / features.num_revisions)[0]
    features['percent_talk_minor'] = (features.num_talk_minor / features.num_talk_revisions)[0]
    features['percent_talk_minor'] = (features.percent_talk_minor.fillna(0))[0]
    features['percent_art_minor'] = (features.num_art_minor / features.num_art_revisions)[0]
    features['percent_minor_are_art'] = (features.num_art_minor / (features.num_art_minor + features.num_talk_minor))[0]
    features['percent_minor_are_art'] = (features.percent_minor_are_art.fillna(0))[0]

    features['num_editors'] = (revs.groupby('title').user.nunique())[0]
    features['num_talk_editors'] = (revs[revs.is_talk == 1].groupby('title').user.nunique())[0]
    features['num_art_editors'] = (revs[revs.is_talk == 0].groupby('title').user.nunique())[0]
    features['num_art_talk_diff_editors'] = (features['num_art_editors'] - features['num_talk_editors'])[0]
    # features['num_art_editors_are_talk'] = revs[revs]

    features['num_anon'] = (revs.groupby('title').anon.sum())[0]
    features['num_talk_anon'] = (revs[revs.is_talk == 1].groupby('title').anon.sum())[0]
    features['num_art_anon'] = (revs[revs.is_talk == 0].groupby('title').anon.sum())[0]
    features['percent_talk_anon'] = (features.num_talk_anon / (features.num_talk_revisions))[0]
    features['percent_art_anon'] = (features.num_art_anon / (features.num_art_revisions))[0]

    features['num_suppressed'] = (revs.groupby('title').anon.sum())[0]
    features['num_talk_suppressed'] = (revs[revs.is_talk == 1].groupby('title').suppressed.sum())[0]
    features['num_art_suppressed'] = (revs[revs.is_talk == 0].groupby('title').suppressed.sum())[0]
    features['percent_talk_suppressed'] = (features.num_talk_suppressed / features.num_talk_revisions)[0]
    features['percent_art_suppressed'] = (features.num_art_suppressed / features.num_art_revisions)[0]

    return features

#call this when you save new files. will return pandas dataframe with the features of that sample
def process_data():
    [titles, revs] = read_files()
    return compute_features(titles, revs)

#this will return a classifier object
def load_classifier():
    return joblib.load('classifier.pk')

#writes a classifier object
def write_classifier(clf):
    joblib.dump(clf, 'classifier.pk')

#predict probability given a classifier and the features for a sample
def predict(clf, data):
    return clf.predict_proba(data[[c for c in data.columns if c !='title']])

#update weights of classifier
def update_weights(clf, data):
    pass

#write our data
def write_data(x):
    data.to_pickle('data')

#load our data
def load_data():
    return pd.read_pickle('data')

#return a dictionary with summary statistics
def summary_stats(revs):
    stats = {}
    stats['Number of Editors'] = revs.user.nunique()
    stats['Number of Anonymous Edits'] = revs.anon.sum()
    stats['Number of Revisions'] = len(revs)
