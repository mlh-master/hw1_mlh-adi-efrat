# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    df = CTG_features.applymap(lambda x: pd.to_numeric(x, errors='coerce')) #replaces all non-numeric to NaN
    df = df.drop(columns=extra_feature) # drop a column
    c_ctg = {k: df[k].dropna() for k in df}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.applymap(lambda x: pd.to_numeric(x, errors='coerce')) #replaces all non-numeric to NaN
    c_cdf = c_cdf.drop(columns=extra_feature) #drop a column 
    for column in c_cdf.columns:
        c_cdf[column] = c_cdf[column].apply(lambda x: np.random.choice(c_cdf[column]) if np.isnan(x) else x)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for col in c_feat.columns:
        d_summary[col] = {'min': c_feat[col].min(),
                          'Q1': np.percentile(c_feat[col], 25),
                          'median': np.percentile(c_feat[col], 50),
                          'Q3': np.percentile(c_feat[col], 75),
                          'max': c_feat[col].max()}
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for col in c_feat.columns:
        Q1 = d_summary[col]['Q1']
        Q3 = d_summary[col]['Q3']
        outl1 = Q1 - 1.5 * (Q3 - Q1)
        outl2 = Q3 + 1.5 * (Q3 - Q1)
        c_no_outlier[col] = {k: k if (outl1 <= k <= outl2) else np.nan for k in c_feat[col]}
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature].apply(lambda x: np.nan if (x > thresh) else x)    
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = {}
    if mode == 'standard':
        for col in CTG_features.columns:
            col_mean = np.mean(CTG_features[col])
            col_std = np.std(CTG_features[col])
            nsd_res[col] = CTG_features.loc[:,col].apply(lambda x: ((x - col_mean) / col_std))
    elif mode == 'MinMax':
        for col in CTG_features.columns:
            col_min = np.min(CTG_features[col])
            col_max = np.max(CTG_features[col])
            nsd_res[col] = CTG_features.loc[:,col].apply(lambda x: ((x - col_min) / (col_max - col_min)))
    elif mode == 'mean':
        for col in CTG_features.columns:
            col_mean = np.mean(CTG_features[col])
            col_min = np.min(CTG_features[col])
            col_max = np.max(CTG_features[col])
            nsd_res[col] = CTG_features.loc[:,col].apply(lambda x: ((x - col_mean) / (col_max - col_min)))
    else:
        nsd_res = CTG_features.to_dict('series')
    if (flag):
        xlbl = ['beats/min','%']
        axarr = pd.DataFrame(nsd_res).hist(column=[x, y], bins=100, figsize=(5, 5))
        for i,ax in enumerate(axarr.flatten()):
            ax.set_xlabel(xlbl[i])
            ax.set_ylabel("Count")
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
