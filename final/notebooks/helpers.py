import os
import numpy as np
import pandas as pd


def load_data(data_path_red, data_path_white):
    column_header = ["fixed_acidity", "volatile_acidity", "citric_acid", \
                     "residual_sugar", "chlorides", "free_sulfur_dioxide", \
                     "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

    df_red = pd.read_csv(data_path_red, sep = ';', names = column_header, header=0)
    df_red['color'] = 1

    df_white = pd.read_csv(data_path_white, sep = ';', names = column_header, header=0)
    df_white['color'] = 0
    
    total_rows = len(df_white) + len(df_red)
    df_all = df_red.append(df_white)
    assert(len(df_all) == total_rows)

    return df_red, df_white, df_all


def get_features_and_labels(df_all):
    features_all = df_all.iloc[:, 0:11] #syntax is 'UP to but NOT (<) when a range.'
    labels_all = df_all.iloc[:, 11]
    return features_all, labels_all

def get_df_no_color(df_all):
    df = df_all.iloc[:, 0:12]
    return df