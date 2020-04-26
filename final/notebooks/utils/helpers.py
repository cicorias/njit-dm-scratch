import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown, Latex


def download_wine_data(urls, force = False):
    import os
    from pathlib import Path, PurePath
    from urllib.parse import urlparse
    import requests

    data_dir = Path(os.path.join(os.getcwd(), 'data'))
    data_dir.mkdir(exist_ok=True)

    filenames = []
    for url in urls:
        p = urlparse(url)
        filename = os.path.basename(p.path)
        
        filename = os.path.basename(p.path)
        filenames.append(os.path.join(data_dir, filename))
        
        if os.path.exists(data_dir) and not force:
            print('path exist and not forced')
        else:
            r = requests.get(url, stream = True)
            full_path = os.path.join(data_dir, filename)
            with open(full_path, 'wb') as file:  
                for block in r.iter_content(chunk_size = 1024): 
                    if block:  
                        file.write(block)
        
    return filenames

def pull_data(force = False):
    red_url = 'https://scicorianjit.blob.core.windows.net/njit/winequality-red.csv'
    white_url = 'https://scicorianjit.blob.core.windows.net/njit/winequality-white.csv'
    data_urls = [red_url, white_url]
    return download_wine_data(data_urls, force = force)
    

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


def get_features_and_labels(df_all, binary=False):
    features_all = df_all.iloc[:, 0:11] #syntax is 'UP to but NOT (<) when a range.'
    if binary:
        labels_all = pd.Series(np.where(df_all.iloc[:, 11].to_numpy() > 5, 1, 0))
    else:
        labels_all = df_all.iloc[:, 11]
    return features_all, labels_all
    

def get_labels(df_all):
    labels_all = df_all.iloc[:, 11]
    return labels_all

def get_df_no_color(df_all, binary=False):
    df = df_all.iloc[:, 0:12]
    
    if binary:
        df.loc[:, 'quality'] = pd.Series(np.where(df.loc[:, 'quality'].to_numpy() > 5, 1, 0))
    
    return df


def pull_and_load_data(force = False):
    files = pull_data(force = force)
    return load_data(files[0], files[1])
    

def bins_labels(bins, shift = 0.25, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins) + bin_w/2, max(bins), bin_w) - shift, bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


def convert_to_quantiles(df, labels = ['d-1stQ', 'd-2ndQ', 'd-3rdQ', 'd-4thQ']):
    df_out = pd.DataFrame()
    df_bins = pd.DataFrame()

    for k, s in df.iteritems():
        df_out[k], df_bins[k] = pd.qcut(s, 4, labels = labels, retbins = True)
        
    

    df_bins['sum_stats'] = ['min', 'd-1stQ', 'd-2ndQ', 'd-3rdQ', 'max']
    return df_out, df_bins

    

### display stuff

def emit_md(markdown_text):
    display(Markdown(markdown_text))


def emit_latex(latext_text):
    display(Latex(latext_text))

#this is for development only and not relevant
def in_script():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            print('running in a Jupyter notebook')
            return False
        elif shell == 'TerminalInteractiveShell':
            print('running in a IPython terminal')
            return True
        else:
            print('unknow shell: {}'.format(shell))
            print('but doesn\'t appear to be a python script')

    except NameError:
        return True
    
def in_notebook():
    return not in_script()
    
def plot_to_file(plt, filename):
    from matplotlib import pyplot as plt
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')
    plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')

