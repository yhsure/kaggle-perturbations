import os
import subprocess
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.colors as colors
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from scipy import sparse
import anndata as ad
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
seed = 0
np.random.seed(seed)

# -----------------------
# -  Plotting defaults  -
# -----------------------
pio.templates["myname"] = go.layout.Template(
    layout=go.Layout(
    colorway=['#636EFA', '#F8766D', '#00BF7D', '#A3A500', '#E76BF3'],
    autosize=True,
    width=700,
    height=500,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0,
        pad=0),
    font=dict(size=24,
          family="Palatino"),
    yaxis=dict(title=dict(font=dict(size=24))),
    xaxis=dict(title=dict(font=dict(size=24))),
    yaxis_tickfont_size=24,
    xaxis_tickfont_size=24,
    yaxis_titlefont_size=30,
    xaxis_titlefont_size=30,
    )
)
pio.templates.default = 'myname+ggplot2'

def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def remove_axes(fig):
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def set_plt_layout():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 18})
    

def reset_plt_layout():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('default')


def mrrmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2, axis=1)).mean()

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2, axis=1))


def save_preds(preds, gene_names, model_name="", dtype=np.float32):
    # prepare df
    df = pd.DataFrame(preds, columns=gene_names).astype(dtype)
    df.insert(0, "id", np.arange(len(df))) 

    # save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"submissions/{timestamp}_{model_name}.csv", index=False)


def submit_latest(msg=""):
    files = [f for f in os.listdir("submissions/") if f.endswith(".csv")]
    files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join("submissions/", x)), reverse=True)
    newest_csv = files_sorted[0]
    cmd = f"kaggle competitions submit -c open-problems-single-cell-perturbations -f 'submissions/{newest_csv}' -m '{msg}'"
    subprocess.run(cmd, shell=True)


def save_and_submit(preds, gene_names, model_name, msg=""):
    save_preds(preds, gene_names, model_name)
    submit_latest(msg=msg)


# --------------------
# -  Coloring utils  -
# --------------------
def truncate_colormap(name, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def take_cmap_colors(cmap, N, cmap_range=(0, 1), return_fmt="float"):
    """
    From https://cmasher.readthedocs.io/_modules/cmasher/utils.html#take_cmap_colors
    Takes `N` equally spaced colors from the provided colormap `cmap` and
    returns them.

    Parameters
    ----------
    cmap : str or :obj:`~matplotlib.colors.Colormap` object
        The registered name of the colormap in :mod:`matplotlib.cm` or its
        corresponding :obj:`~matplotlib.colors.Colormap` object.
    N : int or None
        The number of colors to take from the provided `cmap` within the given
        `cmap_range`.
        If *None*, take all colors in `cmap` within this range.

    Optional
    --------
    cmap_range : tuple of float. Default: (0, 1)
        The normalized value range in the colormap from which colors should be
        taken.
        By default, colors are taken from the entire colormap.
    return_fmt : {'float'/'norm'; 'int'/'8bit'; 'str'/'hex'}. Default: 'float'
        The format of the requested colors.
        If 'float'/'norm', the colors are returned as normalized RGB tuples.
        If 'int'/'8bit', the colors are returned as 8-bit RGB tuples.
        If 'str'/'hex', the colors are returned using their hexadecimal string
        representations.

    Returns
    -------
    colors : list of {tuple; str}
        The colors that were taken from the provided `cmap`.
    """
    from matplotlib.colors import Colormap, ListedColormap as LC, to_hex, to_rgb
    from matplotlib import cm as mplcm

    # Convert provided fmt to lowercase
    return_fmt = return_fmt.lower()

    # Obtain the colormap
    cmap = mplcm.get_cmap(cmap)

    # Check if provided cmap_range is valid
    if not ((0 <= cmap_range[0] <= 1) and (0 <= cmap_range[1] <= 1)):
        raise ValueError("Input argument 'cmap_range' does not contain "
                         "normalized values!")

    # Extract and convert start and stop to their integer indices (inclusive)
    start = int(np.floor(cmap_range[0]*cmap.N))
    stop = int(np.ceil(cmap_range[1]*cmap.N))-1

    # Pick colors
    if N is None:
        index = np.arange(start, stop+1, dtype=int)
    else:
        index = np.array(np.rint(np.linspace(start, stop, num=N)), dtype=int)
    colors = cmap(index)

    # Convert colors to proper format
    if return_fmt in ('float', 'norm', 'int', '8bit'):
        colors = np.apply_along_axis(to_rgb, 1, colors)
        if return_fmt in ('int', '8bit'):
            colors = np.array(np.rint(colors*255), dtype=int)
        colors = list(map(tuple, colors))
    else:
        colors = list(map((lambda x: to_hex(x).upper()), colors))

    # Return colors
    return(colors)


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    # From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
    import matplotlib.colors
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

def save_dgd(model, gmm, rep, val_rep, test_rep, loss_hist, model_params, path):
    if not os.path.exists('models'):
        os.makedirs('models')
    full_path = f"models/{datetime.now().strftime('%m%d_%H%M%S')}_{path}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'gmm_state_dict': gmm.state_dict(),
        'rep_state_dict': rep.state_dict(),
        'val_rep_state_dict': val_rep.state_dict(),
        'test_rep_state_dict': test_rep.state_dict(),
        'loss_hist': loss_hist,
        'model_params': model_params
    }, full_path)
    print(f"Model saved to {full_path}.")

def load_dgd(path, dev="cpu", flavor="normal"):
    from _models import DGD, softball, GaussianMixture, RepresentationLayer, DGDTaskFromLatent
    from _training import train_dgd

    if not os.path.exists(path):
        path = f"models/{path}"
    checkpoint = torch.load(path, map_location=dev)
    p = checkpoint['model_params']    
    nmix = p['nmix']; nsample = p['nsample']; nsample_val = p['nsample_val']; nsample_test = p['nsample_test']; 
    dim_list = p['dim_list']; extra_outputs = p['extra_outputs']; n_conditional_vars = p['n_conditional_vars'] if 'n_conditional_vars' in p else 0

    # initialize
    mean_prior = softball(dim=dim_list[0],radius=1,a=5)
    gmm = GaussianMixture(Nmix=nmix, dim=dim_list[0], type='diagonal', mean_prior=mean_prior, sd_init=(0.05,1), alpha=1)
    rep = RepresentationLayer(nrep=dim_list[0],nsample=nsample,values=torch.zeros(size=(nsample,dim_list[0])))
    val_rep = RepresentationLayer(nrep=dim_list[0],nsample=nsample_val,values=torch.zeros(size=(nsample_val,dim_list[0])))
    test_rep = RepresentationLayer(nrep=dim_list[0],nsample=nsample_test,values=torch.zeros(size=(nsample_test,dim_list[0])))

    if flavor == "normal":
        model = DGD(dim_list=dim_list, r_init=2, scaling_type="library", extra_outputs=0, n_conditional_vars=92)
    elif flavor == "from_latent":
        model = DGDTaskFromLatent(dim_list=dim_list, r_init=2, scaling_type="library", 
                                  extra_outputs=extra_outputs, n_conditional_vars=9)

    # load
    model.load_state_dict(checkpoint['model_state_dict'])
    gmm.load_state_dict(checkpoint['gmm_state_dict'])
    rep.load_state_dict(checkpoint['rep_state_dict'])
    val_rep.load_state_dict(checkpoint['val_rep_state_dict'])
    test_rep.load_state_dict(checkpoint['test_rep_state_dict'])
    loss_hist = checkpoint['loss_hist']
    model = model.to(dev); gmm = gmm.to(dev); rep = rep.to(dev); val_rep = val_rep.to(dev); test_rep = test_rep.to(dev)
    print(f"Model loaded from {path}.")
    return model, gmm, rep, val_rep, test_rep, loss_hist, p


def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    """
    Adapted from this forum post: 
    https://discourse.scverse.org/t/group-sum-rows-based-on-jobs-feature/371/4
    """
    
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    # sum `.X` entries for each unique value in `col`
    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (
            np.broadcast_to(True, adata.n_obs),
            (cat.codes, np.arange(adata.n_obs))
        ),
        shape=(len(cat.categories), adata.n_obs),
    )
    sum_adata = ad.AnnData(
        indicator @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=cat.categories),
        dtype=adata.X.dtype,
    )
    
    # copy over `.obs` values that have a one-to-one-mapping with `.obs[col]`
    obs_cols = adata.obs.columns
    obs_cols = list(set(adata.obs.columns) - set([col]))
    
    one_to_one_mapped_obs_cols = []
    nunique_in_col = adata.obs[col].nunique()
    for other_col in obs_cols:
        if len(adata.obs[[col, other_col]].drop_duplicates()) == nunique_in_col:
            one_to_one_mapped_obs_cols.append(other_col)

    joining_df = adata.obs[[col] + one_to_one_mapped_obs_cols].drop_duplicates().set_index(col)
    assert (sum_adata.obs.index == sum_adata.obs.join(joining_df).index).all()
    sum_adata.obs = sum_adata.obs.join(joining_df)
    sum_adata.obs.index.name = col
    sum_adata.obs = sum_adata.obs.reset_index()
    sum_adata.obs.index = sum_adata.obs.index.astype('str')

    return sum_adata

def _run_limma_for_cell_type(bulk_adata, de_pert_cols, control_compound, design='~0+Rpert+donor_id+plate_name+row'):
    import scripts.limma_utils as limma_utils
    import binascii

    print("Running limma for cell type", bulk_adata.obs["cell_type"][0])

    bulk_adata = bulk_adata.copy()    
    compound_name_col = de_pert_cols[0]
    
    # limma doesn't like dashes etc. in the compound names
    rpert_mapping = bulk_adata.obs[compound_name_col].drop_duplicates() \
        .reset_index(drop=True).reset_index() \
        .set_index(compound_name_col)['index'].to_dict()
    
    bulk_adata.obs['Rpert'] = bulk_adata.obs.apply(
        lambda row: rpert_mapping[row[compound_name_col]], 
        axis='columns',
    ).astype('str')

    compound_name_to_Rpert = bulk_adata.obs.set_index(compound_name_col)['Rpert'].to_dict()
    ref_pert = compound_name_to_Rpert[control_compound]
            
    random_string = binascii.b2a_hex(os.urandom(15)).decode()
    
    limma_utils.limma_fit(
        bulk_adata, 
        design=design,
        output_path=f'output/{random_string}_limma.rds',
        plot_output_path=f'output/{random_string}_voom',
        exec_path='scripts/limma_fit.r',
        verbose=True,
    )

    pert_de_dfs = []
    


    for pert in bulk_adata.obs['Rpert'].unique():
        if pert == ref_pert:
            continue

        pert_de_df = limma_utils.limma_contrast(
            fit_path=f'output/{random_string}_limma.rds',
            contrast='Rpert'+pert+'-Rpert'+ref_pert,
            exec_path='scripts/limma_contrast.r',
        )

        pert_de_df['Rpert'] = pert

        pert_obs = bulk_adata.obs[bulk_adata.obs['Rpert'].eq(pert)]
        for col in de_pert_cols:
            pert_de_df[col] = pert_obs[col].unique()[0]
        pert_de_dfs.append(pert_de_df)

    de_df = pd.concat(pert_de_dfs, axis=0)

    try:
        os.remove(f'output/{random_string}_limma.rds')
        os.remove(f'output/{random_string}_voom')
    except FileNotFoundError:
        pass
    
    return de_df

def convert_de_df_to_anndata(de_df, pert_cols, de_sig_cutoff):
    de_df = de_df.copy()
    zero_pval_selection = de_df['P.Value'].eq(0)
    de_df.loc[zero_pval_selection, 'P.Value'] = np.finfo(np.float64).eps

    de_df['sign_log10_pval'] = np.sign(de_df['logFC']) * -np.log10(de_df['P.Value'])
    de_df['is_de'] = de_df['P.Value'].lt(de_sig_cutoff)
    de_df['is_de_adj'] = de_df['adj.P.Val'].lt(de_sig_cutoff)

    de_feature_dfs = {}
    for feature in ['is_de', 'is_de_adj', 'sign_log10_pval', 'logFC', 'P.Value', 'adj.P.Val']:
        df = de_df.reset_index()
        df.index = df.index.astype('string')
        df = df.pivot_table(
            index=['gene'], 
            columns=pert_cols,
            values=[feature],
            dropna=True,
        )
        de_feature_dfs[feature] = df

    pval = de_feature_dfs['sign_log10_pval'].T
    pval = pval.reset_index().drop(columns=['level_0'], errors='ignore')
    de_adata = ad.AnnData(pval.drop(columns=pert_cols), dtype=np.float64, obs=pval[pert_cols])
    de_adata.obs.index = de_adata.obs.index.astype('string')

    for feature in ['is_de', 'is_de_adj', 'logFC', 'P.Value', 'adj.P.Val']:
        de_adata.layers[feature] = de_feature_dfs[feature].to_numpy().T

    return de_adata



def check_system_usage():
    import psutil
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    print(f"System Usage:\n CPU: {cpu_usage}%\n Memory: {memory_usage}%\n Disk: {disk_usage}%")


def save_model(model, optimizer, loss_hist, path):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    full_path = f"checkpoints/{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hist': loss_hist
    }, full_path)
    print(f"Model saved to {full_path}.")


def make_dataset():
    from scipy.sparse import coo_matrix
    import polars as pl

    add = pl.read_parquet('data/adata_train.parquet')
    excluded_ids = pl.read_csv('data/adata_excluded_ids.csv')

    adata_filtered = add.join(excluded_ids, on=['obs_id', 'gene'], how='anti')
    df = adata_filtered.to_pandas()
    del adata_filtered
    gc.collect()

    p = "data/"
    # Create sparse matrix
    rows = df['obs_id'].astype('category').cat.codes
    cols = df['gene'].astype('category').cat.codes
    data = df['count'].values
    sparse_matrix = coo_matrix((data, (rows, cols)))
    sparse_matrix = sparse_matrix.tocsr()

    # Create AnnData object
    obs = pd.read_csv('data/adata_obs_meta.csv')
    var = pd.DataFrame(index=df['gene'].astype('category').cat.categories)
    adata = ad.AnnData(X=sparse_matrix, var=var, obs=obs)

    adata.write(p+'given_filtered.h5ad')

    pos_controls = ["Dabrafenib", "Belinostat"]
    adata = adata[~adata.obs["sm_name"].isin(pos_controls)].copy()

    adata.write(p+'given_filtered_nocontrols.h5ad')

    import gc
    del adata, df, obs, var, sparse_matrix, rows, cols, data
    gc.collect()

def make_predicted_counts():
    adata = ad.read_h5ad("data/given_filtered.h5ad")
    adata = adata[adata.obs["sm_name"] != "Dimethyl Sulfoxide"]
    adata = adata[~adata.obs["control"]]
    adata.obs = adata.obs.sort_values(["cell_type", "sm_name"]).reset_index(drop=True)

    X = adata.obs[["sm_name", "cell_type", "plate_name", "well"]]

    # make a mapping from (plate_name, well) to sm_name
    plate_well_to_sm_name = {}
    for plate_name in X["plate_name"].unique():
        for well in X[X["plate_name"] == plate_name]["well"].unique():
            plate_well_to_sm_name[(plate_name, well)] = X[(X["plate_name"] == plate_name) & (X["well"] == well)]["sm_name"].unique()[0]

    max_counts = np.asarray(adata.X.max(axis=1).todense()).squeeze()
    df = pd.concat([X, pd.Series(max_counts, name="max")], axis=1)
    max_df = df.groupby(["cell_type", "plate_name", "well"])["max"].median().reset_index()
    max_df["count"] = df.groupby(["cell_type", "plate_name", "well"])["max"].count().reset_index()["max"]


    # split into train and test
    train_i = max_df["count"] > 0
    test_i = max_df["cell_type"].isin(["B cells", "Myeloid cells"]) & (max_df["count"] == 0)

    train = max_df[train_i].iloc[:, :-2]
    test = max_df[test_i].iloc[:, :-2]

    ytrain = max_df[train_i].iloc[:, -2:]
    ytest = max_df[test_i].iloc[:, -2:]

    one_hot = ce.OneHotEncoder(cols=["cell_type", "plate_name", "well"], use_cat_names=True)
    one_hot.fit(train)

    train, test = one_hot.transform(train).values, one_hot.transform(test).values
    ytrain, ytest = ytrain.values, ytest.values

    # generate validation set
    train, val, ytrain, yval = train_test_split(train, ytrain, test_size=0.1, random_state=seed)


    # train random forest regressors
    rf_max = RandomForestRegressor(n_estimators=1000, random_state=seed, n_jobs=-1)
    rf_max.fit(train, ytrain[:, 0])
    print("max score:", rf_max.score(val, yval[:, 0]))
        
    rf_count = RandomForestRegressor(n_estimators=1000, random_state=seed, n_jobs=-1)
    rf_count.fit(train, ytrain[:, 1])
    print("count score:", rf_count.score(val, yval[:, 1]))
        

    # predict on test set
    max_preds = rf_max.predict(test)
    count_preds = rf_count.predict(test)

    preds = np.stack([max_preds, (count_preds + 0.5).astype(int)], axis=1)
    preds = pd.DataFrame(preds, columns=["max", "count"])
    preds["cell_type"] = max_df[test_i]["cell_type"].values
    preds["plate_name"] = max_df[test_i]["plate_name"].values
    preds["well"] = max_df[test_i]["well"].values
    preds["sm_name"] = preds.apply(lambda x: plate_well_to_sm_name[(x["plate_name"], x["well"])], axis=1)

    # sort according to cell type and sm_name
    preds = preds.sort_values(["cell_type", "sm_name"]).reset_index(drop=True)

    # asssert that all sm_names in id_map are in preds
    id_map = pd.read_csv("data/id_map.csv")
    assert set(id_map["sm_name"].unique()) - set(preds["sm_name"].unique()) == set()

    # save to csv
    preds.to_csv("data/predicted_counts.csv", index=False)
