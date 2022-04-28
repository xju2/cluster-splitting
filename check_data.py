#!/usr/bin/env python
# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from acts_utils import get_file_names

# %%
def cal_pt(px, py):
    return np.sqrt(px**2 + py**2)

def plot_cluster_shape(cluster, outname=None):
    min0 = np.min(cluster['channel0'])
    max0 = np.max(cluster['channel0'])
    min1 = np.min(cluster['channel1'])
    max1 = np.max(cluster['channel1'])
    print(min0, max0, min1, max1)
    # the matrix
    matrix = np.zeros(((max1-min1+3),(max0-min0+3)))
    for pixel in cluster.values :
        i0 = int(pixel[2]-min0+1)
        i1 = int(pixel[3]-min1+1)
        value = pixel[5]
        matrix[i1][i0] = value 

    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    im = ax.imshow(matrix, interpolation='none', cmap=plt.get_cmap('Greys'))
    fig.colorbar(im)
    if outname:
        plt.savefig(outname)

data_dir = '/media/DataOcean/projects/tracking/dense/data/qcd_1800GeV'
evtid = 10
fnames = get_file_names(data_dir, evtid)
# %%
fnames
# %%
df_cluster = pd.read_csv(fnames.cluster)
# %%
df_cluster
#%%
df_sp = pd.read_csv(fnames.spacepoint)
# %%
df_sp

# %%
df_measurements = pd.read_csv(fnames.measurement)
df_measurements
# %%
df_hit = pd.read_csv(fnames.hit)
# %%
df_hit
# %%
df_particle = pd.read_csv(fnames.particle)
df_particle['pt'] = cal_pt(df_particle.px, df_particle.py)
plt.hist(df_particle.pt, bins=100, histtype='step', lw=2, log=True)
plt.xlabel("pT [GeV]", fontsize=14)
plt.ylabel("Tracks", fontsize=14)
# %%
df_particle[df_particle.pt > 100]
# %%
ap = df_hit[df_hit.particle_id == 4503600332013568]

# %%
df_measure_map = pd.read_csv(fnames.measure2hits)
# %%
df_sp = pd.read_csv(fnames.spacepoint)
am = df_sp[df_hit.particle_id == 4503600332013568]
# %%
plt.scatter(ap.tx, ap.ty, marker='x', edgecolors='none')
plt.scatter(am.x, am.y, s=80, marker='o', facecolor='none', edgecolors='r')

# %%
tr = cal_pt(ap.tx, ap.ty)
rr = cal_pt(am.x, am.y)
plt.scatter(ap.tz, tr, marker='x', edgecolors='none')
plt.scatter(am.z, rr, s=80, marker='o', facecolor='none', edgecolors='r')
# %%
merged_clusters = df_hit.groupby("geometry_id")['particle_id'].count()
merged_clusters = merged_clusters[merged_clusters > 1]
# %%
merged_clusters[merged_clusters > 10]

# %%
plt.hist(merged_clusters.values, bins=50, range=(0, 50))
# %%
np.max(merged_clusters)
# %%
df_hit[df_hit.geometry_id == 576460889742377075]
# %%
df_hit[df_hit.particle_id == 4503599778365440]
# %%
df_particle[df_particle.particle_id == 4503599778365440]
# %%
df_cluster[df_cluster.geometry_id == 576460889742377075]

# %%
# 504403295704449036
# 576461164620284203
cc = df_cluster[df_cluster.geometry_id == 504403295704449036]
# %%
plot_cluster_shape(cc)

# %%
cc
# %%
plt.hist(cc['value'])
# %%
aa = df_cluster.groupby("hit_id")['channel0'].count()
# %%
aa [aa > 6]
# %%
plot_cluster_shape(df_cluster[df_cluster.hit_id == 268])
# %%
df_cluster[df_cluster.hit_id == 268]
# %%
df_hit
# %%
np.unique(df_cluster.hit_id)
# %%
