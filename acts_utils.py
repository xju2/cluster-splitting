
from argparse import Namespace
import os
def get_file_names(inputdir, evtid):
    evtid = "{:09}".format(evtid)
        
    cluster_fname = os.path.join(inputdir, "event{}-cells.csv".format(evtid))
    particle_fname = os.path.join(inputdir, "event{}-particles_final.csv".format(evtid))
    spacepoint_fname = os.path.join(inputdir, "event{}-spacepoint.csv".format(evtid))
    measurement_fname = os.path.join(inputdir, "event{}-measurements.csv".format(evtid))
    hit_fname = os.path.join(inputdir, "event{}-hits.csv".format(evtid))
    measure2hits_fname = os.path.join(inputdir, "event{}-measurement-simhit-map.csv".format(evtid))

    return Namespace(
        cluster=cluster_fname, particle=particle_fname,
        spacepoint=spacepoint_fname, measurement=measurement_fname,
        hit=hit_fname, measure2hits=measure2hits_fname,
    )