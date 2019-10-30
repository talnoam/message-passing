import uproot
import pandas as pd
import numpy as np
import itertools
import h5py

graph_variables = [
'trk_node_index',
 'trk_node_d0',
 'trk_node_z0',
 'trk_node_pt',
 'trk_node_eta',
 'trk_node_signed_d0',
 'trk_node_signed_z0',
 'trk_node_phi0',
 'trk_node_theta',
 'trk_node_qoverp',
 'trk_node_jetfitter_index',
 'jf_node_index',
 'jf_node_x',
 'jf_node_y',
 'jf_node_z',
 'particle_node_index',
 'particle_node_charge',
 'particle_node_status',
 'particle_node_inJet',
 'particle_node_pdgid',
 'particle_node_E',
 'particle_node_px',
 'particle_node_py',
 'particle_node_pz',
 'particle_node_prod_x',
 'particle_node_prod_y',
 'particle_node_prod_z',
 'particle_node_decay_x',
 'particle_node_decay_y',
 'particle_node_decay_z',
 'edge_start',
 'edge_end']

eventvars = ['eventnb','actmu','PVx','PVy','PVz','truth_PVx','truth_PVy','truth_PVz']

jetvars = ['jet_pt', 'jet_eta',  'jet_phi','jet_LabDr_HadF','jet_DoubleHadLabel','jet_JVT',
          'jet_index']

selectionVars =  ['jet_aliveAfterOR','jet_aliveAfterORmu']

def load_df_column(tree,varname,vartype='jetlevel',selected_index=[]):
    if varname=='jet_index':
        col = tree.pandas.df(['eventnb','jet_pt'],flatten=True)
        j_idx = col.index.get_level_values(1).values
        col = pd.DataFrame( j_idx ,columns = [varname])
    elif varname=='n_jets':
        col = tree.pandas.df(['eventnb','jet_pt'],flatten=True)
        n_jets = col.groupby('entry')['jet_pt'].nunique().values
        col = pd.DataFrame( np.repeat(n_jets,n_jets)  ,columns = [varname])
    elif vartype == 'eventlevel':
        col = tree.pandas.df([varname,'jet_pt'],flatten=True)
        col = col.reset_index(drop=True)
        col = col[varname]
    else:
        col = tree.pandas.df([varname],flatten=True)
        col = col.reset_index(drop=True)
    
    if vartype=='perjetlevel':
        flat = []
        
        for eventlist in col.values:
            
            for jetlist in eventlist[0]:
                flat.append(jetlist)
        col = pd.DataFrame( np.array(flat) )
        col.columns = [varname]
        
    if len(selected_index) > 0:
        col = col.loc[selected_index]
        print(varname)
    
    return col


def get_df(tree,lists_of_vars,vartypes,selected_index=[]):
    dfs = []
    
    for i, list_of_vars in enumerate(lists_of_vars):
        df = pd.concat([ load_df_column(tree,varname,vartypes[i],selected_index) for varname in list_of_vars],axis=1,sort=False)
        dfs.append(df)
    return pd.concat(dfs,axis=1,sort=False)