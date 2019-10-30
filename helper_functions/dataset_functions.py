import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl import BatchedDGLGraph as bg_custom
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def custom_to(self,device='',non_blocking=True):
	return self

bg_custom.to = custom_to


def create_dgl(graph_dict):
	node_feature_size = 8
	pt_eta_size = 2
	
	node_list = graph.nodes
	
	if shuffle:
		np.random.shuffle(node_list)
	
	n_nodes = len( node_list )
	
	g = dgl.DGLGraph()
	g.add_nodes(n_nodes)

	node_features = np.zeros((n_nodes,node_feature_size))
	node_labels = np.zeros((n_nodes))
	pt_eta = np.zeros((n_nodes,2))
	e_label = np.zeros((n_nodes*(n_nodes-1))) #label for edge, (secondary vertex or not)

	pt = graph.pt
	eta = graph.eta

	pt_mean, pt_var = 62321.473, 34734.258
	pt = (pt-pt_mean)/pt_var

	for node_i,node in enumerate(node_list):
		
		pt_eta[node_i] = np.array([pt,eta])

		node_features[node_i] = np.array(node.matched_track)
		
		node_labels[node_i] = node.vertex_idx
	e_idx = -1
	
	for node_i,node in enumerate(node_list):
		for node_j,node2 in enumerate(node_list):
			if node_i==node_j:
				continue
			e_idx+=1
			g.add_edge(node_i,node_j)
			if node.vertex_idx == node2.vertex_idx: #and node.vertex_idx > 1:
				e_label[e_idx] = 1
			else:
				e_label[e_idx] = 0
	
	return_dict = {'graph':g,'flav':graph.flav,
	'pt_eta' : torch.FloatTensor(pt_eta),
	'node_features' : torch.FloatTensor(node_features),
	'node_labels':  torch.LongTensor(node_labels),
	'edge_labels' : torch.LongTensor(e_label),
	}
 
	return return_dict


class JetGraphDataset(Dataset):
	
	def __init__(self, df_list ):
		
		self.df = pd.concat( df_list )
		self.df = self.df.sample(frac=1).reset_index(drop=True)
		n_nodes = np.array( [len(x) for x in self.df.node_labels] )
		self.df = self.df.drop(np.where(n_nodes<2)[0])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		
		sample_df = self.df.iloc[idx]
		n_nodes = len(sample_df.node_labels)

		g = dgl.DGLGraph()
		g.add_nodes(n_nodes)

		flav = sample_df.jet_DoubleHadLabel
		n_lables = sample_df.node_labels
		
		# if flav==0:
		# 	n_lables = np.zeros(len(n_lables))

		e_label = []
		for node_i in range(n_nodes):
			for node_j in range(n_nodes):
				if node_i==node_j:
					continue
				g.add_edge(node_i,node_j)

				if n_lables[node_i] == n_lables[node_j]:  #and n_lables[node_i]  > 1:
					e_label.append( 1 )
				else:
					e_label.append( 0 )

		pt = sample_df.jet_pt
		eta = sample_df.jet_eta

		pt_mean, pt_var = 62321.473, 34734.258
		pt = (pt-pt_mean)/pt_var

		
		pt_eta = np.repeat( [[pt,eta]] , n_nodes,axis=0)

		pt_eta[node_i] = np.array([pt,eta])
		sample = {'node_features': torch.FloatTensor(sample_df.node_features_v2),
		'node_labels' : torch.LongTensor(n_lables),
		'pt_eta' : torch.FloatTensor(pt_eta),
		'flav': flav,
		'graph' : g,
		'edge_labels': torch.LongTensor(e_label),
		'jetfitter' : sample_df.jf_vtx_idxs
		}
		

		return sample

def transform_features(arr):
	
	mean_std_dict = {
	 	0: [0, 0.5],
 		1: [0, 0.5],
 		2: [0, 0.2],
 		3: [0, 0.2],
 		4: [0.994139641899691, 0.010220],
 		5: [15500, 18151]
	}

	new_arr = np.zeros(arr.shape)
	for col_i in range(len(mean_std_dict)):
		
		mean,std = mean_std_dict[col_i]
		new_arr[:,col_i] = (arr[:,col_i]-mean)/std
	return new_arr


def create_batch(batch):
	
	graphs = [x['graph'] for x in batch]
	batched_graph = bg_custom(graphs,"__ALL__","__ALL__")
	
	node_labels = torch.cat([x['node_labels'] for x in batch],dim=0).view(-1).long()
	#edge_labels = torch.cat([x['edge_labels'] for x in batch],dim=0).view(-1,1).long()

	node_features = torch.cat([x['node_features'] for x in batch],dim=0)
	
	node_features = torch.FloatTensor(transform_features(node_features.data.numpy()))
	
	pt_eta = torch.cat([x['pt_eta'] for x in batch],dim=0)
	
	for x_i, x in enumerate(batch):
		if len(x['pt_eta']) < 1:
			print(x_i, graphs[x_i].nodes(),len(batch), x['pt_eta'], pt_eta)
	pt_eta_jet = torch.stack([x['pt_eta'][0] for x in batch],dim=0)
	
	
	jet_label = [0 if x['flav']==5 else 1 for x in batch]
	
	pt_eta = torch.FloatTensor(pt_eta)
	jet_label = torch.LongTensor(jet_label)
	pt_eta_jet = torch.FloatTensor(pt_eta_jet)
	
	return (pt_eta,batched_graph,node_features),node_labels


def create_batch_edges(batch):
	
	graphs = [x['graph'] for x in batch]
	batched_graph = bg_custom(graphs,"__ALL__","__ALL__")
	
	node_labels = torch.cat([x['node_labels'] for x in batch],dim=0).view(-1).long()
	edge_labels = torch.cat([x['edge_labels'] for x in batch],dim=0).view(-1,1).float()

	node_features = torch.cat([x['node_features'] for x in batch],dim=0)
	
	node_features = torch.FloatTensor(transform_features(node_features.data.numpy()))
	
	pt_eta = torch.cat([x['pt_eta'] for x in batch],dim=0)
	
	for x_i, x in enumerate(batch):
		if len(x['pt_eta']) < 1:
			print(x_i, graphs[x_i].nodes(),len(batch), x['pt_eta'], pt_eta)
	pt_eta_jet = torch.stack([x['pt_eta'][0] for x in batch],dim=0)
	
	
	jet_label = [0 if x['flav']==5 else 1 for x in batch]
	
	pt_eta = torch.FloatTensor(pt_eta)
	jet_label = torch.LongTensor(jet_label)
	pt_eta_jet = torch.FloatTensor(pt_eta_jet)
	
	#y = []
	return (pt_eta,batched_graph,node_features),(edge_labels,node_labels)


def create_single_batch(graph):

	dgl_g = create_dgl(graph.get_reco_graph(),shuffle=False)

	single_batch = create_batch([dgl_g])
	x,y  = single_batch
	return x,y
