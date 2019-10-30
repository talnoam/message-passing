
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
sys.path.append('./helper_functions/')

import uproot
import numpy as np
import pandas as pd
import glob
import networkx as nx
import dgl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from file_loading import *
from pdg_id_dict import *
from helper_functions import *
from graph_edm import *
from graph_plotting import *
from dataset_functions import *
from gnn_model import *
from edge_model_full import *

flist = ['data/f_11_bjets.h5',
 'data/f_11_cjets.h5']
 #'data/f_11_ujets.h5']

flist_ujets = ['data/f_11_ujets.h5']


flist_valid = [
 'data/f_0_bjets.h5',
 'data/f_0_cjets.h5',
 'data/f_0_ujets.h5']


df_list = []
for fname in flist:
	f = h5py.File(fname,'r')
	keylist = [x for x in f.keys()]
	f.close()
	print(keylist)
	df_list.append(pd.concat( [pd.read_hdf(fname,key=x)  for x in keylist] ) )

graph_df = pd.concat( df_list )

df_list = []
for fname in flist_ujets:
	f = h5py.File(fname,'r')
	keylist = [x for x in f.keys()]
	f.close()
	print(keylist)
	df_list.append(pd.concat( [pd.read_hdf(fname,key=x)  for x in keylist] ) )

graph_df_ujets = pd.concat( df_list )

graph_df_ujets = graph_df_ujets.sample(frac=0.2).reset_index(drop=True)

graph_df = pd.concat([graph_df,graph_df_ujets])

df_list = []
for fname in flist_valid:
	f = h5py.File(fname,'r')
	keylist = [x for x in f.keys()]
	f.close()
	print(keylist)
	df_list.append(pd.concat( [pd.read_hdf(fname,key=x)  for x in keylist] ) )

graph_df_valid = pd.concat( df_list )

ds = JetGraphDataset([graph_df])
ds_valid = JetGraphDataset([graph_df_valid])

dataset_loader = torch.utils.data.DataLoader(ds,
											 batch_size=10, shuffle=True,collate_fn=create_batch_edges,
											 num_workers=1)

dataset_loader_valid = torch.utils.data.DataLoader(ds_valid,
											 batch_size=150, shuffle=False,collate_fn=create_batch_edges,
											 num_workers=1)


ce_loss = nn.CrossEntropyLoss(reduction='mean')
bce_loss = nn.BCEWithLogitsLoss()
bce_loss_withoutLogists = nn.BCELoss()
def loss_function(yhat,edge_labels,node_labels):
	edge_class = yhat[1]
	edge_edge_attention = yhat[3]

	batch_s = len(edge_class)
	
	edge_loss = bce_loss(edge_class,edge_labels)
	
	edge_edge_loss = bce_loss_withoutLogists(edge_edge_attention,edge_labels)

	return edge_loss+edge_edge_loss

def edge_loss_function(yhat,edge_labels,node_labels):
	node_class = yhat[0]
	edge_class = yhat[1]
	
	edge_loss = bce_loss(edge_class,edge_labels)
	
	return edge_loss

#gnn_edge = JetEdgeClassifier_1(hidden_size=128) #no message
#gnn_node = JetEdgeClassifier_1(hidden_size=128) #GNN
gnn_double = JetEdgeClassifierDoubleChannel(hidden_size=128) #no message no hidden


import fastai
from fastai import *
from fastai.vision import *


db = DataBunch(train_dl=dataset_loader,valid_dl=dataset_loader_valid,collate_fn=create_batch_edges,fix_dl=dataset_loader)

#learn = Learner(db,gnn_edge,loss_func=edge_loss_function)
#learn_1 = Learner(db,gnn_node,loss_func=node_loss_function)
learn = Learner(db,gnn_double,loss_func=loss_function)

def write_to_file(filename,trainloss,valloss):
	f = open(filename,'a')
	n_batches = len(trainloss)
	for batch_i in range(n_batches):
		if batch_i!=n_batches-1:
			f.write(str(trainloss[batch_i])+'\n')
		else:
			f.write(str(trainloss[batch_i])+'\t'+str(valloss)+'\n')
	f.close()

for epoch_i in range(100):
	
	if epoch_i in [0,1]:
		lr = 1e-03
	elif epoch_i in range(2,10):
		lr = 1e-04
	else:
		lr = 1e-05

	# learn.fit(1,lr=lr)
	# train_loss = np.mean([x.item() for x in learn.recorder.losses])
	# val_loss = learn.recorder.val_losses[0]

	# write_to_file('model_gnn_edge_loss.txt',train_loss,val_loss)
	# torch.save(gnn_edge, 'model_gnn_edge_'+str(epoch_i)+'.pt')


	# learn_1.fit(1,lr=lr)
	# train_loss = np.mean([x.item() for x in learn_1.recorder.losses])
	# val_loss = learn_1.recorder.val_losses[0]
	# write_to_file('model_gnn_node_loss.txt',train_loss,val_loss)
	# torch.save(gnn_node, 'model_gnn_node_'+str(epoch_i)+'.pt')
	
	learn.fit(1,lr=lr)
	train_loss = np.array([x.item() for x in learn.recorder.losses])
	val_loss = learn.recorder.val_losses[0]
	write_to_file('model_gnn_auxEloss_loss.txt',train_loss,val_loss)

	torch.save(gnn_double, 'model_gnn_auxEloss_'+str(epoch_i)+'.pt')

