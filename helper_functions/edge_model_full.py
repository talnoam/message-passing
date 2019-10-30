import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader



class NodeEmbedding(nn.Module):
	def __init__(self,input_size,outputsize):
		
		super(NodeEmbedding,self).__init__()
		
		self.net =  nn.Sequential( 
			nn.Linear(input_size,100),
			nn.ReLU(),
			 nn.Linear(100,100),
			 nn.ReLU(),
			 nn.Linear(100,50),
			 nn.ReLU(),
			 nn.Linear(50,10),
			 nn.ReLU(),
			nn.Linear(10,outputsize), nn.Tanh() )

	def forward(self,ndata):
		return self.net(ndata)



class EdgeAttention(nn.Module):
	def __init__(self,input_size):
		super(EdgeAttention,self).__init__()

		self.params = ['node_features','h']

		self.net =  nn.Sequential( 
			nn.Linear(input_size,100),
			nn.ReLU(),
			 nn.Linear(100,100),
		  nn.ReLU(),
			nn.Linear(100,50),
		  nn.ReLU(),
			nn.Linear(50,20),
			 nn.ReLU(),
			nn.Linear(20,10), nn.ReLU(),
			nn.Linear(10,1), nn.Sigmoid() )


	def forward(self,edge):
		 
		parm_list = [edge.src['jet_features']]+[edge.src[x] for x in self.params]+[edge.dst[x] for x in self.params]
		z = torch.cat( parm_list ,dim=1)
		out = self.net(z)

		return {'e_weight': out}


def message_func(edges):
	
	return {'z':  torch.cat([edges.src['node_features'],edges.src['h']],dim=1), 'e' : edges.data['e_weight'] }



class NodeNetwork(nn.Module):
	def __init__(self,input_size,output_size,output_name='new_h'):
		super(NodeNetwork,self).__init__()
		
		self.output_name = output_name
		self.net = nn.Sequential( 
			nn.Linear(input_size,500,bias=False),
			nn.ReLU(),
			 nn.Linear(500,400),
		  nn.ReLU(),
			nn.Linear(400,400),
		  nn.ReLU(),
			nn.Linear(400,200),
			 nn.ReLU(),
			nn.Linear(200,output_size)
					)

		
	def forward(self,nodes):
		   
		alpha = nodes.mailbox['e']
		
		h =  torch.sum(alpha * nodes.mailbox['z'], dim=1) 
		
		out = torch.cat([nodes.data['node_features'],nodes.data['h'], h , nodes.data['jet_features']],dim=1)
		
		out = self.net(out)
		
		return {self.output_name: out}


class NodeUpdate(nn.Module):
	def __init__(self,edge_attention_input_size,input_size,output_size):
		super(NodeUpdate,self).__init__()
		
		self.edge_network = EdgeAttention(input_size=edge_attention_input_size)
		self.message_func = message_func
		self.node_network = NodeNetwork(input_size,output_size)
	
	def forward(self,g):
		   
		g.apply_edges(self.edge_network)

		edge_atten = g.edata['e_weight']

		g.update_all(self.message_func,self.node_network)
		
		g.ndata['h'] = g.ndata.pop('new_h')

		return g,edge_atten

class NodeClassifier(nn.Module):
	def __init__(self,inputsize,outputsize):
		super(NodeClassifier,self).__init__()
		
		self.net =  nn.Sequential(
			nn.Linear(inputsize,300),
			nn.ReLU(),
			 nn.Linear(300,200),
		  nn.ReLU(),
			nn.Linear(200,100),
			 nn.ReLU(),
			nn.Linear(100,outputsize) )


	def forward(self,ndata):
		return self.net(ndata) 
    
class NodeClassifier_2(nn.Module):
	def __init__(self,inputsize,outputsize):
		super(NodeClassifier_2,self).__init__()
		
		self.net =  nn.Sequential(
			nn.Linear(inputsize,300),
			nn.ReLU(),
			 nn.Linear(300,200),
		  nn.ReLU(),
			nn.Linear(200,100),
			 nn.ReLU(),
			nn.Linear(100,outputsize) )


	def forward(self,ndata):
		return self.net(ndata) 

class EdgeClassifier(nn.Module):
	def __init__(self,inputsize,outputsize):
		super(EdgeClassifier,self).__init__()
		
		self.params = ['node_features','h']

		self.net =  nn.Sequential( 
			nn.Linear(inputsize,100),
			nn.ReLU(),
			 nn.Linear(100,100),
		  nn.ReLU(),
			nn.Linear(100,50),
		  nn.ReLU(),
			nn.Linear(50,20),
			 nn.ReLU(),
			nn.Linear(20,10), nn.ReLU(),
			nn.Linear(10,outputsize) )


	def forward(self,edge):
		 
		parm_list = [edge.src['jet_features']]+[edge.src[x] for x in self.params]+[edge.dst[x] for x in self.params]
		z = torch.cat( parm_list ,dim=1)
		
		out = self.net(z)

		return {'e_class': out}
    
class EdgeClassifier_2(nn.Module):
	def __init__(self,inputsize,outputsize):
		super(EdgeClassifier_2,self).__init__()
		
		self.params = ['node_features']

		self.net =  nn.Sequential( 
			nn.Linear(inputsize,100),
			nn.ReLU(),
			 nn.Linear(100,100),
		  nn.ReLU(),
			nn.Linear(100,50),
		  nn.ReLU(),
			nn.Linear(50,20),
			 nn.ReLU(),
			nn.Linear(20,10), nn.ReLU(),
			nn.Linear(10,outputsize) )


	def forward(self,edge):
		 
		parm_list = [edge.src['jet_features']]+[edge.src[x] for x in self.params]+[edge.dst[x] for x in self.params]
		z = torch.cat( parm_list ,dim=1)
		
		out = self.net(z)

		return {'e_class': out}

class JetEdgeClassifier(nn.Module):
	def __init__(self,jet_feature_size=2,node_feature_size=6,hidden_size=128):
		super(JetEdgeClassifier,self).__init__()
		

		self.node_init = NodeEmbedding(input_size=jet_feature_size+node_feature_size,outputsize=hidden_size)
		self.edge_class = EdgeClassifier(inputsize=jet_feature_size+2*(node_feature_size+hidden_size),outputsize=1)
		self.node_class = NodeClassifier(inputsize=jet_feature_size+node_feature_size+hidden_size,outputsize=2)

		edge_attention_size = 2*(node_feature_size+hidden_size)+jet_feature_size
		node_update_size = 2*(node_feature_size+hidden_size)+jet_feature_size

		self.node_updates = nn.ModuleList()
		
		for i in range(2):
			self.node_updates.append(NodeUpdate(edge_attention_input_size=edge_attention_size,
				input_size=node_update_size,output_size=hidden_size))

		
	def forward(self,jet_features,g,node_features):
		

		g.ndata['node_features'] = node_features
		g.ndata['jet_features'] = jet_features
		

		node_in = torch.cat([g.ndata['node_features'],g.ndata['jet_features']],dim=1)
		
		g.ndata['h'] = self.node_init(node_in)


		node_in = torch.cat([g.ndata['node_features'],g.ndata['h'],g.ndata['jet_features']],dim=1)

		node_class = self.node_class(node_in)


		g.apply_edges(self.edge_class)


		return node_class,g.edata['e_class']
    

class JetEdgeClassifier_1(nn.Module):
	def __init__(self,jet_feature_size=2,node_feature_size=6,hidden_size=128):
		super(JetEdgeClassifier_1,self).__init__()
		

		self.node_init = NodeEmbedding(input_size=jet_feature_size+node_feature_size,outputsize=hidden_size)
		self.edge_class = EdgeClassifier(inputsize=jet_feature_size+2*(node_feature_size+hidden_size),outputsize=1)
		self.node_class = NodeClassifier(inputsize=jet_feature_size+node_feature_size+hidden_size,outputsize=2)

		edge_attention_size = 2*(node_feature_size+hidden_size)+jet_feature_size
		node_update_size = 2*(node_feature_size+hidden_size)+jet_feature_size

		self.node_updates = nn.ModuleList()
		
		for i in range(1):
			self.node_updates.append(NodeUpdate(edge_attention_input_size=edge_attention_size,
				input_size=node_update_size,output_size=hidden_size))

		
	def forward(self,jet_features,g,node_features):
		

		g.ndata['node_features'] = node_features
		g.ndata['jet_features'] = jet_features
		

		node_in = torch.cat([g.ndata['node_features'],g.ndata['jet_features']],dim=1)
		
		g.ndata['h'] = self.node_init(node_in)

		for i in range(1):
			
			g, edge_attention = self.node_updates[i](g)

		node_in = torch.cat([g.ndata['node_features'],g.ndata['h'],g.ndata['jet_features']],dim=1)

		node_class = self.node_class(node_in)


		g.apply_edges(self.edge_class)


		return node_class,g.edata['e_class'],edge_attention
    
    
    
class JetEdgeClassifier_2(nn.Module):
	def __init__(self,jet_feature_size=2,node_feature_size=6,hidden_size=128):
		super(JetEdgeClassifier_2,self).__init__()
		

		self.node_init = NodeEmbedding(input_size=jet_feature_size+node_feature_size,outputsize=hidden_size)
		self.edge_class = EdgeClassifier_2(inputsize=jet_feature_size+2*(node_feature_size),outputsize=1)
		self.node_class = NodeClassifier_2(inputsize=jet_feature_size+node_feature_size,outputsize=2)

		edge_attention_size = 2*(node_feature_size+hidden_size)+jet_feature_size
		node_update_size = 2*(node_feature_size+hidden_size)+jet_feature_size

		self.node_updates = nn.ModuleList()
		
		for i in range(2):
			self.node_updates.append(NodeUpdate(edge_attention_input_size=edge_attention_size,
				input_size=node_update_size,output_size=hidden_size))

		
	def forward(self,jet_features,g,node_features):
		

		g.ndata['node_features'] = node_features
		g.ndata['jet_features'] = jet_features
		

		node_in = torch.cat([g.ndata['node_features'],g.ndata['jet_features']],dim=1)
		
		#g.ndata['h'] = self.node_init(node_in)

		node_in = torch.cat([g.ndata['node_features'],g.ndata['jet_features']],dim=1)

		node_class = self.node_class(node_in)


		g.apply_edges(self.edge_class)


		return node_class,g.edata['e_class']



class JetEdgeClassifierDoubleChannel(nn.Module):
	def __init__(self,jet_feature_size=2,node_feature_size=6,hidden_size=128):
		super(JetEdgeClassifierDoubleChannel,self).__init__()
		

		self.node_classifier = JetEdgeClassifier_1(jet_feature_size=jet_feature_size,node_feature_size=node_feature_size,hidden_size=hidden_size)
		self.edge_classifier = JetEdgeClassifier_1(jet_feature_size=jet_feature_size,node_feature_size=node_feature_size,hidden_size=hidden_size)

		
	def forward(self,jet_features,g,node_features):
		

		node_class, _ , node_edge_attention = self.node_classifier(jet_features,g,node_features)
		g.ndata.pop('h')
		_, edge_class, edge_edge_attention = self.edge_classifier(jet_features,g,node_features)


		return node_class,edge_class, node_edge_attention, edge_edge_attention



