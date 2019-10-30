from helper_functions import *
import numpy as np
from track_trajectory_functions import *



def build_jet_graph(jet_graph,merging_dist=0):
	
	pt = jet_graph.jet_pt
	eta = jet_graph.jet_eta
	phi = jet_graph.jet_phi
	flav = jet_graph.jet_DoubleHadLabel

	axis = get_jet_axis(pt, eta, phi)
	rotMatrix = rotationMatrix(axis,[0,0,1])

	eventnumber = jet_graph.eventnb
	jet_idx = jet_graph.jet_index

	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]

	pv_x,pv_y,pv_z = jet_graph.truth_PVx, jet_graph.truth_PVy, jet_graph.truth_PVz
	PV = (pv_x,pv_y,pv_z)
	vtxlist, vtx_dict,hadron_list, additional_vtx = compute_jet_vtx(jet_graph,merging_dist=merging_dist)

	n_tracks = len(jet_graph['trk_node_index'])

	n_locations = 2+len(additional_vtx)+len(vtxlist)
	
	j_graph_obj = JetGraph(pt,eta,flav,axis,n_tracks,eventnumber,jet_idx,PV,hadron_list)

	full_vtx_list = [(pv_x,pv_y,pv_z)]+vtxlist+additional_vtx
	full_vtx_dict = {}
	for vtx in full_vtx_list:
		full_vtx_dict[vtx] = []
	sorted_vtx = sorted( zip( [ np.linalg.norm(vtx-np.array([pv_x,pv_y,pv_z])) for vtx in full_vtx_list ], full_vtx_list ) )
	
	particle_array = np.stack( (jet_graph.particle_node_index, jet_graph.particle_node_pdgid,
											  jet_graph.particle_node_prod_x, 
											  jet_graph.particle_node_prod_y,
											  jet_graph.particle_node_prod_z,
											  jet_graph.particle_node_decay_x, 
											  jet_graph.particle_node_decay_y,
											  jet_graph.particle_node_decay_z,
											 jet_graph.particle_node_status,
											  jet_graph.particle_node_inJet,
											   jet_graph.particle_node_charge,
												jet_graph.particle_node_E,
												 jet_graph.particle_node_px,
												 jet_graph.particle_node_py,
												 jet_graph.particle_node_pz,) ,axis=1)

	for idx, pdgid,x0,y0,z0, x,y,z, stat,injet,charge,e,px,py,pz in particle_array:
		if stat!=1:
			continue

		prod_vtx = (x0,y0,z0)
		decay_vtx = (x,y,z)
		p4 = (e,px,py,pz)
		matched_track_idx = MatchedTrack(idx,jet_graph)
		matched_track = np.nan
		jf_vtx_idx = np.nan
		reconstructed = False
		xy_p_at_z0 = np.nan
		if not np.isnan(matched_track_idx):
			reconstructed = True
			track_i = np.where(np.array(jet_graph.trk_node_index) == matched_track_idx)[0][0]
			pt = jet_graph.trk_node_pt[track_i]
			phi = jet_graph.trk_node_phi0[track_i]
			theta = jet_graph.trk_node_theta[track_i]
			d0 = jet_graph.trk_node_d0[track_i]
			z0 = jet_graph.trk_node_z0[track_i]
			d0signed = jet_graph.trk_node_signed_d0[track_i]
			z0signed = jet_graph.trk_node_signed_z0[track_i]
			qoverp = jet_graph.trk_node_qoverp[track_i]
			jf_vtx_idx = jet_graph.trk_node_jetfitter_index[track_i]

			matched_track = (pt,d0,d0signed,z0,z0signed,phi,theta,qoverp)

			phi_z = get_phi(0,(d0,z0,phi,theta,qoverp),rotMatrix)
			xy, p_unit, p3 = get_xy_p(phi_z, (d0,z0,phi,theta,qoverp),rotMatrix )
			xy_p_at_z0 = (xy[0],xy[1],p_unit[0],p_unit[1],p_unit[2])

		vertex_idx = 0

		
		
		for f_indx, f_vtx in enumerate( full_vtx_dict ):
			if np.linalg.norm(np.array(prod_vtx)-np.array(f_vtx) ) < 0.01:
				full_vtx_dict[f_vtx].append(idx)
				vertex_idx = np.where( [f_vtx==vtx for dist,vtx in sorted_vtx] )[0][0]
				vertex_idx = vertex_idx+1
				break
		

		origin = np.array(prod_vtx)-PV
		had_idx = np.nan
		ending_edges = edge_list[edge_list[:,1]==idx]
		if len(ending_edges) > 0:
			parent_idx = ending_edges[0][0]
			for had_i, had in enumerate( hadron_list ):
				if had[0]==parent_idx:
					had_idx = had_i

		node = JetNode(p4,charge,pdgid,reconstructed,matched_track,origin,vertex_idx,had_idx,xy_p_at_z0,jf_vtx_idx)
		j_graph_obj.add_node(node)
		

	for track_i, idx in enumerate( jet_graph['trk_node_index'] ):
		pt = jet_graph.trk_node_pt[track_i]
		phi = jet_graph.trk_node_phi0[track_i]
		theta = jet_graph.trk_node_theta[track_i]
		d0 = jet_graph.trk_node_d0[track_i]
		z0 = jet_graph.trk_node_z0[track_i]
		d0signed = jet_graph.trk_node_signed_d0[track_i]
		z0signed = jet_graph.trk_node_signed_z0[track_i]
		qoverp = jet_graph.trk_node_qoverp[track_i]
		jf_vtx_idx = jet_graph.trk_node_jetfitter_index[track_i]

		matched_track = (pt,d0,d0signed,z0,z0signed,phi,theta,qoverp)

		track_parent = edge_list[edge_list[:,1]==idx]

		origin = np.nan

		if len(track_parent)==0: #not truth matched 
			p4 = np.nan
			charge = np.sign(qoverp)
			pdgid = np.nan
			reconstructed = True
			vertex_idx = 0
			had_idx = np.nan


			phi_z = get_phi(0,(d0,z0,phi,theta,qoverp),rotMatrix)
			xy, p_unit, p3 = get_xy_p(phi_z, (d0,z0,phi,theta,qoverp),rotMatrix )
			xy_p_at_z0 = (xy[0],xy[1],p_unit[0],p_unit[1],p_unit[2])


			node = JetNode(p4,charge,pdgid,reconstructed,matched_track,origin,vertex_idx,had_idx,xy_p_at_z0,jf_vtx_idx)
			j_graph_obj.add_node(node)

	return j_graph_obj




class JetNode:

	def __init__(self,p4,charge,pdgid,reconstructed,matched_track,origin,vertex_idx,had_idx,xy_p_at_z0,jf_vtx_idx):
		self.charge = charge
		self.pdgid = pdgid
		self.reconstructed = reconstructed
		self.matched_track = matched_track
		self.origin = origin
		self.vertex_idx = vertex_idx
		self.jf_vtx_idx = jf_vtx_idx
		self.p4 = p4
		self.had_idx = had_idx
		self.xy_p_at_z0 = xy_p_at_z0

	def __str__(self):
		return ' node: '
	
	def copy_node(self):
		new_node = JetNode(self.p4,self.charge,self.pdgid,self.reconstructed,
			self.matched_track,self.origin,self.vertex_idx,self.had_idx,self.xy_p_at_z0,self.jf_vtx_idx)
		return new_node

class JetGraph:
	def __init__(self,pt,eta,flav,axis,n_tracks,eventnumber,jet_idx,PV,hadron_list):
		self.pt = pt
		self.eta = eta
		self.flav = flav
		self.axis = axis
		self.n_tracks = n_tracks
		self.nodes = []
		self.eventnumber = eventnumber
		self.jet_idx = jet_idx
		self.PV = PV
		self.hadron_list = hadron_list

	def add_node(self,node):
		self.nodes.append(node)
	
	def sort_nodes(self):
		node_list = self.nodes
	
		n_idx_list = np.array([node.vertex_idx for node in node_list])
		secondary_nodes = np.where(n_idx_list > 1)[0]
		primary_and_fakes = np.where(n_idx_list < 2)[0]
	
		sorted_indices = list( secondary_nodes[ np.argsort(n_idx_list[secondary_nodes]) ] )+list( primary_and_fakes[ np.argsort(n_idx_list[primary_and_fakes]) ] )
		node_list = [node_list[x] for x in sorted_indices]

		self.nodes = node_list

	def get_reco_graph(self):
		reco_graph = JetGraph(self.pt,self.eta,self.flav,self.axis,
			self.n_tracks,self.eventnumber,self.jet_idx,self.PV,self.hadron_list)

 

		reco_nodes = []
		for node in self.nodes:
			if node.reconstructed == True:
				new_node = node.copy_node()
				reco_nodes.append(new_node)

		vtx_indices  = np.array( list( set([node.vertex_idx for node in reco_nodes])  ) )
		
		#we need to "translate" the index of non-primary and non fake indices
		over1s = vtx_indices[vtx_indices > 1]

		listlen = len(over1s)

		translate_indx = {
		0 : 0,
		1 : 1
		}
		for new_idx ,old_idx in zip(range(2,listlen+2),over1s):
			translate_indx[old_idx] = new_idx 

		for node in reco_nodes:
			node.vertex_idx = translate_indx[node.vertex_idx]
			reco_graph.add_node(node)
		return reco_graph
	
	def get_jet_node_metrics(self,vertex_assignment):

		true_vtx = np.array([node.vertex_idx for node in self.nodes])
		true_vtx = np.array([x if not np.isnan(x) else -1 for x in true_vtx])


		true_secondary = np.where(true_vtx > 0)[0]
		n_true = len(true_secondary)
		n_found = len(np.where(vertex_assignment[true_secondary] > -1)[0])

		assigned_secondary = np.where(vertex_assignment > -1)[0]

		n_in_vtx = len(assigned_secondary)

		n_fakes = len(np.where(true_vtx[assigned_secondary] < 1)[0])

		if n_true > 0:
			id_rate = float(n_found)/n_true
		else:
			id_rate = -1
		if n_in_vtx > 0:
			fake_rate = float(n_fakes)/n_in_vtx
		else:
			fake_rate = 0

		return id_rate, n_found, fake_rate, n_fakes

	
	def check_vertexing(self,vertex_assignment):
		vertexing_category = -1
		true_vtx = np.array([node.vertex_idx for node in self.nodes])
		true_vtx = np.array([x if not np.isnan(x) else -1 for x in true_vtx])

		merged = self.has_merged(vertex_assignment,true_vtx)
		split = self.has_split(vertex_assignment,true_vtx)
		vertex_id_cat = self.node_id_category(vertex_assignment,true_vtx)
		found_fakes = self.has_fakes(vertex_assignment,true_vtx)
		#print('v cat ',vertex_id_cat,'split ',split,'merged ',merged)
		return [vertex_id_cat,split,merged,found_fakes]
	
	def has_fakes(self,vertex_assignment,true_vtx):
		found_fakes = False

		uniqe_assigned_vtx = list(set(vertex_assignment))

		for uniq_vtx in uniqe_assigned_vtx:
			if np.isnan(uniq_vtx) or uniq_vtx < 0:
				continue
			locs = np.where(vertex_assignment==uniq_vtx)[0]
			in_vtx = true_vtx[locs]

			if 0 in in_vtx or -1 in in_vtx or np.nan in in_vtx:
				found_fakes=True

		return found_fakes


	def node_id_category(self,vertex_assignment,true_vtx):
		
		uniqe_true_vtx = list(set(true_vtx))
		
		vtx_cat = []
		for uniq_vtx in uniqe_true_vtx:
			if uniq_vtx < 1:
				continue
			if np.isnan(uniq_vtx):
				continue
			locs = np.where(true_vtx==uniq_vtx)[0]
			vertexing_ids = vertex_assignment[locs]


			vertexing_ids = [x for x in vertexing_ids if (x > -1 and not np.isnan(x))]
			
			if len(vertexing_ids) == len(locs):
				if len(locs) > 1:
					vtx_cat.append('perfect-2')
				else:
					vtx_cat.append('perfect-1')
			elif len(vertexing_ids) > 0:
				vtx_cat.append('partial')
			else:
				vtx_cat.append('failed')

		vtx_cat = list(set(vtx_cat))

		if len(vtx_cat) ==0:
			return 'no_vertex'
		elif 'perfect-2' in vtx_cat and len(vtx_cat)==1:
			return 'perfect'
		elif 'perfect-1' in vtx_cat and len(vtx_cat)==1:
			return 'perfect'
		elif 'perfect-2' in vtx_cat and 'perfect-1' in vtx_cat and len(vtx_cat)==2:
			return 'perfect'
		elif 'perfect-2' in vtx_cat or 'perfect-1' in vtx_cat:
			if 'perfect-2' in vtx_cat:
				return 'partial-2'
			elif 'perfect-1' in vtx_cat:
				return 'partial-1' 
		elif 'partial' in vtx_cat:
			return 'partial'
		elif 'failed' in vtx_cat:
			return 'failed'
		else:
			return 'something you didnt think of'

	def assigned_vtx_categories(self,vertex_assignment):
		true_vtx = np.array([node.vertex_idx for node in self.nodes])
		true_vtx = np.array([x if not np.isnan(x) else -1 for x in true_vtx])

		vtx_cats = []
		
		uniqe_assigned_vtx = list(set(vertex_assignment))

		for uniq_vtx in uniqe_assigned_vtx:
			if np.isnan(uniq_vtx) or uniq_vtx < 0:
				continue

			
			locs = np.where(vertex_assignment==uniq_vtx)[0]
			in_vtx = true_vtx[locs]

			in_vtx = [x for x in in_vtx if (x > 0)]

			n_found = len(in_vtx)
			n_fake = len(locs)-n_found
			n_tracks = len(locs)

			n_in_vtx = len( list(set(in_vtx)) )

			cat = ''
			if n_in_vtx == 1:
				cat = 'single'
				n_true_vtx = len(np.where(true_vtx==in_vtx[0])[0])
				if len(in_vtx)==n_true_vtx:
					cat = 'perfect'
			elif n_in_vtx> 1:
				cat = 'merged'
			elif n_in_vtx==0:
				cat = 'fake'
			vtx_cat = [cat,float(n_found)/n_tracks, n_found,n_fake, n_tracks]
			vtx_cats.append(vtx_cat)

		return vtx_cats

	def true_vtx_categories(self,vertex_assignment):
		true_vtx = np.array([node.vertex_idx for node in self.nodes])
		true_vtx = np.array([x if not np.isnan(x) else -1 for x in true_vtx])

		vtx_cats = []
		
		uniqe_true_vtx = list(set(true_vtx))

		for uniq_vtx in uniqe_true_vtx:
			if uniq_vtx < 1:
				continue
			if np.isnan(uniq_vtx):
				continue
			locs = np.where(true_vtx==uniq_vtx)[0]
			n_tracks = len(locs)
			vertexing_ids = vertex_assignment[locs]

			vertexing_ids = [x for x in vertexing_ids if (x > -1 and not np.isnan(x))]
			
			n_found = len(vertexing_ids)
			n_in_vtx = len( list(set(vertexing_ids)) )

			cat = ''
			if n_in_vtx ==1:
				cat = 'single'
				if len(vertexing_ids)==n_tracks:
					cat = 'perfect'
			elif n_in_vtx > 1:
				cat = 'split'
			elif n_in_vtx < 1:
				cat = 'failed'
			vtx_cat = [cat,float(n_found)/n_tracks,n_found,n_tracks]
			vtx_cats.append(vtx_cat)

		return vtx_cats

	def has_merged(self,vertex_assignment,true_vtx):
		has_merged = False

		uniqe_assigned_vtx = list(set(vertex_assignment))

		for uniq_vtx in uniqe_assigned_vtx:
			if np.isnan(uniq_vtx) or uniq_vtx < 0:
				continue
			locs = np.where(vertex_assignment==uniq_vtx)[0]
			in_vtx = true_vtx[locs]

			in_vtx = [x for x in in_vtx if (x > 0)]

			n_in_vtx = len( list(set(in_vtx)) )

			if n_in_vtx > 1:
				has_merged = True
			#print(uniq_vtx,in_vtx, len( list(set(in_vtx)) ), list(set(in_vtx)))

		return has_merged

	def has_split(self,vertex_assignment,true_vtx):
		has_split = False
		
		uniqe_true_vtx = list(set(true_vtx))

		for uniq_vtx in uniqe_true_vtx:
			if uniq_vtx < 1:
				continue
			if np.isnan(uniq_vtx):
				continue
			locs = np.where(true_vtx==uniq_vtx)[0]
			vertexing_ids = vertex_assignment[locs]

			vertexing_ids = [x for x in vertexing_ids if (x > -1 and not np.isnan(x))]
			
			n_in_vtx = len( list(set(vertexing_ids)) )

			if n_in_vtx > 1:
				has_split = True

		return has_split

	def __str__(self):
		return '** jet graph **'
	def __repr__(self):
		return ''

def get_jet_axis(jetpt, jeteta, jetphi):
	#return the unit vector along jet directon

	jet_axis = [0,0,0]
	jet_axis[0] = abs(jetpt)*np.cos(jetphi)
	jet_axis[1] = abs(jetpt)*np.sin(jetphi)
	jet_axis[2] = abs(jetpt)*np.sinh(jeteta)

	jet_axis = np.array(jet_axis)/np.linalg.norm(jet_axis)

	return jet_axis