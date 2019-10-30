import uproot
import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import DBSCAN


def to_cylindrical(xyz):
	x,y,z = xyz
	r = np.sqrt( x**2+y**2 )
	phi = np.arctan2(x,y)
	return r,phi,z

def printvtx(xyz):
	x,y,z = xyz
	return '( {0:.2f}'.format(x)+','+ '{0:.2f}'.format(y)+','+ '{0:.2f} )'.format(z)

def vtx_dist(vtx1,vtx2):
	return np.linalg.norm(np.array(vtx1)-np.array(vtx2))

def n_charged(vtx):
	charged =0
	
	for child in vtx:
		if abs(child[0]) > 0:
			charged+=1
	return charged
def n_reconstructed(vtx):
	recod = 0
	
	for child in vtx:
		_, charge,p4,isReco, _ ,pdgid,had_idx = child
		if isReco:
			recod+=1
	return recod

def p4mass(p4):
	
	e = p4[0]
	px,py,pz = p4[1],p4[2],p4[3]
	
	if np.linalg.norm([px,py,pz])**2 > e**2:
		m = 0
	else:
		m = np.sqrt( e**2 - np.linalg.norm([px,py,pz])**2)
	
	return m

def vtx_masses(vtx):
	
	charged_total_p = np.array([0,0,0,0])
	total_p = np.array([0,0,0,0])
	reco_p = np.array([0,0,0,0])
	
	for child in vtx:
		_, charge,p4,isReco, _ ,pdgid,had_idx = child
		child_is_charged = abs(charge) > 0
		if child_is_charged:

			charged_total_p = charged_total_p+np.array(p4)
		if isReco:
			reco_p = reco_p+np.array(p4)
		total_p = total_p+np.array(p4)
	
	charged_mass = p4mass(charged_total_p)
	total_mass = p4mass(total_p)
	reco_mass = p4mass(reco_p)
	
	return charged_mass, total_mass,reco_mass

def compute_combined_masses(sorted_vertices,vtx_dict):
	
	charged_total_p = np.array([0,0,0,0])
	total_p = np.array([0,0,0,0])
	reco_p = np.array([0,0,0,0])
		
	for vtx_loc in sorted_vertices:
		for child in vtx_dict[vtx_loc]:
			
			_, charge,p4,isReco, _ ,pdgid,had_idx = child
			child_is_charged = abs(charge) > 0
			if child_is_charged:
				charged_total_p = charged_total_p+np.array(p4)
			if isReco:
				reco_p = reco_p+np.array(p4)
			total_p = total_p+np.array(p4)
	
	charged_mass = p4mass(charged_total_p)
	total_mass = p4mass(total_p)
	reco_mass = p4mass(reco_p)
		
	if np.isnan( reco_mass ):
		for vtx_loc in sorted_vertices:
			for child in vtx_dict[vtx_loc]:
				_, charge,p4,isReco, _ ,pdgid,had_idx = child
				print(p4,' charged ',abs(charge), ' pdgid ',pdgid, ' isReco ',isReco)

	return total_mass,charged_mass,reco_mass
	

def check_reco(jet_df,child_barcode):
	track_barcode_list = jet_df['jet_trk_barcode']
	if child_barcode in track_barcode_list:
		return True
	else:
		return False
def MatchedTrack(child_idx,jet_graph):
	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]
	
	for edge in edge_list[edge_list[:,0]==child_idx]:
		s,e = edge
		if e in jet_graph['trk_node_index']:
			return e
	return np.nan
	
def particle_children(jet_graph,idx):
	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]
	
	return [np.where(np.array(jet_graph.particle_node_index) == x )[0][0] for x in edge_list[:,1][np.where(edge_list[:,0]==idx)]]


def parent_is_bc_hadron(jet_graph,idx):
	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]
	for edge in edge_list[edge_list[:,1]==idx]:
		s,e = edge
		parent_index = np.where( np.array(jet_graph.particle_node_index) == s )[0]
		if len(parent_index) > 0:
			stat = jet_graph.particle_node_status[parent_index[0]]
			injet = jet_graph.particle_node_inJet[parent_index[0]]
			if stat==2 and injet==1:
				return True
			parent_of_parent = parent_is_bc_hadron(jet_graph,s)
			if parent_of_parent:
				return True
	return False


def collect_jet_vtx(jet_graph):
	
	pt = jet_graph.jet_pt
	eta = jet_graph.jet_eta
	flav = jet_graph.jet_DoubleHadLabel

	pv_x,pv_y,pv_z = jet_graph.truth_PVx, jet_graph.truth_PVy, jet_graph.truth_PVz


	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]

	vtxlist = []
	vtxdict = {}
	hadron_list = []
	additional_vtx = []

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

	children_in_vtx = []

	for idx, pdgid,x0,y0,z0, x,y,z, stat,injet,charge,_,_,_,_ in particle_array:
		prod_vtx = (x0,y0,z0)
		decay_vtx = (x,y,z)
		if (stat==2 and injet==1) or (stat==2 and parent_is_bc_hadron(jet_graph,idx)): #b/c hadrons
			#check if they have any outgoing stable particles
			hasChildren = False
			for child in particle_array[particle_children(jet_graph,idx)]:
				child_stat = child[8]
				child_charge = child[10]
				if child_stat==1: #and abs(child_charge) > 0:
					hasChildren = True
			if not hasChildren:
				continue
			hadron_list.append([idx,pdgid,decay_vtx])
			if decay_vtx not in vtxlist:
				vtxlist.append(decay_vtx)
				vtxdict[decay_vtx] = []
			
			for child in particle_array[particle_children(jet_graph,idx)]:
				status = child[8]
				if status==2:
					continue
				charge = child[10]
				e,px,py,pz = child[11],child[12],child[13],child[14]
				pdgid = child[1]
				had_idx = len(hadron_list)-1
				
				child_idx = child[0]
				matched_track_idx = MatchedTrack(child_idx,jet_graph)
				isReco = False
				if not np.isnan(matched_track_idx):
					isReco = True
				
				children_in_vtx.append(child_idx)
				vtxdict[decay_vtx].append([child_idx,charge,(e,px,py,pz),isReco,matched_track_idx,pdgid,had_idx])
	## look for additional stable particles that came from/b/c but dont fit into the vertices that we collected already 
	# this would be second level decays, like V0 decays, material interactions
	# also add here any particles in the evnet that have matched tracks, and come from some other location
	for idx, pdgid,x0,y0,z0, x,y,z, stat,injet,charge,e,px,py,pz in particle_array:
		prod_vtx = (x0,y0,z0)
		decay_vtx = (x,y,z)
		if idx in children_in_vtx:
			continue

		matched_track_idx = MatchedTrack(idx,jet_graph)

		if (stat==1 and ( not np.isnan( matched_track_idx ) or parent_is_bc_hadron(jet_graph,idx)) ):
			if prod_vtx not in additional_vtx:
				if np.linalg.norm(np.array(prod_vtx)-np.array([pv_x,pv_y,pv_z])) > 0.01:
					additional_vtx.append(prod_vtx)

			
	return vtxlist,vtxdict,hadron_list, additional_vtx

def compute_jet_vtx(jet_df,merging_dist=0):
	
	vtxlist,vtxdict,hadron_list, additional_vtx = collect_jet_vtx(jet_df)
	
	if merging_dist==0 or len(vtxlist) == 0:
		return vtxlist,vtxdict,hadron_list,additional_vtx
	
	merged_vtx_list = []
	merged_vtx_dict = {}
	
	labels = DBSCAN(eps=merging_dist, min_samples=2).fit(vtxlist).labels_
	n_merged_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	n_clusters = n_merged_clusters + sum([1 if x==-1 else 0 for x in labels ])
	
	avg_vtx_positions = {}
	
	for mergedvtx_i in range(n_merged_clusters):
		
		idxs = np.where(labels==mergedvtx_i)
		cluster_members = np.array(vtxlist)[idxs]
		avg_x = np.mean( cluster_members[:,0] )
		avg_y = np.mean( cluster_members[:,1] )
		avg_z = np.mean( cluster_members[:,2] )
		avg_vtx_positions[mergedvtx_i] = (avg_x,avg_y,avg_z)
		
	
	for i, vtx in enumerate(vtxlist): 
		label = labels[i]
		if label== -1:
			merged_vtx_list.append(vtx)
			merged_vtx_dict[vtx] = vtxdict[vtx]
		else:
			vtxpos = avg_vtx_positions[label]
			if vtxpos not in merged_vtx_list:
				merged_vtx_list.append(vtxpos)
				merged_vtx_dict[vtxpos] = []
			merged_vtx_dict[vtxpos] = merged_vtx_dict[vtxpos]+vtxdict[vtx]
			
	
	return merged_vtx_list, merged_vtx_dict, hadron_list, additional_vtx

def compute_vtx_properties(vtx_list, vtx_dict):
	
	vtx_distances = [ np.linalg.norm(x) for x in vtx_list]
	sorted_vertices = [x for _,x in sorted(zip(vtx_distances,vtx_list))]
	
	cols = np.array([ ['L'+str(x),'L_r'+str(x),'n'+str(x),'n_c'+str(x),'n_r'+str(x),'mass'+str(x),'mass_c'+str(x),
								 'mass_r'+str(x)] 
								for x in range(1,4)]).flatten()
	
	cols = np.concatenate((['n_vtx','n_reco_vtx'],cols))
	cols = np.concatenate((['total_mass','total_c_mass','total_r_mass'],cols))
	df = pd.DataFrame(columns=cols)
	df.loc[0] = list( -99*np.ones(len(cols)))
	vtx_index = -1
	



	t_mass, c_mass, r_mass = compute_combined_masses(sorted_vertices,vtx_dict)
	df.iloc[0]['total_mass'] = t_mass
	df.iloc[0]['total_c_mass'] = c_mass
	df.iloc[0]['total_r_mass'] = r_mass
	
	for vtx in sorted_vertices:
		n_charge = n_charged(vtx_dict[vtx])
		if n_charge == 0:
			continue
		vtx_index+=1
		charged_mass, total_mass,_ = vtx_masses(vtx_dict[vtx])
		
		n_children = len(vtx_dict[vtx])
		
		idx = str(vtx_index+1)
		df.iloc[0]['L'+idx] = np.linalg.norm(vtx)
		df.iloc[0]['n'+idx] = n_children
		df.iloc[0]['n_c'+idx] = n_charge
		df.iloc[0]['mass'+idx] = total_mass
		df.iloc[0]['mass_c'+idx] = charged_mass
	df.iloc[0]['n_vtx'] = vtx_index+1
	
	vtx_index = -1
	for vtx in sorted_vertices:
		n_reco = n_reconstructed(vtx_dict[vtx])
		if n_reco==0:
			continue
		vtx_index+=1
		_, _,reco_mass = vtx_masses(vtx_dict[vtx])
		
		n_children = len(vtx_dict[vtx])
		
		idx = str(vtx_index+1)
		df.iloc[0]['L_r'+idx] = np.linalg.norm(vtx)
		df.iloc[0]['n_r'+idx] = n_reco
		df.iloc[0]['mass_r'+idx] = reco_mass
	df.iloc[0]['n_reco_vtx'] = vtx_index+1
	
	return df