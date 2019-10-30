from helper_functions import *
from graph_edm import *
from track_trajectory_functions import *
from scipy.spatial import ConvexHull
import networkx as nx
import dgl
from pdg_id_dict import *

edge_color_dict ={
	1: 'cornflowerblue',
	2: 'mediumseagreen',
	3: 'gold',
	4: 'magenta',
	5: 'red',
	6: 'orange'

}

node_color_dict = {
	'charged': {np.nan: 'darkgray',0:'midnightblue', 1:'cornflowerblue', 2:'mediumseagreen',3:'gold',4:'magenta',5:'red',6:'orange'},
	'neutral': 'olive'
}


def plot_tree_graph(jet_graph,ax):

	ax.set_axis_off()
	
	vtxlist,vtxdict,hadron_list, additional_vtx_dict = compute_jet_vtx(jet_graph)
	
	pt = jet_graph.jet_pt
	eta = jet_graph.jet_eta
	flav = jet_graph.jet_DoubleHadLabel
	
	ax.set_title('Flavour : '+str(flav)+'   pt: '+'{0:.2f}'.format(pt/1000.0)+'   eta: '+'{0:.2f}'.format(eta) ,
				  fontsize=20)
	
	g = dgl.DGLGraph()
	n_nodes = len( jet_graph['trk_node_index'] )+len(jet_graph['jf_node_index'])+len(jet_graph['particle_node_index'])
	g.add_nodes(n_nodes)
	
	edge_list = np.dstack([ jet_graph.edge_start , jet_graph.edge_end ])[0]
	
	labels = {}
	
	for edge in edge_list:
		s,e = edge
		
		g.add_edge(int(s),int(e))
	
	pv_x,pv_y,pv_z = jet_graph.truth_PVx, jet_graph.truth_PVy, jet_graph.truth_PVz
	

	
	g.add_nodes(1)
	
	particles_in_primary = []
	for idx, pdgid,x0,y0,z0, x,y,z, stat,injet in zip(jet_graph['particle_node_index'], 
										  jet_graph['particle_node_pdgid'],
										  jet_graph.particle_node_prod_x, 
										  jet_graph.particle_node_prod_y,
										  jet_graph.particle_node_prod_z,
										  jet_graph.particle_node_decay_x, 
										  jet_graph.particle_node_decay_y,
										  jet_graph.particle_node_decay_z,
										 jet_graph.particle_node_status,
											   jet_graph.particle_node_inJet):
		if np.linalg.norm([x0-pv_x,y0-pv_y,z0-pv_z]) < 0.01:
			
			particles_in_primary.append(idx)
	
	for p_in_primary in particles_in_primary:
		has_parent = False
		#loop over the other children of the vtx, see if one of them is the parent
		for p_in_primary_j in particles_in_primary:
			for edge in edge_list:
				s,e = edge
				if p_in_primary_j==s and p_in_primary==e:
					has_parent=True
		if not has_parent:
			g.add_edge(n_nodes,int(p_in_primary))
				
	G = g.to_networkx()
	
	node_colors = []
	for idx, pdgid, charge, in zip(jet_graph['particle_node_index'], jet_graph['particle_node_pdgid'],jet_graph.particle_node_charge):
		if abs(pdgid) in [6,24]:
			node_colors.append('lightgreen')
		elif charge==0:
			node_colors.append('khaki')
		else:
			node_colors.append('lightsalmon')
	
	
	pos = nx.nx_agraph.graphviz_layout(G,prog='dot')
	
	min_max_x = list(pos[0])
	y_min = -10
	
	for key_i, key in enumerate( jet_graph['particle_node_index'] ):
		if key_i==0:
			min_max_x = list(pos[key])
		x,y = pos[key]
		if x < min_max_x[0]:
			min_max_x[0] = x
		if x > min_max_x[1]:
			min_max_x[1] = x
		if y < y_min:
			y_min = y-10
	x_range = min_max_x[1]-min_max_x[0]
	
	
	n_tracks = len(jet_graph['trk_node_index'])
	
	track_x_positions = []

	for track_i, idx in enumerate( jet_graph['trk_node_index'] ):
		x_orig, y_orig = pos[idx]
		if idx not in jet_graph.edge_end:
			track_x_positions.append( (min_max_x[0]+track_i*x_range/n_tracks, idx) )
		else:
			track_x_positions.append((x_orig, idx))
		
	
	track_x_positions = sorted(track_x_positions, key=lambda x: x[0])
	
	
	spacing = 50
	for track_i in range(1,len(track_x_positions)):
		previous_pos = track_x_positions[track_i-1][0]
		current_pos = track_x_positions[track_i][0]
		
		if current_pos < previous_pos+spacing:
			track_x_positions[track_i] = ( previous_pos+spacing ,track_x_positions[track_i][1] ) 
	
	for track_x,idx in track_x_positions:
		pos[idx] = ( track_x ,y_min)
		
	n_jf_vtx = len(jet_graph['jf_node_index'])
	
	for idx, vtx_i in zip(jet_graph['jf_node_index'], range(len(jet_graph['jf_node_index']))):
		pos[idx] = (min_max_x[0]+x_range/2+(vtx_i)*x_range/n_jf_vtx/2.,y_min-80)
		labels[idx] = 'JF'+str(vtx_i)
	
	nx.draw_networkx_nodes(G,pos,node_color='orchid',node_size=1200,ax=ax,nodelist=jet_graph['jf_node_index'])
	nx.draw_networkx_nodes(G,pos,node_color='lightskyblue',node_size=300,ax=ax,nodelist=jet_graph['trk_node_index'])
	nx.draw_networkx_nodes(G,pos,node_color=node_colors,node_size=800,ax=ax,nodelist=jet_graph['particle_node_index'])
	nx.draw_networkx_edges(G,pos,ax=ax)
	
	
	
	for idx, pdgid,x0,y0,z0, x,y,z, stat,injet in zip(jet_graph['particle_node_index'], 
										  jet_graph['particle_node_pdgid'],
										  jet_graph.particle_node_prod_x, 
										  jet_graph.particle_node_prod_y,
										  jet_graph.particle_node_prod_z,
										  jet_graph.particle_node_decay_x, 
										  jet_graph.particle_node_decay_y,
										  jet_graph.particle_node_decay_z,
										 jet_graph.particle_node_status,
											   jet_graph.particle_node_inJet):
		#labels[idx] = str(idx)
		#labels[idx] = str(stat)+' '+str(injet)
		#labels[idx] = '{0:.2f}'.format(x0)+'\n'+ '{0:.2f}'.format(y0)+'\n'+ '{0:.2f}'.format(z0)
		#labels[idx] = '{0:.4f}'.format(np.linalg.norm(np.array([pv_x,pv_y,pv_z])-np.array([x0,y0,z0])))
		if pdgid in pdg_id_dict:
			labels[idx] = pdg_id_dict[pdgid]
		else:
			labels[idx] = str(pdgid)

	nx.draw_networkx_labels(G,pos,labels,ax=ax)


def plot_locations(graph,ax):

	had_list = graph.hadron_list
	node_list = graph.nodes

	vertices = list(set([node.vertex_idx for node in node_list]))

	if 0 not in vertices:
		vertices.append(0)
	if 1 not in vertices:
		vertices.append(1)

	n_vertices = len(vertices)


	locations_g = dgl.DGLGraph()
	locations_g.add_nodes(n_vertices)
	loc_labels = {}
	loc_spacing =  500

	x_range = n_vertices*loc_spacing

	loc_labels[0] = 'pileup/\nfakes'
	loc_labels[1] = 'primary'

	
	G = locations_g.to_networkx()
	pos = {} 

	for pos_i in range(n_vertices):
		pos[pos_i] = (loc_spacing*pos_i, 0)
		if pos_i < 2:
			continue
		for node in node_list:
			if node.vertex_idx == pos_i:
				loc_labels[pos_i] = '{0:.2f}'.format( np.linalg.norm(node.origin) )
				break
	
	nx.draw_networkx_nodes(G,pos,node_color='mediumaquamarine',node_size=1800,ax=ax)
	nx.draw_networkx_edges(G,pos,ax=ax)
	nx.draw_networkx_labels(G,pos,loc_labels,ax=ax)


	#fake/pileup vertex 
	center_point = pos[0]
	sub_g = dgl.DGLGraph()
	sub_g.add_nodes(1)
	sub_g_pos = {0: center_point}
	n_children = 0
	r_x = loc_spacing/3.5
	r_y = 30
	for node in node_list:
		if node.vertex_idx == 0:
			n_children+=1
			sub_g.add_nodes(1)
			sub_g.add_edge(0,n_children)

	child_idx = 0
	n_tracks = 0
	for node in node_list:
		if node.vertex_idx == 0:
			child_idx+=1
				
			sub_g_pos[child_idx] = (center_point[0]+r_x*np.cos( ((child_idx-1)/float(n_children))*2*np.pi ),
							  center_point[1]+r_y*np.sin( ((child_idx-1)/float(n_children))*2*np.pi ))
	G = sub_g.to_networkx()
		
	nx.draw_networkx_nodes(G,sub_g_pos,node_color='skyblue',node_size=800,ax=ax,nodelist=range(1,n_children+1))
	nx.draw_networkx_edges(G,sub_g_pos,ax=ax)

	for vtx_i in range(1,n_vertices):

		center_point = pos[vtx_i]


		sub_g = dgl.DGLGraph()
		sub_g.add_nodes(1)
		sub_g_labels = {}

		sub_g_pos = {0: center_point}



		n_children = 0
		for node in node_list:
			if node.vertex_idx == vtx_i:
				n_children+=1
				sub_g.add_nodes(1)
				sub_g.add_edge(0,n_children)

				if node.pdgid in pdg_id_dict:
					sub_g_labels[n_children] = pdg_id_dict[node.pdgid]
				else:
					sub_g_labels[n_children] = str(node.pdgid)


				

		r_x = loc_spacing/3.5
		r_y = 30

		child_idx = 0
		n_tracks = 0
		for node in node_list:
			if node.vertex_idx == vtx_i:
				child_idx+=1
				
				sub_g_pos[child_idx] = (center_point[0]+r_x*np.cos( ((child_idx-1)/float(n_children))*2*np.pi ),
							  center_point[1]+r_y*np.sin( ((child_idx-1)/float(n_children))*2*np.pi ))

				if node.reconstructed:
					sub_g.add_nodes(1)
					n_tracks+=1
					sub_g.add_edge(child_idx,n_children+n_tracks)

					sub_g_pos[n_children+n_tracks] = (center_point[0]+1.5*r_x*np.cos( ((child_idx-1)/float(n_children))*2*np.pi ),
							  center_point[1]+1.5*r_y*np.sin( ((child_idx-1)/float(n_children))*2*np.pi ))



		G = sub_g.to_networkx()
		
		nx.draw_networkx_nodes(G,sub_g_pos,node_color='darksalmon',node_size=800,ax=ax,nodelist=range(1,n_children+1))
		nx.draw_networkx_nodes(G,sub_g_pos,node_color='skyblue',node_size=300,ax=ax,nodelist=range(n_children+1,n_children+n_tracks+1))
		nx.draw_networkx_edges(G,sub_g_pos,ax=ax)
		nx.draw_networkx_labels(G,sub_g_pos,sub_g_labels,ax=ax,nodelist=range(1,n_children+1))


def create_dgl_for_plot(graph,addJF=False):
	g = dgl.DGLGraph()

	node_list = graph.nodes

	n_nodes = len( node_list )

	g.add_nodes(n_nodes)

	node_colors = []
	jf_nodes = {}
	node_pdgids = []

	for node_i,node in enumerate(node_list):
		if node.pdgid in pdg_id_dict:
			node_pdgids.append(pdg_id_dict[node.pdgid])
		else:
			node_pdgids.append(str(node.pdgid))
		

		if abs(node.charge) > 0:
			
			if node.vertex_idx not in node_color_dict['charged']:
				node_colors.append(node_color_dict['charged'][np.nan])
			else:
				node_colors.append(node_color_dict['charged'][node.vertex_idx])
		else:
			node_colors.append(node_color_dict['neutral'])
		if node.jf_vtx_idx > -1:
			if node.jf_vtx_idx not in jf_nodes:
				jf_nodes[node.jf_vtx_idx] = []
			jf_nodes[node.jf_vtx_idx].append(node_i)
			
	#loop over edges
	edge_colors = []
	for node_i,node in enumerate(node_list):
		for node_j,node2 in enumerate(node_list):
			
			edge_color = 'gray'
			if node.vertex_idx > 1:
				if node.vertex_idx == node2.vertex_idx:
					if node.vertex_idx in edge_color_dict:
						edge_color = edge_color_dict[node.vertex_idx]
					else:
						edge_color = 'r'
					g.add_edge(node_i,node_j)
					edge_colors.append(edge_color)
				#if node.jf_vtx_idx > -1 and node.jf_vtx_idx==node2.jf_vtx_idx and addJF:
					#g.add_edge(node_i,node_j)
					#edge_colors.append('gray')
				
	if not addJF:
		return g, node_colors,node_pdgids,edge_colors
	else:
		return g, node_colors,node_pdgids,edge_colors,jf_nodes

def create_hadron_G(graph):
	had_list = graph.hadron_list
	G = nx.Graph()
	for h_i, h in enumerate(had_list):
		
		G.add_node(h_i,label = h[1]) 

	return G

def draw_hadron_plot(ax,G):

	pos = nx.circular_layout(G)

	n_data = G.nodes.data()
	n_label = {}
	for key in pos:
		pos[key] = pos[key]*0.3
		if n_data[key]['label'] in pdg_id_dict:
			n_label[key] = pdg_id_dict[ n_data[key]['label'] ]
		else:
			n_label[key] = n_data[key]['label']
		ax.text(pos[key][0],pos[key][1]+0.2,n_label[key],fontsize=20,
			horizontalalignment='center',verticalalignment='center')
	nx.draw_networkx(G,pos=pos,with_labels=False, style='-',ax=ax,arrows=False,
		node_color='k')
	return pos

def add_circle(p,radius):
	x, y = p
	added_points = []
	for n in range(36):
		ang = (2*np.pi)*(float(n)/36.0)
		new_p = [ x+radius*np.cos(ang),y+radius*np.sin(ang) ]
		added_points.append(new_p)
		
	return added_points

def fill_points(points_list,r=0.18):
	extended = []
	for p in points_list:
		extended.append(p)
		added = add_circle(p,r)
		for ad in added:
			extended.append(ad)
	return np.array(extended)


def create_graph_plot(graph,ax,draw_JF=True):
	
	graph.sort_nodes()

	g, node_colors,node_pdgids,e_colors, jf_nodes = create_dgl_for_plot(graph,addJF=True)

	hadron_G = create_hadron_G(graph)
	hadron_positions = draw_hadron_plot(ax,hadron_G)
	hadron_node_indices = [n.had_idx for n in graph.nodes]
	
	G = g.to_networkx()

	pos = nx.circular_layout(G)

	e_widths=[0.2 if e_c=='gray' else 3 for e_c in e_colors]

	nx.drawing.draw_circular(G,with_labels=False,node_color=node_colors,edge_color=e_colors, style='-',ax=ax
						,width=e_widths,arrows=False)

	for n_i, n_idx in enumerate(hadron_node_indices):
		if np.isnan(n_idx):
			continue
		origin_point = hadron_positions[n_idx]
		node_point = pos[n_i]
		lcolor = node_colors[n_i]
		ax.plot([origin_point[0],node_point[0]],[origin_point[1],node_point[1]],
			c=lcolor,linestyle='-',alpha=0.3)

	points_dict = {}
	if draw_JF:
		for key in jf_nodes:
			points_dict[key] = np.array([pos[x] for x in jf_nodes[key]])
			fill_p = fill_points(points_dict[key])
			hull = ConvexHull(fill_p)
			for simplex in hull.simplices:

				ax.plot(fill_p[simplex, 0], fill_p[simplex, 1], 'k--')

def plot_graph_jet_image(graph,ax):
	g, node_colors,node_pdgids,e_colors = create_dgl_for_plot(graph)
	# PVx,PVy,PVz = graph.
	j_axis = graph.axis

	rotMatrix = rotationMatrix(j_axis,[1,0,0])

	
	for node_i, node in enumerate( graph.nodes ):
		if not  type(node.p4) == float:
			_ ,px,py,pz = node.p4
			p = np.array([px,py,pz])
			p = 0.5*(p/np.linalg.norm(p))
			px,py,pz = 5.0*np.dot(rotMatrix,list(p))
			x,y,z = np.dot(rotMatrix,list(node.origin) )

			lcolor = node_colors[node_i] #'purple'
			lstyle = '-'
			lwidth = 6

			ax.plot([x,x+px],[y,y+py],c=lcolor,linestyle=lstyle,lw=lwidth)

	ylim = list(ax.get_ylim())
	xlim = list(ax.get_xlim())

	for node_i, node in enumerate( graph.nodes ):
		if not node.reconstructed:
			continue
		pt,d0,d0signed,z0,z0signed,phi,theta,qoverp = node.matched_track
		track = build_track_trajectory( [phi,theta,d0,z0,qoverp] ) 

		plot_track(ax,track,rotMatrix,node_colors[node_i])

	
	ax.set_xlabel('distance along jet axis [mm]',fontsize=20)
	ax.set_ylabel('distance transverse to jet axis [mm]',fontsize=20)

	ax.plot([0,1.5*xlim[1]],[0,0],c='k')
	ax.set_xlim(0,1.5*xlim[1])
	ax.set_ylim(-1.2*max(abs(ylim[0]),abs(ylim[1])), 1.2*max(abs(ylim[0]),abs(ylim[1])) )


