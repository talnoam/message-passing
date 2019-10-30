import numpy as np
from functools import partial
import scipy.optimize
from scipy.optimize.nonlin import NoConvergence
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        

def xv(phi,d0,phi0,theta,qoverp,z0):
	return d0*np.cos(phi0+np.pi/2.0)+(np.sin(theta)/(2.0*qoverp*0.299792))*(np.cos(phi+np.pi/2.0)-np.cos(phi0+np.pi/2.0))

def yv(phi,d0,phi0,theta,qoverp,z0):
	return d0*np.sin(phi0+np.pi/2.0)+(np.sin(theta)/(2.0*qoverp*0.299792))*(np.sin(phi+np.pi/2.0)-np.sin(phi0+np.pi/2.0))

def zv(phi,d0,phi0,theta,qoverp,z0):
	return z0-(np.cos(theta)/(2.0*qoverp*0.299792))*(phi-phi0)




def rotationMatrix(jet_axis,des_directon):

	norm_des_dir = des_directon/np.linalg.norm(des_directon)

	axis = np.cross(jet_axis,norm_des_dir)

	dotprod = np.dot(jet_axis,norm_des_dir)

	angle = np.arccos(min([dotprod,1.0]))

	r = np.zeros((3,3))


	if angle < 0.00001:
		r[0][0] = 1.0
		r[1][1] = 1.0
		r[2][2] = 1.0

		return np.array(r)

	ca = np.cos(angle)
	sa = np.sin(angle)
	axislength = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
	dx = axis[0]/axislength
	dy = axis[1]/axislength
	dz = axis[2]/axislength


	r[0][0] = ca+(1-ca)*dx*dx
	r[0][1] = (1-ca)*dx*dy-sa*dz
	r[0][2] = (1-ca)*dx*dz+sa*dy

	r[1][0] = (1-ca)*dy*dx+sa*dz
	r[1][1] = ca+(1-ca)*dy*dy
	r[1][2] =  (1-ca)*dy*dz-sa*dx

	r[2][0] = (1-ca)*dz*dx-sa*dy
	r[2][1] = (1-ca)*dz*dy+sa*dx
	r[2][2] = ca+(1-ca)*dz*dz

	return np.array(r)

def build_track_trajectory(track,true_range=300):
	phi0,theta,d0,z0,qoverp = track
	
	
	phiV = (phi0-1.0*np.sign(qoverp)*np.pi/320.0)
	xv_end = xv(phiV,d0,phi0,theta,qoverp,z0)
	yv_end = yv(phiV,d0,phi0,theta,qoverp,z0)
	zv_end = zv(phiV,d0,phi0,theta,qoverp,z0)

	distance = np.linalg.norm([xv_end,yv_end,zv_end])


	fraction_to_target = (float(2.0*true_range)/distance)

	phiFinal = (phi0-1.0*np.sign(qoverp)*fraction_to_target*(np.pi/320.0))

	npoints = 800

	track_trajectory = []
	
	for phiV in np.linspace(phi0,phiFinal,npoints):
		xV = xv(phiV,d0,phi0,theta,qoverp,z0)
		yV = yv(phiV,d0,phi0,theta,qoverp,z0)
		zV = zv(phiV,d0,phi0,theta,qoverp,z0)
		
		track_trajectory.append([xV,yV,zV])
		
	return np.array(track_trajectory)

def plot_track(ax, track,rotMatrix,track_color):
	
	rotated_track = np.array( [np.dot(rotMatrix,[x,y,z]) for x,y,z in track] )
	
	linestyle='--'

	ax.plot(rotated_track[:,0],rotated_track[:,1],c=track_color,linestyle=linestyle)


def G(data, phi):
	d0, z0 , phi0,theta,qoverp, rotMatrix, z_target = data
	
	xV = xv(phi,d0,phi0,theta,qoverp,z0)
	yV = yv(phi,d0,phi0,theta,qoverp,z0)
	zV = zv(phi,d0,phi0,theta,qoverp,z0)
	x,y,z = np.dot(rotMatrix,[xV,yV,zV])
	
	return abs(z-z_target)


def get_phi(target_z,track,rotMatrix):
	d0, z0 , phi0,theta,qoverp = track
	
	data = (d0, z0 , phi0,theta,qoverp,rotMatrix,target_z)
	G_partial = partial(G, data)
	
	try:
		phi_z = scipy.optimize.broyden1(G_partial,phi0, f_tol=1e-3)
	except NoConvergence as e:
		phi_z = e.args[0]

	return float(phi_z)

def get_xy_p(phi,track,rotMatrix):
	d0, z0 , phi0,theta,qoverp = track

	p3 = abs(1.0/qoverp)

	xV = xv(phi,d0,phi0,theta,qoverp,z0)
	yV = yv(phi,d0,phi0,theta,qoverp,z0)
	zV = zv(phi,d0,phi0,theta,qoverp,z0)
	
	pVx = xV+np.cos(phi)*np.sin(theta)
	pVy = yV+np.sin(phi)*np.sin(theta)
	pVz = zV+np.cos(theta)
	
	x,y,z = np.dot(rotMatrix,[xV,yV,zV])
	px,py,pz = np.dot(rotMatrix,[pVx,pVy,pVz])

	px,py,pz = px-x,py-y,pz-z

	return (x,y),(px,py,pz),p3

