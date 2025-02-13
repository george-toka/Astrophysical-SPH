import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W( x, y, z, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T
	
	return dx, dy, dz
	

def getDensity( r, pos, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos )
	
	rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P
	

def getAcc(pos, vel, m, h, k, G, n, alpha, beta):
	"""
	Calculate the acceleration on each SPH particle with artificial viscosity (Balsara switch)

	Parameters:
	pos   : N x 3 matrix of positions
	vel   : N x 3 matrix of velocities
	m     : particle mass
	h     : smoothing length
	k     : equation of state constant
	n     : polytropic index
	lmbda : external force constant
	nu    : viscosity
	alpha, beta : artificial viscosity parameters (default: α=1, β=2)

	Returns:
	a     : N x 3 matrix of accelerations
	a_ampl: magnitude of acceleration per particle
	"""

	N = pos.shape[0]

	# Compute density at particle positions
	rho = getDensity(pos, pos, m, h)

	# Compute pressure
	P = getPressure(rho, k, n)

	# Get pairwise distances and kernel gradients
	dx, dy, dz = getPairwiseSeparations(pos, pos)
	dWx, dWy, dWz = gradW(dx, dy, dz, h)

	# Compute velocity differences
	vx, vy, vz = getPairwiseSeparations(vel, vel)

	# Compute divergence of velocity (∇·v)
	div_v = np.sum(m * (vx * dWx + vy * dWy + vz * dWz) / rho, axis=1)

	# Compute vorticity (∇×v)
	omega_x = np.sum(m * ((vy * dWz - vz * dWy) / rho), axis=1)
	omega_y = np.sum(m * ((vz * dWx - vx * dWz) / rho), axis=1)
	omega_z = np.sum(m * ((vx * dWy - vy * dWx) / rho), axis=1)
	vorticity = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

	# Compute sound speed at each position
	c = (k * (1 + 1/n) * rho**(1/n)) ** (1/2)

	# Compute Balsara correction factor
	f = np.abs(div_v) / (np.abs(div_v) + np.abs(vorticity) + 1e-4 * c / h)

	# Compute artificial viscosity term
	r2 = dx**2 + dy**2 + dz**2 + 0.01 * h**2
	v_dot_r = vx * dx + vy * dy + vz * dz
	mu = np.where(v_dot_r < 0, h * v_dot_r / r2, 0)

	# Compute averages
	c_avg = (c + c.T) / 2
	rho_avg = (rho + rho.T) / 2
	f_avg = (f + f.T) / 2

	Pi_ij = (-alpha * c_avg * mu + beta * mu**2) / rho_avg * f_avg

	# Compute acceleration due to pressure and artificial viscosity
	ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2 + Pi_ij) * dWx, axis=1)
	ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2 + Pi_ij) * dWy, axis=1)
	az = -np.sum(m * (P / rho**2 + P.T / rho.T**2 + Pi_ij) * dWz, axis=1)

	# Gravity contribution
	e = 0.01
	ct = m * G
	r_inv = (dx**2 + dy**2 + dz**2 + e)**(3/2)
	gx = ct * dx / r_inv
	gy = ct * dy / r_inv
	gz = ct * dz / r_inv

	ax -= np.sum(gx, axis=1)
	ay -= np.sum(gy, axis=1)
	az -= np.sum(gz, axis=1)

	# Pack acceleration components
	a = np.column_stack((ax, ay, az))
	a_ampl = np.sqrt(np.sum(a**2, axis=1))

	return a, a_ampl

	


def main():
	""" SPH simulation """
	
	# Simulation parameters
	N         = 400    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 40     # time at which simulation ends
	dt        = 0.04   # timestep
	M         = 2      # star mass
	R         = 0.75   # star radius
	h         = 0.1    # smoothing length
	G		  = 1	   # Universal constant		
	k         = 0.1    # equation of state constant
	n         = 1      # polytropic index
	alpha     = 1      # constant to handle bulk viscocity
	beta      = 2      # constant to handle particle interpenetration
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
	m     = M/N                    # single particle mass
	pos   = np.random.randn(N,3)   # randomly selected positions and velocities
	vel   = np.zeros(pos.shape)
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	rr = np.zeros((100,3))
	rlin = np.linspace(0,1,100)
	rr[:,0] =rlin
	rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)
    	
	# calculate initial gravitational accelerations
	acc, a_ampl = getAcc( pos, vel, m, h, k, G, n, alpha, beta)
	dt = 0.3 * np.sqrt(h / a_ampl.max())
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc,a_ampl = getAcc( pos, vel, m, h, k, G, n, alpha, beta)

		# Courant Condition
		#dt = min((0.3 * (2*h) / (1.2*nu + 1) * vel.max()), 0.3 * np.sqrt((2*h) / a_ampl.max()))

		# (1/2) kick
		vel += acc * dt/2

		# Courant Condition
		#dt = min((0.3 * (2*h) / (1.2*nu + 1) * vel.max()), 0.3 * np.sqrt((2*h) / a_ampl.max()))

		# update time
		t += dt
		
		# get density for plotting
		rho = getDensity( pos, pos, m, h )
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			cval = np.minimum((rho-3)/3,1).flatten()
			plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
			ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-1,0,1])
			ax1.set_yticks([-1,0,1])
			ax1.set_facecolor('black')
			ax1.set_facecolor((.1,.1,.1))
			
			plt.sca(ax2)
			plt.cla()
			ax2.set(xlim=(0, 1), ylim=(0, 3))
			ax2.set_aspect(0.1)
			plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
			rho_radial = getDensity( rr, pos, m, h )
			plt.plot(rlin, rho_radial, color='blue')
			plt.pause(0.001)
	    
	
	
	# add labels/legend
	plt.sca(ax2)
	plt.xlabel('radius')
	plt.ylabel('density')
	
	# Save figure
	plt.savefig('clean_folder/results/MOCZ&PAIRWISE',dpi=240)
	plt.show()
	    
	return 0
	


  
if __name__== "__main__":
  main()