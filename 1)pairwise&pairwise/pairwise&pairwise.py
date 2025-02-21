import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time


def W( r, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x relative positions
	y     is a vector/matrix of y relative positions
	z     is a vector/matrix of z relative positions
	r	  is the M x N matrix of pairwise euclidean distances
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, r, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x relative positions
	y     is a vector/matrix of y relative positions
	z     is a vector/matrix of z relative positions
	r	  is the M x N matrix of pairwise euclidean distances
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz

def gravityKernel(x, y, z, r, h):
	N = r.shape[0]
	PHI = np.zeros((N,N))

	ct = 1 / h

	for i in range(N):
		ri = np.array(r[i])
		qi = ri / h 

		indices_mask1 = (qi <= 1)
		indices_mask2 = (qi > 1) & (qi <= 2)
		indices_mask3 = ~(indices_mask1 | indices_mask2)
		

		if np.any(indices_mask1) != 0:
			PHI[i][indices_mask1] = ct * (2/3 * qi[indices_mask1]**2 - 3/10 * qi[indices_mask1]**3 + 1/10 * qi[indices_mask1]**5 - 7/5)

		if np.any(indices_mask2) != 0:
			PHI[i][indices_mask2] = ct * (4/3 * qi[indices_mask2]**2 - qi[indices_mask2]**3 + 3/10 * qi[indices_mask2]**4 - 1/30 * qi[indices_mask2]**5 - 8/5 + 1/15 / qi[indices_mask2])

		if np.any(indices_mask3) != 0:
			PHI[i][indices_mask3] = - 1 / ri[indices_mask3]

	return PHI


	
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

	r = np.sqrt(dx**2 + dy**2 + dz**2)
	
	return dx, dy, dz, r
	

def getDensity( r, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	M = r.shape[0]

	w = W(r, h)
	rho = np.sum( m * w, 1 ).reshape((M,1))
	
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
	G	  : Universal constant of gravity
	n     : polytropic index
	alpha, beta : artificial viscosity parameters

	Returns:
	a     : N x 3 matrix of accelerations
	"""

	dx, dy, dz, r = getPairwiseSeparations( pos, pos )

	# Compute density at particle positions
	rho = getDensity(r, m, h)

	# Compute pressure
	P = getPressure(rho, k, n)

	# Kernel gradients
	dWx, dWy, dWz = gradW(dx, dy, dz, r, h)

	# Compute velocity differences
	vx, vy, vz, _ = getPairwiseSeparations(vel, vel)

	# Compute divergence of velocity (∇·v)
	div_v = np.sum(m * (vx * dWx + vy * dWy + vz * dWz) / rho, axis=1)
	#div_v = np.sum(m * ((vx-vx.T) * dWx + (vy-vy.T) * dWy + (vz-vz.T) * dWz) , axis=1) / rho # needs smaller timestep

	# Compute vorticity (∇×v)
	omega_x = np.sum(m * ((vy * dWz - vz * dWy) / rho), axis=1)
	omega_y = np.sum(m * ((vz * dWx - vx * dWz) / rho), axis=1)
	omega_z = np.sum(m * ((vx * dWy - vy * dWx) / rho), axis=1)
	vorticity = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

	# Compute sound speed at each position
	c = (k * (1 + 1/n) * rho**(1/n)) ** (1/2)
	c_max = c.max() # used for adaptive timestepping

	# Compute Balsara correction factor
	f = np.abs(div_v) / (np.abs(div_v) + np.abs(vorticity) + 1e-4 * c / h)
	
	# Compute averages
	c_avg = (c + c.T) / 2
	f_avg = (f + f.T) / 2

	# Compute artificial viscosity term
	r2 = dx**2 + dy**2 + dz**2 + 0.01 * h**2
	v_dot_r = vx * dx + vy * dy + vz * dz
	mu = np.where(v_dot_r < 0, h * v_dot_r / r2, 0) * f_avg / c_avg

	Pi_ij = (-alpha * mu + beta * mu**2)

	# Compute acceleration due to pressure and artificial viscosity
	p_term = (P / rho**2 + P.T / rho.T**2)
	pAV_term = m * (p_term + p_term * Pi_ij)
	ax = -np.sum(pAV_term * dWx, axis=1)
	ay = -np.sum(pAV_term * dWy, axis=1)
	az = -np.sum(pAV_term * dWz, axis=1)

	# Gravity contribution
	e = 0.001
	ct = m * G
	r_g = dx**2 + dy**2 + dz**2 + e
	r_inv = r_g**(3/2)
	gx = ct * dx / r_inv
	gy = ct * dy / r_inv
	gz = ct * dz / r_inv

	ax -= np.sum(gx, axis=1)
	ay -= np.sum(gy, axis=1)
	az -= np.sum(gz, axis=1)

	# Pack acceleration components
	a = np.column_stack((ax, ay, az))
	a_max = np.sqrt(np.sum(a**2, axis=1)).max() # used for adaptive timestepping

	return a, r_g, rho, a_max, dWx, dWy, dWz, c_max


def timestepSelector(vel, alpha, beta, h, eta, m, dWx, dWy, dWz, rho, c_max, a_max):
	"""
	Timestep Selector Algorithm

	Parameters:
	vel   : N x 3 matrix of velocities
	c_max : maximum sound speed observed
	a_max : maximum acceleration observed
	eta   : scaling timestep factor
	alpha, beta : artificial viscosity parameters (default: α=1, β=2)
	"""
	# Compute relative velocities
	vx, vy, vz, _ = getPairwiseSeparations(vel, vel)

	# Find divergence of v
	abs_div_v = np.abs(np.sum(m * (vx * dWx + vy * dWy + vz * dWz) / rho, axis=1)).max()
	
	# Courant–Friedrichs–Lewy condition for AV 
	dt_cfl = 0.3 * h / (c_max + h * abs_div_v + 1.2 * (alpha * c_max + beta * h * abs_div_v)) # NUM and DENOM factors are empirical (0.3 and 1.2 in documentation)

	# Use timestep method from Goswamiy and Pajarola
	v_max = np.sqrt(np.sum(vel**2, axis=1)).max()
	v_max =  v_max + np.sqrt(h*a_max*m)

	if(v_max < alpha * c_max):
		dt = eta * dt_cfl
	else:
		dt = dt_cfl

	return dt


def main():
    """ SPH simulation """
    
    # Start clock
    start = time.time()

    # Simulation parameters
    N         = 400    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 40     # time at which simulation ends
    dt        = 0.01   # timestep
    M         = 2      # star mass
    R         = 0.75   # star radius
    h         = 0.1    # smoothing length
    G		  = 1	   # Universal constant		
    k         = 1    # equation of state constant
    n         = 3      # polytropic index
    alpha     = 1      # constant to handle bulk viscocity
    beta      = 2      # constant to handle particle interpenetration
    plotRealTime = 	False # switch on for plotting as the simulation goes along
    
    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed
    
    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
	
    # Compute radius in the xy-plane
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

    # Define angular velocity (can be constant or depend on r)
    omega = 0.1  # Adjust as needed

    # Compute velocity components for rotation around the z-axis
    vel = np.zeros_like(pos)  # Initialize velocity array
    vel[:, 0] = -omega * pos[:, 1]  # v_x = -ωy
    vel[:, 1] = omega * pos[:, 0]   # v_y = ωx
    vel[:, 2] = 0  # No motion in the z-direction

    # Add random noise to velocity if desired
    vel += 0.01 * np.random.randn(N, 3)  # Small random perturbation
    
    # Prep figure for rho profile and particle simulation
    fig1 = plt.figure(figsize=(4,5), dpi=80)
    grid1 = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid1[0:2,0])
    ax2 = plt.subplot(grid1[2,0])
    rr = np.zeros((100,3))
    rlin = np.linspace(0,1,100)
    rr[:,0] =rlin
    rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)

    # Prep figure for conservation checks
    fig2 = plt.figure(figsize=(4,5), dpi=80)
    grid2 = plt.GridSpec(3,1, hspace=0.4)
    nrg_plot = plt.subplot(grid2[0,0])
    p_plot = plt.subplot(grid2[1,0])
    l_plot = plt.subplot(grid2[2,0])
        
    # Calculate initial gravitational accelerations
    acc, r_g, rho, a_max, dWx, dWy, dWz, c_max = getAcc(pos, vel, m, h, k, G, n, alpha, beta)
    dt = timestepSelector(vel, alpha, beta, h, 4, m, dWx, dWy, dWz, rho, c_max, a_max)
    
    # Statistics for real time simulation
    t_all = [t]

    KE_hist = [0.5 * m * np.sum(np.sum(vel**2, axis=1))]

    #dx, dy, dz, r = getPairwiseSeparations(pos, pos)
    #PE_hist = [G * m**2 / 2  * np.sum(np.sum(gravityKernel(dx, dy, dz, r, h), axis=1))]
    PE_hist = [- G * m**2 / 2 * np.sum(np.sum(1/(r_g)**(1/2), axis=1))]
	
    #U = m * (n*k) * np.sum(rho**(1/n))
    #U_hist = [U]

    Etot_hist = [KE_hist[0] + PE_hist[0]]

    r_com = np.sum(pos, axis=0) / N
    p = m * np.sum(vel, axis=0)
    l = m * np.sum(np.cross(pos-r_com, vel), axis=0)

    p = np.sqrt(np.sum(p**2))
    l = np.sqrt(np.sum(l**2))

    p_hist = [p]
    l_hist = [l]

    # Simulation Main Loop /w KDK 
    while(t < tEnd):
        # (1/2) Kick
        vel += acc * dt/2
        
        # Drift
        pos += vel * dt
        
        # Update accelerations
        acc, r_g, rho, a_max, dWx, dWy, dWz, c_max = getAcc(pos, vel, m, h, k, G, n, alpha, beta)

        # (1/2) Kick
        vel += acc * dt/2

        # Adaptive Timestep after a whole KDK cycle
        dt = timestepSelector(vel, alpha, beta, h, 4, m, dWx, dWy, dWz, rho, c_max, a_max)

        # Update time
        t += dt
        
        # Calculate all statistics
        
        # Get Kinetic Energy
        KE = 0.5 * m * np.sum(np.sum(vel**2, axis=1))

        # Get Potential Energy
        #PHI = gravityKernel(dx, dy, dz, r, h)
        #PE = G * m**2 / 2 * np.sum(np.sum(PHI, axis=1))
        PE = - G * m**2 / 2 * np.sum(np.sum(1/r_g**(1/2), axis=1))

        #U = m * (n*k) * np.sum(rho**(1/n))				

        # Get Center of Mass + Linear & Angular Momentum on each axis
        r_com = np.sum(pos, axis=0) / N 
        p = m * np.sum(vel, axis=0)
        l = m * np.sum(np.cross(pos-r_com, vel), axis=0)
        
        p = np.sqrt(np.sum(p**2))
        l = np.sqrt(np.sum(l**2)) 

        # Append time series of statistics
        t_all.append(t)
        KE_hist.append(KE)
        PE_hist.append(PE)
        Etot_hist.append(KE+PE)
        p_hist.append(p)
        l_hist.append(l)

        # Plot in real time
        if plotRealTime or (t >= tEnd):
            # Get density for plotting
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            
            plt.sca(ax2)
            plt.cla()
            rplot = getPairwiseSeparations(rr, pos)[-1]
            rho_radial = getDensity( rplot, m, h )
            ax2.set(xlim=(0, 1), ylim=(0, np.max(rho_radial)))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)

            # Plot statistics in figure 2
            plt.sca(nrg_plot)
            plt.cla()
            plt.plot(t_all, KE_hist, color="red")
            plt.plot(t_all, PE_hist, color="blue")
            plt.plot(t_all, Etot_hist, color="black")

            plt.sca(p_plot)
            plt.cla()
            plt.plot(t_all, p_hist, color = "orange")

            plt.sca(l_plot)
            plt.cla()
            plt.plot(t_all, l_hist, color = "pink") 

        
    # End clock
    end = time.time()
    
    # Print runtime and errors
    print("Runtime with pairwise pressure & gravity was:", end-start)
    #print("Errors: ")

    # Add labels/legend
    plt.sca(ax2)
    plt.xlabel('radius')
    plt.ylabel('density')

    plt.sca(nrg_plot)
    plt.xlabel('time')
    plt.ylabel('energy')

    plt.sca(p_plot)
    plt.xlabel('time')
    plt.ylabel('Linear Momentum')
    
    plt.sca(l_plot)
    plt.xlabel('time')
    plt.ylabel('Angular Momentum')


    # Save figure
    fig1.savefig('clean_folder/1)pairwise&pairwise/star.png',dpi=240)
    fig2.savefig('clean_folder/1)pairwise&pairwise/statistics.png',dpi=240)
    plt.show()
        
    return 0
    



if __name__== "__main__":
    main()