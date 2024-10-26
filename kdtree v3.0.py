import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.spatial import cKDTree
import time

# experiment with changing where the tree is created, DONE & ALMOST 0 DIFFERENCE
# try parsing the r from the pairwise function to WGRADW -> data structs for r,dx,dy,dz as sparse matrtices and parse only them 1) ftiakse ta masks na pairnoun ta swsta indices 2) h kai r tetoia wste oti den einai mask1 einai mask2
# Can i get rid of the for loop in pairwise UNFORTUNATELY NO

#----OPTIMISATIONS-----
# try poly6 kernel
# squeeze method used twice (1 neighbors, 1 gradW-W so change that)
# instead of NX3 arrays create NX3X3 arrays for contiguous memory
# stop initialising over and over again the w, gradW arrays inside W, gradW functions
# You don't need rho for plotting
# Fix the allocation problem for memory 

def W(r, h, masks):
    """
    Gaussian Smoothing kernel (3D)
    x, y, z: vectors/matrices of positions
    h: smoothing length
    w: evaluated smoothing function
    """
    N = len(r)
    M = 500
    w = np.zeros((N,M))

    ct = 1 / (np.pi*h**3)
    
    for i in range(N):
        ri = np.array(r[i]).squeeze()            

        mask1 = masks[i][0][0]
        mask2 = masks[i][0][1]
        indices_mask1 = masks[i][0][2]
        indices_mask2 = masks[i][0][3]

        q = ri / h

        if np.any(indices_mask1) != 0:
            w[i][indices_mask1] = ct * (1 - 3/2 * q[mask1]**2 + 3/4 * q[mask1]**3)

        if np.any(indices_mask2) != 0:
            w[i][indices_mask2] = ct * 1/4 * (2 - q[mask2])**3

    return w

def gradW(x, y, z, r, h, masks):
    N = len(r)
    M = 500
    gradWx = np.zeros((N,M))
    gradWy = np.zeros((N,M))
    gradWz = np.zeros((N,M))

    ct = 1 / (np.pi*h**3)
    
    for i in range(N):
        ri = np.array(r[i]).squeeze()            

        mask1 = masks[i][0][0]
        mask2 = masks[i][0][1]
        indices_mask1 = masks[i][0][2]
        indices_mask2 = masks[i][0][3]

        if np.any(indices_mask1) != 0:
            n = ct / h * (9/4 * ri[mask1] / h**2 - 3 / h)
            gradWx[i][indices_mask1] = n * x[i][mask1]
            gradWy[i][indices_mask1] = n * y[i][mask1]
            gradWz[i][indices_mask1] = n * z[i][mask1]

        if np.any(indices_mask2) != 0:
            n = ct / h * (-3/4 * (4 / ri[mask2] - 4 / h + ri[mask2] / h**2))
            gradWx[i][indices_mask2] = n * x[i][mask2]
            gradWy[i][indices_mask2] = n * y[i][mask2]
            gradWz[i][indices_mask2] = n * z[i][mask2]

    return gradWx, gradWy, gradWz


def getPairwiseSeparations(ri, rj, h, tree):
    """
    Get pairwise separations between 2 sets of coordinates
    ri: M x 3 matrix of positions
    rj: N x 3 matrix of positions
    dx, dy, dz: M x N matrices of separations
    """
    # Builds the KD-Tree and returns each point's neighbors in index order
    neighbors = tree.query_ball_point(ri, r=2*h)
   
    # Initialise with big values so that distant points are not taken into account
    dx, dy, dz, r, masks = [], [], [], [], []

    # Broadcast the difference calculation for all neighbors
    for i, indices in enumerate(neighbors):
        # Calculate differences
        diff = ri[i] - rj[indices]
        
        # Append as NumPy arrays instead of lists
        dx.append(diff[:, 0])
        dy.append(diff[:, 1])
        dz.append(diff[:, 2])
        
        # Compute squared distances
        r.append([np.sqrt(dx[-1]**2 + dy[-1]**2 + dz[-1]**2)])

        # prepare masks for W and gradW
        rr = np.array(r[i]).squeeze()
        indices = np.array(indices)
        q = rr / h
        
        mask1 = q <= 1
        mask2 = ~ mask1

        indices_mask1 = indices[mask1]
        indices_mask2 = indices[mask2] 

        masks.append([])
        masks[i].append([mask1, mask2, indices_mask1, indices_mask2])

    return dx, dy, dz, r, masks


def getDensity(r, m, h, masks):
    """
    Get Density at sampling locations from SPH particle distribution
    r: distances
    m: particle mass
    h: smoothing length
    rho: M x 1 vector of densities
    """
    # Can't take pairwise outside getDensity and put it first in getAcc because it's used for plotting as well, 
    # except if i change the main function structure as well - DONE
    w = W(r, h, masks)
    rho = np.sum(m * w, 1).reshape((-1, 1))
    return rho

def getPressure(rho, k, n):
    """
    Equation of State
    rho: vector of densities
    k: equation of state constant
    n: polytropic index
    P: pressure
    """
    P = k * rho**(1 + 1/n)
    return P

def getAcc(pos, vel, m, h, k, n, lmbda, nu, tree):
    """
    Calculate the acceleration on each SPH particle
    pos: N x 3 matrix of positions
    vel: N x 3 matrix of velocities
    m: particle mass
    h: smoothing length
    k: equation of state constant
    n: polytropic index
    lmbda: external force constant
    nu: viscosity
    a: N x 3 matrix of accelerations
    """
    dx, dy, dz, r, masks = getPairwiseSeparations(pos, pos, h, tree)
    rho = getDensity(r, m, h, masks)
    P = getPressure(rho, k, n)
    dWx, dWy, dWz = gradW(dx, dy, dz, r, h, masks)
    p_term = m * (P/rho**2 + (P/rho**2).T)
    ax = - np.sum(p_term * dWx, 1).reshape((-1,1))
    ay = - np.sum(p_term * dWy, 1).reshape((-1,1))
    az = - np.sum(p_term * dWz, 1).reshape((-1,1))
    a = np.hstack((ax, ay, az))
    a -= lmbda * pos
    a -= nu * vel
    return a

def main():
    """ SPH simulation """

    start = time.time()

    # Simulation parameters with units
    #gia n = 1.4, k=1e-3

    solar_mass = 1.989e30  # kg
    km = 1e3  # m
    M0 = solar_mass  # characteristic mass
    L0 = 1 * km  # characteristic length - radius

    N = 500  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 0.6  # time at which simulation ends
    dt = 0.001  # timestep in seconds
    M = 2 * M0  # star mass (2 solar masses)
    R = 11 * L0  # star radius (11 km)
    h = 2 * L0  # smoothing length (2 km)
    k = 1e-7 # equation of state constant (kg/(m·s^2))
    n = 1 # polytropic index
    nu = 25  # damping (1/s)
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    lmbda = (2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n)) / R**2
    m = M / N  # single particle mass
    pos = np.random.randn(N, 3) * R # randomly selected positions
    vel = np.zeros(pos.shape)

    # Initialise KDTree 
    tree = cKDTree(pos)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu, tree)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, R, 100)
    rr[:, 0] = rlin
    rho_analytic = ((lmbda / (2*k*(1+n)) * (R**2 - rlin**2)) ** n ) 

    # To make the analytic and arithmetic solution viewable
    plot_scale_factor = 1e14

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2
    
        # drift
        pos += vel * dt

        # Update Tree
        tree = cKDTree(pos)

        # update accelerations
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu, tree)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt

        # get density for plotting
        r, masks = getPairwiseSeparations(pos, pos, h, tree)[3:]
        rho = getDensity(r, m, h, masks)
        
        #to determine timestep
        print(str(vel.max()))

        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho - 3) / 3, 1).flatten()
            plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-2*R, 2*R), ylim=(-2*R, 2*R))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2*R, 0, 2*R])
            ax1.set_yticks([-2*R, 0, 2*R])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1, .1, .1))

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, R), ylim=(0, np.max(rho_analytic/plot_scale_factor)))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic/plot_scale_factor, color='gray', linewidth=2) 
            r, masks = getPairwiseSeparations(rr, pos, h, tree)[3:]
            rho_radial = getDensity(r, m, h, masks)
            plt.plot(rlin, rho_radial/plot_scale_factor, color='blue')
            plt.pause(0.001)
    
    end = time.time()

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel('radius (m)')
    plt.ylabel('density (kg/m^3)')

    # Save figure
    plt.savefig('sph.png', dpi=240)
    plt.show()

    # Error in density and runtime
    print("Error in %\n", (rho_radial.T-rho_analytic)/rho_analytic * 100)
    print("Dimensional kdtree v3.0 with cubic spline kernel N = " + str(N) + " - Runtime was: " + str(end-start))


    return 0

if __name__ == "__main__":
    main()

