import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
import time 

# grid-boundary exceeding
# find mini-optimisations 
# write for each var its dimensions
# TODAY - final accelaration and create the integration - simulation script

# Have it as a func for future improvements (e.g. AMR)
def meshgrid3D(L, gridN):
    x = np.linspace(-L, L, gridN)
    y = np.linspace(-L, L, gridN)
    z = np.linspace(-L, L, gridN)
    
    return np.meshgrid(x, y, z, indexing='ij') 

def selfGravityAcc(pos, rho, grid, grid_rho, phi, L, d, r_centroid, distances, A, G, yIT, zIT):
    X, Y, Z = grid

    # Counting from bottom left and out to top right and inwards due to meshgrid indexing
    cell_indices = np.floor((pos + L) / d + 1).astype(int)

    # Offsets for neighboring grid points (including the cell itself)
    offsets = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
    
    # Density distribution using CIC (Cloud-In-Cell) or NPG (Nearest-Grid-Point)
    for i, offset in enumerate(offsets):
        # Compute the indices of the neighboring grid points
        neighbor_indices = cell_indices + offset
        x, y, z = np.clip(neighbor_indices[:, 0],0,19), np.clip(neighbor_indices[:, 1],0,27), np.clip(neighbor_indices[:, 2],0,27)
        
        # Extract coordinates of neighboring grid points to calculate distances
        neighbor_coords = np.array([
            X[x, y, z],
            Y[x, y, z],
            Z[x, y, z]
        ]).T      

        # Compute the distance between particles and neighboring grid points
        distances[:, i] = np.sqrt(np.sum((pos - neighbor_coords)**2, axis=1))
        grid_rho[x + yIT*y + zIT*z] += (r_centroid / distances[:, i]) * (rho[:, 0] / 8)
    
    # Solve for phi using spsolve or any appropriate solver
    grid_phi = spsolve(A, 4 * np.pi * G * grid_rho)
    
    # Interpolate phi back to particle positions
    for i, offset in enumerate(offsets):
        neighbor_indices = cell_indices + offset

        # For readability
        x, y, z = np.clip(neighbor_indices[:, 0],0,19), np.clip(neighbor_indices[:, 1],0,27), np.clip(neighbor_indices[:, 2],0,27)

        # Finally find phi
        phi[:, 0] += (r_centroid / distances[:,i]) * grid_phi[x + yIT*y + zIT*z] / 8
    
    return phi

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
    neighbors = tree.query_ball_point(ri, r=3500)
   
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
        mask2 = (q > 1) & (q <= 2)

        indices_mask1 = indices[mask1]
        indices_mask2 = indices[mask2] 

        masks.append([])
        masks[i].append([mask1, mask2, indices_mask1, indices_mask2])

    return dx, dy, dz, r, masks

def getParticleDensity(r, m, h, masks):
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

def getAcc(pos, vel, m, h, k, n, nu, tree, grid, grid_rho, phi, L, d, r_centroid, distances, A, G, yIT, zIT):
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
    rho = getParticleDensity(r, m, h, masks)
    P = getPressure(rho, k, n)
    phi = selfGravityAcc(pos, rho, grid, grid_rho, phi, L, d, r_centroid, 
        distances, A, G, yIT, zIT)
    dWx, dWy, dWz = gradW(dx, dy, dz, r, h, masks)
    pg_term = m * (P/rho**2 + (P/rho**2).T + phi/rho + rho * (phi/rho**2).T)
    ax = - np.sum(pg_term * dWx, 1).reshape(-1,1)
    ay = - np.sum(pg_term * dWy, 1).reshape(-1,1)
    az = - np.sum(pg_term * dWz, 1).reshape(-1,1)
    a = np.hstack((ax, ay, az))
    a -= nu * vel
    return a
 
def main():
    # -----MAIN PROGRAM-----
    start = time.time()

    # Problem and simulation parameters
    M0 = 1.989e30 #kg
    L0 = 1000 #m

    N = 500 # number of particles
    t = 0 # start of simulation
    tEnd = 0.7 # end of simulation
    dt = 0.001 # timestep
    M = 2 * M0 # mass of the star
    R = 11 * L0 # radius of the star 
    G = 6.67e-11
    h = 2 * L0 # smoothing radius
    k = 1e-9 # equation of state constant (kg/(m*s^2)) 
    n = 1 # polytropic index of state constant
    nu = 10000 # damping coefficient (1/s)
    plotIt = True
    Nt = int(np.ceil(tEnd/dt))

    # Generate Initial Conditions
    np.random.seed(42)
    m = M / N # mass of a single particle
    pos = np.random.randn(N, 3) * R # random initial positions 
    vel = np.zeros((N, 3))
    lmbda = (2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n)) / R**2

    # Parameters for PM method
    L = 22 * L0 # Length of single dimension
    d = L / 14 # Distance between grid points in a single dimension - 440 is the limit (memory dependent)
    gridN = int(2*L/d)
    r_centroid = np.sqrt(3) * d / 2 
    distances = np.zeros((N, 8)) # distance between particles and their respective cell points
    grid_rho = np.zeros(gridN**3) # in flat form
    yIT = gridN
    zIT = gridN**2
    phi = np.zeros((N,1))

    # Simple construction of arrays for kronecker product
    I = np.eye(gridN,gridN)
    T = (-2) * I 
    T[1:gridN] += I[:gridN-1]
    T[:gridN-1] += I[1:gridN]
    A = np.kron(I,np.kron(I,T)) + np.kron(I,np.kron(T,I)) + np.kron(T,np.kron(I,I)) / d**2 # Calculate the RHS - 3D Laplacian for the Gauss' Law for Gravity

    # Grid construction
    grid = meshgrid3D(L, gridN)

    # Prep the plotting figure
    """ fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Real Time Simulation')

    ax.grid(False)

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')

    ax.xaxis.label.set_color('green')       
    ax.yaxis.label.set_color('blue')    
    ax.zaxis.label.set_color('red')

    plot_lims = 3*R
    ax.set_xlim([-plot_lims, plot_lims])
    ax.set_ylim([-plot_lims, plot_lims])
    ax.set_zlim([-plot_lims, plot_lims])

    ax.set_xticks([-plot_lims/2, 0, plot_lims/2])
    ax.set_yticks([-plot_lims/2, 0, plot_lims/2])
    ax.set_zticks([-plot_lims/2, 0, plot_lims/2]) """
    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    plotGrid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(plotGrid[0:2, 0])
    ax2 = plt.subplot(plotGrid[2, 0])
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, R, 100)
    rr[:, 0] = rlin
    rho_analytic = ((lmbda / (2*k*(1+n)) * (R**2 - rlin**2)) ** n )
    plot_scale_factor = 1e14 # To make the analytic and arithmetic solution viewable

    # Initialise KDTree 
    tree = cKDTree(pos)

    # calculate initial accelarations
    acc = getAcc(pos, vel, m, h, k, n, nu, tree, grid, grid_rho, phi, L, d, r_centroid, 
        distances, A, G, yIT, zIT)

 
    # Integration - Simulation
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2

        # drift
        pos += vel * dt

        # Update Tree 
        tree = cKDTree(pos)

        # update accelarations
        acc = getAcc(pos, vel, m, h, k, n, nu, tree, grid, grid_rho, phi, L, d, r_centroid, 
            distances, A, G, yIT, zIT)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt
            
        # get density for plotting
        r, masks = getPairwiseSeparations(pos, pos, h, tree)[3:]
        rho = getParticleDensity(r, m, h, masks)

        print(str(vel.max()))

        # Plotting 
        if plotIt or (i == Nt-1):
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
            rho_radial = getParticleDensity(r, m, h, masks)
            plt.plot(rlin, rho_radial/plot_scale_factor, color='blue')
            plt.pause(0.001)
        #plot the evolution of gravitational field and quiver plot pressure forces and velocity as well
    
    # End of simulation - print runtimes and errors
    end = time.time()
    print("Runtime with kdtree v3.0 + PM-FDM for self-gravitation was:", end-start)
    print("Errors: ")
    return 0

if __name__ == "__main__":
    main()