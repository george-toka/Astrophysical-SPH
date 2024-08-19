import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.spatial import KDTree
import time

# experiment with neighbor distance, plotting points and see if it gets fixed


def WgradW(x, y, z, h):
    """
    Gaussian Smoothing kernel (3D)
    x, y, z: vectors/matrices of positions
    h: smoothing length
    w: evaluated smoothing function
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    w = np.zeros(r.shape)
    gradWx = np.zeros(r.shape)
    gradWy = np.zeros(r.shape)
    gradWz = np.zeros(r.shape)

    q = r / h
    ct = 1 / (np.pi*h**3)

    mask1 = q <= 1
    mask2 = (q <= 2) & (q > 1)

    w[mask1] = ct * (1 - 3/2 * q[mask1]**2 + 3/4 * q[mask1]**3)
    n = ct / h * (9/4 * r[mask1] / h**2 - 3 / h)
    gradWx[mask1] = n * x[mask1]
    gradWy[mask1] = n * y[mask1]
    gradWz[mask1] = n * z[mask1]

    w[mask2] = ct * 1/4 * (2 - q[mask2])**3
    n = ct / h * (-3/4 * (4 / r[mask2] - 4 / h + r[mask2] / h**2))
    gradWx[mask2] = n * x[mask2]
    gradWy[mask2] = n * y[mask2]
    gradWz[mask2] = n * z[mask2]

    return w, gradWx, gradWy, gradWz


def getPairwiseSeparations(ri, rj, tree):
    """
    Get pairwise separations between 2 sets of coordinates
    ri: M x 3 matrix of positions
    rj: N x 3 matrix of positions
    dx, dy, dz: M x N matrices of separations
    """
    # Builds the KD-Tree and returns each point's neighbors in index order
    neighbors = tree.query_ball_point(ri, r=4000)  

    M = ri.shape[0]
    N = rj.shape[0]

    # Initialise with big values so that distant points are not taken into account
    dx = np.full((M, N), 1e6)
    dy = np.full((M, N), 1e6)
    dz = np.full((M, N), 1e6)

    # Broadcast the difference calculation for all neighbors
    for i, indices in enumerate(neighbors):
        diff = ri[i] - rj[indices]
        dx[i, indices] = diff[:, 0]
        dy[i, indices] = diff[:, 1]
        dz[i, indices] = diff[:, 2]

    return dx, dy, dz


def getDensity(r, pos, m, h, tree):
    """
    Get Density at sampling locations from SPH particle distribution
    r: M x 3 matrix of sampling locations
    pos: N x 3 matrix of SPH particle positions
    m: particle mass
    h: smoothing length
    rho: M x 1 vector of densities
    """
    dx, dy, dz = getPairwiseSeparations(r, pos, tree)
    rho = np.sum(m * WgradW(dx, dy, dz, h)[0], 1).reshape((-1, 1))
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
    rho = getDensity(pos, pos, m, h, tree)
    P = getPressure(rho, k, n)
    dx, dy, dz = getPairwiseSeparations(pos, pos, tree)
    dWx, dWy, dWz = WgradW(dx, dy, dz, h)[1:]
    ax = - np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWx, 1).reshape((-1, 1))
    ay = - np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWy, 1).reshape((-1, 1))
    az = - np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWz, 1).reshape((-1, 1))
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

    N = 400  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 0.6  # time at which simulation ends
    dt = 0.001  # timestep in seconds
    M = 2 * M0  # star mass (2 solar masses)
    R = 11 * L0  # star radius (11 km)
    h = 2 * L0  # smoothing length (2 km)
    k = 1e-7 # equation of state constant (kg/(m·s^2))
    n = 1 # polytropic index
    nu = 20  # damping (1/s)
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    lmbda = (2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n)) / R**2
    m = M / N  # single particle mass
    pos = np.random.randn(N, 3) * R # randomly selected positions
    vel = np.zeros(pos.shape)

    # Build tree
    tree = KDTree(pos)

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

        # Update tree
        tree = KDTree(pos)

        # update accelerations
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu, tree)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt

        # get density for plotting
        rho = getDensity(pos, pos, m, h, tree)


        #to determine timestep
        #print(str(vel.max()))

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
            rho_radial = getDensity(rr, pos, m, h, tree)  
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
    print("Dimensional kdtree with cubic spline kernel - Runtime was: ", end-start)


    return 0

if __name__ == "__main__":
    main()

