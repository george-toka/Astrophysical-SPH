include("adiabatic_SnapshotRW.jl")
using .SnapshotRW

include("adiabatic_hydro.jl")
using .PJL

using Random

# ====== Random Initial conditions for Isothermal Sphere in CGS Units =========


M0 = 1.9891e33 # [g] solar mass in grams
pc = 5.38552341e16	   # [cm] parsec in cm

# Simulation parameters
N         = 10000      # Number of particles
Kh		  = 50         # Number of neighbors for hydro_end
Kgr       = 20		   # Bucket size for gravity Octree
t         = 0          # current time of the simulation
tEnd      = 5e12       # time at which simulation ends
M         = 2 * M0     # star mass
R         = 0.1 * pc   # scale used for initial distribution / Radius for the star
alpha     = 1.0        # constant to handle bulk viscocity
beta      = 2.0        # constant to handle particle interpenetration
theta     = 0.576      # Opening angle for BH Tree

# Physical constants (in CGS units) 
G = 6.67430e-8        # Gravitational constant [cm^3 g^-1 s^-2]
kB = 1.380649e-16     # Boltzmann constant [erg K^-1]
mH = 1.6735575e-24    # Mass of hydrogen atom [g]

# Gas properties 
m = M / N
gamma = 5/3
mu = 2.33             # Mean molecular weight for molecular cloud (H2 + He)
T = 10                # Temperature [K]

# Sample random initial conditions taken from uniform distribution
Random.seed!(42)
pos = randn(N,3) * R
r_com = sum(pos, dims=1) / N
omega = 0.0 
vel = zeros(N,3)  # Initialize velocity array
vel[:, 1] = -omega * (pos[:, 2] .- r_com[2])  # v_x = -ωy
vel[:, 2] = omega * (pos[:, 1] .- r_com[1])  # v_y = ωx
vel[:, 3] .= 0  # No motion in the z-direction 

#vel += 0.01 * randn(N, 3) * R # Small random perturbation

# Find the maximum extend of the distribution and multiply by some factor η, to determine domain size
max_domain = 5*maximum(abs.(pos))

# Calculate the initial vector of polytropic constants (one for each particle) / all equal in the beginning
r_com = sum(pos, dims=1) / N
rho0 = PJL.density_plot(m, r_com, pos, Kh, max_domain); rho0 = rho0[1]
K = fill(kB * T / (mu * mH * rho0^(gamma-1)), N)  # Polytropic constant vector [erg⋅cm⁶/g²]


constants = Dict(
    "iterID" => 1, # Used for snaps and statistics book-keeping
    "N" => N, 
    "Kh" => Kh,
    "Kgr" => Kgr,
    "t" => t,
    "tEnd" => tEnd,
    "M" => M,
    "R" => R,
    "alpha" => alpha,
    "beta" => beta,
    "G" => G,
    "theta" => theta,
    "m" => m,
    "gamma" => gamma,
    "max_domain" => max_domain
)

SnapshotRW.write_snapshot(string(1), pos, vel, K; constants=constants)