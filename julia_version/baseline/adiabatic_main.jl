using Random
using GLMakie
using Mmap
include("./adiabatic_SnapshotRW.jl")
using .SnapshotRW

include("./adiabatic_forces.jl")
using .FJL


function getAcc(pos::Matrix{Float64}, vel::Matrix{Float64}, m::Float64, K::Vector{Float64}, gamma::Float64, G::Float64, alpha::Float64, beta::Float64, Kh::Int64, max_domain::Float64)
	#=
	Calculate the acceleration on each SPH particle with artificial viscosity (Balsara switch)

	Parameters:
	pos   : N x 3 matrix of positions
	vel   : N x 3 matrix of velocities
	m     : particle mass
	h     : smoothing length
	cs    : sound speed / equation of state constant
	G	  : Universal constant of gravity
	theta : opening angle for BH Tree
	alpha, beta : artificial viscosity parameters

	=#

	# Domain re-evaluation
	l_domain = min(maximum(abs.(pos)), max_domain)

	# Get acceleration of fluid
	acc, rho, dWx, dWy, dWz, h, cs, mu, vij_x, vij_y, vij_z, Pi_ij, PHI = FJL.Accs(pos, vel, m, K, gamma, G, alpha, beta, Kh)

	
	return acc, rho, dWx, dWy, dWz, h, cs, mu, vij_x, vij_y, vij_z, Pi_ij, PHI, l_domain
end


function fast_cross_threads(A::Matrix{<:Real}, B::Matrix{<:Real})
    @assert size(A) == size(B)
    @assert size(A, 2) == 3

    N = size(A, 1)
    C = similar(A)

    Threads.@threads for i in 1:N
        ax, ay, az = A[i, 1], A[i, 2], A[i, 3]
        bx, by, bz = B[i, 1], B[i, 2], B[i, 3]

        C[i, 1] = ay * bz - az * by
        C[i, 2] = az * bx - ax * bz
        C[i, 3] = ax * by - ay * bx
    end

    return C
end


# Find star radius for further stat investigation / Works for a fully formed star
function find_star_radius(rlin::AbstractVector{<:Real}, rho_radial::AbstractVector{<:Real}; threshold=1e-20)
    @assert length(rlin) == length(rho_radial) "rlin and rho_radial must be the same length"

    idx = findfirst(x -> x < threshold, rho_radial)
    return isnothing(idx) ? rlin[end] : rlin[idx]
end



function main()
	# Start clock
	start = time()

	# Read snapshot
	snap = SnapshotRW.read_snapshot("snapshots/bin/evrard_collapse_iniconds.csv")
	
	# Initial Distribution 
	pos = snap[:pos] 
	vel = snap[:vel]
	
	constants = snap[:constants]
	
	# === Simulation Parameters ===
	iterID    = constants["iterID"] # Iteration ID for snapshots and statistics book-keeping
	N         = constants["N"]      # Number of particles
	Kh		  = constants["Kh"]     # Number of neighbors for hydro_end
	Kgr       = constants["Kgr"]	# Bucket size for gravity Octree
	t         = constants["t"]      # current time of the simulation
	tEnd      = constants["tEnd"]   # time at which simulation ends
	M         = constants["M"]     	# star mass
	R         = constants["R"]  	# radius used for initial distribution
	alpha     = constants["alpha"]  # constant to handle bulk viscocity
	beta      = constants["beta"]   # constant to handle particle interpenetration
	theta     = constants["theta"]  # Opening angle for BH Tree Monopole approximation
	max_domain= constants["max_domain"]# Domain maximum extend in a single axis / outside that we don't care for interactions

	# Physical constants (in CGS units) 
	G = constants["G"]        		# Gravitational constant [cm^3 g^-1 s^-2]

	# Gas properties 
	m 	  = constants["m"]
	gamma = constants["gamma"]
	K = snap[:K]
	K = Vector{Float64}(K)

	# Initialise static arrays for speed
	pos_half = zeros(N, 3)
	vel_half = zeros(N, 3)

	# Plotting parameters 
	snapInterval 	= 5    # switch on for snapshots every now and then	
	intervalCounter = 0    # Counter that resets everytime we plot
	keepSnaps 		= true
	showPlots 		= true 

	# === Prepare plot figures and axis ===

	# Prep figure for rho profile and particle simulation
	plotN = 10000
	rr = zeros(plotN, 3)
	rlin = LinRange(0, 0.2*max_domain, plotN)  # scale sample points to domain scale
	rho_radial = zeros(plotN)

	# Add plots 
	fig1 = Figure(size=(500, 500))
	ax1 = Axis(fig1[1,1])
	ax1.limits=(-1.4, 1.4, -1.4, 1.4)
	ax2 = Axis(fig1[2,1], xlabel="radius", ylabel="density")

	# Prep figure for conservation checks
	fig2 = Figure(size=(500, 500))
	nrg_plot = Axis(fig2[1,1], xlabel="Time", ylabel="Energy")
	l_plot = Axis(fig2[2,1], xlabel="Time", ylabel="L Mom")
	p_plot = Axis(fig2[3,1], xlabel="Time", ylabel="Ang Mom")

	# Create GUI panes
	screen1 = display(GLMakie.Screen(), fig1)
	screen2 = display(GLMakie.Screen(), fig2)


	# === Initiate simulation ===

	# Initialise binary file that stores statistics
	stats_arr, stats_io = SnapshotRW.open_or_create_stats_mmap("./snapshots/stats")

	println("Starting simulation...")

	while t < tEnd 
	
		# Synchronise acceleration with pos, vel
		acc, rho, dWx, dWy, dWz, h, cs, mu, vij_x, vij_y, vij_z, Pi_ij, PHI, l_domain = getAcc(pos, vel, m, K, gamma, G, alpha, beta, Kh, max_domain)
		
		
		# ------- Adaptive Timestep -------
		vel_r = sqrt.(sum(vel.^2, dims=2))
		a_r = sqrt.(sum(acc.^2, dims=2))
		v_dot_dW = vij_x .* dWx + vij_y .* dWy + vij_z .* dWz
		abs_div_v = abs.(-sum(m * v_dot_dW , dims=2) ./ rho)
		dt = 0.3 * minimum([
					minimum(1 ./ abs_div_v),
					minimum(h ./ vel_r),
					minimum(sqrt.(h ./ a_r)),
					minimum(h ./ (cs .+ 1.2 * (alpha * cs .+ beta * maximum(mu, dims=2))))
				])
		

		# ------- Statistics -------

		# Get Kinetic Energy
		T = 0.5 * m * sum(vel_r.^2)

		# Get Potential Energy
		V = G / 2 * m^2 * sum(PHI)

		# Get Internal Energy
		U = m * sum(K ./ (gamma-1) .* rho.^(gamma-1))

		# Get total energy
		Etot = T + V + U
		println("Virial Ratio " * string(abs(V/U)))

		# Get Center of Mass + Linear & Angular Momentum 
		r_com = sum(pos, dims=1) / N
		rcom_x, rcom_y, rcom_z = r_com
		
		p = m * sum(vel, dims=1)
		linear_momentum = sqrt(sum(p.^2))

		l = m * sum(fast_cross_threads(pos.-r_com, vel), dims=1)
		angular_momentum = sqrt(sum(l.^2)) 

		stats_vector = [t, T, V, U, Etot, rcom_x, rcom_y, rcom_z, linear_momentum, angular_momentum]
		
		# Append time series of statistics in a binary File
		SnapshotRW.update_stats_row!(stats_arr, iterID, stats_vector)

		
		# ------- Leapfrog w/ Predictor - Corrector Mod -------

		# Drift (1/2) 
		pos_half .= pos + vel * dt/2

		# Kick (1/2) - just to extrapolate acceleration at half step
		vel_half .= vel + acc * dt/2
		
		# Evolve K at half step / Find acceleration at half step
		FJL.evolve_K!(K, m, Pi_ij, v_dot_dW, gamma ,rho, dt)
		acc, rho, dWx, dWy, dWz, h, cs, mu, vij_x, vij_y, vij_z, Pi_ij, PHI, l_domain = getAcc(pos_half, vel_half, m, K, gamma, G,alpha, beta, Kh, max_domain)
		
		# Evolve K another half step - Go to whole step
		FJL.evolve_K!(K, m, Pi_ij, vij_x, vij_y, vij_z, dWx, dWy, dWz, gamma, rho, dt)

		# Find v_t+1 = v_t + dt * a_t+1/2 - Do whole step
		vel .+= acc * dt

		# Correct position term - Do whole step
		pos .+= vel*dt - (1/2)*acc*dt^2


		# Update time
		t += dt
		println("Time: ", t)


		# ------- Plotting -------
		if keepSnaps * intervalCounter == snapInterval || t >= tEnd
			
			# Use r_com to sample radial density profile starting from the true center
			rr[:, 1] = rlin .+ rcom_x 	# samples lie on the x-axis
			rr[:, 2:3] .= r_com[:, 2:3]
			rho_radial = FJL.density_plot(m, rr, pos, Kh)
			R = find_star_radius(rlin, rho_radial; threshold=0.01*rho_radial[1])
			
			# Reset Interval Counter and increment snapID
			intervalCounter = 0
			constants["iterID"] = iterID
			constants["t"] = t
			constants["N"] = N
			constants["R"] = R

			println("Saving snapshot with ID: $(iterID)")

			# To plot or not to plot
			if showPlots || t >= tEnd
				# Avoid lines from overlapping with old data
				empty!(ax1)
				empty!(ax2)
				empty!(nrg_plot)
				empty!(l_plot)
				empty!(p_plot)

				# === Figure 1 ===
				# Subplot 1: Star scatter - Normalise values to plot, based on farthest star's particle or domain size
				scatterBaseScale=0.2*max_domain
				cval = min.((rho .- 3) ./ 3, 1)
				scatter!(ax1, pos[:, 1]/scatterBaseScale, pos[:, 2]/scatterBaseScale,
					color=cval, 
					colormap=:autumn, 
					markersize=10, 
					alpha=0.5, 
					)

				# Subplot 2: radial density plot
				lines!(ax2, rlin, rho_radial, color=:blue, 
					linewidth=2)
				autolimits!(ax2)

				# === Figure 2 ===

				# Read history up to now
				hist = SnapshotRW.get_stats_up_to(stats_arr, iterID)
				t_all = hist[1:iterID, 1]
				
				# Subplot: energy
				lines!(nrg_plot, t_all, hist[1:iterID, 2], color=:red, label="T")
				lines!(nrg_plot, t_all, hist[1:iterID, 3], color=:blue, label="V")
				lines!(nrg_plot, t_all, hist[1:iterID, 4], color=:green, label="U")
				lines!(nrg_plot, t_all, hist[1:iterID, 5], color=:black, label="E")
				#axislegend(nrg_plot)

				# Subplot: linear momentum
				lines!(p_plot, t_all, hist[1:iterID, 9], color=:orange, label="Linear P")

				# Subplot: angular momentum
				lines!(l_plot, t_all, hist[1:iterID, 10], color=:pink, label="Angular L")

				# Ensure write back / Bin statistics and graphs
				Mmap.sync!(stats_arr)
				SnapshotRW.write_snapshot(string(iterID), pos, vel, K; constants,rlin, rho_radial, fig1, fig2)

			else
				# Ensure write back / Bin statistics and graphs
				Mmap.sync!(stats_arr)
				SnapshotRW.write_snapshot(string(iterID), pos, vel, K; constants,rlin, rho_radial)
			end

		end

		# Increment plot interval counter and iteration counter
		iterID += 1
		intervalCounter += 1

	end 

	# End clock
	end_time = time()

	# Print runtime
	println("Faster: KD-Tree for Pressure/AV + Octree for Smoothed Gravity. Runtime: $(end_time - start) seconds")

	# Close I/O interface
	close(stats_io)
end

# Run the goddamn script
main()
