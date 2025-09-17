module isothermalSim
	
	using Random
	using GLMakie
	using Mmap
	include("./SnapshotRW.jl")
	using .SnapshotRW

	include("./gravOctree_Single.jl")
	using .GJL

	include("./isothermal_hydroKDTree.jl")
	using .HJL


	function getAcc(pos::Matrix{Float64}, vel::Matrix{Float64}, m::Float64, cs::Float64, G::Float64, theta::Float64, alpha::Float64, beta::Float64, Kh::Int64)
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
		l_domain = maximum(abs.(pos))

		# Get hydrodynamics of fluid
		ax, ay, az, rho, dWx, dWy, dWz, h, vij_x, vij_y, vij_z, mu = HJL.hydrodynamics(pos, vel, m, cs, alpha, beta, Kh)

		# Get gravity accelaration
		g, PHI = GJL.gravity(l_domain, m, pos, theta, h)

		ax -= G * g[:, 1]
		ay -= G * g[:, 2]
		az -= G * g[:, 3]
		
		# Pack acceleration components
		acc = hcat(ax, ay, az)
		
		return acc, rho, dWx, dWy, dWz, h, vij_x, vij_y, vij_z, mu, PHI
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


	function run_simulation(ic_type::String, snapID::Int, snapInterval::Int, keepSnaps::Bool, showPlots::Bool)

		# Start clock
		start = time()

		# Read snapshot
		snap = SnapshotRW.read_snapshot("snapshots/" * ic_type * "/bin/" * string(snapID) * "snap.csv")

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
		R         = constants["R"]   	# radius used for initial distribution
		alpha     = constants["alpha"]  # constant to handle bulk viscocity
		beta      = constants["beta"]   # constant to handle particle interpenetration
		theta     = constants["theta"]  # Opening angle for BH Tree Monopole approximation

		# Physical constants (in CGS units) 
		G = constants["G"]        		# Gravitational constant [cm^3 g^-1 s^-2]

		# Gas properties 
		m = constants["m"]
		cs = constants["cs"] 			# Isothermal sound speed [cm/s]
		U =  constants["U"]  			# Internal energy [erg] / constant in isothermal processes

		# Plotting parameters 
		intervalCounter = snapInterval    # Counter that resets everytime we plot

		# Initialise static arrays for speed
		pos_half = zeros(N, 3)
		vel_half = zeros(N, 3)

		# === Prepare plot figures and axis ===

		# Get (approximately) the virial mass
		#= ext = maximum(sqrt.(sum(pos.^2, dims=2)))
		M = 5 * ext * cs^2 / G * 1
		m = M/N =#

		# Prep figure for rho profile and particle simulation
		plotN = 1000
		rr = zeros(1000, 3)
		rlin = LinRange(-1, 1, 1000) * R # scale sample points to domain scale
		rho_analytic = zeros(plotN)
		rho_radial = zeros(plotN)

		# Add plots 
		fig1 = Figure(size=(500, 500))
		ax1 = Axis(fig1[1,1])
		ax1.limits=(-1.4, 1.4, -1.4, 1.4)
		ax2 = Axis(fig1[2,1], xlabel="radius", ylabel="density")

		# Prep figure for conservation checks
		fig2 = Figure(size=(500, 500))
		nrg_plot = Axis(fig2[1,1], xlabel="Time", ylabel="Energy")
		p_plot = Axis(fig2[2,1], xlabel="Time", ylabel="L Mom")
		l_plot = Axis(fig2[3,1], xlabel="Time", ylabel="Ang Mom")

		# Create GUI panes
		screen1 = display(GLMakie.Screen(), fig1)
		screen2 = display(GLMakie.Screen(), fig2)


		# === Initiate simulation ===
		
		# Initialise binary file that stores statistics
		stats_arr, stats_io = SnapshotRW.open_or_create_stats_mmap("./snapshots/" * ic_type * "/stats")

		println("Starting simulation...")

		while t < tEnd 
			
			# Synchronise acceleration with pos, vel
			acc, rho, dWx, dWy, dWz, h, vij_x, vij_y, vij_z, mu, PHI = getAcc(pos, vel, m, cs, G, theta, alpha, beta, Kh)

			# ------- Adaptive Timestep -------
			vel_r = sqrt.(sum(vel.^2, dims=2))
			a_r = sqrt.(sum(acc.^2, dims=2))
			abs_div_v = abs.(-sum(m * (vij_x .* dWx + vij_y .* dWy + vij_z .* dWz) , dims=2) ./ rho)
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
			V = G / 2 * m * sum(PHI)

			# Get total energy
			Etot = T + V + 2*U

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
			
			# Evaluate acceleration at half step
			acc, rho = getAcc(pos_half, vel_half, m, cs, G, theta,alpha, beta, Kh)[1:2]
			
			# Find v_t+1 = v_t + dt * a_t+1/2 - Do whole step
			vel .+= acc * dt

			# Correct position term - Do whole step
			pos .+= vel*dt - (1/2)*acc*dt^2
			
			# Update time
			t += dt
			println("Time: ", t)

			# ------- Plotting -------
			if keepSnaps * intervalCounter == snapInterval || t >= tEnd
				# Important features for snapshots (and density plot)
				rr[:, 1] = rlin .+ rcom_x 	# samples lie on the x-axis
				rr[:, 2:3] .= r_com[:, 2:3]
				rho_radial = HJL.density_plot(m, rr, pos, Kh)

				# Reset Interval Counter and increment snapID
				intervalCounter = 0
				constants["iterID"] = iterID
				constants["t"] = t

				# To plot or not to plot
				if showPlots || t >= tEnd
					# Avoid lines from overlapping with old data
					empty!(ax1)
					empty!(ax2)
					empty!(nrg_plot)
					empty!(l_plot)
					empty!(p_plot)

					# === Figure 1 ===
					# Subplot 1: Star scatter - Normalise values to plot, based on initial star radius
					cval = min.((rho .- 3) ./ 3, 1)
					scatter!(ax1, pos[:, 1]/R, pos[:, 2]/R,
						color=cval, 
						colormap=:autumn, 
						markersize=10, 
						alpha=0.5, 
						)

					# Subplot 2: radial density plot

					# Use r_com to sample radial density profile starting from the true center

					lines!(ax2, rlin, rho_analytic, color=:gray, 
						linewidth=2, label="analytic")
					lines!(ax2, rlin, rho_radial, color=:blue, 
						linewidth=2, label="numerical")

					# === Figure 2 ===

					# Read history up to now
					hist = SnapshotRW.get_stats_up_to(stats_arr, iterID)
					t_all = hist[1:iterID, 1]
					
					# Subplot: energy
					lines!(nrg_plot, t_all, hist[1:iterID, 2], color=:red, label="T")
					lines!(nrg_plot, t_all, hist[1:iterID, 3], color=:blue, label="V")
					lines!(nrg_plot, t_all, hist[1:iterID, 5], color=:black, label="T+V+U")

					# Subplot: linear momentum
					lines!(p_plot, t_all, hist[1:iterID, 9], color=:orange, label="Linear P")

					# Subplot: angular momentum
					lines!(l_plot, t_all, hist[1:iterID, 10], color=:pink, label="Angular L")

					# Ensure write back / Bin statistics and graphs
					Mmap.sync!(stats_arr)
					SnapshotRW.write_snapshot(string(iterID), ic_type, pos, vel; constants,rlin, rho_radial, fig1, fig2)

				else
					# Ensure write back / Bin statistics and graphs
					Mmap.sync!(stats_arr)
					SnapshotRW.write_snapshot(string(iterID), ic_type, pos, vel; constants,rlin, rho_radial)
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


end
