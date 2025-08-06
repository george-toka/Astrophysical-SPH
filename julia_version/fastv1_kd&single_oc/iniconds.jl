module INICONDS

    include("SnapshotRW.jl")
    using .SnapshotRW

    include("polytrope_hydroKDTree.jl")
    using .HJL

    using Random
    using LinearAlgebra
    using Statistics
    using DifferentialEquations, Interpolations, QuadGK
    using FFTW


    function sample_isothermal_sphere(N::Int, R::Float64, cs::Float64)
        # Approximate density profile: rho(r) ∝ exp(-r^2 / (2 * H^2)), where H = cs / sqrt(4πGρ_c)
        # Assume ρ(r) ∝ exp(-r^2 / (2 * σ^2)), for simplicity in sampling
        σ = R / 3  # Rough estimate so most particles are inside R

        # Sample radial distances from Gaussian density
        radii = abs.(σ .* randn(N))
        radii = clamp.(radii, 0, R)

        # Uniform angular distribution
        θ = acos.(2 .* rand(N) .- 1)  # polar angle
        φ = 2π .* rand(N)             # azimuthal angle

        # Convert to Cartesian coordinates
        x = radii .* sin.(θ) .* cos.(φ)
        y = radii .* sin.(θ) .* sin.(φ)
        z = radii .* cos.(θ)

        pos = hcat(x, y, z)

        # Maxwell-Boltzmann distributed velocities (mean 0, std = cs)
        vel = cs .* randn(N, 3)

        return pos, vel
    end

    function sample_plummer_sphere(N::Int, M::Float64, a::Float64)
        G = 6.67430e-8  # CGS units

        # Position sampling using inverse transform method
        pos = zeros(N, 3)
        for i in 1:N
            # Generate radius using cumulative mass inversion
            ξ = rand()
            r = a * (ξ^(-2/3) - 1)^(-0.5)

            # Uniformly sample direction on sphere
            θ = acos(2rand() - 1)
            φ = 2π * rand()

            x = r * sin(θ) * cos(φ)
            y = r * sin(θ) * sin(φ)
            z = r * cos(θ)

            pos[i, :] .= [x, y, z]
        end

        # Velocity sampling using rejection sampling
        vel = zeros(N, 3)
        for i in 1:N
            r = norm(pos[i, :])
            ψ = -G * M / sqrt(r^2 + a^2)

            # Max velocity at r is escape speed
            v_esc = sqrt(-2ψ)

            accepted = false
            while !accepted
                x1 = rand()
                x2 = rand()

                v = x1^2 * v_esc
                g = v^2 * (1 - v^2 / v_esc^2)^(3.5)
                if 0.1 * x2 < g
                    # Sample random velocity direction
                    θ = acos(2rand() - 1)
                    φ = 2π * rand()

                    vx = v * sin(θ) * cos(φ)
                    vy = v * sin(θ) * sin(φ)
                    vz = v * cos(θ)

                    vel[i, :] .= [vx, vy, vz]
                    accepted = true
                end
            end
        end

        return pos, vel
    end


    function bonnor_ebert_sphere(N::Integer,
                                cs::Float64,
                                rho_c::Float64,
                                ξ_max::Float64;
                                velocity_mode::Symbol = :none,
                                mach_number::Float64 = 1.0,
                                alpha_vir::Float64   = 1.0,
                                rng                  = Random.GLOBAL_RNG)

        # ---------------- Lane–Emden solution (unchanged) ----------------
        G = 6.67430e-8                    # cm^3 g^-1 s^-2

        function lane_emden!(dψ, ψ, p, ξ)
            if ξ == 0.0
                dψ[1] = 0.0
                dψ[2] = 0.0
            else
                dψ[1] = ψ[2]
                dψ[2] = -2/ξ*ψ[2] + exp(-ψ[1])
            end
        end

        ψ0  = [0.0, 0.0]
        sol = solve(ODEProblem(lane_emden!, ψ0, (1e-8, ξ_max)),
                    reltol = 1e-8, abstol = 1e-8)

        ξs, ψs  = sol.t, sol[1, :]
        interp_ψ = Interpolations.LinearInterpolation(ξs, ψs, extrapolation_bc = Throw())

        a = cs / sqrt(4π * G * rho_c)     # cm

        ρ(ξ) = rho_c * exp(-interp_ψ(ξ))

        function M_of_ξ(ξ)
            integrand(x) = x^2 * exp(-interp_ψ(x))
            I, _ = quadgk(integrand, 0, ξ)
            return 4π * a^3 * rho_c * I     # g
        end

        Mtot = M_of_ξ(ξ_max)               # total mass (g)

        # -------------- sample radii by inverse‑CDF ----------------------
        function sample_ξ(N)
            radii = zeros(Float64, N)
            for i in 1:N
                y = rand(rng) * Mtot
                lo, hi = 0.0, ξ_max
                @inbounds for _ in 1:40             # bisection
                    mid = (lo + hi)/2
                    if M_of_ξ(mid) < y
                        lo = mid
                    else
                        hi = mid
                    end
                end
                radii[i] = (lo + hi)/2
            end
            radii
        end

        ξ_samples = sample_ξ(N)
        r_samples = a .* ξ_samples                    # cm

        # isotropic directions
        θ = acos.(2rand(rng, N) .- 1)
        ϕ = 2π .* rand(rng, N)

        x = r_samples .* sin.(θ) .* cos.(ϕ)
        y = r_samples .* sin.(θ) .* sin.(ϕ)
        z = r_samples .* cos.(θ)

        positions  = hcat(x, y, z)                    # N×3  (cm)
        velocities = zeros(Float64, N, 3)             # fill below

        # -------------- velocity presets ---------------------------------
        if velocity_mode == :mach
            # Gaussian components with σ = cs * Mach / √3  (3‑D rms = Mach*cs)
            velocities .= randn(rng, N, 3) .* (mach_number * cs / √3)
            velocities .-= mean(velocities, dims = 1)           # zero net momentum

        elseif velocity_mode == :virial
            # start with Gaussian field, then rescale kinetic energy
            velocities .= randn(rng, N, 3)
            m_part = Mtot / N
            cur_Ekin = 0.5 * m_part * sum(abs2, velocities)     # g cm^2 s^-2
            R_eff    = maximum(norm.(eachrow(positions)))        # cm (outer edge)
            Egrav_est = - (3/5) * G * Mtot^2 / R_eff            # uniform‑sphere approx
            desired_Ekin = 0.5 * alpha_vir * abs(Egrav_est)
            velocities .*= sqrt(desired_Ekin / cur_Ekin)
            velocities .-= mean(velocities, dims = 1)

        elseif velocity_mode != :none
            error("velocity_mode must be :none, :mach, or :virial")
        end

        return positions, velocities
    end



    function turbulent_molecular_cloud(N::Int, R_cloud::Float64, M_cloud::Float64, spectrum::String, cs::Float64, seed::Int)
        Random.seed!(seed)

        G = 6.67430e-8
        ρ_cloud = M_cloud / ((4/3) * π * R_cloud^3)

        # --- Sample N particles in a uniform sphere ---
        positions = zeros(N, 3)
        for i in 1:N
            while true
                x = 2R_cloud * (rand(3) .- 0.5)
                if norm(x) ≤ R_cloud
                    positions[i, :] .= x
                    break
                end
            end
        end

        # --- Generate random velocity field on a 3D grid ---
        grid_size = 32
        box_size = 2R_cloud

        velx = zeros(grid_size, grid_size, grid_size)
        vely = similar(velx)
        velz = similar(velx)

        function k_index_shifted(k, N)
            return k <= N ÷ 2 ? k : k - N
        end

        for i in 1:grid_size, j in 1:grid_size, k in 1:grid_size
            kx = k_index_shifted(i, grid_size)
            ky = k_index_shifted(j, grid_size)
            kz = k_index_shifted(k, grid_size)
            kvec = [kx, ky, kz]
            k_mag = norm(kvec)

            if k_mag == 0
                continue
            end

            power = spectrum == "burgers" ? -2.0 : -11/3
            amp = randn() * k_mag^power
            φ = 2π * rand()
            direction = randn(3); direction /= norm(direction)
            vx, vy, vz = amp * cos(φ) .* direction

            velx[i, j, k] = vx
            vely[i, j, k] = vy
            velz[i, j, k] = vz
        end

        # --- Trilinear interpolation for particle velocities ---
        dx = box_size / grid_size
        velocities = zeros(N, 3)

        for idx in 1:N
            pos = positions[idx, :] .+ R_cloud  # shift to [0, box_size]
            fx, fy, fz = pos ./ dx
            i = clamp(Int(floor(fx)), 1, grid_size - 1)
            j = clamp(Int(floor(fy)), 1, grid_size - 1)
            k = clamp(Int(floor(fz)), 1, grid_size - 1)
            wx, wy, wz = fx - i, fy - j, fz - k

            function interp(cube)
                return (1-wx)*(1-wy)*(1-wz)*cube[i, j, k] +
                    wx*(1-wy)*(1-wz)*cube[i+1, j, k] +
                    (1-wx)*wy*(1-wz)*cube[i, j+1, k] +
                    (1-wx)*(1-wy)*wz*cube[i, j, k+1] +
                    wx*wy*(1-wz)*cube[i+1, j+1, k] +
                    wx*(1-wy)*wz*cube[i+1, j, k+1] +
                    (1-wx)*wy*wz*cube[i, j+1, k+1] +
                    wx*wy*wz*cube[i+1, j+1, k+1]
            end

            velocities[idx, 1] = interp(velx)
            velocities[idx, 2] = interp(vely)
            velocities[idx, 3] = interp(velz)
        end

        # Remove net momentum
        velocities .-= mean(velocities, dims=1)
        velocities .*= cs / std(vec(norm.(eachrow(velocities))))
        return positions, velocities, fill(ρ_cloud, N)
    end


    function rotating_cloud(N::Int;
                            Mtot::Float64 = 1.99e33,         # total mass [g] (default = 1 Msun)
                            Rcloud::Float64 = 3e17,           # cloud radius [cm]
                            rho_c::Float64 = 1e-18,           # central density [g/cm^3]
                            Ω_frac::Float64 = 0.5,            # fraction of centrifugal to gravity (0 = none, ~0.5 = moderate)
                            add_turbulence::Bool = false,
                            turb_frac::Float64 = 0.1)         # RMS turbulence as fraction of v_rot

        G = 6.67430e-8  # gravitational constant [cm^3 g^-1 s^-2]

        # Generate radius profile with Plummer-like density
        r0 = Rcloud / 3
        function sample_r()
            while true
                r = Rcloud * rand()^(1/3)  # uniform in volume
                ρ = rho_c / (1 + (r/r0)^2)^(2.5)
                if rand() < ρ / rho_c
                    return r
                end
            end
        end

        r_samples = [sample_r() for _ in 1:N]

        # Isotropic direction
        θ = acos.(2 .* rand(N) .- 1)
        ϕ = 2π .* rand(N)

        x = r_samples .* sin.(θ) .* cos.(ϕ)
        y = r_samples .* sin.(θ) .* sin.(ϕ)
        z = r_samples .* cos.(θ)
        positions = hcat(x, y, z)

        # Compute rotation velocities
        R = sqrt.(x.^2 + y.^2)
        v_circ = sqrt.(G * Mtot .* R ./ (Rcloud^3))   # simplified Keplerian scaling
        v_rot = Ω_frac .* v_circ

        vx = -v_rot .* y ./ R
        vy =  v_rot .* x ./ R
        vz = zeros(N)

        # Handle R = 0
        vx[isnan.(vx)] .= 0.0
        vy[isnan.(vy)] .= 0.0

        velocities = hcat(vx, vy, vz)

        # Add turbulence if desired
        if add_turbulence
            rms_turb = turb_frac * mean(norm.(eachrow(velocities)))
            velocities .+= rms_turb .* (randn(N, 3) ./ sqrt(3))
        end

        return positions, velocities
    end

    function polytropic_sphere(N::Int, n::Float64, K::Float64, ρ_c::Float64, ξ_max::Float64)
        # Physical constants
        G = 6.67430e-8  # cm^3 g^-1 s^-2

        # Lane-Emden Equation for polytropic sphere
        function lane_emden!(dθ, θ, p, ξ)
            if ξ == 0.0
                dθ[1] = 0.0
                dθ[2] = 0.0
            else
                dθ[1] = θ[2]
                dθ[2] = -2/ξ * θ[2] - θ[1]^n
            end
        end

        θ0 = [1.0, 0.0]  # θ(0) = 1, θ'(0) = 0
        prob = ODEProblem(lane_emden!, θ0, (1e-8, ξ_max))
        sol = solve(prob, reltol=1e-8, abstol=1e-10)

        ξs = sol.t
        θs = sol[1, :]
        interp_θ = LinearInterpolation(ξs, θs, extrapolation_bc=Throw())

        # Scaling factor: a = sqrt((n+1)K / (4πG) * ρ_c^(1/n - 1))
        a = sqrt((n + 1) * K / (4π * G) * ρ_c^(1/n - 1))

        # Define density profile ρ(ξ)
        ρ(ξ) = ρ_c * interp_θ(ξ)^n

        # Mass profile M(ξ) = 4π a^3 ρ_c ∫₀^ξ ξ'^2 θ(ξ')^n dξ'
        function M(ξ)
            integrand(x) = x^2 * interp_θ(x)^n
            integral, _ = quadgk(integrand, 0, ξ)
            return 4π * a^3 * ρ_c * integral
        end

        Mtot = M(ξ_max)

        # Sample ξ values according to mass profile
        function sample_radius(N)
            radii = zeros(N)
            for i in 1:N
                y = rand() * Mtot
                ξ_low, ξ_high = 0.0, ξ_max
                for _ in 1:30
                    ξ_mid = (ξ_low + ξ_high) / 2
                    if M(ξ_mid) < y
                        ξ_low = ξ_mid
                    else
                        ξ_high = ξ_mid
                    end
                end
                radii[i] = (ξ_low + ξ_high) / 2
            end
            return radii
        end

        ξ_samples = sample_radius(N)
        r_samples = a .* ξ_samples

        # Isotropic angular sampling
        θ = acos.(2 .* rand(N) .- 1)
        ϕ = 2π .* rand(N)

        x = r_samples .* sin.(θ) .* cos.(ϕ)
        y = r_samples .* sin.(θ) .* sin.(ϕ)
        z = r_samples .* cos.(θ)

        positions = hcat(x, y, z)

        velocities = zeros(N, 3)  # static equilibrium, no motion
        
        return positions, velocities, Mtot
    end


    function gaussian_sphere(N::Int, R::Float64; axis::Vector{Float64} = nothing, Ω_frac::Float64 = 0.0, rng = nothing)
        #=
        gaussian_sphere(N; axis = nothing, omega = 0.0, seed = nothing)

        Generates a Gaussian-distributed 3D particle sphere with optional solid-body rotation.

        Arguments:
        - `N`: Number of particles
        - `axis`: Unit vector `[x, y, z]` defining the rotation axis e.g [1 0 0] rotation around x axis (default: no rotation)
        - `Ω_frac`: Angular velocity in radians per unit time (default: 0.0)
        - `rng`: Optional RNG seed

        Returns:
        - `pos`: Nx3 matrix of particle positions
        - `vel`: Nx3 matrix of particle velocities
        =#
        if rng !== nothing
            Random.seed!(rng)
        end

        pos = randn(N, 3) * R
        r_com = mean(pos, dims=1)
        pos .-= r_com  # Center around origin

        vel = zeros(N, 3)

        if axis !== nothing && Ω_frac != 0.0
            axis = normalize(axis)
            for i in 1:N
                r = pos[i, :]
                v = Ω_frac * cross(axis, r)
                vel[i, :] = v
            end
        end

        return pos, vel
    end


    function iniconds_setup(EOS::String, ic_type::String)
        #=
            EOS: Can either be "isothermal" or "polytropic"
            ic_type: Can take any name of the above initial conditions functions
        =#

        # ====== Initial conditions Setup in CGS Units =========


        M0 = 1.9891e33 # [g] solar mass in grams
        pc = 5.38552341e16	   # [cm] parsec in cm

        # Common Simulation parameters
        N         = 10000      # Number of particles
        R         = 2.0 * pc
        Kh		  = 50         # Number of neighbors for hydro_end
        Kgr       = 20		   # Bucket size for gravity Octree
        t         = 0          # current time of the simulation
        tEnd      = 5e12       # time at which simulation ends
        alpha     = 1.0        # constant to handle bulk viscocity
        beta      = 2.0        # constant to handle particle interpenetration
        theta     = 0.576      # Opening angle for BH Tree
        M = 1 * M0             # in case Mtot is not provided from the above initial conditions' functions
        rho_c =  150.0         # Central Density [g/cm^3]
        ξ_max =  7.5           # Dimensionless truncation radius   
        Ω_frac = 0.5           # Angular Velocity

        # Physical constants (in CGS units) 
        G = 6.67430e-8        # Gravitational constant [cm^3 g^-1 s^-2]
        kB = 1.380649e-16     # Boltzmann constant [erg K^-1]
        mH = 1.6735575e-24    # Mass of hydrogen atom [g]


        # Gas properties 
        gamma = 5/3           # Ideal monoatomic gas
        mu = 0.61             # Mean molecular weight for molecular cloud (H2 + He)
        T = 15_000_000                # Temperature [K]
        cs = sqrt(kB * T / (mu * mH))  # Isothermal sound speed [cm/s]
        m = M / N

        # Other initial conditions needed for the functions

        # Plummer sphere 
        a = 0.01 # smoothing factor

        # Bonnor Ebert sphere 
        velocity_mode = :virial              # :virial or :mach or :none
        mach_number = 1.0                    
        alpha_vir = 1.0                       
        rng = MersenneTwister(42)


        # Turbulent Molecular Cloud
        spectrum = "burgers"                 # 'burgers' or anything else

        # Rotating Cloud
        add_turbulence = false
        turb_frac = 0.1

        # Polytropic Sphere
        n = 3.0 ; gamma = 1 + 1/n           
        if n==3.0
            β = 0.9995
            αλφα = 4 * (5.670e-5) / 29979245800
            K = (kB / (mH * mu))^(4/3) * (3*(1-β)/(αλφα*β^4))^(1/3) # from eddington model
        else
            K = kB * T / (mu * mH * rho_c^(1/n)) # For other polytropes
        end

        # Gaussian sphere
        axis = nothing
        
        # Get initial distribution
        if ic_type == "sample_isothermal_sphere"
            pos, vel = sample_isothermal_sphere(N, R, cs)
        elseif ic_type == "sample_plummer_sphere"
            pos, vel = sample_plummer_sphere(N, M, a)
        elseif ic_type == "bonnor_ebert_sphere"
            pos, vel = bonnor_ebert_sphere(N, cs, rho_c, ξ_max; velocity_mode, mach_number, alpha_vir, rng)
        elseif ic_type == "turbulent_molecular_cloud"
            pos, vel, rho_vec = turbulent_molecular_cloud(N, R, M, spectrum, cs, rng)
            # Find K for given cs and central density
            K = cs^2 / gamma * rho_vec.^(1-gamma)
        elseif ic_type == "rotating_cloud"
            pos, vel = rotating_cloud(N; Mtot=M, Rcloud=R, rho_c, Ω_frac, add_turbulence, turb_frac)
            # Find K for given cs and central density
            K = fill(kB * T / (mu * mH * rho_c^(gamma-1)), N)  # Polytropic constant vector [erg⋅cm⁶/g²]
        elseif ic_type == "polytropic_sphere"
            pos, vel, M = polytropic_sphere(N, n, K, rho_c, ξ_max)
            K = fill(K, N)
            m = M / N
        elseif ic_type == "gaussian_sphere"
            pos, vel = gaussian_sphere(N, R; axis, Ω_frac, rng)
            # Find K from central density in this distribution
            r_com = sum(pos, dims=1) / size(pos, 1)
            r_com_vec = vec(r_com)  # Convert 1×3 to plain 3-element vector
            rho0 = HJL.density_plot(m, r_com, pos, Kh); rho0 = rho0[1]
            K = fill(kB * T / (mu * mH * rho0^(gamma-1)), N)  # Polytropic constant vector [erg⋅cm⁶/g²]

        else
            println("The ic_type you provided is invalid. Please check all available options")
            exit()
        end

        # Compute radius of distribution
        r_com = sum(pos, dims=1) / size(pos, 1)
        r_com_vec = vec(r_com)  # Convert 1×3 to plain 3-element vector
        R = maximum([norm(pos[i, :] .- r_com_vec) for i in 1:N])

        if EOS == "isothermal"
            U = cs^2/(gamma-1) * M # Internal energy [erg] / constant in isothermal processes

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
                "cs" => cs,
                "U" => U
            )

            SnapshotRW.write_snapshot(string(1), ic_type, pos, vel; constants=constants)
            println("Initial conditions for an isothermal $(ic_type) of an ideal gas, have been produced.")

        elseif EOS == "polytropic"
            

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
                "gamma" => gamma
            )

            SnapshotRW.write_snapshot(string(1), ic_type, pos, vel; K, constants=constants)
            println("Initial conditions for a polytropic $(ic_type) of an ideal gas, have been produced.")

        else 
            println("No EOS of type $(EOS) exists. Available options are either: 'isothermal' or 'polytropic'")
        end
    end

end