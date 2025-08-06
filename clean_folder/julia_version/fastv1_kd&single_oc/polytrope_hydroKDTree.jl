module HJL

    using NearestNeighbors, Logging

    function W(h::Vector{Float64}, q::Matrix{Float64}, masks::BitArray{3})
        #= 
        Cubic Spline Kernel (3D)
        h: smoothing length
        q: point-distance to h, ratio
        masks: boolean masks for q < 1 and 1 ≤ q < 2, shape (..., 2)
        =#

        mask1 = masks[:, :, 1]
        mask2 = masks[:, :, 2]

        N, Kh = size(q)
        w = zeros(N, Kh)

        for i in 1:N
            mask1_i = mask1[i, :]
            mask2_i = mask2[i, :]
            ct = 1 / (pi * h[i]^3)

            if any(mask1_i)
                q1 = q[i, mask1_i]
                w[i, mask1_i] = ct * (1 .- 3/2 * q1.^2 + 3/4 * q1.^3)
            end

            if any(mask2_i)
                w[i, mask2_i] = (ct * 1/4 .* (2 .- q[i, mask2_i]).^3)
            end
        end

        return w
    end


    function gradW(x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64}, r::Matrix{Float64}, h::Vector{Float64}, q::Matrix{Float64}, masks::BitArray{3})
        #= 
        Cubic Spline Kernel Gradient (3D)
        x, y, z, r: point distance in x,y,z and r coords
        h: smoothing length
        q: point-distance to h, ratio
        masks: hold the bool values for the respective q ratios 
        =#

        mask1 = masks[:, :, 1]
        mask2 = masks[:, :, 2]

        N, Kh = size(r)
        dWdr = zeros(N,Kh)

        # With increased N (number of particles) masking like below makes it faster 
        for i in 1:N
            mask1_i = mask1[i, :]
            mask2_i = mask2[i, :]
            ct = 1 / (pi * h[i]^4)
            if any(mask1_i) != 0
                dWdr[i, mask1_i] = ct * (9/4 * r[i, mask1_i] ./ h[i]^2 .- 3 / h[i]) # already divided with r for less operations and to avoid zero division
            end

            if any(mask2_i) != 0
                q2 = q[i, mask2_i]
                dWdr[i, mask2_i] = ct * (-3/4 * (2 .- q2).^2) ./ r[i, mask2_i] 
            end
        end

        gradWx = dWdr .* x 
        gradWy = dWdr .* y 
        gradWz = dWdr .* z 
        
        return gradWx, gradWy, gradWz
    end


    function getTreeDiffs(fi::Matrix{Float64}, fj::Matrix{Float64}, indices::Matrix{Int32})
        #= 
        Calculate the NxK Matrix of some quantity, based on each particle's K Neighbors 
        =#

        N, Kh = size(indices)
        D = size(fi, 2)  # D = 3

        # Preallocate
        fj_neighbors = Array{Float64}(undef, N, Kh, D)

        # Fill fj_neighbors[i, j, :] = fj[indices[i, j], :]
        for i in 1:N, j in 1:Kh
            fj_neighbors[i, j, :] = fj[indices[i, j], :]
        end

        # Reshape fi to broadcast
        fi_reshaped = reshape(fi, N, 1, D)
        diff = fi_reshaped .- fj_neighbors

        return diff[:, :, 1], diff[:, :, 2], diff[:, :, 3]
    end


    function getVectorTreeAvgs(f::Vector{Float64}, indices::Matrix{Int32})
        #= 
        Get the NxK that holds averages of some quantity
        =#

        N, Kh = size(indices)
        f_avg = zeros(N, Kh)

        for j in 1:Kh
            for i in 1:N
                f_avg[i, j] = (f[i] + f[indices[i, j]]) / 2
            end
        end

        return f_avg
    end


    function getNeighbors(ri::Matrix{Float64}, rj::Matrix{Float64}, Kh::Int64)
        
        #= 
        Get separations between 2 sets of coordinates
        ri: M x 3 matrix of positions
        rj: N x 3 matrix of positions
        K: number of neighbors
        =#

        # Initialise tree
        tree = KDTree(rj') # it expects the data in columns

        # Builds the KD-Tree and returns each point's neighbors in index order
        indices_vv, r_vv = NearestNeighbors.knn(tree, ri', Kh, true) 
        
        # Convert vector of vectors to matrix
        conv_start=time()

        N = size(indices_vv, 1)
        r = Matrix{Float64}(undef, N, Kh)
        indices = Matrix{Int32}(undef, N, Kh)
        for i in 1:N
            r[i, :] = r_vv[i]
            indices[i, :] = indices_vv[i]
        end

        # Calculate differences
        dx, dy, dz = getTreeDiffs(ri, rj, indices)

        conv_end=time()
        @debug "Conversion Processes: $(conv_end-conv_start)"
        
        # Smoothing length of each particle based on farthest neighbor
        h = r[:, end] ./ 2    # shape (N,)

        # Compute q = r / h
        q = r ./ h

        # Generate masks
        mask1 = (q .<= 1.0)
        mask2 = .!mask1 

        masks = cat(mask1, mask2; dims=3)  # shape (N, K, 2)
        
        return dx, dy, dz, r, h, q, masks, indices
    end


    function getDensity(m::Float64, w::Matrix{Float64})
        #=
        Get Density at sampling locations from SPH particle distribution

        m:    N-element vector of particle masses
        w:    M × N matrix of kernel weights (M sample points, N particles)
        rho:  M-element vector of densities
        =#

        rho = m * sum(w, dims=2)   
        
        return vec(rho)              
    end


    function getAV(dx::Matrix{Float64}, dy::Matrix{Float64}, dz::Matrix{Float64},
        r::Matrix{Float64}, vel::Matrix{Float64}, gamma::Float64, rho::Vector{Float64}, K::Vector{Float64}, 
        alpha::Float64, beta::Float64, h::Vector{Float64}, indices::Matrix{Int32})
        
        # Compute sound speed at each position
        cs = (gamma * K .* rho.^(gamma-1)).^(1/2)

        # Get h_avg for AV consistency
        h_avg = getVectorTreeAvgs(h, indices)

        # Compute velocity differences between i-th particle and its neighbors
        vij_x, vij_y, vij_z = getTreeDiffs(vel, vel, indices)

        # Compute averages
        rho_avg = getVectorTreeAvgs(rho, indices)
        
        # Compute artificial viscosity term
        v_dot_r = vij_x .* dx + vij_y .* dy + vij_z .* dz
        mu = min.(h_avg .* v_dot_r ./ (r.^2 .+ 0.01 .* h_avg.^2), 0)
        
        Pi_ij = (-alpha * cs .* mu + beta * mu.^2) ./ rho_avg

        return Pi_ij, cs, mu, vij_x, vij_y, vij_z
    end


    function getPressure(rho::Vector{Float64}, K::Vector{Float64}, gamma::Float64)
        #=
        Equation of State
        rho   vector of densities
        k     equation of state constant
        n     polytropic index
        P     pressure
        =#
        
        P = K .* rho.^gamma 
        
        return P
    end


    function hydroCalculation(m, dWx, dWy, dWz, rho, P, Pi_ij, indices)
        N, Kh = size(indices)
        ax = zeros(size(rho))
        ay = zeros(size(rho))
        az = zeros(size(rho))
        
        # j starts from 2, because there's no self effect in hydrodynamic forces (pressure, AV)
        for j in 2:Kh
            neighbors = indices[:, j]
            
            for i in 1:N
                nj = neighbors[i]
                
                ct = m * (( (P[i] / rho[i]^2) + (P[nj] / rho[nj]^2) ) + Pi_ij[i, j]) / 2
                
                ax[i] -= ct * dWx[i, j]
                ay[i] -= ct * dWy[i, j]
                az[i] -= ct * dWz[i, j]

                ax[nj] += ct * dWx[i, j]
                ay[nj] += ct * dWy[i, j]
                az[nj] += ct * dWz[i, j]
            end
        end

        return ax, ay, az
    end


    function hydrodynamics(pos::Matrix{Float64}, vel::Matrix{Float64}, m::Float64, K::Vector{Float64}, gamma::Float64, alpha::Float64, beta::Float64, Kh::Int64)

        tree_start=time()

        # Neighbors of each particle
        dx, dy, dz, r, h, q, masks, indices = getNeighbors(pos, pos, Kh)
        tree_end=time()
        @debug "KD-Tree Processes: $(tree_end-tree_start)"

        # Compute density at particle positions
        w = W(h, q, masks)
        w_end = time()
        @debug "W Calc Processes: $(w_end-tree_end)"

        rho = getDensity(m, w)
        rho_end = time()
        @debug "Rho Processes: $(rho_end-w_end)"

        # Artificial Viscosity term Π 
        Pi_ij, cs, mu, vij_x, vij_y, vij_z = getAV(dx, dy, dz, r, vel, gamma, rho, K, alpha, beta, h, indices)
        AV_end = time()
        @debug "AV Processes: $(AV_end-rho_end)"

        # Compute pressure
        P = getPressure(rho, K, gamma)
        P_end = time()
        @debug "P Processes: $(P_end-AV_end)"

        # Kernel gradients
        dWx, dWy, dWz = gradW(dx, dy, dz, r, h, q, masks) 
        gradW_end = time()
        @debug "gradW Processes:  $(gradW_end-P_end)"


        # Symmetric Pressure + AV calculation
        ax, ay, az = hydroCalculation(m, dWx, dWy, dWz, rho, P, Pi_ij, indices)
        
        hydro_end = time()
        @debug "Hydro Calc Processes: $(hydro_end-gradW_end)"

        return ax, ay, az, rho, dWx, dWy, dWz, h, cs, mu, indices, vij_x, vij_y, vij_z, Pi_ij
    end

    
    # This version of the function is used when fnishing a whole step
    function evolve_K!(K::Vector{Float64}, m::Float64, indices::Matrix{Int32}, Pi_ij::Matrix{Float64}, vij_x::Matrix{Float64}, vij_y::Matrix{Float64}, vij_z::Matrix{Float64}, dWx::Matrix{Float64}, dWy::Matrix{Float64}, dWz::Matrix{Float64}, gamma::Float64, rho::Vector{Float64}, dt::Float64)
        N, Kh = size(Pi_ij)
        dk_dt = zeros(N)

        # j starts from 2, because there's no self effect in hydrodynamic forces (pressure, AV)
        for j in 2:Kh
            neighbors = indices[:, j]
            
            for i in 1:N
                nj = neighbors[i]
                
                ct = m * Pi_ij[i, j] * (vij_x[i, j] * dWx[i, j] + vij_y[i, j] * dWy[i, j] + vij_z[i, j] * dWz[i, j]) / 2
                
                dk_dt[i] += ct 
                dk_dt[nj] += ct

            end
        end 

        K .= K + (1/2 * (gamma-1) ./ rho.^(gamma-1) .* dk_dt) * (dt/2)  
    end


    # Overload function, to take pre-calculated v dot dW matrix, from the adaptive timestep scheme
    function evolve_K!(K::Vector{Float64}, m::Float64, indices::Matrix{Int32}, Pi_ij::Matrix{Float64}, v_dot_dW::Matrix{Float64}, gamma::Float64, rho::Vector{Float64}, dt::Float64)
        N, Kh = size(Pi_ij)
        dk_dt = zeros(N)

        # j starts from 2, because there's no self effect in hydrodynamic forces (pressure, AV)
        for j in 2:Kh
            neighbors = indices[:, j]
            
            for i in 1:N
                nj = neighbors[i]
                
                ct = m * Pi_ij[i, j] * v_dot_dW[i, j] / 2
                
                dk_dt[i] += ct 
                dk_dt[nj] += ct

            end
        end 

        K .= K + (1/2 * (gamma-1) ./ rho.^(gamma-1) .* dk_dt) * (dt/2)   
   
    end


    function density_plot(m::Float64, rr::Matrix{Float64}, pos::Matrix{Float64}, Kh::Int64)
        h_plot, q_plot, masks_plot = getNeighbors(rr, pos, Kh)[5:7]
        w = W(h_plot, q_plot, masks_plot)
        rho_radial = getDensity(m, w)

        return rho_radial
    end

end