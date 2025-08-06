module FJL

    using NearestNeighbors, Logging

    function W(h::Vector{Float64}, masks)
        #= 
        Cubic Spline Kernel (3D)
        h: smoothing length
        q: point-distance to h, ratio
        masks: boolean masks for q < 1 and 1 ≤ q < 2, shape (..., 2)
        =#

        q = masks[1]
        indices_mask1 = masks[2]
        indices_mask2 = masks[3]

        N = size(q,1)
        w = zeros(N,N)

        for i in 1:N
            mask1_i = indices_mask1[i, :]
            mask2_i = indices_mask2[i, :]
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


    function gradW(x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64}, r::Matrix{Float64}, h::Vector{Float64}, masks)
        #= 
        Cubic Spline Kernel Gradient (3D)
        x, y, z, r: point distance in x,y,z and r coords
        h: smoothing length
        q: point-distance to h, ratio
        masks: hold the bool values for the respective q ratios 
        =#

        q = masks[1]
        indices_mask1 = masks[2]
        indices_mask2 = masks[3]

        N = size(r,1)
        dWdr = zeros(N,N)

        # With increased N (number of particles) masking like below makes it faster 
        for i in 1:N
            mask1_i = indices_mask1[i, :]
            mask2_i = indices_mask2[i, :]
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


    function PhiKernel(r, h, masks)
        PHI = zeros(size(r))

        q = masks[1]
        indices_mask1 = masks[2]
        indices_mask2 = masks[3]
        indices_mask3 = masks[4]
            
        if any(indices_mask1) != 0
            q1 = q[indices_mask1]
            PHI[indices_mask1] = (1 ./ h[indices_mask1]) .* (2/3 * q1.^2 - 3/10 * q1.^4 + 1/10 * q1.^5 .- 7/5)
        end

        if any(indices_mask2) != 0
            q2 = q[indices_mask2]
            PHI[indices_mask2] = ((1 ./ h[indices_mask2]) .* (4/3 * q2.^2 - q2.^3 + 
                                3/10 * q2.^4 - 1/30 * q2.^5 .- 8/5 .+ 1/15 ./ q2))
        end

        if any(indices_mask3) != 0
            PHI[indices_mask3] = - 1 ./ r[indices_mask3]
        end

        return sum(PHI, dims=2)
    end


    function gradPhiKernel(x, y, z, r, h, masks)

        q = masks[1]
        indices_mask1 = masks[2]
        indices_mask2 = masks[3]
        indices_mask3 = masks[4]

        gPHI = zeros(size(r))

        if any(indices_mask1)
            r1, h1 = r[indices_mask1], h[indices_mask1]
            gPHI[indices_mask1] = ((1 ./ h1.^2) .* (4/3 ./ h1 - 6/5 * (r1.^2 ./ h1.^3) + 
                                            1/2 * (r1.^3 ./ h1.^4)))
        end

        if any(indices_mask2)
            q2 = q[indices_mask2]
            gPHI[indices_mask2] = ((1 ./ h[indices_mask2].^2) .* (8/3 * q2 - 3 * q2.^2 + 
                                            6/5 * q2.^3 - 1/6 * q2.^4 - 
                                            1/15 * (1 ./ q2.^2))) ./ r[indices_mask2]
        end

        if any(indices_mask3)
            gPHI[indices_mask3] = (1 ./ r[indices_mask3].^3)
        end 

        gPHIx = gPHI .* x
        gPHIy = gPHI .* y
        gPHIz = gPHI .* z

        return gPHIx, gPHIy, gPHIz
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


    function getPairwiseSeparations(ri::Matrix{Float64}, rj::Matrix{Float64}, h)
        #=
        Get pairwise desprations between 2 sets of coordinates
        ri    is an M x 3 matrix of positions
        rj    is an N x 3 matrix of positions
        dx, dy, dz   are M x N matrices of separations
        =#
        
        # positions ri = (x,y,z)
        rix = ri[:, 1]
        riy = ri[:, 2]
        riz = ri[:, 3]

        # other set of points positions rj = (x,y,z)
        rjx = rj[:, 1]
        rjy = rj[:, 2]
        rjz = rj[:, 3]

        # matrices that store all pairwise particle separations: r_i - r_j
        dx = rix .- rjx'
        dy = riy .- rjy'
        dz = riz .- rjz'

        # compute distances
        r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        q = r ./ h

        # masks
        indices_mask1 = q .<= 1
        indices_mask2 = (q .> 1) .& (q .<= 2)
        indices_mask3 = q .> 2

        # pack into a list (tuple or array depending on use case)
        masks = (q, indices_mask1, indices_mask2, indices_mask3)
        
        return dx, dy, dz, r, masks
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
        alpha::Float64, beta::Float64, h_avg::Matrix{Float64})
        
        # Compute sound speed at each position
        cs = (gamma * K .* rho.^(gamma-1)).^(1/2)

        # Compute velocity differences between i-th particle and its neighbors
        vij_x, vij_y, vij_z = getPairwiseSeparations(vel, vel, h_avg)[1:3]

        # Compute averages
        rho_avg = (rho .+ rho') ./ 2
        
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


    function Acc(m, G, dWx, dWy, dWz, gPHIx, gPHIy, gPHIz, rho, P, Pi_ij)
        N = size(rho)
        ax = zeros(N)
        ay = zeros(N)
        az = zeros(N)
        
        # Pairwise calculations
        ct_g = G * m
        ct_h = (P./rho.^2 .+ P'./rho'.^2 + Pi_ij)
        ax -= ct_g * sum(gPHIx, dims=2) + m * sum(ct_h .* (dWx .- dWx') ./ 2, dims=2)
        ay -= ct_g * sum(gPHIy, dims=2) + m * sum(ct_h .* (dWy .- dWy') ./ 2, dims=2)
        az -= ct_g * sum(gPHIz, dims=2) + m * sum(ct_h .* (dWz .- dWz') ./ 2, dims=2)

        return ax, ay, az
    end


    function Accs(pos::Matrix{Float64}, vel::Matrix{Float64}, m::Float64, K::Vector{Float64}, gamma::Float64, G::Float64, alpha::Float64, beta::Float64, Kh::Int64)

        start = time()

        # Neighbors of each particle - just to determine h of each particle to do even comparisons
        dx, dy, dz, r, h, q, masks, indices = getNeighbors(pos, pos, Kh)

        # Calculate pairwise separations
        dx, dy, dz, r, masks = getPairwiseSeparations( pos, pos, h )
        
        # Compute density at particle positions
        w = W(h, masks)

        rho = getDensity(m, w)

        # Artificial Viscosity term Π 
        h_avg = (h .+ h') / 2
        Pi_ij, cs, mu, vij_x, vij_y, vij_z = getAV(dx, dy, dz, r, vel, gamma, rho, K, alpha, beta, h_avg)

        # Compute pressure
        P = getPressure(rho, K, gamma)

        # Kernel gradients
        dWx, dWy, dWz = gradW(dx, dy, dz, r, h, masks) 

        # Gravity kernels    
        gPHIx, gPHIy, gPHIz = gradPhiKernel(dx, dy, dz, r, h_avg, masks) 
        PHI = PhiKernel(r, h_avg, masks)

        # Symmetric Pressure + AV calculation
        ax, ay, az = Acc(m, G, dWx, dWy, dWz, gPHIx, gPHIy, gPHIz, rho, P, Pi_ij)

        # Pack acceleration components
        acc = hcat(ax, ay, az)

        end_ = time()
        @debug "All basic calcs:  $(end_-start)"
        
        return acc, rho, dWx, dWy, dWz, h, cs, mu, vij_x, vij_y, vij_z, Pi_ij, PHI
    end

    
    # This version of the function is used when fnishing a whole step
    function evolve_K!(K::Vector{Float64}, m::Float64, Pi_ij::Matrix{Float64}, vij_x::Matrix{Float64}, vij_y::Matrix{Float64}, vij_z::Matrix{Float64}, dWx::Matrix{Float64}, dWy::Matrix{Float64}, dWz::Matrix{Float64}, gamma::Float64, rho::Vector{Float64}, dt::Float64)
        N = size(Pi_ij, 1)
        dk_dt = zeros(N)

        for j in 1:N
            for i in 1:N                
                ct = m * Pi_ij[i, j] * (vij_x[i, j] * dWx[i, j] + vij_y[i, j] * dWy[i, j] + vij_z[i, j] * dWz[i, j]) / 2
                
                dk_dt[i] += ct 
                dk_dt[j] += ct

            end
        end 

        K .= K + (1/2 * (gamma-1) ./ rho.^(gamma-1) .* dk_dt) * (dt/2)  
    end


    # Overload function, to take pre-calculated v dot dW matrix, from the adaptive timestep scheme
    function evolve_K!(K::Vector{Float64}, m::Float64, Pi_ij::Matrix{Float64}, v_dot_dW::Matrix{Float64}, gamma::Float64, rho::Vector{Float64}, dt::Float64)
        N = size(Pi_ij, 1)
        dk_dt = zeros(N)

        for j in 1:N
            
            for i in 1:N
                
                ct = m * Pi_ij[i, j] * v_dot_dW[i, j] / 2
                
                dk_dt[i] += ct 
                dk_dt[j] += ct

            end
        end 

        K .= K + (1/2 * (gamma-1) ./ rho.^(gamma-1) .* dk_dt) * (dt/2)   
   
    end


    function density_plot(m::Float64, rr::Matrix{Float64}, pos::Matrix{Float64}, Kh::Int64)
        h_plot = getNeighbors(rr, pos, Kh)[5]
        masks_plot = getPairwiseSeparations(rr, pos, h_plot)[5]
        w = W(h_plot, masks_plot)
        rho_radial = getDensity(m, w)

        return rho_radial
    end

end