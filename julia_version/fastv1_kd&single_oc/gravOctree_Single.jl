module GJL

    using DataStructures, Logging

    function Kernels(x::Float64, y::Float64, z::Float64, r::Float64, h::Float64)
        q = r/h

        if q <= 1
            gPHI = ((1 / h^2) * (4/3 / h - 6/5 * (r^2 / h^3) + 
                                        1/2 * (r^3 / h^4)))
            PHI = (1 / h) * (2/3 * q^2 - 3/10 * q^4 + 1/10 * q^5 - 7/5)
        elseif (q > 1) & (q <= 2)
            gPHI = ((1 / h^2) * (8/3 * q - 3 * q^2 + 
                                        6/5 * q^3 - 1/6 * q^4 - 
                                        1/15 * (1 / q^2))) / r
            PHI = ((1 / h) * (4/3 * q^2 - q^3 + 
                                3/10 * q^4 - 1/30 * q^5 - 8/5 + 1/15 / q))
        else
            gPHI = 1 / r^3
            PHI = - 1 / r
        end

        gPHIx = gPHI * x
        gPHIy = gPHI * y
        gPHIz = gPHI * z

        return [gPHIx, gPHIy, gPHIz], PHI

    end


    const AxisBounds = NamedTuple{(:x, :y, :z), Tuple{NTuple{2, Float64}, NTuple{2, Float64}, NTuple{2, Float64}}}
    mutable struct CellNode
        # Properties set during construction from Octree
        ID::Int32
        Length::Float64
        Center::Vector{Float64}
        axis_bounds::AxisBounds

        # Randomly initialised - TBC later
        Mass::Float64
        rCOM::Vector{Float64}
        parentID::Int32
        particle_count::Int32
        particle_list::Vector{Int32}  # Controlled by the wise octree - stores particle index
        child_nodes::Vector{Int32} # Controlled by the wise octree - stores children nodes' index
        is_leaf::Bool

        function CellNode(CellID::Int32, CellLength::Float64, CellCenter::Vector{Float64}, axis_bounds::AxisBounds)
            return new(
                CellID,
                CellLength,
                CellCenter,
                axis_bounds,
                0.0, # Mass
                [0.0, 0.0, 0.0], # rCOM
                -1, # parentID
                0, # particle_count
                Int32[], # particle_list
                Int32[], # child_nodes index
                false # is_leaf
            )
        end
    end

    function addParticles!(self::CellNode, m::Float64, p_index::Vector{Int32})
        # Add mass to this cell and increment count of particles
        nop = length(p_index)
        self.Mass += m * nop
        self.particle_count += nop

        # Set particle list for this cell with the new particles' index
        self.particle_list = p_index
    end


    # BFS Octree
    mutable struct Octree
        #=
        l     is the length of the computational space in a single dimension    
        pos   is an N x 3 matrix of SPH particle positions
        theta is the maximum opening angle to determine going further in the tree
        =#
        l::Float64 # Domain length (div by 2 to avoid constant divisions later)
        m::Float64 # Single particle mass
        pos::Matrix{Float64} # Matrix of particle positions
        theta_sq::Float64 # Theta squared to avoid repetetive squaring operations
        h::Vector{Float64} # Particles' smoothing lengths
        nodes::Vector{CellNode} # Node objects in a list
        node_count::Int32 # Number of nodes - used for Cell ID as well
        leaf_list::Vector{Int32} # Leaf-node indices in a list
        leaf_counter::Int32

        function Octree(l::Float64, m::Float64, pos::Matrix{Float64}, theta::Float64, h::Vector{Float64})
            N = size(pos, 1)
            axis_bounds_root = (x = (-l,l), y = (-l,l), z = (-l,l))

            # "this" assignment works like the dotted syntax in python OOP (self.)
            this = new(l, m, pos, theta^2, h, CellNode[], 1, zeros(Int32, N), 0)
            push!(this.nodes, CellNode(Int32(1), l, [0.0, 0.0, 0.0], axis_bounds_root))
            this.nodes[1].particle_list = collect(Int32, 1:N)

            return this
        end
    end

    function addNodes!(self::Octree, parentNode::CellNode)

        # Determine child cell centers based on parent
        parent_l = parentNode.Length
        parent_x, parent_y, parent_z = parentNode.Center

        child_l = parent_l / 2

        # From SW outermost up to NE innermost (0,1,...,7)
        left_x = parent_x - child_l; down_y = parent_y - child_l; outw_z = parent_z - child_l
        right_x = parent_x + child_l; up_y = parent_y + child_l; inw_z = parent_z + child_l
        child_center = [left_x down_y outw_z;
                        right_x down_y outw_z;
                        left_x up_y outw_z;
                        right_x up_y outw_z;
                        left_x down_y inw_z;
                        right_x down_y inw_z;
                        left_x up_y inw_z;
                        right_x up_y inw_z
        ]

        left_minx = left_x - child_l; centerx = left_x + child_l; right_maxx = right_x + child_l 
        down_miny = down_y - child_l; centery = down_y + child_l; up_maxy = up_y + child_l  
        outw_minz = outw_z - child_l; centerz = outw_z + child_l; inw_maxz = inw_z + child_l  

        child_ax_bounds = [(x = (left_minx,centerx), y = (down_miny,centery), z = (outw_minz,centerz)),
            (x = (centerx,right_maxx), y = (down_miny,centery), z = (outw_minz,centerz)),
            (x = (left_minx,centerx),  y = (centery,up_maxy),   z = (outw_minz,centerz)),
            (x = (centerx,right_maxx), y = (centery,up_maxy),   z = (outw_minz,centerz)),
            (x = (left_minx,centerx),  y = (down_miny,centery), z = (centerz,inw_maxz)),
            (x = (centerx,right_maxx), y = (down_miny,centery), z = (centerz,inw_maxz)),
            (x = (left_minx,centerx),  y = (centery,up_maxy),   z = (centerz,inw_maxz)),
            (x = (centerx,right_maxx), y = (centery,up_maxy),   z = (centerz,inw_maxz))
        ]

        # Which parent's particles belong to each child (vectorised)
        relative = (self.pos[parentNode.particle_list, :] .- parentNode.Center')
        oct_z = (relative[:,3] .> 0) # first four or latter four children (based on z-placement)
        oct_y = (relative[:,2] .> 0) # of those four chosen, choose the first two or latter two (based on y-placement) 
        oct_x = (relative[:,1] .> 0) # of those two chosen, choose the first or second (based on x-placement)
        
        ci = 4 * oct_z + 2 * oct_y + oct_x .+ 1 # child index 1–8 

        # Create a list to store particles for each child (8 children in total)
        particles_per_child = [Int32[] for _ in 1:8]

        # Group particles by their child index (no need for ci == i check)
        for (particle_idx, child_idx) in zip(parentNode.particle_list, ci)
            push!(particles_per_child[child_idx], particle_idx)
        end

        # Create child nodes and assign particles to each child
        for i in 1:8
            child_particles_idxs = particles_per_child[i]
            if length(child_particles_idxs) != 0  # Skip dead nodes to save memory and avoid overhead / errors
                # Increase node counter to be correctly used for ID of new_node
                self.node_count += 1

                # Append node list for tree and child_node list for parent
                new_node = CellNode(self.node_count, child_l, child_center[i,:], child_ax_bounds[i])
                push!(self.nodes, new_node)
                push!(parentNode.child_nodes, self.node_count)
                
                # Add particles / check for leaf node - FIX THAT
                addParticles!(new_node, self.m, child_particles_idxs)

                # Add to child the parent's ID
                new_node.parentID = parentNode.ID
            end
        end
        
        # Free up space by deleting the parent's particle list
        parentNode.particle_list = Int32[]

    end

    function setCOMs!(self::Octree) 
        for i in reverse(1:length(self.nodes))
            node = self.nodes[i]
            if node.particle_count==1
                # Set leaf properties for further processes 
                node.is_leaf = true
                self.leaf_counter += 1
                self.leaf_list[self.leaf_counter] = node.ID

                # Calculate leaf node's COM using its particle (list) that is kept
                j = node.particle_list[1]
                node.rCOM = self.pos[j, :]
            else
                # Calculate node's COM and h from its precalculated children
                total_mass = 0
                weighted_pos_sum = zeros(3)
                for child_id in node.child_nodes
                    child = self.nodes[child_id]

                    # Accumulate mass
                    total_mass += child.Mass
                    weighted_pos_sum += child.Mass * child.rCOM

                end           
                node.Mass = total_mass
                node.rCOM = weighted_pos_sum / total_mass
            end
        end
    end

    function build_octree!(self::Octree)
        i = 1
        
        # Breadth First Order
        while i <= length(self.nodes)
            parentNode = self.nodes[i]
            if parentNode.particle_count != 1
                addNodes!(self, parentNode)
            end
            i += 1
        end
        
        # Set COMs
        setCOMs!(self)
    end


    # Helper function to compute minimum squared distance between cell point and cell bounds 
    function min_distance2_point_to_cell(p::Vector{Float64}, bounds::AxisBounds)
        dx = max(bounds.x[1] - p[1], 0, p[1] - bounds.x[2])
        dy = max(bounds.y[1] - p[2], 0, p[2] - bounds.y[2])
        dz = max(bounds.z[1] - p[3], 0, p[3] - bounds.z[2])
        return dx^2 + dy^2 + dz^2
    end


    function compute_g(self::Octree, i::Int32)
        
        g = zeros(3)
        PHI = 0
        pos_i = self.pos[i, :]
        h_i = self.h[i]
        
        nodes_to_visit = Deque{Int32}()
        for child in self.nodes[1].child_nodes
            push!(nodes_to_visit, child)
        end

        while length(nodes_to_visit) > 0
            node_id = popfirst!(nodes_to_visit)
            node = self.nodes[node_id]

            dx, dy, dz = pos_i - node.rCOM
            d_sq = dx^2 + dy^2 + dz^2
            s = node.Length * 2
            
            if node.is_leaf 
                j = node.particle_list[1]
                h_ij = (h_i + self.h[j]) / 2
                gPHI, pot = Kernels(dx, dy, dz, sqrt(d_sq), h_ij)
                g += node.Mass * gPHI
                PHI += node.Mass * pot
            elseif (s^2 / d_sq < self.theta_sq) && (h_i^2 / min_distance2_point_to_cell(pos_i, node.axis_bounds) < 0.25)
                d = sqrt(d_sq)
                factor = node.Mass / (d^3)
                g += [factor * dx, factor * dy, factor * dz]
                PHI += - node.Mass / d
            else
                for child in node.child_nodes
                    push!(nodes_to_visit, child)
                end
            end
        end
        
        return g, PHI
    end

    function gravity_acc(self::Octree)
        N = size(self.pos, 1)
        g = zeros((N, 3))
        PHI = zeros(N) 
        
        # For each particle calculate gravitational acc
        for i in 1:self.leaf_counter
            leafID = self.leaf_list[i]
            leaf = self.nodes[leafID]

            p_i = leaf.particle_list[1]
            
            # Temporarily remove leaf from node list to avoid redundant checks   
            parent = self.nodes[leaf.parentID] 
            deleteat!(parent.child_nodes, findfirst(==(leaf.ID), parent.child_nodes))                
                               
            # Calculate leaf node's / particle's gravitational acc
            g[p_i, :], PHI[p_i] = compute_g(self, p_i)

            # Place the node back
            push!(parent.child_nodes, leafID)
        end

        return g, PHI - (self.m * (7/5) ./ self.h) # self effect on potential due to smoothing
    end


    function gravity(l_domain::Float64, m::Float64, pos::Matrix{Float64}, theta::Float64, h::Vector{Float64})
        hydro_end = time()
        g_tree = Octree(l_domain, m, pos, theta, h)
        build_octree!(g_tree)
        build_end = time()
        @debug "Octree build time: $(build_end - hydro_end)"

        g, PHI = gravity_acc(g_tree)
        gcalc_end = time()
        @debug "Grav Calc: $(gcalc_end-build_end)"

        return g, PHI
    end

end