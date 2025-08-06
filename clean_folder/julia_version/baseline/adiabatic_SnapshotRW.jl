module SnapshotRW

using CSV, DataFrames, GLMakie, Mmap, DelimitedFiles

"""
    write_snapshot(filename, pos, vel, rlin, rho_radial; constants=Dict(), type="particle")

Save a simulation snapshot to CSV with:
- Per-particle data (position, velocity)
- `rlin` and `rho_radial` saved as serialized vectors in separate rows
- Constants added as a final row

Arguments:
- `filename::String` — path to save the CSV
- `pos::Matrix{Float64}` — Nx3 matrix of positions
- `vel::Matrix{Float64}` — Nx3 matrix of velocities
- `rlin::Vector{Float64}` — radial coordinate vector
- `rho_radial::Vector{Float64}` — radial density vector
- `constants::Dict` — (optional) dictionary of scalar constants
- `type::String` — (optional) label for the particle rows, default `"particle"`
"""
function write_snapshot(snapID::String,
                        pos::Matrix{Float64},        # For warm starts
                        vel::Matrix{Float64},        # For warm starts
                        K::Vector{Float64};          # For warm starts
                        constants=Dict(),            # For warm starts
                        rlin=nothing,                # For plotting
                        rho_radial=nothing,          # For plotting
                        fig1=nothing,
                        fig2=nothing,
                        type::String="particle")

    N = size(pos, 1)

    # Create DataFrame with all necessary columns (no rho)
    df = DataFrame(
        type = fill(type, N),
        x = Vector{Union{Missing, Float64}}(pos[:, 1]),
        y = Vector{Union{Missing, Float64}}(pos[:, 2]),
        z = Vector{Union{Missing, Float64}}(pos[:, 3]),
        vx = Vector{Union{Missing, Float64}}(vel[:, 1]),
        vy = Vector{Union{Missing, Float64}}(vel[:, 2]),
        vz = Vector{Union{Missing, Float64}}(vel[:, 3]),
        K  = Vector{Union{Missing, Float64}}(K),
        rlin = Vector{Union{Missing, String}}(missing, N),
        rho_radial = Vector{Union{Missing, String}}(missing, N),
        constants = Vector{Union{Missing, String}}(missing, N)
    )


    if rlin !== nothing && !isempty(rlin)
        # Serialize rlin and add a separate row
        rlin_str = join(string.(rlin), ";")
        push!(df, (
            type = "rlin",
            x = missing, y = missing, z = missing,
            vx = missing, vy = missing, vz = missing,
            K = missing,
            rlin = rlin_str,
            rho_radial = missing,
            constants = missing
        ))
    end

    if rho_radial !== nothing && !isempty(rho_radial)
        # Serialize rho_radial and add a separate row
        rho_radial_str = join(string.(rho_radial), ";")
        push!(df, (
            type = "rho_radial",
            x = missing, y = missing, z = missing,
            vx = missing, vy = missing, vz = missing,
            K = missing,
            rlin = missing,
            rho_radial = rho_radial_str,
            constants = missing
        ))
    end

    # Serialize constants dictionary and add a separate row
    if !isempty(constants)
        const_str = join(["$k=$(v)" for (k, v) in constants], ";")
        push!(df, (
            type = "constants",
            x = missing, y = missing, z = missing,
            vx = missing, vy = missing, vz = missing,
            K = missing,
            rlin = missing,
            rho_radial = missing,
            constants = const_str
        ))
    end

    CSV.write("./snapshots/bin/" * snapID * "snap.csv", df)

    # Save figures if provided
    if fig1 !== nothing 
        save("./snapshots/graphs/" * snapID * "_star.png", fig1)
    end
    if fig2 !== nothing 
        save("./snapshots/graphs/" * snapID * "_stats.png", fig2)
    end

end



"""
    read_snapshot(filename)

Reads the CSV snapshot file and returns a dictionary with keys:
- `:pos` => Nx3 matrix of positions
- `:vel` => Nx3 matrix of velocities
- `:rlin` => vector of radial coordinates
- `:rho_radial` => vector of radial densities
- `:constants` => dictionary of constants (may be empty)
"""
function read_snapshot(filename::String)
    df = CSV.read(filename, DataFrame)

    # Extract particle rows
    particle_rows = df[df.type .== "particle", :]
    pos = hcat(particle_rows.x, particle_rows.y, particle_rows.z)
    vel = hcat(particle_rows.vx, particle_rows.vy, particle_rows.vz)
    K   = particle_rows.K

    # Find and parse rlin row
    rlin_row = filter(row -> row[:type] == "rlin", eachrow(df))
    if length(rlin_row) == 1
        rlin_str = rlin_row[1][:rlin]
        rlin = parse.(Float64, split(rlin_str, ';'))
    else
        rlin = Float64[]
    end

    # Find and parse rho_radial row
    rho_radial_row = filter(row -> row[:type] == "rho_radial", eachrow(df))
    if length(rho_radial_row) == 1
        rho_radial_str = rho_radial_row[1][:rho_radial]
        rho_radial = parse.(Float64, split(rho_radial_str, ';'))
    else
        rho_radial = Float64[]
    end

    # Find and parse constants row
    constants_row = filter(row -> row[:type] == "constants", eachrow(df))
    constants = Dict{String, Real}()
    if length(constants_row) == 1
        const_str = constants_row[1][:constants]
        for pair in split(const_str, ';')
            k, v = split(pair, '=')
            if occursin(".", v) || occursin("e", v) || occursin("E", v)
                constants[k] = parse(Float64, v)
            else
                constants[k] = parse(Int, v)
            end
        end
    end

    return Dict(
        :pos => pos,
        :vel => vel,
        :K   => K,
        :rlin => rlin,
        :rho_radial => rho_radial,
        :constants => constants
    )
end


"""
    open_or_create_stats_mmap(filename::String, nsteps::Int, nfields::Int)

Creates or opens a memory-mapped Float64 matrix of size `nsteps × nfields`.
Returns `(mmap_array, io)`.
If the file doesn't exist, it will be initialized to zeros.
"""

const nsteps = 100000
const nfields = 10  

function open_or_create_stats_mmap(filename::String)
    filesize = nsteps * nfields * sizeof(Float64)
    isnew = !isfile(filename)
    io = open(filename, isnew ? "w+" : "r+")
    if isnew
        write(io, zeros(UInt8, filesize))
        seekstart(io)
    end
    arr = Mmap.mmap(io, Matrix{Float64}, (nsteps, nfields))
    return arr, io
end

"""
    update_stats_row!(arr::Matrix{Float64}, iterID::Int, stats::Vector{Float64})

Updates a row of the memory-mapped stats array at `iterID` with `stats` vector.
Assumes iter is 1-based.
"""
function update_stats_row!(arr::Matrix{Float64}, iterID::Int, stats::Vector{Float64})
    @assert 1 <= iterID <= size(arr, 1) "Iteration index out of bounds"
    @assert length(stats) == size(arr, 2) "Mismatch in stats length"
    arr[iterID, :] .= stats
end

"""
    get_stats_up_to(arr::Matrix{Float64}, iterID::Int) -> Matrix{Float64}

Returns a copy of the statistics data from iteration 1 to `iterID` (inclusive).
Useful for plotting.
"""
function get_stats_up_to(arr::Matrix{Float64}, iterID::Int)
    return arr[1:iterID, :]
end


end # module
