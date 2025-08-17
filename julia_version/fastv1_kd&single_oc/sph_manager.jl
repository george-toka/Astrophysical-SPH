#!/usr/bin/env julia
using ArgParse

include("iniconds.jl")

include("isothermal_sim.jl")

include("polytrope_sim.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--generate"
            help = "Generate initial conditions only"
            action = :store_true

        "--run"
            help = "Run simulation"
            action = :store_true

        "--EOS"
            help = "Equation of State: isothermal or polytropic"
            arg_type = String
            required = true

        "--ic_type"
            help = "Type of initial condition. Available options: sample_isothermal_sphere, 
                    sample_plummer_sphere, bonnor_ebert_sphere, turbulent_molecular_cloud, 
                    rotating_cloud, polytropic_sphere, gaussian_sphere"
            arg_type = String
            required = true
        
        "--kwargs"
            help = "Extra keyword arguments for initial conditions, in format key1=val1,key2=val2"
            arg_type = String
            required = false
            default = ""

        "--snapID"
            help = "Snapshot number to use for cold/warm start"
            arg_type = Int
            required = false
            default = 1
        
        "--snapInterval"
            help = "Interval in which we take a single snapshot of the simulation"
            arg_type = Int
            required = false
            default = 10

        "--keepSnaps"
            help = "Keep or not the snapshots"
            arg_type = Bool
            required = false
            default = true

        "--showPlots"
            help = "Only useful when keepSnaps is active"
            arg_type = Bool
            required = false
            default = true
    end

    return parse_args(ARGS, s)
end

function main()
    args = parse_command_line()
    
    if args["generate"]
        println("Generating $(args["EOS"]) initial conditions for the test case of : $(args["ic_type"])")

        # parse kwargs string into Dict
        kwargs_dict = Dict{Symbol, Any}()

        if !isempty(args["kwargs"])
            for kv in split(args["kwargs"], ",")
                k, v = split(kv, "=")
  
                # convert to Bool if "true"/"false"
                v_lower = lowercase(v)
                if v_lower == "true"
                    v_parsed = true
                elseif v_lower == "false"
                    v_parsed = false
                else
                    # try parsing Int first, then Float, else keep string
                    parsed_int = tryparse(Int, v)
                    parsed_float = tryparse(Float64, v)
                    v_parsed = parsed_int !== nothing ? parsed_int :
                            parsed_float !== nothing ? parsed_float :
                            v
                end

                kwargs_dict[Symbol(k)] = v_parsed
            end
        end

        INICONDS.iniconds_setup(args["EOS"], args["ic_type"]; kwargs_dict...)  
    end

    if args["run"]
        if args["EOS"] == "isothermal"
            println("Running $(args["EOS"]) simulation from snapshot $(args["snapID"]) with IC type: $(args["ic_type"])")
            isothermalSim.run_simulation(args["snapID"], args["ic_type"], args["snapInterval"], args["keepSnaps"], args["showPlots"])  
        
        elseif args["EOS"] == "polytropic"
            println("Running $(args["EOS"]) simulation from snapshot $(args["snapID"]) with IC type: $(args["ic_type"])")
            polytropeSim.run_simulation(args["ic_type"], args["snapID"], args["snapInterval"], args["keepSnaps"], args["showPlots"])  
        
        else
            println("No EOS of type $(args["EOS"]) exists. Available options are either: 'isothermal' or 'polytropic'")
        end
    end
end

main()

