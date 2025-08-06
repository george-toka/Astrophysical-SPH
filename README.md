We have created an engine for Astrophysical SPH simulations. This engine uses ArgParse package from julia to emulate a CLI-like 
package. The usage guide is pretty simple.

1) Clone Git Repository
2) Activate Julia REPL
3) Change directory inside "fastv1_kd&single_oc" folder
4) Load the ArgParse package, and set the variable ARGS with every argument that is needed
  -> Looking at the sph_manager.jl we have two base options :
   a) generate
   b) run

   With the "generate" option, you create a csv file with initial conditions of your choice, based on the available functions
   included in the iniconds.jl file. All special arguments and constants are inside the iniconds.jl file and if one wants to change
   something specific e.g the temperature of the gas, they should do so by altering them inside the file. That way the CLI-like package
   and the instructions needed in the terminal are short and clean.

   With the "run" option, one can either run a simulation based on a gas that has a polytropic EOS or an isothermal EOS, where some characteristics of an isothermal gas
   simplify certain operations, thus the separate simulation file along with its tailored modules.

   Now for a single example. Let's say we want to generate initial conditions for the test case of a gaussian sphere with a polytropic EOS. The ARGS variable should be assigned like so:
   ARGS = ["--generate", "--EOS", "polytropic", "--ic-type", "gaussian_sphere"].
   Then, what happens is that a file with name 1snap.csv is generated and saved in the folder that corresponds to the ic-type name
   A second example, where we want to run a simulation with the above initial conditions now.
   ARGS = ["--run", "--EOS", "polytropic", "--ic-type", "gaussian_sphere", "1", "5". "true", "false"].
   Now a polytropic simulation will start running. The last 4 arguments correspond to snapshot ID, snapshot Interval to know that every K steps we take a snapshot, a snapshot flag to decide whether we keep the snapshots, and if
   that happens, we have a plotting flag lastly, to decide whether we keep the plots and graphs in png files.

6) After assigning the ARGS variable simply run the sph_manager.jl file and you are ready to go.

   
