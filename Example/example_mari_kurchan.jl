import Pkg;
Pkg.activate(".");
using Random, Statistics
import SimulationCode

# Mari-Kurchan model: the Berthier soft sphere potential acting on the separations
# r_i - r_j + A_ij, with quenched random shifts A_ij. Equilibrium configurations are
# generated directly by planting, so no MC equilibration is needed.

dims = 3
kBT = 0.1
ρ = 1.0
N = 1000                        # Number of particles
force_cutoff = 1.25             # Force cutoff
σ_ratio = 2.219                 # Ratio of the diameters of the smallest and largest particles
box_size = (N / ρ)^(1 / dims)   # Cubic box dimension

random_seed = rand(1:10^9)      # Seed for the random number generator
Random.seed!(random_seed)       # Set before creating the potential, which draws random shifts
println("Random seed = $random_seed")

base_potential = SimulationCode.Berthier(
    n=12,
    rc=force_cutoff,
    σ_ratio=σ_ratio,
    ζ=0.2)
interaction_potential = SimulationCode.MariKurchan(base_potential, N, dims, box_size)

# System
Δt = 0.01                       # Time step
m = 1.0                         # mass
system = SimulationCode.Newtonian(kBT, Δt, m, false, dims)

q_cutoff = 0.0                  # Steinhardt order parameters are meaningless for the MK model
N_stepsMD = 10^5                # Number of time steps to take
swap_probability = 0.0
max_MC_displacement = 0.1
N_MD_equilibration_steps = 0    # The planted configuration is already in equilibrium

simulation_folder = "Data"      # Name of folder in which to store datafile
simulation_name = joinpath(@__DIR__, simulation_folder, "MK_T_$(kBT)_seed_$(random_seed)")

skin_distanceMD = 0.4           # Size of the verlet cells for MD

dump_info = SimulationCode.DumpInfo(
    true, #save
    simulation_name * "_MD.h5",
    SimulationCode.create_when_to_save_array(1.3, 100, N_stepsMD), #when save
    true, # r
    true, # v
    true, # F
    false, # D
    false, # Epot
)

cb(x...) = nothing

parameters = SimulationCode.ParameterStruct(
    N_MD_equilibration_steps, random_seed, ρ, N, box_size,
    N_stepsMD, swap_probability, max_MC_displacement,
    force_cutoff^2, q_cutoff, system, interaction_potential, dump_info, cb)
arrays = SimulationCode.ArrayStruct(N, dims)
SimulationCode.generate_diameters!(arrays, parameters, interaction_potential)
output = SimulationCode.OutputStruct()
neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMD,
    box_size, force_cutoff, N, arrays.D_array, dims)

### Planting: equilibrium positions and shifts at temperature kBT ###
SimulationCode.plant_configuration!(arrays, parameters, output, neighborlist)

### MD Production ###
SimulationCode.perform_molecular_dynamics!(arrays, parameters, output, neighborlist)
