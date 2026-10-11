"""
    pair_displacement(ri, rj, particle_i, particle_j, box_size, U::MariKurchan)

Calculates the minimum image of the shifted separation `ri - rj + A_ij` of the
Mari-Kurchan model.
"""
@inline function pair_displacement(ri, rj, particle_i, particle_j, box_size, U::MariKurchan)
    # A_ij = -A_ji. Reading column particle_i makes loops over the neighbors of particle_i cache friendly
    dr = ri - rj - U.shifts[particle_j, particle_i]
    dr -= round.(dr / box_size) * box_size
    return dr
end

"""
    randomize_shifts!(U::MariKurchan, box_size)

Draws the quenched random shifts `A_ij = -A_ji` of the Mari-Kurchan model
independently and uniformly in the box `[-box_size/2, box_size/2)^dims`.
"""
function randomize_shifts!(U::MariKurchan{P, dims}, box_size) where {P, dims}
    shifts = U.shifts
    N = size(shifts, 1)
    for particle_i = 1:N
        shifts[particle_i, particle_i] = zero(SVector{dims, Float64})
        for particle_j = particle_i+1:N
            A = (rand(SVector{dims, Float64}) .- 0.5) * box_size
            shifts[particle_i, particle_j] = A
            shifts[particle_j, particle_i] = -A
        end
    end
end

"""
    find_neighbor_pairs!(arrays, parameters, neighborlist, r_cutoff, U::MariKurchan)

Calls `add_pair!` for every pair of particles whose shifted separation lies within the Verlet sphere.

# Notes
- The shifts couple particles that are anywhere in the box, so cell lists cannot be used and all N(N-1)/2 pairs are checked. Since shifted separations change exactly as much as unshifted ones when particles move, the usual skin criterion for rebuilding the lists remains valid.
"""
function find_neighbor_pairs!(arrays, parameters, neighborlist, r_cutoff, interaction_potential::MariKurchan)
    r_array = arrays.r_array
    box_size = parameters.box_size
    verlet_sphere_size2 = neighborlist.verlet_sphere_size^2
    N = length(r_array)
    for particle_i = 1:N
        ri = r_array[particle_i]
        for particle_j = particle_i+1:N
            dr = pair_displacement(ri, r_array[particle_j], particle_i, particle_j, box_size, interaction_potential)
            d2 = sum(dr.^2)
            if d2 < verlet_sphere_size2
                add_pair!(particle_i, particle_j, d2, neighborlist, arrays, r_cutoff, interaction_potential)
            end
        end
    end
end

"""
    save_interaction_potential(file, U::MariKurchan)

Stores the fields of the base potential as attributes of the HDF5 file, and the
shifts as the dataset `miscellaneous/MK_shifts` of size (dims, N, N), such
that `MK_shifts[:, i, j]` holds `A_ij`.
"""
function save_interaction_potential(file, interaction_potential::MariKurchan)
    attributes(file)["base_potential"] = string(typeof(interaction_potential.potential))
    save_interaction_potential(file, interaction_potential.potential)
    file["miscellaneous/MK_shifts"] = reinterpret(reshape, Float64, interaction_potential.shifts)
end

"""
    sample_planted_shift(ri, rj, mean_d_squared, box_size, kBT, r²_cutoff, interaction_potential)

Samples a shift `A_ij` from P(A_ij) ∝ exp(-V(|ri - rj + A_ij|)/kBT) in the box.

# Notes
- Uses rejection sampling: uniform shifts are accepted with probability
  exp(-V/kBT), which is exact for non-negative pair potentials. Since a uniform
  shift rarely lands within the range of the potential, almost every first
  attempt is accepted.
"""
function sample_planted_shift(ri::SVector{dims, Float64}, rj, mean_d_squared, box_size, kBT, r²_cutoff, interaction_potential) where dims
    while true
        A = (rand(SVector{dims, Float64}) .- 0.5) * box_size
        dr = ri - rj + A
        dr -= round.(dr / box_size) * box_size
        rij2 = sum(dr.^2)
        if r²_cutoff * mean_d_squared < rij2
            return A
        end
        E = potential(rij2, mean_d_squared, interaction_potential)
        if E < 0.0
            error("Planting requires a non-negative pair potential, found V = $E")
        end
        if rand() < exp(-E/kBT)
            return A
        end
    end
end

"""
    plant_shifts!(U::MariKurchan, r_array, D_array, box_size, kBT, r²_cutoff)

Draws every shift `A_ij = -A_ji` of the Mari-Kurchan model conditioned on the
configuration `r_array`, from P(A_ij) ∝ exp(-V(|r_i - r_j + A_ij|)/kBT).
"""
function plant_shifts!(U::MariKurchan{P, dims}, r_array, D_array, box_size, kBT, r²_cutoff) where {P, dims}
    shifts = U.shifts
    base_potential = U.potential
    N = length(r_array)
    for particle_i = 1:N
        shifts[particle_i, particle_i] = zero(SVector{dims, Float64})
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        for particle_j = particle_i+1:N
            mean_d = find_mean_D(Di, D_array[particle_j], base_potential)
            A = sample_planted_shift(ri, r_array[particle_j], mean_d * mean_d, box_size, kBT, r²_cutoff, base_potential)
            shifts[particle_i, particle_j] = A
            shifts[particle_j, particle_i] = -A
        end
    end
end

"""
    plant_configuration!(arrays, parameters, output, neighborlist)

Generates an equilibrium configuration of the Mari-Kurchan model by planting.

The particles are placed uniformly at random in the box, after which the shifts
stored in `parameters.interaction_potential` are redrawn conditioned on these
positions (see `plant_shifts!`). This samples the joint distribution
P(r, A) ∝ ∏_{i<j} exp(-V(|r_i - r_j + A_ij|)/kBT), whose marginal over the
positions is uniform because each pair factor integrates over A_ij to a
constant. In the liquid phase the annealed and quenched free energies of the
model coincide, so this is the same as first drawing the shifts and then
equilibrating the particles: the result is an equilibrium configuration for
the planted shifts, also at densities beyond the dynamical transition where
equilibration by MC or MD is impossible.

# Arguments
- `arrays`: The ArrayStruct, `D_array` must already be set (`generate_diameters!`).
- `parameters`: The ParameterStruct, with a `MariKurchan` interaction potential.
- `output`: The OutputStruct, its potential energy is updated.
- `neighborlist`: The NeighborListStruct, rebuilt for the planted configuration.

# Notes
- Only exact for non-negative pair potentials, an error is raised otherwise.
- The planted state is an equilibrium state at temperature `parameters.system.kBT`.
"""
function plant_configuration!(arrays, parameters, output, neighborlist)
    interaction_potential = parameters.interaction_potential
    if !(interaction_potential isa MariKurchan)
        error("Planting is only possible for the Mari-Kurchan model")
    end
    N = parameters.N
    r_array = arrays.r_array
    @assert size(interaction_potential.shifts) == (N, N)
    @assert eltype(interaction_potential.shifts) == eltype(r_array)
    box_size = parameters.box_size
    kBT = parameters.system.kBT

    for particle_i = 1:N
        r_array[particle_i] = rand(eltype(r_array)) * box_size
    end
    plant_shifts!(interaction_potential, r_array, arrays.D_array, box_size, kBT, parameters.force_cutoff2)

    output.steps_done = 0
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)
    println("Planted an equilibrium configuration of the Mari-Kurchan model at kBT = $(kBT).")
    println("The volume fraction is φ = $(find_volume_fraction(arrays.D_array, box_size, parameters.system.dims))")
    println("E_pot/N = $(round(output.potential_energy/N, digits=6))")
end
