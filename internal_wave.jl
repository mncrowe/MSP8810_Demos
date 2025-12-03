"""
Oceananigans internal wave simulation

"""

using Oceananigans, CUDA

grid = RectilinearGrid(GPU(), size=(512, 512), x=(-π, π), z=(-π, π), topology=(Periodic, Flat, Periodic))

coriolis = FPlane(f=0.2)

# Background fields are functions of `x, y, z, t`, and optional parameters.
# Here we have one parameter, the buoyancy frequency

N = 1       # buoyancy frequency [s⁻¹]
B_func(x, z, t, N) = N^2 * z
B = BackgroundField(B_func, parameters=N)

model = NonhydrostaticModel(; grid, coriolis,
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B)) # `background_fields` is a `NamedTuple`

m = 16      # vertical wavenumber
k = 8       # horizontal wavenumber
f = coriolis.f

# Dispersion relation for inertia-gravity waves
ω² = (N^2 * k^2 + f^2 * m^2) / (k^2 + m^2)

ω = sqrt(ω²)

# Some Gaussian parameters
gaussian_amplitude = 1e-9
gaussian_width = grid.Lx / 15

# A Gaussian envelope centered at `(x, z) = (0, 0)`
a(x, z) = gaussian_amplitude * exp( -( x^2 + z^2 ) / 2gaussian_width^2 )

u₀(x, z) = a(x, z) * k * ω   / (ω^2 - f^2) * cos(k * x + m * z)
v₀(x, z) = a(x, z) * k * f   / (ω^2 - f^2) * sin(k * x + m * z)
w₀(x, z) = a(x, z) * m * ω   / (ω^2 - N^2) * cos(k * x + m * z)
b₀(x, z) = a(x, z) * m * N^2 / (ω^2 - N^2) * sin(k * x + m * z)

set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

simulation = Simulation(model, Δt = 0.01 * 2π/ω, stop_iteration = 1000)

filename = "internal_wave.jld2"
simulation.output_writers[:velocities] = JLD2Writer(model, model.velocities; filename,
                                                    schedule = IterationInterval(1),
                                                    overwrite_existing = true)

run!(simulation)