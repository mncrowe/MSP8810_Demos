"""
Beta-plane turbulence simulation with finite Rossby radius

"""

using GeophysicalFlows, Random, CUDA, NetCDF

using GeophysicalFlows: peakedisotropicspectrum
using Random: seed!

# Set parameters:

dev = GPU()

n, L = 1024, 2π

deformation_radius = 0.1 # 0.35     # number or Inf
β = 0.0 # 10.0

k₀, E₀ = 20, 0.25 # initial position (dimensionless) of peak in spectrum and total initial energy

forcing_wavenumber = 14.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0 # 0.001                         # energy input rate by the forcing

stepper = "FilteredRK4"
dt = 2e-3
Nt = 1e5
Ns = 1000
savename = "data_R"

# Build Forcing:

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = FourierFlows.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0       # normalize forcing to inject energy at rate ε

if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end

function calcF!(Fh, sol, t, clock, vars, params, grid)
  randn!(Fh)
  @. Fh *= sqrt(forcing_spectrum) / sqrt(clock.dt)
  return nothing
end

# Build problem:

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β, deformation_radius, dt, stepper,
                             calcF=calcF!, stochastic=true)

# Set initial condition:

q₀ = peakedisotropicspectrum(prob.grid, k₀, E₀, mask=prob.timestepper.filter)

SingleLayerQG.set_q!(prob, q₀)

# Define saves:

to_CPU(f) = device_array(CPU())(f)

fstring(num) = string(round(num,sigdigits=8))
istring(num) = string(Int(num))

filename = savename * ".nc"
if isfile(filename); rm(filename); end

t = LinRange(0, Nt * dt, Ns+1)
nccreate(filename, "psi", "x", grid.x, "y", grid.y, "t", t)
nccreate(filename, "Q", "x", grid.x, "y", grid.y, "t", t)

function Save_NetCDF(i)
  ncwrite(reshape(to_CPU(prob.vars.ψ[:,:]),(n, n)), filename, "psi", start = [1, 1, i+1], count = [n, n, 1])
  ncwrite(reshape(to_CPU(prob.vars.q[:,:]),(n, n)), filename, "Q", start = [1, 1, i+1], count = [n, n, 1])
  print("Iteration: " * istring(i*Nt/Ns) * ", t = " * fstring(prob.clock.t) * "\n")
  if maximum(isnan.(prob.vars.q)); error("Infinite q, reduce timestep"); end
  return nothing
end

Save_NetCDF(0)

# Run simulation:

for i1 in 1:Ns
  stepforward!(prob, Int(Nt/Ns))
  SingleLayerQG.updatevars!(prob)
  Save_NetCDF(i1)
end