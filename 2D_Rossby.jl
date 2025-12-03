# Simulates 2D Rossby waves, if a jet profile is used, the waves couple becoming barotropic instability.
# Inputs:
# Nx,Ny    - number of gridpoints, enter single integer for both
# Nt       - number of time saves
# filename - string, name of saved data file, do not include nc_data/

using Oceananigans, Oceananigans.Advection
using Statistics

# parameters:
Nx = Ny = 512       # 1024, 2048
Lx = Ly =  2π
Nt = 1200        # 1000, 500
Δt = 0.005      # 0.01, 0.001
T = 120
filename = "nc_data/2D_Rossby_512"
U(x, y, z, t) = tanh(4/π*y) # exp(-y^2) #+ y*(tanh(y)+1)/2

# overwrite parameters with ARGS if entered:
if length(ARGS) > 0 Nx = Ny = parse(Int64, ARGS[1]) end
if length(ARGS) > 1 Nt = parse(Int64, ARGS[2]) end
if length(ARGS) > 2 filename = "nc_data/"*ARGS[3] end

# describe setup:
println("Running simulations for (Nx,Ny) = ("*string(Nx)*", "*string(Ny)*"), with "*string(Nt)*" time saves.")
println("Data saved in NETCDF format as "*filename*".nc.")

# derived parameters:
ν = 1e-4 #1*((Lx/Nx)^2+(Ly/Ny)^2)
Δt_save = T/Nt

grid = RectilinearGrid(GPU(), size=(Nx, Ny-1), x=(-Lx/2,Lx/2), y = (-Ly/2,Ly/2), topology=(Periodic, Bounded, Flat))

model = NonhydrostaticModel(; grid,
	background_fields = (u=U,),
	coriolis = FPlane(f=1),
	timestepper = :RungeKutta3,
	advection = UpwindBiasedFifthOrder(),
	closure = ScalarDiffusivity(ν=ν))

u, v, w = model.velocities
ui(x,y,z) =  0.1*(y-1)*exp(-100*x^2-100*(y-1)^2) +  0.1*(y+1)*exp(-100*x^2-100*(y+1)^2)
vi(x,y,z) = -0.1*x*exp(-100*x^2-100*(y-1)^2)     + -0.1*x*exp(-100*x^2-100*(y+1)^2)

set!(model, u=ui, v=vi)

simulation = Simulation(model, Δt=Δt, stop_time=T)

#wizard = TimeStepWizard(cfl = 0.2, diffusive_cfl = 0.2, max_Δt = Δt_save/5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities
zeta = ∂x(v) - ∂y(u)
# s = sqrt(u^2 + v^2)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; zeta, u, v), schedule = TimeInterval(Δt_save), filename = filename, overwrite_existing = true)

run!(simulation)