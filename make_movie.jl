"""
Oceananigans internal wave simulation

"""

using CairoMakie
set_theme!(Theme(fontsize = 20))

fig = Figure(size = (600, 600))

ax = Axis(fig[2, 1]; xlabel = "x", ylabel = "z",
          limits = ((-π, π), (-π, π)), aspect = AxisAspect(1))

n = Observable(1)

w_timeseries = FieldTimeSeries(filename, "w")
w = @lift w_timeseries[$n]
w_lim = 1e-8

contourf!(ax, w;
          levels = range(-w_lim, stop=w_lim, length=10),
          colormap = :balance,
          extendlow = :auto,
          extendhigh = :auto)

title = @lift "ωt = " * string(round(w_timeseries.times[$n] * ω, digits=2))
fig[1, 1] = Label(fig, title, fontsize=24, tellwidth=false)

using Printf

frames = 1:length(w_timeseries.times)

@info "Animating a propagating internal wave..."

record(fig, "internal_wave.mp4", frames, framerate=20) do i
    n[] = i
    # @info i
end

