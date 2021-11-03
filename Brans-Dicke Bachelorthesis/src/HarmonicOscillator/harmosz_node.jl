using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, PlotThemes, PlotUtils, Measures
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets
include("../AwesomeTheme.jl")

theme(:awesome)
resetfontsizes()
scalefontsizes(1.5)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=256))
u0 = [3.0, 0.0] # contains x0 and dx0
p = [2.0]
stderr = 0.2

# Define potential and get dataset
V0(x,p) = p[1]*x^2 
data = MechanicsDatasets.potentialproblem(V0, u0, p, t0, addnoise=true, σ=stderr)
### End ------------------------------------------------ #




ps = [2.0, 0.5, 1.0]
allp = vcat(u0, ps)

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -p[1] * u[1] + p[2] * du[1] + p[3]
end
prob = ODEProblem(neuraloscillator!, u0, tspan, ps)
sol = solve(prob, Tsit5())

#=function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3], saveat=t0))
end

println(predict(allp)[1:2])=#


function loss(ps)
    #pred = predict(params)
    #return sum(abs2, (pred .- data[2:3,:])./stderr) / (size(data, 2) - size(params, 1)), pred
    sol = solve(prob, Tsit5(), p=ps, saveat = t0)
    loss = sum(abs2, sol.-data[2:3,:])
    return loss, sol
end

cb = function(p,l,pred)
    display(l)
    plt = plot(pred, ylim = (-4, 4))
    #display(plt)
    return false
end
opt = ADAM(0.1)

@time result = DiffEqFlux.sciml_train(loss, allp, opt, cb=cb, maxiters=1000)
res = Array(solve(prob, Tsit5(), p=result.minimizer[1:3], saveat=t0))
println("params : ", result.minimizer[1:3])

traj_plot = plot(t0, res[1, :], title="Trajectory",ylabel="Displacement x", xlabel="Time t", ylim=(-4,4), label="fit with NN")
traj_plot = scatter!(traj_plot, t0, data[2,:], label="data")



# Plotting the potential
x0 = Array(range(-u0[1], u0[1], step=0.01))
predicted_potential = map(x -> 0.5*result.minimizer[1]*x^2, x0)
true_potential = map(x -> V0(x, p), x0)
pot_plot = plot(x0, predicted_potential,  title="Potential", ylabel="V", xlabel="Displacement x", label = "fit with NN")
pot_plot = plot!(pot_plot, x0,true_potential, ylims=(-0.25,3.5), xlims=(-u0[1],u0[1]), label = "data")
resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800))


savefig(resultplot,"C:/Jana/Studium/8. sem phys/bachelor/harmosz_node.png")