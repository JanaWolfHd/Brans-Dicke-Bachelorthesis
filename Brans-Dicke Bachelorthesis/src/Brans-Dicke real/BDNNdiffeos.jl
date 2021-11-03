using Flux: Zygote
using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, PlotThemes, PlotUtils, Measures
using QuadGK
include("../Qtils.jl")
using .Qtils
include("../AwesomeTheme.jl")

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)
const w_BD = 40000 #4.538e-6

# we cannot vary the initial conditions much, otherwise we get inconsistent results!!!
p = [3.0, 0.0]
u0 = vcat(p[1:2], [1.0, 0.0])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
#Ω_ϕ(Q, dQ, E, V, z) = dQ.^2 .*w_BD .*(1+z)^2 ./(6 .*Q) .+  V ./(3 .*Q.*E.^2) .- sqrt(6).*dQ.*(1+z) .*E./Q #brans dicke

V = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1, exp) # maybe choose exp as output function to enforce positive potentials only
)

dV(Q, p) = Flux.gradient(q -> V(q, p)[1], Q)[1]

# Defining the EoS for some possible fluid
ω = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 8, relu),
    FastDense(8, 1, tanh)
)

ps = vcat(p, initial_params(V), initial_params(ω))
println(length(initial_params(V)))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 1/3 #- Ω_ϕ(Q, dQ, E, V(Q, p), z) # 8pi/3 * (0.5*((1+z)*dQ)^2 + V(Q, p)[1]/E^2)
    #dE = (1+1.5*E/(2*w_BD + 6)) *w_BD* (1+z)dQ^2/Q^2 +(1+ E*w_BD/(w_BD+3)) * (2*dQ/Q) +(6*E/(1+z) - dV(Q,p)[1]/((1+z)*E)) * (1/(2*w_BD + 6)) + 3* Ω_m/(Q*8*pi*(1+z)) #dE brans dicke
    dE = 1/E*(1+z)^3 * ((1.5*E^2 *(1+z)^2 - 0.5) *((ω(1/(1+z), p[(length(initial_params(V))+1):end])[1] +1) - 1/3))

    #println("dE ", dE)
    du[1] = dQ
    du[2] = (dQ^2)*0.5/Q - dQ*(dE/E - 2/(1+z)) + Q/w_BD *(6/(1+z)^2 - dV(Q,p[1:length(initial_params(V))])[1]/((1+z)*E)^2 - 3*dE/((1+z)*E)) #ddF brans dicke
    du[3] = dE
    du[4] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=uniquez))
end
#println(predict(ps))

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedchisquared(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    println("Q ", pred[1, 1])
    println("dQ ", pred[2, 1])
    #println("ddQ ", pred[2, 1])
   # println("dE ",1.5*pred[3, 1]/(1+z)*(1+ ω(Q, p[(length(initial_params(V))+1):end])[1]))
    #potential = V(pred[1,1], p[3:end])[1]
    #println("Dark matter density parameter: ", 1 - Ω_ϕ(pred[1,1], pred[2,1], pred[3,1], potential, uniquez[1])[1])
    return l < 1.05
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=2)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)



plot1 = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

dV0 = map(x -> dV(x, result.minimizer[3:99])[1], res[1,:])
V0 = map(x -> V(x, result.minimizer[3:99])[1], res[1,:])
ddNN(Q, P) = Zygote.hessian(q -> V(q, P)[1], Q)[1]
ddV0 = map(x -> ddNN(x, result.minimizer[3:99])[1], res[1,:])
EoS1 = Qtils.calculateEOS(V0, res[2,:], res[3,:], uniquez)
slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:99], res[1,:], verbose=true)
ϵ = 1/(16π) .* (dV0 ./V0).^2
η = 1/(16π) .* (ddV0 ./V0)
y0 = map(x -> ω(x, result.minimizer[length(initial_params(V))+3:end])[1], uniquez)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
plot2 = Plots.plot(uniquez, EoS1, title="Equation of State 1", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
plot3 = Plots.plot(uniquez, y0, title="Equation of State 2", xlabel="redshift z", ylabel="equation of state w", legend=:topright) #hier ist meine eos die in der diff berechnet wurde
plot4 = Plots.plot(uniquez, V0, title="Potential", xlabel="scalar field Ψ", ylabel="V(Ψ)", legend=:bottomright)
#plot4 = Plots.plot(uniquez, 1 .- Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
#plot4 = Plots.plot!(plot4, uniquez, Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez), legend=:topright, label="Ω_ϕ")
plot5 = Plots.plot(res[1,:], ϵ, title = "Slow-roll Parameters", xlabel = "scalar-field Ψ", ylabel = "slow-roll parameter", label = "ϵ")
plot5 = Plots.plot!(plot5, res[1,:], η, legend=:topright, label = "η" )
plot6 = Plots.plot(res[1,:], res[2,:], title= " Ψ' in dependence of Ψ", xlabel = "scalar-field Ψ", ylabel = "Ψ'", legend=:topright)

println("Cosmological parameters: ")
#println("Dark matter density parameter Ω_m = ", 1 - Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])


resultplot = Plots.plot(plot1, plot2, plot3, layout=(3, 1), size=(1200, 800))
Plots.plot(resultplot)

savefig(plot1,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos1.png")
savefig(plot2,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos2.png")
savefig(plot3,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos3.png")
savefig(plot4,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos4.png")
savefig(plot5,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos5.png")
savefig(plot6,"C:/Jana/Studium/8. sem phys/bachelor/real universe/NNpot/BDNNdiffeos6.png")
 






 #=dE = E/((1-(dQ+6*Q/(w_BD*(1+z)))/(dQ-2*Q/(1+z)))*(dQ - 2*Q/(1+z))) * (-(1.5*dQ^2/Q + 2*dQ/(1+z) +Q/w_BD * (6/(1+z)^2 - dV(Q,p)/(((1+z)*E)^2))) - dQ^2*w_BD/Q - 2*dQ/(1+z)- 3 *Ω_m/(8*pi*G*(1+z)^2)) #dE brans dicke
 dE = 1.5*(E/(1+z)) * (Ω_m + 8pi/3 * ((1+z)*dQ)^2)

 du[1] = dQ
 du[2] = (dQ^2)*1.5/Q - dQ*(dE/E - 2/(1+z)) + Q/w_BD *(6/(1+z)^2 - dV(Q,p)/((1+z)*E)^2 - 6*dE/((1+z)*E)) #ddF brans dicke
 Ω_ϕ(Q, dQ, E, V, z) = #=dQ.^2 .*w_BD .*(1+z)^2 ./(6 .*Q) .+  V(Q, p) ./(3 .*Q.*E.^2) .=#- sqrt(6).*dQ.*(1+z) .*E./Q #brans dicke
 =#