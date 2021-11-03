using Flux: Zygote
using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, PlotThemes, PlotUtils, Measures
using QuadGK
include("../Qtils.jl")
using .Qtils
include("../AwesomeTheme.jl")

theme(:awesome)
resetfontsizes()
scalefontsizes(1.6)

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

V(Q, p) =  #=p[1] * Q^4 =# p[1]*Q^4 
dV(Q, p) = Zygote.gradient(q -> V(q, p)[1], Q)[1]

ps = vcat(p, rand(Float32, 1))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 0 #- Ω_ϕ(Q, dQ, E, V(Q, p), z) # 8pi/3 * (0.5*((1+z)*dQ)^2 + V(Q, p)[1]/E^2)
    # println(Ω_m)
    dE = (E/(2+3/w_BD)) * ((1+z)*(0.5 + w_BD)*dQ^2/ Q^2 + 4*dQ/Q + (6-dV(Q,p)[1]/E^2)/(w_BD*(1+z)) + 3*Ω_m/(1+z)) #dE brans dicke

    # println("dE ", dE)
    du[1] = dQ
    du[2] = (dQ^2)*0.5/Q - dQ*(dE/E - 2/(1+z)) + Q/w_BD *(6/(1+z)^2 - dV(Q,p)/((1+z)*E)^2 - 3*dE/((1+z)*E)) #ddF brans dicke
    du[3] = dE
    du[4] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-3)

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
    #println("Dark matter density parameter: ", 1 - Ω_ϕ(pred[1,1], pred[2,1], pred[3,1], potential, uniquez[1])[1])
    return l < 1.05
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)


U(Q, p) =  #=p[1] * Q^4 =# p[1] .*Q .^4 
dU(Q, p) = Zygote.gradient(q -> U(q, p)[1], Q)[1]
ddU(Q, p) = Zygote.hessian(q -> U(q,p)[1], Q)[1]

# Hier neue Sachen, die die EoS für ein mögliches Fluid ausrechnen sollen
Ω_m = 0
v = rand(Float32, 7)
println( "v: ", v)
Enachz = (res[3,:] ./(2+3/w_BD)) .* ((1 .+uniquez) .*(0.5 + w_BD) .*res[2,:].^2 ./ res[1,:].^2 .+ 4 .*res[2,:] ./res[1,:] .+ (6 .-dU(res[1,:],result.minimizer[3:end])[1] ./res[3,:].^2) ./(w_BD .*(1 .+uniquez)) .+ 3*Ω_m ./(1 .+uniquez))



EoSs = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1, tanh) # maybe choose exp as output function to enforce positive potentials only
)

y0 = 2.0/3.0 .* (1 .+ uniquez) .*(Enachz) ./(res[3,:]) .- 1

function loss(p)
    y = map(z -> EoSs(1/(1+z),p)[1], uniquez) # [EoSs(1/(1+z),p)[1] for z ∈ uniquez]
    return sum(abs2, y .-  y0 )
end

opt = ADAM(0.1)

ps = initial_params(EoSs)
# dat = zip( uniquez, -1.5 .*(1 .+uniquez) .*(Enachz) ./(res[3,:]) .+ 1)

epochs = 1000
for epoch ∈ 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    # gs = Flux.gradient(ps) do
    #     loss(d...)
    # end
    _loss = loss(ps)
    println("Loss: ", _loss)
    gs = Flux.gradient(p -> loss(p), ps)[1]
    Flux.Optimise.update!(opt, ps, gs)
end

# neue Sachen zu Ende 

plot1 = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = map(q -> V(q, result.minimizer[3:end])[1], res[1,:])
slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)
EoS1 = Qtils.calculateEOS(potential, res[2,:], res[3,:], uniquez)
EoS2 = map(x-> EoSs(1/(1+x), initial_params(EoSs))[1], uniquez)
#slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)
ϵ = 1/(16π) .* (dU(res[1,:],result.minimizer[3:end]) ./U(res[1,:],result.minimizer[3:end])).^2
η = 1/(16π) .* (ddU(res[1,:],result.minimizer[3:end])[1] ./U(res[1,:],result.minimizer[3:end]))

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
plot2 = Plots.plot(uniquez, EoS1, title="Equation of State, method 4", xlabel="redshift z", ylabel="eos w", legend=:topright)
plot3 = Plots.plot(uniquez, EoS2, title="Equation of State, method 2", xlabel="redshift z", ylabel="eos w", legend=:topright) #hier ist meine eos die in der diff berechnet wurde
plot4 = Plots.plot(uniquez, potential, title="Potential", xlabel="scalar field Ψ", ylabel="V(Ψ)", legend=:bottomright)
#plot4 = Plots.plot(uniquez, 1 .- Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
#plot4 = Plots.plot!(plot4, uniquez, Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez), legend=:topright, label="Ω_ϕ")
plot5 = Plots.plot(res[1,:], ϵ, title = "Slow-roll Parameters", xlabel = "scalar-field Ψ", ylabel = "slow-roll parameter", label = "ϵ")
plot5 = Plots.plot!(plot5, res[1,:], η, legend=:topright, label = "η" )
plot6 = Plots.plot(res[1,:], res[2,:], title= " Ψ' in dependence of Ψ", xlabel = "scalar-field Ψ", ylabel = "Ψ'", legend=:topright)

println("Cosmological parameters: ")
#println("Dark matter density parameter Ω_m = ", 1 - Ω_ϕ(res[1,:], res[2,:], res[3,:], potential, uniquez)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])


resultplot1 = Plots.plot(plot1, plot3, plot2, layout = (3,1),  size=(1200, 1100))
resultplot2 = Plots.plot(plot4, plot6, plot5, layout = (3,1), size=(1200, 1100))
savefig(resultplot1,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotNNeos1e.png")
savefig(resultplot2,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotNNeos2e.png")

#=
savefig(plot1,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN1d.png")
savefig(plot2,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN2d.png")
savefig(plot3,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN3d.png")
savefig(plot4,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN4d.png")
savefig(plot5,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN5d.png")
savefig(plot6,"C:/Jana/Studium/8. sem phys/bachelor/fake universe/quartpot/BDquartpotEoSNN6d.png")=#





 #=dE = E/((1-(dQ+6*Q/(w_BD*(1+z)))/(dQ-2*Q/(1+z)))*(dQ - 2*Q/(1+z))) * (-(1.5*dQ^2/Q + 2*dQ/(1+z) +Q/w_BD * (6/(1+z)^2 - dV(Q,p)/(((1+z)*E)^2))) - dQ^2*w_BD/Q - 2*dQ/(1+z)- 3 *Ω_m/(8*pi*G*(1+z)^2)) #dE brans dicke
 dE = 1.5*(E/(1+z)) * (Ω_m + 8pi/3 * ((1+z)*dQ)^2)

 du[1] = dQ
 du[2] = (dQ^2)*1.5/Q - dQ*(dE/E - 2/(1+z)) + Q/w_BD *(6/(1+z)^2 - dV(Q,p)/((1+z)*E)^2 - 6*dE/((1+z)*E)) #ddF brans dicke
 Ω_ϕ(Q, dQ, E, V, z) = #=dQ.^2 .*w_BD .*(1+z)^2 ./(6 .*Q) .+  V(Q, p) ./(3 .*Q.*E.^2) .=#- sqrt(6).*dQ.*(1+z) .*E./Q #brans dicke
 =#