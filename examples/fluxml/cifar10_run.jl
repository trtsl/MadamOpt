# An example of using Madam with FluxML on GPU's.  Preliminary tests suggest
# that it's initially slower at minimizing the loss function than standard
# Adam, but leads to better convergence on long runs (similar to SGD relative
# to Adam).  This may be due to the fact that Adam's step size is invariant to
# the magnitude of the gradient, thereby ameliorating vanishing gradients and
# potentially resulting in greater exploration.

using Pkg

Pkg.develop(path=realpath(joinpath(@__DIR__,"../../.")))

Pkg.add("Flux")
Pkg.add("Metalhead")
Pkg.add("Parameters")
Pkg.add("Images")
Pkg.add("CUDAapi")
Pkg.add("CuArrays")

include("cifar10.jl")

# Pkg.add("BSON")
# using BSON: @save, @load
using Random
using MadamOpt

struct MadamSplice
    dict::IdDict
end

MadamSplice() = MadamSplice(IdDict())

function Flux.Optimise.apply!(d::MadamSplice, theta, grad)
    m = get!(d.dict, theta) do
        Madam(theta; alpha = 1e-1)
    end
    return .-step!(m, grad; l1_penalty = 1e-4, weight_decay = 1e-4)
end

Random.seed!(0)

m = vgg16()
opt = MadamSplice()
# opt = ADAM(3e-4)
# opt = ADAMW(3e-4, (0.9, 0.999), 0.001)
train!(m, opt; epochs=20)
test(m)
