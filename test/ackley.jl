using MadamOpt
using ..TestHelpers: @testset_filtered
using Test, Printf, Random, LinearAlgebra

Random.seed!(0)
N = 100
SOLUTION = 10*randn(N)

# idea from https://github.com/polixir/ZOOpt
# consider implementing "Shekel’s Foxholes", though those likely won't lead to
# great results as the test is more challenging
function ackley(x::AbstractArray, bias::Union{Nothing, AbstractArray}=nothing)
    if bias != nothing
        x = x - bias
    end
    -20.0 * exp(-0.2 * sqrt(dot(x,x)/length(x))) - exp(sum(cos.(2.0*pi*x)) / length(x)) + 20.0 + exp(1)
end

const ackley_biased = x -> ackley(x, SOLUTION)

@testset_filtered "Ackley - gradient free" begin
    maxIter = Int(5e5)

    theta = zeros(N)
    m = Madam(theta; alpha = 0.01, beta1 = 0.9, max_steps=maxIter, max_temp=10.0)

    @time for ii = 0:maxIter
        update!(
            m
            , ackley_biased
            ; grad_samples = max(1,length(theta)>>5)
        )

        if ii%Int(div(maxIter,20))==0
            @printf("%10d %e %e\n"
                , ii
                , norm(current(m) .- SOLUTION)
                , ackley_biased(current(m))
            )
        end
    end
    # println("SOLUTION: ", current(m))
    @test norm(current(m) .- SOLUTION) < 1e-2
    @test ackley_biased(current(m)) < 1e-2
end


# # For comparison:
# using Optim
# for _ in 1:2
# @testset_filtered "Ackley - Nelder Mead" begin
#     opts = Optim.Options(iterations = 10^5, show_trace = true, show_every = 10^4)
#     res = @time optimize(ackley_biased, zeros(N), NelderMead(), opts)
#     theta_hat = Optim.minimizer(res)
#     # println("solution: ", theta_hat)
#     println("solution: ", ackley_biased(theta_hat))
# end
# end
