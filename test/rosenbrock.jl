using MadamOpt
using ..TestHelpers: @testset_filtered
using Test, OptimTestProblems, Printf, Random, LinearAlgebra

Random.seed!(0)

const MVP = OptimTestProblems.MultivariateProblems
const PROBLEMS = MVP.UnconstrainedProblems.examples

@testset_filtered "Rosenbrock - gradient" begin
    p = PROBLEMS["Rosenbrock"]
    obj = MVP.objective(p)
    m = Madam(deepcopy(p.initial_x); alpha = 0.01, beta1 = 0.9)
    grad = begin
        grad_storage = zeros(size(p.initial_x))
        x -> (MVP.gradient(p)(grad_storage, x); grad_storage)
    end

    maxIter = Int(5e4)
    @time for ii = 0:maxIter
        update!(
            m
            , grad(current(m))
            ; l1_penalty = nothing
            , l2_penalty = nothing
        )

        if ii%Int(div(maxIter,20))==0
            @printf("%10d %10.2f %s %s %e %e\n"
                , ii
                , p.minimum
                , p.solutions
                , current(m)
                , norm(current(m) .- p.solutions)
                , obj(current(m))
            )
        end
    end

    @test norm(current(m) .- p.solutions) < 1e-5
end

@testset_filtered "Rosenbrock - gradient free" begin
    p = PROBLEMS["Rosenbrock"]
    obj = MVP.objective(p)
    m = Madam(deepcopy(p.initial_x); alpha = 0.01, beta1 = 0.9)

    maxIter = Int(5e4)
    @time for ii = 0:maxIter
        update!(
            m
            , obj
            ; grad_samples = max(1,length(p.initial_x)>>3)
            , l1_penalty = nothing
            , l2_penalty = nothing
        )

        if ii%Int(div(maxIter,20))==0
            @printf("%10d %10.2f %s %s %e %e\n"
                , ii
                , p.minimum
                , p.solutions
                , current(m)
                , norm(current(m) .- p.solutions)
                , obj(current(m))
            )
        end
    end

    @test norm(current(m) .- p.solutions) < 1e-5
end

# For comparison:
# using Optim
# for _ in 1:2
# @testset_filtered "Rosenbrock - Nelder Mead" begin
#     p = PROBLEMS["Rosenbrock"]
#     obj = MVP.objective(p)
# 
#     opts = Optim.Options(iterations = 10^4, show_trace = true, show_every = 10)
#     res = @time optimize(obj, deepcopy(p.initial_x), NelderMead(), opts)
#     theta_hat = Optim.minimizer(res)
#     println("solution: ", theta_hat)
# end
# end
