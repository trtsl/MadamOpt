using MadamOpt
using ..TestHelpers: @testset_filtered
using Test, LinearAlgebra, Printf, Random, SparseArrays

#############
# Test
#############
# Problem initialized from https://stanford.edu/~boyd/papers/prox_algs/lasso.html

Random.seed!(0)

M = 500;    # number of examples
N = 250;    # number of features

theta = sprandn(N,0.05);
A = randn(M,N);
A = A * SparseArrays.spdiagm(0=>(1.0 ./ sqrt.(sum(A.^2;dims=1)[:]))) # normalize columns
v = sqrt(0.01)*randn(M,1)
b = A*theta + v
l1_penalty = 4.0 * norm(A'*b, 2) / length(b)

@printf("solving instance with %d examples, %d variables\n", M, N);
@printf("nnz(theta) = %d; signal-to-noise ratio: %.2f\n", nnz(theta), norm(A*theta)^2/norm(v)^2);

thetaLinReg = A\b
objective(A, b, l1_penalty, theta) = norm(A*theta - b) + l1_penalty*norm(theta,1)

@testset_filtered "Sparse matrix - prox_L1 - gradient" begin
    m = Madam(zeros(N); alpha = 0.1)

    maxIter = Int(1e3)
    @time for ii = 0:maxIter
        rowIdx  = rand(1:length(b), 32)
        ai      = view(A,rowIdx,:)
        bi      = view(b,rowIdx)
        update_ols!(m, ai, bi; l1_penalty = l1_penalty)

        if ii%Int(div(maxIter,20))==0
            @printf(
                "%10d %10d %10.2f %10.2f %10.2f\n"
                , ii
                , count(eps().<abs.(current(m)))
                , norm(current(m)-theta)
                , norm(A*current(m)-b)
                , objective(A, b, l1_penalty, current(m))
            )
        end
    end

    println("theta:\n", theta)
    println("theta_hat:\n", sparse(current(m)))
    println("l1 norm: ", norm(current(m),1))
    println("l2 norm: ", norm(current(m),2))
    println("non-zero: ", count(eps().<abs.(current(m))))

    println("true error: ", norm(A*theta-b))
    println("ols error (overfitted): ", norm(A*thetaLinReg-b))
    println("madam error: ", norm(A*current(m)-b))

    println("ols params error: ", norm(thetaLinReg-theta))
    println("zero params error: ", norm(theta))
    println("madam params error: ", norm(current(m)-theta))

    @test norm(current(m) .- theta) < norm(theta)*0.5
    @test norm(A*current(m) .- b) < norm(b)
    @test count(eps().<abs.(current(m))) < nnz(theta)*2
end

@testset_filtered "Sparse matrix - prox_L1 - gradient-free" begin
    m = Madam(zeros(N); alpha = 0.1)
    maxIter = Int(1e3)
    @time for ii = 0:maxIter
        rowIdx      = rand(1:length(b), 32)
        ai          = view(A,rowIdx,:)
        bi          = view(b,rowIdx)
        batch_obj(x) = objective(ai, bi, 0.0, x)
        update!(m, batch_obj; grad_samples = N>>4, l1_penalty = 2*l1_penalty)
        if ii%Int(div(maxIter,20))==0
            @printf("%10d %10d %10.2f %10.2f %10.2f\n"
                , ii
                , count(eps().<abs.(current(m)))
                , norm(current(m)-theta)
                , norm(A*current(m)-b)
                , objective(A, b, 2*l1_penalty, current(m))
            )
        end
    end

    println("theta:\n", theta)
    println("theta_hat:\n", sparse(current(m)))
    println("l1 norm: ", norm(current(m),1))
    println("l2 norm: ", norm(current(m),2))
    println("non-zero: ", count(eps().<abs.(current(m))))

    println("true error: ", norm(A*theta-b))
    println("ols error (overfitted): ", norm(A*thetaLinReg-b))
    println("madam error: ", norm(A*current(m)-b))

    println("ols params error: ", norm(thetaLinReg-theta))
    println("zero params error: ", norm(theta))
    println("madam params error: ", norm(current(m)-theta))

    @test norm(current(m) .- theta) < norm(theta)*0.5
    @test norm(A*current(m) .- b) < norm(b)
    @test count(eps().<abs.(current(m))) < nnz(theta)*2
end

# Nelder-Mead batch style
# using Optim
# @testset_filtered "Sparse matrix - prox_L1 - gradient: NelderMead batch style" begin
#     maxIter = 10
#     opts = Optim.Options(iterations = 10^4, show_trace = false, show_every = 1000)
#     local theta_hat = zeros(size(theta))
#     local res
#     @time for ii = 0:maxIter
#         rowIdx      = rand(1:length(b), 128+0*Int(round(32.0*ii/maxIter)))
#         ai          = view(A,rowIdx,:)
#         bi          = view(b,rowIdx)
#         batch_obj(x) = norm(ai*x - bi)
#         res = @time optimize(batch_obj, theta_hat, NelderMead(), opts)
#         theta_hat = Optim.minimizer(res)
#         if ii%Int(1e3)==0
#             @printf("%10d %10.2f %10.2f %10.2f\n"
#                 , ii
#                 , norm(A*theta_hat-b)
#                 , norm(theta_hat-theta)
#                 , objective(A, b, l1_penalty, theta_hat)
#             )
#         end
#     end
#     println("l1 norm: ", norm(theta_hat,1))
#     println("l2 norm: ", norm(theta_hat,2))
# 
#     println("true error: ", norm(A*theta-b))
#     println("ols error (overfitted): ", norm(A*thetaLinReg-b))
#     println("nelder-mead error: ", norm(A*theta_hat-b))
# 
#     println("ols params error: ", norm(thetaLinReg-theta))
#     println("zero params error: ", norm(theta))
#     println("nelder-mead params error: ", norm(theta_hat-theta))
# 
#     @test norm(theta_hat .- theta) < norm(theta)/2
#     @test norm(A*theta_hat .- b) < norm(b)/2
# end
