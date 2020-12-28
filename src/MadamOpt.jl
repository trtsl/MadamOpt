module MadamOpt

export Madam, step!, step_ols!, update!, update_ols!, predict_ols, current

using LinearAlgebra, Random

abstract type AbstractAdam end

mutable struct Madam{T} <: AbstractAdam
    alpha::Float64                      # step-size
    beta1::Float64                      # 1st moment moving average retention factor 
    beta2::Float64                      # 2nd moment moving average retention factor
    beta3::Float64                      # gradient decay in gradient-free usages
    eps::Float64                        # don't divide by zero
    soft_clip::Float64                  # a gradient-clipping factor where Inf can be used to emulate standard Adam (only use without L1 regularization)
    max_temp::Float64                   # incremental temperature parameter for non-convex problems
    max_steps::Union{UInt64, Nothing}   # temperate decreases until t==max_steps
    dx::Float64                         # controls gradient estimates in gradient-free settings
    t::UInt64                           # timestep
    theta::T                            # parameter vector
    m::T                                # 1st moment vector
    v::T                                # uncentered 2nd moment vector
    a::T                                # for discrete gradient approximation
end

"""
    Madam{T}(
        theta
        ; alpha     = 0.01
        , beta1     = 0.9
        , beta2     = 0.999
        , beta3     = 0.9
        , eps       = 1e-8
        , max_temp  = 0.0
        , max_steps = nothing
        , dx        = 1e-8
    )

# Arguments
- `theta::T`: the initial parameters (i.e. staring point).
- `alpha::Float64`: learning-rate / step-size.
- `beta1::Float64`: controls the exponential decay rate for the 1st moment.
- `beta2::Float64`: controls the exponential decay rate for the 2nd moment.
- `beta3::Float64`: in gradient-free usage, `beta3` determines over what period
  a gradient approximation that was not recently sampled should decay
- `eps::Float64`: small-constant for numerical stability.
- `max_temp::Float64`: starting temperature representing the search space
  perturbations used to estimate the gradient; the perturbations will approach
  zero when `max_steps` is reached; setting this parameter can help with
  non-convex problems
- `max_steps::Union{Integer, Nothing}`: if an `Integer` is set, the algorithm
  will estimate the gradient using a larger region around the current estimate to
  allow optimizing non-convex functions
- `dx::Float64`: the steady state perturbations used to estimate the gradient
  after `max_steps` have been taken

Construct the Madam optimizer.
"""
Madam(
    theta::T
    ; alpha::Float64                        = 0.01
    , beta1::Float64                        = 0.9
    , beta2::Float64                        = 0.999
    , beta3::Float64                        = 0.9
    , eps::Float64                          = 1e-8
    , soft_clip::Float64                    = 1.0
    , max_temp::Float64                     = 0.0
    , max_steps::Union{Integer, Nothing}    = nothing
    , dx::Float64                           = 1e-8
) where T = begin
    t           = 0
    m           = zero(theta)
    v           = zero(theta)
    a           = zero(theta)
    
    Madam{T}(alpha,beta1,beta2,beta3,eps,soft_clip,max_temp,max_steps,dx,t,theta,m,v,a)
end

step_adam!(m::Madam{T}, grad, idxs = :) where {T} = begin
    m.t += 1
    # calculate moving averages of the gradient and squared gradient
    m.m[idxs]  .= m.beta1.*m.m[idxs] .+ (1.0-m.beta1).*grad[idxs]
    m.v[idxs]  .= m.beta2.*m.v[idxs] .+ (1.0-m.beta2).*grad[idxs].^2
    m.a       .*= m.beta3
    m.a[idxs]  .= 1.0

    return m
end

# implements a function that basically performs soft gradient clipping
# parameterized by `m` where for x>=0: df/dx = m/(m+x); note that standard
# gradient clipping clips the input gradient such that `m.m` and `m.v`
# would be based on clipped gradients; the implementation below clips the
# output gradients, preserving the full variance of the parameters to be
# reflected in `m.v`
function log_scale(x, m)
    @assert(m>0.0)
    isinf(m) ? abs.(x) : m.*log.(1.0 .+ abs.(x)./m)
end

calc_step(m::Madam) = begin
    # NB: there are some differences here form standard Adam; in particular,
    # Adam does not include the log_scale term, which is used to allow the
    # proximal operator to be effective (`mHat ./ sqrt.(vHat)` is always
    # between -1.0 and 1.0, assuming `beta1==beta2`)
    #
    # bias adjustments
    mHat        = m.m./(1.0-m.beta1^m.t)
    vHat        = m.v./(1.0-m.beta2^m.t)
    # the adam paper refers to the ratio mHat/sqrt(vHat) as the signal-to-noise
    # ratio; eps is used for numerical stability
    canonical_step = -m.alpha .* mHat ./ (sqrt.(vHat) .+ m.eps)
    # the `log_scale` adjustment is required for the proximal step and `adam.a`
    # reflects an adjustment related to sampling of the gradient in
    # gradient-free settings
    return log_scale(mHat, m.soft_clip) .* m.a .* canonical_step
end

_step!(
    m::Madam{T}
    , grad
    ; l1_penalty::Union{Nothing, Float64} = nothing
    , l2_penalty::Union{Nothing, Float64} = nothing
    , weight_decay::Union{Nothing, Float64} = nothing
    , idxs = :
) where T = begin
    # for l2_penalty / weight_decay details see:
    # https://arxiv.org/pdf/1711.05101.pdf
    # https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
    if !isnothing(l2_penalty)
        grad .+= l2_penalty.*m.theta
    end
    step_adam!(m, grad, idxs)
    if !isnothing(weight_decay)
        m.theta .*= (1.0 .- weight_decay*m.alpha.*m.a)
    end
    # adam.theta .+= calc_step(adam)
    adamStep = calc_step(m)
    if !isnothing(l1_penalty)
        # adam.theta .= prox_L1(adam.theta, l1_penalty*adam.alpha.*adam.a)
        adamStep .+= prox_L1_adj(m.theta.+adamStep, l1_penalty*m.alpha.*m.a)
    end

    # FluxML requires the step to be returned (rather than updating `adam.theta` directly)
    adamStep
end

"""
    step!(
        adam
        , grad
        ; l1_penalty    = nothing
        , l2_penalty    = nothing
        , weight_decay  = nothing
    )

Advances the algorithm by one step based on the current gradient, but does *not* update
theta estimates.  This method is included for interfaces such as those of `FluxML` which
require that only the step size is returned.

# Arguments
- `adam::Madam{T}`
- `grad::T`: the gradient for the next set of observations (i.e mini-batch)
- `l1_penalty::Union{Nothing, Float64}`: penalty for L1 regularization
  (implemented via the proximal operator)
- `l2_penalty::Union{Nothing, Float64}`: penalty for L2 regularization
- `weight_decay::Union{Nothing, Float64}`: weight decay regularization
"""
step!(
    m::Madam{T}
    , grad
    ; l1_penalty::Union{Nothing, Float64} = nothing
    , l2_penalty::Union{Nothing, Float64} = nothing
    , weight_decay::Union{Nothing, Float64} = nothing
) where T = _step!(
    m
    , grad
    ; l1_penalty = l1_penalty
    , l2_penalty = l2_penalty
    , weight_decay = weight_decay
)

"""
    step!(adam, loss; grad_samples=max(1,length(adam.theta) >> 4), kwargs...)

Advances the algorithm by one step based on a loss function.

# Arguments
- `adam::Madam{T}`
- `loss::Function`: objective or loss function (e.g. mini-batch loss)
- `grad_samples::Integer`: The number of samples taken from the loss function
  to estimate the gradient.  This value must be in the range `0 <= grad_samples
  <= length(parameters)`.
- `kwargs....`: `l1_penalty`, `l2_penalty`, and `weight_decay` penalties described above
"""
function step!(
    m::Madam
    , loss::Function
    ; grad_samples::Integer = max(1,length(m.theta) >> 4)
    , kwargs...
)
    (grad, idxs) = gradient_approx(m, loss, grad_samples)
    _step!(m, grad; idxs=idxs, kwargs...)
end

"""
    update!(adam, args...; kwargs...)

Calls [`step!`](@ref) and then also updates the parameter estimates `theta`.
"""
update!(m, args...; kwargs...) = m.theta .+= step!(m, args...; kwargs...)

"""
    current(adam::Madam)

Retrieves a reference to the current parameter estimates (i.e. `theta` in
constructor [`Madam`](@ref)).
"""
current(m::Madam) = m.theta

# multiplication by 2.0 is not strictly necessary here, but included in case
# these functions are used in other contexts
grad_ols(m::Madam, x::AbstractArray{Float64,2}, y::AbstractArray{Float64,1}) = 2.0*x'*(x*m.theta.-y)
grad_ols(m::Madam, x::AbstractArray{Float64,1}, y::Float64                 ) = 2.0*x*(dot(x,m.theta)-y)

"""
    step_ols!(adam, A, b; kwargs...)

For linear models of the form `Ax = b`, this is a convenience method that
calculates the gradient given `A` and `b`.

# Arguments
- `adam::Madam`
- `A::AbstractArray{Float64, Union{1,2}}`: Design / independent variables.
- `A::Union{AbstractArray{Float64, 1}, Float64}`: Response / dependent variables.
- `kwargs....`: same keyword arguments as for `step!`.
"""
step_ols!(m::Madam, x, y; kwargs...) = step!(m, grad_ols(m,x,y); kwargs...)

"""
    update_ols!(adam, args...; kwargs...)

Calls [`step_ols!`](@ref) and then also updates the parameter estimates `theta`.
"""
update_ols!(m, args...; kwargs...) = m.theta .+= step_ols!(m, args...; kwargs...)

predict_ols(m::Madam, x::AbstractArray{Float64,2}) = x*m.theta
predict_ols(m::Madam, x::AbstractArray{Float64,1}) = dot(x,m.theta)

"""
Soft-thresholding operator for proximal gradient
"""
# A nice explanation of iterative shrinkage thresholding is:
# https://towardsdatascience.com/unboxing-lasso-regularization-with-proximal-gradient-method-ista-iterative-soft-thresholding-b0797f05f8ea
# Here, we use a faster version of: sign.(u).*max(abs.(u).-t, 0.0)
prox_L1(u, t) = max.(0.0,u.-t) .+ min.(0.0,u.+t)
prox_L1_adj(u ,t) = max.(min.(-u,t),-t)


function gradient_approx(
    m::Madam
    , loss::Function
    , grad_samples::Integer
)
    l = length(m.theta)
    @assert(0 < grad_samples <= l)
    grad = zeros(l)
    y = loss(m.theta)
    idxs = view(Random.randperm(l), 1:grad_samples)
    for idx in idxs
        dx = m.dx
        if m.max_steps != nothing
            h = huber_loss(m.t, m.max_steps, m.max_temp)
            dx += h # * abs(randn())
        end
        if rand() < 0.5
            dx *= -1.0
        end
        m.theta[idx] += dx
        try
            grad[idx] = (loss(m.theta) - y) / dx
        finally
            m.theta[idx] -= dx
        end
    end
    (grad, idxs)
end

function huber_loss(t, max_steps, max_temp)
    # temperate based on huber loss (i.e. temperature changes are sublinear as
    # t approaches max_steps)
    # https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
    if t > max_steps
        t = max_steps
    end
    x = max_temp * (1.0 - t/max_steps)
    del = max_temp / 2.0
    del^2*(sqrt(1.0+(x/del)^2) - 1.0)
end

end # module
