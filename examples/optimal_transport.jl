using SequentialMeasureTransport
import SequentialMeasureTransport as SMT
using ApproxFun
using LinearAlgebra
using Distributions
using Plots
using Hypatia

function compute_Sinkhorn(rng, left_marg_d, right_marg_d, c, ϵ; iter=100)
    M_epsilon = [exp(-c([x,y])/ϵ) for x in rng, y in rng]
    left_mat = [left_marg_d([x]) for x in rng]
    right_mat = [right_marg_d([x]) for x in rng]
    left_margin(M) = M * ones(size(M, 1))
    right_margin(M) = M' * ones(size(M, 2))
    for _=1:iter
        M_epsilon = diagm(left_mat ./ left_margin(M_epsilon)) * M_epsilon
        M_epsilon = M_epsilon * diagm(right_mat ./ right_margin(M_epsilon))
    end
    M_epsilon = length(rng)^2 * M_epsilon / sum(M_epsilon)
    # for (i, ii) in enumerate(rng), (j, jj) in enumerate(rng)
    #     M_epsilon[i, j] *= 1/(left_marg_d([ii]) * right_marg_d([jj]))
    # end
    return M_epsilon
end



c(x) = (x[1] - x[2])^2

left_dist = MixtureModel(Normal[
    Normal(0.3, 0.1),
    Normal(0.7, 0.1)
])
right_dist = Normal(0.5, 0.2)
left_marg_d(x) = pdf(left_dist, x[1])
right_marg_d(x) = pdf(right_dist, x[1])

## model creation
model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 4)
model.B .= Hermitian(rand(size(model.B)...))

left_marg = SMT.marginal_model(model, [2])
right_marg = SMT.marginal_model(model, [1])


# transport cost
N_XY = 5000
c(x, y) = (x - y)^2
c(x) = c(x[1], x[2])
XY = rand(2, N_XY)
C = map(c, eachcol(XY))
ϵ = 0.3
ξ_c = map(x->exp(-c(x)/ϵ), eachcol(XY))

X_left = rand(left_dist, 2000)
X_right = rand(right_dist, 2000)

e_1 = zeros(2)
e_1[1] = 1
e_2 = zeros(2)
e_2[2] = 1
D, C = SMT.get_semialgebraic_domain_constraints(model)
SMT._OT_JuMP!(model, eachcol(XY), ξ_c,
            optimizer=Hypatia.Optimizer,
            λ_2 = 0.0, λ_marg_reg=10000.0,
            normalization=false, trace=true,
            mat_list=D, coef_list=C,
            # marg_regularization=[(left_marg, new_model_left.B), (right_marg, new_model_right.B)],
            marg_data_regularization=[(e_1, eachrow(X_left)), (e_2, eachrow(X_right))],)



rng = 0.0:0.01:1.0
contourf(rng, rng, (x, y) -> exp(-c([x,y])/ϵ), alpha=0.5, label="cost")
contourf(rng, rng, (x, y) -> model([x, y]), alpha=1.0, label="transported model")
left_marg2 = SMT.marginal_model(model, [2])
right_marg2 = SMT.marginal_model(model, [1])
normalize!(model)
plot(rng, x->left_marg2([x]), label="left")
plot!(rng, x->left_marg_d(x), label="left orig")
plot(rng, x->right_marg2([x]), label="right")
plot!(rng, x->right_marg_d(x), label="right orig")

M_sink = compute_Sinkhorn(rng, left_marg_d, right_marg_d, c, 0.3)
contourf(M_sink')