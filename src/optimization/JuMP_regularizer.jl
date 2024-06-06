

function JuMP_regularization(a::PSDModelPolynomial{d, T}, model, B;
        λ_1=0.0, λ_2=0.0, λ_id=0.0, λ_custom=0.0, cust_regu_func=nothing) where {d, T<:Number}
    
    JuMP.@expression(model, expr_l1, nuclearnorm(B))
    if λ_2 > 0.0
        JuMP.@expression(model, expr_l2, sum(B[i,j]^2 for i=1:N, j=1:N))
    else
        JuMP.@expression(model, expr_l2, 0.0)
    end
    
    if λ_id > 0.0
        JuMP.@expression(model, expr_id, _JuMP_identity_map_regularizer(a, model, B))
    else
        JuMP.@expression(model, expr_id, 0.0)
    end

    if λ_custom > 0.0 && cust_regu_func !== nothing
        JuMP.@expression(model, expr_custom, cust_regu_func(B))
    else
        JuMP.@expression(model, expr_custom, 0.0)
    end

    JuMP.@expression(model, expr_regul, λ_1 * expr_l1 + λ_2 * expr_l2 + λ_id * expr_id + λ_custom * expr_custom)
    return expr_regul
end

"""
SoS identity map regularizer, exact for SoS polynomials.
"""
function _JuMP_identity_map_regularizer(a::PSDModelPolynomial{d, T}, model, B) where {d, T<:Number}
    integrated, _B_sym = _compute_symbolic_idmap_regularizer(a)
    expr = Sym.substitute(integrated, _B_sym=>B)
    JuMP.@expression(model, ex_id, expr - 2.0 * tr(B))
    return ex_id
end

# function _JuMP_custom_regularizer(cust_regu_func, model, B) where {T<:Number}
#     JuMP.@expression(model, ex_custom, cust_regu_func(B))
#     return ex_custom
# end