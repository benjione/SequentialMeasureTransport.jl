
## kwargs utils
@inline _filter_kwargs(kwargs, filt_keys, ignore) = 
            kwargs[setdiff(intersect(filt_keys, keys(kwargs)), ignore)]
@inline _filter_kwargs(kwargs, filt_keys) = 
            kwargs[intersect(filt_keys, keys(kwargs))]

### Matrix utils
function Hermitian_to_low_vec(A::AbstractMatrix)
    return A[tril!(ones(Bool, size(A)))]
end

function view_mat_for_to_symmetric(N)
    A = zeros(Int, N,N)
    A[tril!(ones(Bool, size(A)))] = collect(1:N*(N+1)/2)
    return Symmetric(A, :L)
end


function low_vec_to_Symmetric(vec::AbstractVector)
    N = Int(round(sqrt(2*length(vec))))
    return low_vec_to_Symmetric(vec, view_mat_for_to_symmetric(N))
end
@inline low_vec_to_Symmetric(vec::AbstractVector, vm::AbstractMatrix) = @view vec[vm]
