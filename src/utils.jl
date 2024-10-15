
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


## threaded map
@inline function map_threaded(f::Function, A::AbstractArray{T}) where {T<:Number}
    return map_threaded(T, f, A)
end

@inline function map_threaded(f::Function, A::AbstractArray{<:Vector{T}}) where {T<:Number}
    return map_threaded(Vector{T}, f, A)
end

## threaded map
function map_threaded(T, f::Function, A::AbstractArray)
    res = Array{T}(undef, size(A))
    Threads.@threads for i=1:length(A)
        res[i] = f(A[i])
    end
    return res
end

## threading based on variable, 
## see https://discourse.julialang.org/t/putting-threads-threads-or-any-macro-in-an-if-statement/41406/6
macro _condusethreads(multithreaded, expr::Expr)
    ex = quote
        if $multithreaded
            Threads.@threads $expr
        else
            $expr
        end
    end
    esc(ex)
end

## slicing
slice_matrix(A::Array{T}) where {T<:Number} = ndims(A)>1 ? Vector{T}[c for c in eachcol(A)] : A
unslice_matrix(A::Vector{Vector{T}}) where {T<:Number} = reduce(hcat, A)

## norm utilitys

nuclearnorm(A::AbstractMatrix) = tr(A)


### macro
macro _StaticArrayAppend(A, a::Int, b::Int, z)
    str = "SA[ "

    for i = a:1:b
        str = "$str $A[$i], "
    end
    str = "$str $z]"
    # return str
    expr = Meta.parse(str)
    return esc(expr)
end