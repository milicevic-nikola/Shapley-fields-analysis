#define functions for picking the n max or n min
#elements from an array
function findpartialsort(v,k; opts...)
    inds = partialsortperm(v,k;opts...)
    length(inds) == 1 ? (inds,v[inds]) : (inds,v[[inds...]])
end

Base.findmax(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts..., rev=true)
Base.findmin(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts...)


#function for taking union of two vectors
function union_vec(x::Vector{T}, y::Vector{T})::Vector{T} where {T}
    (nx, ny) = (1, 1)
    z = T[]
    while nx <= length(x) && ny <= length(y)
        if x[nx] < y[ny]
            push!(z, x[nx])
            nx += 1
        elseif y[ny] < x[nx]
            push!(z, y[ny])
            ny += 1
        else
            push!(z, x[nx])
            nx += 1
            ny += 1
        end
    end
    if nx <= length(x)
        [ push!(z, x[n]) for n = nx:length(x) ]
    elseif ny <= length(y)
        [ push!(z, y[n]) for n = ny:length(y) ]
    end
    return z
end


not_in(inds::AbstractVector{Int}, n::Int)::Vector{Int} = setdiff(1:n, IntSet(inds))

function descr_as_random_variable(masks::Array{Any,2},glom_vs_descr_matrix::Array{Float64,2})
    descr_random_variables=Array{Array{Float64,2},1}(undef,size(glom_vs_descr_matrix,2))
    for l in 1:size(glom_vs_descr_matrix,2)
        descr_random_variable=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                        #animal_1_glom[i,j]=max(animal_1_glom[i,j],glom_vs_descr_matrix[k,idx[l]],0.0000001)
                        descr_random_variable[i,j]=descr_random_variable[i,j]+glom_vs_descr_matrix[k,l]
#                        animal_1_glom[i,j]=max(animal_1_glom[i,j],glom_vs_descr_matrix[k,idx[l]]-2*std(glom_vs_descr_matrix[:,idx[l]]),0.0000001)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(descr_random_variable)
        descr_random_variable=descr_random_variable[min_y:max_y,min_x:max_x]
        descr_random_variables[l]=descr_random_variable
    end
    #need to find min windrow of matrix first
    return descr_random_variables
end