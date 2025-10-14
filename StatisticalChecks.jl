# Statistical checks
using Random
""" This function returns a row of means of a given mask
    Example:
    m=CenterOfMask(msk);
"""
function CenterOfMask(msk::Array{Float64,2})::Array{Float64,1}
    a=findall(msk.>0);
    pixels=zeros(Int, length(a),2)
    for i=1 : length(a)
        pixels[i,:] =  [a[i][1], a[i][2]]
    end
    Centers= mean(pixels, dims=1);
    return  [Centers[1,1], Centers[1,2]]
end

""" This function computes Moran's I, given particular clustering
    Usage:
     M_I=Moran_I(Clusters,Centers);
"""
function Moran_I(Clusters::Array{Array{Int64,1},1}, Centers ::Array{Array{Float64,1},1})::Float64
    # First we compute the weight matrix W, so that w_ij =1 if i \neq  j  and belong to the same cluster
    # otherwise W=0
    N=length(Centers);
    Nclusters=length(Clusters);
    w=zeros(Int,N,N)
    for i=1:N
        for j=i+1:N
            # here we detect if both i and j belong to the same cluser
              w[i,j]=any([(i in Clusters[c]) && (j in Clusters[c]) for c=1:Nclusters]);
              w[j,i]= w[i,j]
        end
    end
   x1=[Centers[i][1] for i=1:N]; x2=[Centers[i][2] for i=1:N];
   Y=hcat(x1.-mean(x1), x2.-mean(x2));
   C= Y*Y'; # this is the matrix of dot products of Y[i,:]
   Morans_I=N/sum(w)* (sum(w.*C)/ LinearAlgebra.tr(C))
   return Morans_I
end

"""
Dunn=Dunn(Clusters,Centers)
"""
function Dunn(Clusters::Array{Array{Int64,1},1}, Centers ::Array{Array{Float64,1},1})::Float64
    # First we compute the weight matrix W, so that w_ij =1 if i \neq  j  and belong to the same cluster
    # otherwise W=0

    N=length(Centers);
	 nonzero_clust_inds=findall(length.(Clusters).>1)
	 Clusters=Clusters[nonzero_clust_inds]
	 Nclusters=length(Clusters);
	 x1=[Centers[i][1] for i=1:N]; x2=[Centers[i][2] for i=1:N];
	 Intracluster_dist=Array{Float64,1}(undef,Nclusters)
	 for i=1:Nclusters
		 x1_diff=x1[Clusters[i]].-mean(x1[Clusters[i]])
		 x2_diff=x2[Clusters[i]].-mean(x2[Clusters[i]])
		 Intracluster_dist[i]=sum(sqrt.(x1_diff.^2 .+ x2_diff.^2))/length(Clusters[i])
	 end
	 Intercluster_dist=zeros(Nclusters,Nclusters)
	 centroids_vector=Array{Array{Float64},1}(undef,Nclusters)
	 for i=1:Nclusters
		 	centroids_vector[i]=[mean(x1[Clusters[i]]),mean(x2[Clusters[i]])]
	 end
    for i=1:Nclusters
		 for j=1:Nclusters
			 Intercluster_dist[i,j]=norm(centroids_vector[i]-centroids_vector[j])
		 end
	 end
	 for i=1:Nclusters
		 Intercluster_dist[i,i]=1000
	 end
	 Dunn_coef=findmin(Intercluster_dist)[1]/findmax(Intracluster_dist)[1]
    return Dunn_coef
end

"""
Gearys_C=Geary_C(Clusters,Centers)
"""
function Geary_C(Clusters::Array{Array{Int64,1},1}, Centers ::Array{Array{Float64,1},1})::Float64
    # First we compute the weight matrix W, so that w_ij =1 if i \neq  j  and belong to the same cluster
    # otherwise W=0
    N=length(Centers);
    Nclusters=length(Clusters);
    w=zeros(Int,N,N)
    for i=1:N
        for j=i+1:N
            # here we detect if both i and j belong to the same cluser
              w[i,j]=any([(i in Clusters[c]) && (j in Clusters[c]) for c=1:Nclusters]);
              w[j,i]= w[i,j]
        end
    end
   x1=[Centers[i][1] for i=1:N]; x2=[Centers[i][2] for i=1:N];
   Y=hcat(x1.-mean(x1), x2.-mean(x2));
   C= Y*Y'; # this is the matrix of dot products of Y[i,:]

   M=zeros(Float64,N,N);
   for i=1:N
       for j=1:N
           M[i,j]=w[i,j]* dot(Y[i,:]-Y[j,:],Y[i,:]-Y[j,:])
       end
   end
   Gearys_C=(N-1)/(2*sum(w))* (sum(M)/ LinearAlgebra.tr(C))
   return   Gearys_C
end


""" This function shuffles every row of A
    B=ShuffleEachRow(A)
"""
function ShuffleEachRow_new(A::AbstractArray{Float64,2})::AbstractArray{Float64,2}
    m,n= size(A)
    B=copy(A)
    B=B[shuffle(1:end),:]
    return B
end

function ShuffleEachRow(A::AbstractArray{Float64,2})::AbstractArray{Float64,2}
    m,n= size(A)
    B=copy(A)
     # for i=1:m
     #     B[i,:]=shuffle(B[i,:])
     # end

# instead, we shuffle every column
 for j=1:n
     B[:,j]=shuffle(B[:,j])
  end


    return B
end
