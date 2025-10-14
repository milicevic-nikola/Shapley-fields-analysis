include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Load_data_from_the_files_all_omp8x_no_amines.jl"));
#include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Load_the_data_from_the_files_no_amines.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/More_functions.jl"));
using JLD;
#relevant_odor_names = vcat(1:26,32:59);

#using DelimitedFiles
#descr_vals_norm = readdlm(joinpath(homedir(),"Dropbox/Nikola-Vladimir/learned_descrs.csv"), ',', Float64);

#schmuker cactvs and chemsense descriptors
#schmuker_descr_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/cactvs_our_odors.csv");
#descr_cact, descr_cact_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_descr_filename));
#descr_cact, descr_cact_names = load_descriptors_no_amines(joinpath(homedir(),schmuker_descr_filename),relevant_odor_names);
#schmuker_descr_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/chemsense_our_odors.csv");
#descr_chemsense, descr_chemsense_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_descr_filename));


#descr_vals_norm=hcat(descr_chemsense,descr_cact)#,descr_MACCS);
#descr_vals_norm=descr_vals_norm[setdiff(1:59,26:32),:];

#descr_names=vcat(descr_chemsense_names,descr_cact_names);

#descr_vals_norm=descr_vals_norm[setdiff(1:59,26:32),:];

#unique_descr_idx=unique(z -> descr_vals_norm[:,z], 1:size(descr_vals_norm,2));
#descr_vals_norm=descr_vals_norm[:,unique_descr_idx];
#descr_names=descr_names[unique_descr_idx];


#cor_feat=DataFrame(cor_feat,descr_names,makeunique=true);

#cor_feat=cor(descr_vals_norm);
#for i in 1:size(cor_feat,1)
#	cor_feat[i,i]=0
#end
#high_cor_feat = cor_feat.>0.9
#high_cor_feat=sum(high_cor_feat, dims=1)[1,:]
#high_cor_idx=findall(high_cor_feat.!=0)
#high_cor_idx=high_cor_idx[8:100]
#descr_vals_norm=descr_vals_norm[:,setdiff(1:end,high_cor_idx)]
#descr_names=descr_names[setdiff(1:end,high_cor_idx)]

#=
using Counters;
function corr_cleaner(features::Array{Float64,2},corr_cutoff::Float64)
	abs_corr_matrix = abs.(cor(features))
	filtered_cols = []
	while true
		offenders=[]
		for i in 1:size(abs_corr_matrix,1)
			for j in 1:size(abs_corr_matrix,2)
				if i != j
					if abs_corr_matrix[i,j] > corr_cutoff
						push!(offenders, features[,:i])
					end
				end
			end
		end
		if length(offenders) > 0:
            c = counter(offenders)
		end
	end
=#
#outfile = joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chemsense_and_cactvs_descrs.txt")
#open(outfile, "w") do f
#	for descr in descr_names
#		println(f, descr)
#	end
#end

UseDatasets=[1,2,3];
Ndatasets=length(UseDatasets);

#don't need to run the following if loading the non-amines response matrices, it was already taken care of by the loading function
for i in 1:length(resp_matr)
	resp_matr[i]=resp_matr[i][nonzero_rows[i],:]
end

Nglomeruli= zeros(Int,length(UseDatasets));

for i in 1:Ndatasets
    resp_matr[UseDatasets[i]]=normalize_matrix_by_max(resp_matr[UseDatasets[i]],dim=1)
end

centroids=Array{Array{Float64,2},1}(undef,Ndatasets);

for i in 1:Ndatasets
    centers=zeros(Float64,length(masks[i]),2)
    for j in 1:length(masks[i])
        (min_x,max_x,min_y,max_y)=window_matr(masks[i][j])
        x=ceil(min_x-64+(max_x-min_x)/2)
        y=ceil(min_y-64+(max_y-min_y)/2)
        centers[j,1]=x
        centers[j,2]=y
    end
                            #centroids[i]=centers
    centroids[i]=convert(Array{Float64},centers)
end


glomeruli_positions=Array{Array{Float64,2},1}(undef,Ndatasets);
for i in 1:Ndatasets
    glomeruli_positions[i]=centroids[i][:,1:2]
end

normalized_glomeruli_positions=Array{Array{Float64,2},1}(undef,Ndatasets);
for i in 1:Ndatasets
    normalized_glomeruli_positions[i]=normalize_matrix_zscore(centroids[i][:,1:2]);
end

glomeruli_areas=Array{Any,1}(undef,Ndatasets);
for i in 1:Ndatasets
    #glomeruli_areas[i]=fill(0.0,size(masks[i],2))
	glomeruli_areas[i]=fill(0.0,length(masks[i]))
    for j in 1:length(glomeruli_areas[i])
        glomeruli_areas[i][j]=sum(masks[i][j])
    end
    glomeruli_areas[i]=glomeruli_areas[i]./findmax(glomeruli_areas[i])[1];
end

binary_responses=Array{BitArray{2},1}(undef,Ndatasets);
for i in 1:length(binary_responses)
    binary_responses[i]=resp_matr[UseDatasets[i]].!=0.0
end

save(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"binary_responses",
										binary_responses,"gloms_positions",normalized_glomeruli_positions,
                                        "gloms_areas",glomeruli_areas,"resp_matrices",resp_matr,"masks",masks,
										"glom_pos",glomeruli_positions);
					


#=
descr_distance_metric = Distances.euclidean;
descr_distance_metric = Distances.hamming;


descr_distances= distance_matrix(descr_vals_norm, descr_distance_metric);

descr_linkage = :average;

descr_clustering_obj = hclust(descr_distances, descr_linkage);

height=5.0;
cluster_idxs=Int64.(cutree(descr_clustering_obj,h=height));
extremums = Vector(undef,length(unique(cluster_idxs)));
for (k,i) in enumerate(unique(cluster_idxs))
    extremums[k] = extrema([findfirst(x -> x .== j,descr_clustering_obj.order) for j in findall(x -> x ==i,cluster_idxs)]) .+ (-0.5,0.5)
end

Plots.plot([Shape([m,M,M,m],[0,0,height,height]) for (m,M) in extremums]; alpha = 0.5)
Plots.plot!(descr_clustering_obj, xticks = (1:length(cluster_idxs),string.(descr_clustering_obj.order)), xlabel="node", ylabel=string(descr_clustering_obj.linkage) * " linkage distance")
unique_clust_nums = unique(cluster_idxs);
clusters = Vector{Vector{Int}}(undef,length(unique_clust_nums));

for i in unique_clust_nums
	 clusters[i] = VERSION>= v"0.7.0" ? findall(x->x==i,cluster_idxs) : find(x->x==i,cluster_idxs);
end


height=0.95;

cluster_idxs=Int64.(cutree(descr_clustering_obj,h=height));

unique_clust_nums = unique(cluster_idxs);

descr_clusters = Vector{Vector{Int}}(undef,length(unique_clust_nums));

for i in unique_clust_nums
	 descr_clusters[i] = VERSION>= v"0.7.0" ? findall(x->x==i,cluster_idxs) : find(x->x==i,cluster_idxs);
end

good_descriptors = select_descriptors(descr_clusters, descr_vals_norm);

good_descr_vals = descr_vals_norm[:, good_descriptors];
good_descr_names=String.(descr_names[good_descriptors]);

=#
