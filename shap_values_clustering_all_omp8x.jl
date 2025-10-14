using PyCall,JLD, Clustering, DataFrames, LinearAlgebra, Statistics,MultivariateStats;#, MLBase;

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/StatisticalChecks.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files//Initialize_analyses.jl"));

#load masks for all odors dataset
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");
#load masks for no amines dataset
#masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"masks");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"masks");

#this_descr_by_gloms=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/cact_shaps_on_bulb.jld"),"shap_per_glom_matrix");
this_descr_by_gloms=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/cactno_amines_shaps_on_bulb_all_omp8x.jld"),"shap_per_glom_matrix");

#descr_by_gloms =this_descr_by_gloms[1]
#this_descr_by_gloms_copy=this_descr_by_gloms;
# for i in 1:length(this_descr_by_gloms)
# 	this_descr_by_gloms[i]=this_descr_by_gloms[i][:,1:154]
# end
this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2], this_descr_by_gloms[3]);

M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

descr_by_gloms_transformed=MultivariateStats.transform(M,descr_by_gloms');

descr_by_gloms=copy(descr_by_gloms_transformed');


# pc_projection_vectors=Array{Any,1}(undef,5);
# for i in 1:length(pc_projection_vectors);
# 	#pc_projection_vectors[i]=abs.(projection(M)[1:154,i]).+abs.(projection(M)[155:308,i])
# 	pc_projection_vectors[i]=abs.(projection(M)[1:143,i])#.+abs.(projection(M)[144:286,i])
# end
# p1=histogram(abs.(pc_projection_vectors[1]),grid=false,legend=false,title="Weights for PC1",xtickfontsize=20,ytickfontsize=20,titlefontsize=30);
# p2=histogram(abs.(pc_projection_vectors[2]),grid=false,legend=false,title="Weights for PC2",xtickfontsize=20,ytickfontsize=20,titlefontsize=30);
# p3=histogram(abs.(pc_projection_vectors[3]),grid=false,legend=false,title="Weights for PC3",xtickfontsize=20,ytickfontsize=20,titlefontsize=30);
# p4=histogram(abs.(pc_projection_vectors[4]),grid=false,legend=false,title="Weights for PC4",xtickfontsize=20,ytickfontsize=20,titlefontsize=30);
# p5=histogram(abs.(pc_projection_vectors[5]),grid=false,legend=false,title="Weights for PC5",xtickfontsize=20,ytickfontsize=20,titlefontsize=30);
# l = @layout [a b; c d e]
# Plots.plot(p1,p2,p3,p4,p5,layout=l,size=(2000,2000))
# Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files","pca_loading_vectors_pos_and_neg_shaps_histogram.png"))

#pc_projection_important_idxs=Array{Any,1}(undef,5);

#for i in 1:length(pc_projection_important_idxs)
#	pc_projection_important_idxs[i]=findmax(pc_projection_vectors[i],1:5)[1]
#end

#pc_descr_names=Array{Any,1}(undef,5)
#for i in 1:length(pc_descr_names)
#	pc_descr_names[i]=descr_names[pc_projection_important_idxs[i]]
#end

UseDatasets=[2,3];

for mask in masks[1]
    M = copy(mask)
    for i=1:size(M,1)
        mask[i,:] = M[257-i,:]
    end
end

#glom_distance_metric = Distances.euclidean;
#glom_distance_metric = Distances.corr_dist;
glom_distance_metric = Distances.cosine_dist;

glom_dist = distance_matrix(descr_by_gloms, glom_distance_metric, dim=1);
glom_clustering_obj = hclust(glom_dist, :ward);#,:U,:optimal);
dend_title="The dendrogram of glomeruli response to particular features";
glom_dend = plot_dendrogram_and_matrix(glom_clustering_obj, descr_by_gloms,title=dend_title, tickfontsize=7);
glom_dend = plot!(; size=(1000, 600));
display(glom_dend)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pom21_pca_dendrogram_coss_pos_and_neg.png"))
#Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files","pca_dendrogram_coss_signed.png"))

height=2.5;
cluster_idxs=Int64.(cutree(glom_clustering_obj,h=height));
extremums = Vector(undef,length(unique(cluster_idxs)));
for (k,i) in enumerate(unique(cluster_idxs))
    extremums[k] = extrema([findfirst(x -> x .== j,glom_clustering_obj.order) for j in findall(x -> x ==i,cluster_idxs)]) .+ (-0.5,0.5)
end

#color=[:red,:green,:yellow,:purple,:blue,:cyan,:pink,:white]

# xticks = (1:length(cluster_idxs),string.(glom_clustering_obj.order))
Plots.plot([Shape([m,M,M,m],[0,0,height,height]) for (m,M) in extremums],xticks=false, color=[:red,:green,:yellow,:purple,:blue,:cyan,:pink,:white]; alpha = 1.0)
plot!(glom_clustering_obj,xticks=false,grid=false)#, ylabel=string(glom_clustering_obj.linkage) * " linkage distance")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig2","cact_dendrogram_all_omp8x.svg"))


unique_clust_nums = unique(cluster_idxs);
clusters = Vector{Vector{Int}}(undef,length(unique_clust_nums));
for i in unique_clust_nums
	 clusters[i] = VERSION>= v"0.7.0" ? findall(x->x==i,cluster_idxs) : find(x->x==i,cluster_idxs);
end

Nclusters=length(clusters);
Ndatasets=length(UseDatasets);

#Nglomeruli=[113,108];
Nglomeruli=[107,128,132];

masks[2][128]=Float64.(masks[2][128]);

separate_glom_clusters=Vector{typeof(clusters)}(undef, Ndatasets);
M_I=zeros(Ndatasets);
Gearys_C=zeros(Ndatasets);
for i=1:Ndatasets
    OR_number_shift = (i==1) ? 0 : sum(Nglomeruli[1:i-1]);
    glumeruli_numbers=collect(OR_number_shift+1: OR_number_shift+Nglomeruli[i]);
# first find the clusters that are specific to the dataset
    separate_glom_clusters[i]=map(x-> intersect(x,glumeruli_numbers),  clusters);
# Now we conver the "combined" cluster numbers into the inside-the-dataset cluster numbers:
    for j=1:Nclusters
        separate_glom_clusters[i][j]=map(x->(x-OR_number_shift), separate_glom_clusters[i][j]);
    end
	for j=1:length(separate_glom_clusters[i])
		if length(separate_glom_clusters[i][j])<4
			separate_glom_clusters[i][j]=[]
		end
	end
    actual_mask = masks[i]

    actual_mask =[actual_mask[a] for a=1:length(actual_mask)];  # convert to the type Array{Array{Float64,2},1}

    Nglom=length(actual_mask);
    ClusterCenters=zeros(Nglom,2);
    Centers = CenterOfMask.(actual_mask);
    Clusters = separate_glom_clusters[i];
    M_I[i]=Moran_I(Clusters,Centers);
    Gearys_C[i]=Geary_C(Clusters,Centers);
end

f1=plot_glomeruli_clusters(masks[1],separate_glom_clusters[1]);
f1
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig2","cact_clusters_new_colors_83L.svg"));

f2=plot_glomeruli_clusters(masks[2],separate_glom_clusters[2]);
f2
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig2","cact_clusters_new_colors_83R.svg"));

f3=plot_glomeruli_clusters(masks[3],separate_glom_clusters[3]);
f3
#f2

this_descr_by_gloms_copy[1]=descr_by_gloms[1:Nglomeruli[1],:];
this_descr_by_gloms_copy[2]=descr_by_gloms[Nglomeruli[1]+1:end,:];


glom_clustering_objs = [];
Nglomeruli= zeros(Int,length(UseDatasets));
#Statistical checks


println("____________________________________________________");
Nshuflings=1000;
Shuffled_M_I=zeros(Float64,Nshuflings,Ndatasets);
Shuffled_Gearys_C=zeros(Float64,Nshuflings,Ndatasets);


for t=1:Nshuflings
   println("Trial $t");
   local this_descr_by_gloms=Array{Array{Float64,2},1}(undef,Ndatasets)
   for counter=1:Ndatasets
       local dataset_n =UseDatasets[counter];
       local responses = this_descr_by_gloms_copy[dataset_n];

       #Shuffled_Responses=ShuffleEachRow(responses);
	   Shuffled_Responses=ShuffleEachRow_new(responses);

       this_descr_by_gloms[counter] =Shuffled_Responses;
       Nglomeruli[counter]=size(this_descr_by_gloms[counter],1);
   end

#   descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2],this_descr_by_gloms[3]);
   	descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);#,this_descr_by_gloms[3]);
   	glom_distance_metric = Distances.cosine_dist

	glom_dist = distance_matrix(descr_by_gloms, glom_distance_metric, dim=1);
	glom_clustering_obj = hclust(glom_dist, :ward);
	optimal_height=2.5;

    cluster_idxs=Int64.(cutree(glom_clustering_obj,h=optimal_height));
	unique_clust_nums = unique(cluster_idxs);
	clusters = Vector{Vector{Int}}(undef,length(unique_clust_nums));
	for i in unique_clust_nums
		 clusters[i] = VERSION>= v"0.7.0" ? findall(x->x==i,cluster_idxs) : find(x->x==i,cluster_idxs);
	end
   	Nclusters=length(clusters);
   	separate_glom_clusters= Vector{typeof(clusters)}(undef, Ndatasets);
   	for i=1:Ndatasets
       	OR_number_shift = (i==1) ? 0 : sum(Nglomeruli[1:i-1]);
       	glumeruli_numbers=collect(OR_number_shift+1: OR_number_shift+Nglomeruli[i]);
       	separate_glom_clusters[i]=map(x-> intersect(x,glumeruli_numbers),  clusters);
       	for j=1:Nclusters
           	separate_glom_clusters[i][j]=map(x->(x-OR_number_shift), separate_glom_clusters[i][j]);
       	end

       	actual_mask = masks[i]

       	actual_mask =[actual_mask[a] for a=1:length(actual_mask)];  # convert to the type Array{Array{Float64,2},1}

       	Nglom=length(actual_mask);
       	ClusterCenters=zeros(Nglom,2);
       	Centers = CenterOfMask.(actual_mask);
       	Clusters = separate_glom_clusters[i];
       	Shuffled_M_I[t,i]=Moran_I(Clusters,Centers);
       	Shuffled_Gearys_C[t,i]=Geary_C(Clusters,Centers);
   	end
end

save(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/Morans_and_Gearys_pom21.jld"),
                    "Shuffled_Gearys_C",Shuffled_Gearys_C,"Shuffled_M_I",Shuffled_M_I,
					"Gearys_C",Gearys_C,"M_I",M_I);#
                    

descr_name="pom21";#"cact"

#load all odors Gearys C's and Moran I's
Gearys_all_odors=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Gearys_C");
Shuffled_Gearys_all_odors=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Shuffled_Gearys_C");
Morans_all_odors=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"M_I");
Shuffled_Morans_all_odors=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Shuffled_M_I");

#load no amines Gearys C's and Moran I's
Gearys_no_amines=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Gearys_C");
Shuffled_Gearys_no_amines=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Shuffled_Gearys_C");
Morans_no_amines=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"M_I");
Shuffled_Morans_no_amines=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/Morans_and_Gearys_$descr_name"*".jld"),"Shuffled_M_I");

Gearys_C_pom21=vcat(Gearys_all_odors,Gearys_no_amines);
M_I_pom21=vcat(Morans_all_odors,Morans_no_amines);
Gearys_C_cact=vcat(Gearys_all_odors,Gearys_no_amines);
M_I_cact=vcat(Morans_all_odors,Morans_no_amines);
Shuffled_Gearys_C=hcat(Shuffled_Gearys_all_odors,Shuffled_Gearys_no_amines);
Shuffled_M_I=hcat(Shuffled_Morans_all_odors,Shuffled_Morans_no_amines);

using PyPlot;

fig = figure("boxplot-Geary's C",figsize=(10,10));
fz=30; # font size
#for i=1: 3;
#subplot(130+ i )
PyPlot.title("Geary's C, shuffled control versus original data ", fontsize=fz);
PyPlot.plt.plot([1,2,3,4],Gearys_C_cact, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.plt.plot([1,2,3,4],Gearys_C_pom21, color = :green, marker=:d, markersize=20, linewidth=0);
#PyPlot.plt.plot([1,2,3],Gearys_C, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.boxplot(Shuffled_Gearys_C, # Each column/cell is one box
   notch=true, # Notched center
   whis=0.75, # Whisker length as a percent of inner quartile range
   widths=0.25, # Width of boxes
   vert=true, # Horizontal boxes
   sym="rs"); # Symbol color and shape (rs = red square)
# PyPlot.plt.scatter([1,2,3],Gearys_C, c=:green, s=10)
#plt.legend(["real data"], fontsize=fz, bbox_to_anchor=(1.0, 1.0));
plt.legend(["Pubchem","GNN"], fontsize=fz, bbox_to_anchor=(1.0, 1.0));
# plt.ylabel("Geary's C values", fontsize=fz);
plt.xticks([1,2,3,4],["omp83L","omp83R","omp84","omp85"],fontsize=fz);
#plt.xticks([1,2,3],["animal 1","animal  2","animal 3"],fontsize=fz);
plt.yticks( fontsize=fz);
#end
display(fig)

#gcf()
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig3","Panel_E1.svg"))
plt.clf()
plt.cla()
plt.close()

fig = figure("Moran's I",figsize=(10,10));
fz=30; # font size
#for i=1: 3;
#subplot(130+ i )
PyPlot.title("Moran's I, shuffled control versus original data ", fontsize=fz);
PyPlot.plt.plot([1,2,3,4],M_I_cact, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.plt.plot([1,2,3,4],M_I_pom21, color = :green, marker=:d, markersize=20, linewidth=0);
#PyPlot.plt.plot([1,2,3,4],M_I, color = :blue, marker=:d, markersize=20, linewidth=0);
#PyPlot.plt.plot([1,2,3],M_I, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.boxplot(Shuffled_M_I, # Each column/cell is one box
	notch=true, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
	sym="rs"); # Symbol color and shape (rs = red square)
	# PyPlot.plt.scatter([1,2,3],Gearys_C, c=:green, s=10)
#plt.legend(["real data"], fontsize=fz, loc=0);
plt.legend(["Pubchem","GNN"], fontsize=fz, bbox_to_anchor=(1.5, 1.0));
#plt.legend(["real data"], fontsize=fz, bbox_to_anchor=(1.35, 1.0));
	# plt.ylabel("Geary's C values", fontsize=fz);
plt.xticks([1,2,3,4],["omp83L","omp83R","omp84","omp85"],fontsize=fz);
#plt.xticks([1,2,3],["animal 1","animal  2","animal 3"],fontsize=fz);
plt.yticks( fontsize=fz);
#end
display(fig)
#gcf()
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig3","Panel_E2.svg"))
plt.clf()
plt.cla()
plt.close()
