using PyCall,JLD, Clustering, DataFrames, LinearAlgebra, Statistics,MultivariateStats#, MLBase;
#ENV["PYTHON"]="/home/nqm5625/anaconda3/bin/python";
#Pkg.build("PyCall");
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/StatisticalChecks.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files//Initialize_analyses.jl"));

masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chem_data_segmented_high_and_low_conc.jld"),"masks");
#masks=load(joinpath(homedir(),"C:/Users/nikol/OneDrive/Desktop/chem_data_segmented.jld"),"masks");

this_descr_by_gloms=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/schmuker_descrs_on_bulb_separate_positive_and_negative_with_amines_high_and_low_conc.jld"),"shap_per_glom_matrix");
#this_descr_by_gloms=load(joinpath(homedir(),"C:/Users/nikol/OneDrive/Desktop/83_and_85_schmuker_descrs_on_bulb.jld"),"shap_per_glom_matrix");
#descr_by_gloms =this_descr_by_gloms[1]
#this_descr_by_gloms_copy=this_descr_by_gloms;
this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2],this_descr_by_gloms[3],this_descr_by_gloms[4]);


#Pkg.build("PyCall");
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/StatisticalChecks.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files//Initialize_analyses.jl"));

masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/chem_data_segmented.jld"),"masks");
#masks=load(joinpath(homedir(),"C:/Users/nikol/OneDrive/Desktop/chem_data_segmented.jld"),"masks");

this_descr_by_gloms=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/schmuker_descrs_on_bulb_signed_no_amines.jld"),"shap_per_glom_matrix");
#this_descr_by_gloms=load(joinpath(homedir(),"C:/Users/nikol/OneDrive/Desktop/83_and_85_schmuker_descrs_on_bulb.jld"),"shap_per_glom_matrix");
#descr_by_gloms =this_descr_by_gloms[1]
this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);

M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=40);

variance_explained=principalvars(M) ./ tvar(M) * 100

descr_by_gloms_transformed=MultivariateStats.transform(M,descr_by_gloms');

descr_by_gloms=copy(descr_by_gloms_transformed');

descr_by_gloms2d=descr_by_gloms[:,1:2];

descr_by_gloms83L_2d=descr_by_gloms2d[185:276,:]-descr_by_gloms2d[1:92,:];
descr_by_gloms83R_2d=descr_by_gloms2d[277:368,:]-descr_by_gloms2d[93:184,:];
angled2(a, b) = atand(norm(cross(a,b)),dot(a,b));
angled1(a, b) = acosd(clamp(a⋅b/(norm(a)*norm(b)), -1, 1));
xaxis_vector=[1.0, 0.0, 0.0];

descr_by_gloms_83L_angles=Array{Any,1}(undef,size(descr_by_gloms83L_2d,1));
descr_by_gloms_83R_angles=Array{Any,1}(undef,size(descr_by_gloms83R_2d,1));

for i in 1:length(descr_by_gloms_83L_angles);
	descr_by_gloms_83L_angles[i]=angled2(xaxis_vector,vcat(descr_by_gloms83L_2d[i,:],0.0))
end

for i in 1:length(descr_by_gloms_83R_angles);
	descr_by_gloms_83R_angles[i]=angled2(xaxis_vector,vcat(descr_by_gloms83R_2d[i,:],0.0))
end

quadrant_class_83L=Array{Int64,1}(undef,length(descr_by_gloms_83L_angles));
quadrant_class_83R=Array{Int64,1}(undef,length(descr_by_gloms_83R_angles));

for i in 1:length(quadrant_class_83L)
	if descr_by_gloms_83L_angles[i]<90 && descr_by_gloms83L_2d[i,2]>0
		quadrant_class_83L[i]=1
	elseif descr_by_gloms_83L_angles[i]<90 && descr_by_gloms83L_2d[i,2]<0
		quadrant_class_83L[i]=4
	elseif descr_by_gloms_83L_angles[i]>90 && descr_by_gloms83L_2d[i,2]>0
		quadrant_class_83L[i]=2
	else
		quadrant_class_83L[i]=3
	end
end

for i in 1:length(quadrant_class_83R)
	if descr_by_gloms_83R_angles[i]<90 && descr_by_gloms83R_2d[i,2]>0
		quadrant_class_83R[i]=1
	elseif descr_by_gloms_83R_angles[i]<90 && descr_by_gloms83R_2d[i,2]<0
		quadrant_class_83R[i]=4
	elseif descr_by_gloms_83R_angles[i]>90 && descr_by_gloms83R_2d[i,2]>0
		quadrant_class_83R[i]=2
	else
		quadrant_class_83R[i]=3
	end
end

for mask in masks[1]
    M = copy(mask)
    for i=1:size(M,1)
        mask[i,:] = M[257-i,:]
    end
end

for mask in masks[3]
    M = copy(mask)
    for i=1:size(M,1)
        mask[i,:] = M[257-i,:]
    end
end

quadrants_83L_matrix=high_vs_low_quadrants_on_bulb_matrices(masks[1],quadrant_class_83L);
quadrants_83R_matrix=high_vs_low_quadrants_on_bulb_matrices(masks[2],quadrant_class_83R);


color_array=[:white,:blue,:green,:red,:black];
p1=heatmap(quadrants_83L_matrix,c=color_array,grid=false,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "omp83L")
p2=heatmap(quadrants_83R_matrix,c=color_array,grid=false,
			xticks=false,yticks=false,xaxis=false,yaxis=false,title = "omp83R")
Plots.plot(p1,p2,layout=(1,2),size=(2000,1000))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","omp83_quadrants_high_vs_low_on_bulb.png"))











Plots.scatter(descr_by_gloms[1:92,1],descr_by_gloms[1:92,2],grid=false,label="high",c=:red,
				size = (1000, 1000),legend=:outerright,xlabel="PC 1", ylabel="PC 2", title="omp83L");
Plots.scatter!(descr_by_gloms[185:276,1],descr_by_gloms[185:276,2],grid=false,label="low",c=:blue);
for i in 1:92
	Plots.plot!([descr_by_gloms[i,1],descr_by_gloms[i+184,1]],[descr_by_gloms[i,2],descr_by_gloms[i+184,2]],c=:green,label="",arrow=true);
end
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","omp83L_pc1_pc2_shaps_high_and_low_conc.png"));

Plots.scatter(descr_by_gloms[1:92,2],descr_by_gloms[1:92,3],grid=false,label="high",c=:red,
				size = (1000, 1000),legend=:outerright,xlabel="PC 2", ylabel="PC 3", title="omp83L");
Plots.scatter!(descr_by_gloms[185:276,2],descr_by_gloms[185:276,3],grid=false,label="low",c=:blue);
for i in 1:92
	Plots.plot!([descr_by_gloms[i,2],descr_by_gloms[i+184,2]],[descr_by_gloms[i,3],descr_by_gloms[i+184,3]],c=:green,label="",arrow=true);
end
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","omp83L_pc2_pc3_shaps_high_and_low_conc.png"));

Plots.scatter(descr_by_gloms[93:184,1],descr_by_gloms[93:184,2],grid=false,label="high",c=:red,
				size = (1000, 1000),legend=:outerright,xlabel="PC 1", ylabel="PC 2", title="omp83R");
Plots.scatter!(descr_by_gloms[277:368,1],descr_by_gloms[277:368,2],grid=false,label="low",c=:blue);
for i in 1:92
	Plots.plot!([descr_by_gloms[92+i,1],descr_by_gloms[i+276,1]],[descr_by_gloms[i+92,2],descr_by_gloms[i+276,2]],c=:green,label="",arrow=true);
end
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","omp83R_pc1_pc2_shaps_high_and_low_conc.png"));

Plots.scatter(descr_by_gloms[93:184,2],descr_by_gloms[93:184,3],grid=false,label="high",c=:red,
				size = (1000, 1000),legend=:outerright,xlabel="PC 2", ylabel="PC 3", title="omp83R");
Plots.scatter!(descr_by_gloms[277:368,2],descr_by_gloms[277:368,3],grid=false,label="low",c=:blue);
for i in 1:92
	Plots.plot!([descr_by_gloms[92+i,2],descr_by_gloms[i+276,2]],[descr_by_gloms[i+92,3],descr_by_gloms[i+276,3]],c=:green,label="",arrow=true);
end
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","omp83R_pc2_pc3_shaps_high_and_low_conc.png"));


UseDatasets=[1,2];
















glom_distance_metric = Distances.euclidean;
glom_distance_metric = Distances.corr_dist;
glom_distance_metric = Distances.cosine_dist;

glom_dist = distance_matrix(descr_by_gloms, glom_distance_metric, dim=1);
glom_clustering_obj = hclust(glom_dist, :ward)#,:U,:optimal);
title="The dendrogram of glomeruli response to particular features";
glom_dend = plot_dendrogram_and_matrix(glom_clustering_obj, descr_by_gloms,title=title, tickfontsize=7)
glom_dend = plot!(; size=(1000, 600));
display(glom_dend)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pca_dendrogram_coss_separate_positive_and_negative.png"))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files","pca_dendrogram_coss_signed.png"))

height=2.5;
cluster_idxs=Int64.(cutree(glom_clustering_obj,h=height));
extremums = Vector(undef,length(unique(cluster_idxs)));
for (k,i) in enumerate(unique(cluster_idxs))
    extremums[k] = extrema([findfirst(x -> x .== j,glom_clustering_obj.order) for j in findall(x -> x ==i,cluster_idxs)]) .+ (-0.5,0.5)
end

Plots.plot([Shape([m,M,M,m],[0,0,height,height]) for (m,M) in extremums]; alpha = 0.5)
plot!(glom_clustering_obj, xticks = (1:length(cluster_idxs),string.(glom_clustering_obj.order)), xlabel="node", ylabel=string(glom_clustering_obj.linkage) * " linkage distance")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pca_dendrogram_with_clusters_coss_separate_pos_and_neg_height_25.png"))


unique_clust_nums = unique(cluster_idxs);
clusters = Vector{Vector{Int}}(undef,length(unique_clust_nums));
for i in unique_clust_nums
	 clusters[i] = VERSION>= v"0.7.0" ? findall(x->x==i,cluster_idxs) : find(x->x==i,cluster_idxs);
end

Nclusters=length(clusters);
Ndatasets=length(UseDatasets);
#masks[1][128]=Float64.(masks[1][128])
Nglomeruli=[113,108]
Nglomeruli=[128,132]

masks[1][128]=Float64.(masks[1][128])

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

f1=plot_glomeruli_clusters(masks[1],separate_glom_clusters[1])
f2=plot_glomeruli_clusters(masks[2],separate_glom_clusters[2])
Plots.plot(f1,f2,layout=(1,2),size=(2000,1000))

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pca_clusters_coss_sep_pos_and_neg_height_25.png"))

this_descr_by_gloms_copy[1]=descr_by_gloms[1:Nglomeruli[1],:];
this_descr_by_gloms_copy[2]=descr_by_gloms[Nglomeruli[1]+1:end,:];


old_masks=Array{Any,1}(undef,2)
for i in 1:length(old_masks)
	old_masks[i]=Array{Any,2}(undef,1,length(masks[i]))
	old_masks[i][1,:]=masks[i]
end

#compute the chemotopy gradient matrices that are to be plotted

shaps_on_bulb=Array{Array{Array{Float64,2},1},1}(undef,Ndatasets)

for i in 1:Ndatasets
    shaps_on_bulb[i]=shaps_on_bulb_matrices(old_masks[i],this_descr_by_gloms_copy[i])
end

color_array=[:blue,:green,:white,:yellow,:red];

for i in 1:size(descr_by_gloms,2)
	max_valp1=maximum(shaps_on_bulb[1][i])
	max_valp2=maximum(shaps_on_bulb[2][i])
	p1=heatmap(shaps_on_bulb[1][i],c = cgrad([:blue,:white,:red]),grid=false,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "omp83L",
		  	clims=(-max_valp1, max_valp1))
	p2=heatmap(shaps_on_bulb[2][i],c = cgrad([:blue,:white,:red]),grid=false,
			xticks=false,yticks=false,xaxis=false,yaxis=false,title = "omp83R",
			clims=(-max_valp2, max_valp2))
	#p1=heatmap(shaps_on_bulb[1][i],legend=false,c=color_array,grid=false,xticks=false,yticks=false,xaxis=false,yaxis=false)
	#p2=heatmap(shaps_on_bulb[2][i],legend=false,c=color_array,grid=false,xticks=false,yticks=false,xaxis=false,yaxis=false)
	Plots.plot(p1,p2,layout=(1,2),size=(2000,1000))
	Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/shapley_fields_pca","shapley_fields_pc$i.png"))
end
color_array=[:white, :gray,:blue,:green,:yellow,:red];

color_array=[:red,:yellow,:green,:blue, :gray,:white];
for i in 1:length(shaps_on_bulb)
	(min_x,max_x,min_y,max_y)=window(shaps_on_bulb[i])
	for j in 1:length(shaps_on_bulb[i])
		#shaps_on_bulb[i][j]=-1*shaps_on_bulb[i][j][(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
		shaps_on_bulb[i][j]=shaps_on_bulb[i][j][(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
	end
end
window(shaps_on_bulb[1])

p1=heatmap(shaps_on_bulb[1][4],legend=false,c=color_array,grid=false,xticks=false,yticks=false,xaxis=false,yaxis=false)
p2=heatmap(shaps_on_bulb[2][4],legend=false,c=color_array,grid=false,xticks=false,yticks=false,xaxis=false,yaxis=false)
Plots.plot(p1,p2,layout=(1,2),size=(2000,1000))


file = matopen(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/shap_chem_fields.mat"), "w");
write(file, "omp84_shapley_chem_field", shaps_on_bulb[1]);
write(file, "omp85_shapley_chem_field", shaps_on_bulb[2]);
write(file, "descriptor_names", descr_names);
close(file)



glom_clustering_objs = [];
Nglomeruli= zeros(Int,length(UseDatasets));
#Statistical checks


println("____________________________________________________");
Nshuflings=500;
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


using PyPlot

fig = figure("boxplot-Geary's C",figsize=(10,10));
fz=20; # font size
#for i=1: 3;
#subplot(130+ i )
PyPlot.title("Geary's C, shuffled control versus original data ", fontsize=fz);
PyPlot.plt.plot([1,2],Gearys_C, color = :blue, marker=:d, markersize=20, linewidth=0)
#PyPlot.plt.plot([1,2,3],Gearys_C, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.boxplot(Shuffled_Gearys_C, # Each column/cell is one box
   notch=true, # Notched center
   whis=0.75, # Whisker length as a percent of inner quartile range
   widths=0.25, # Width of boxes
   vert=true, # Horizontal boxes
   sym="rs"); # Symbol color and shape (rs = red square)
# PyPlot.plt.scatter([1,2,3],Gearys_C, c=:green, s=10)
plt.legend(["real data"], fontsize=fz, loc=1);
# plt.ylabel("Geary's C values", fontsize=fz);
plt.xticks([1,2],["animal 83L","animal  83R"],fontsize=fz)
#plt.xticks([1,2,3],["animal 1","animal  2","animal 3"],fontsize=fz);
plt.yticks( fontsize=fz);
#end
display(fig)

#gcf()
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pca_Geary_coss_sep_pos_and_neg.png"))
plt.clf()
plt.cla()
plt.close()

fig = figure("Moran's I",figsize=(10,10));
fz=20; # font size
#for i=1: 3;
#subplot(130+ i )
PyPlot.title("Moran's I, shuffled control versus original data ", fontsize=fz);
PyPlot.plt.plot([1,2],M_I, color = :blue, marker=:d, markersize=20, linewidth=0)
#PyPlot.plt.plot([1,2,3],M_I, color = :blue, marker=:d, markersize=20, linewidth=0);
PyPlot.boxplot(Shuffled_M_I, # Each column/cell is one box
	notch=true, # Notched center
	whis=0.75, # Whisker length as a percent of inner quartile range
	widths=0.25, # Width of boxes
	vert=true, # Horizontal boxes
	sym="rs"); # Symbol color and shape (rs = red square)
	# PyPlot.plt.scatter([1,2,3],Gearys_C, c=:green, s=10)
plt.legend(["real data"], fontsize=fz, loc=1);
	# plt.ylabel("Geary's C values", fontsize=fz);
plt.xticks([1,2],["animal 83L","animal  83R"],fontsize=fz)
#plt.xticks([1,2,3],["animal 1","animal  2","animal 3"],fontsize=fz);
plt.yticks( fontsize=fz);
#end
display(fig)
gcf()
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files","pca_Moran_coss_sep_pos_and_neg.png"))
plt.clf()
plt.cla()
plt.close()

p1=heatmap(descr_vals_norm,yflip=false,c=:jet1)
p2=heatmap(descrs,yflip=false,c=:jet1)

Plots.plot(p1,p2,layout=(1,2),size=(2000,1000))
