using JLD, Plots;

include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Useful_functions.jl"));

#The code below makes the plot for class 1 vs class 2 on omp83L
dataset_name="no_amines";
masks=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/data_for_resp_prediction_and_clustering.jld"),"masks");
binary_responses=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");

class_1_idxs=findall(x-> x.!=0, vec(sum(binary_responses[1][:,1:5],dims=2)));

function classification_of_glomeruli(masks::Array{Any,1},class_1_gloms::Vector{Int64},bin_resp::BitMatrix)
    odor_mask=fill(NaN,size(masks[1],1),size(masks[1],2))
    for i in 1:size(masks[1],1)
        for j in 1:size(masks[1],2)
            for k in setdiff(1:size(bin_resp,1),class_1_gloms)
                img=masks[k]
                if img[i,j]!=0.0
                    odor_mask[i,j]=0
                end
            end
            for k in class_1_gloms
                img=masks[k]
                if img[i,j]!=0.0
                    if isnan(odor_mask[i,j])
                        odor_mask[i,j]=1
                    else
                        odor_mask[i,j]=max(1,odor_mask[i,j])
                    end
                end
            end
        end
        #odor_matrix[l]=odor_mask
    end
    (min_x,max_x,min_y,max_y)=window_matr_inside_nan(odor_mask)
    odor_mask=odor_mask[(min_y-5):(max_y+5),(min_x-5):(max_x+5)]
    #need to find min windrow of matrix first
    return odor_mask
end

#In case dataset_name="no_amines", run the code below
masks[1][128]=Float64.(masks[1][128]);


class_1_on_bulb=classification_of_glomeruli(masks[1],class_1_idxs,binary_responses[1]);
gr()
heatmap(class_1_on_bulb,c=[:green,:red],background_color=:black,legend=false,yflip=true,xaxis=false,yaxis=false,
                            xticks=false,yticks=false,grid=false,dpi=300,aspectratio=1,right_margin = 10Plots.mm)
Plots.savefig(joinpath(homedir(),"Documents/SuppFig8_panelA_class_1.svg"))

#The code below plots GNN Shapley fields, after PCA projection, the first 3 components only





#The code below plots Panel B
this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/pom21_all_odors_shaps_on_bulb.jld"),"shap_per_glom_matrix");
#this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/pom21_shaps_on_bulb.jld"),"shap_per_glom_matrix");
#for i in 1:length(this_descr_by_gloms)
#	this_descr_by_gloms[i][isnan.(this_descr_by_gloms[i])].=0
#end

this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);

using MultivariateStats;
M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

descr_by_gloms_transformed=MultivariateStats.transform(M,descr_by_gloms');

descr_by_gloms=copy(descr_by_gloms_transformed');

class_1_idxs1=vec(sum(binary_responses[1][:,1:5],dims=2));
class_1_idxs2=vec(sum(binary_responses[2][:,1:5],dims=2));
class_1_idxs=vcat(class_1_idxs1,class_1_idxs2);

alde

A=descr_by_gloms;#cact
B=descr_by_gloms;#pom21

using PyCall;
sci_stats=pyimport("scipy.stats");

n_pc_features=10;
cor_matrix=zeros(n_pc_features,2);
pvalue_matrix=zeros(n_pc_features,2);
for i in 1:n_pc_features
    pvalue_matrix[i,1]=sci_stats.pearsonr(A[:,i],class_1_idxs)[2]
    if pvalue_matrix[i,1]>0.05
        cor_matrix[i,1]=NaN
    else
        cor_matrix[i,1]=sci_stats.pearsonr(A[:,i],class_1_idxs)[1]
    end
end

for i in 1:n_pc_features
    pvalue_matrix[i,2]=sci_stats.pearsonr(B[:,i],class_1_idxs)[2]
    if pvalue_matrix[i,2]>0.05
        cor_matrix[i,2]=NaN
    else
        cor_matrix[i,2]=sci_stats.pearsonr(B[:,i],class_1_idxs)[1]
    end
end

heatmap(abs.(cor_matrix),c=[:blue,:pink,:salmon,:red],grid=false,yflip=true,legend=false,fontsize=12,aspectratio=:equal,xlims=(0.5,2.5),xticks=false,yticks=false)
annotate!([(j, i, Plots.text(round(abs.(cor_matrix[i,j]),digits=2), 12,"Computer Modern",:black)) for i in 1:n_pc_features for j in 1:2 if pvalue_matrix[i,j]<0.05])

#The code below plots Panel C

n_pc_features=10;
cor_matrix=zeros(n_pc_features,n_pc_features);
pvalue_matrix=zeros(n_pc_features,n_pc_features);
for i in 1:n_pc_features
    for j in 1:n_pc_features
        pvalue_matrix[i,j]=sci_stats.pearsonr(A[:,i],B[:,j])[2]
        if pvalue_matrix[i,j]>0.05
            cor_matrix[i,j]=NaN
        else
            cor_matrix[i,j]=sci_stats.pearsonr(A[:,i],B[:,j])[1]
        end
    end
end

heatmap(abs.(cor_matrix),c=[:blue,:pink,:salmon,:red],grid=false,yflip=true,legend=false,fontsize=12,aspectratio=:equal,xlims=(0.5,10.5),xticks=false,yticks=false)
annotate!([(j, i, Plots.text(round(abs.(cor_matrix[i,j]),digits=2), 12,"Computer Modern",:black)) for i in 1:n_pc_features for j in 1:n_pc_features if pvalue_matrix[i,j]<0.05])
Plots.savefig(joinpath(homedir(),"Documents/SuppFig8_panel_C.svg"));


#The code below plots Panel E

this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/cact_no_amines_shaps_on_bulb.jld"),"shap_per_glom_matrix");


this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);
M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

pom21_var_real=principalvars(M) ./ tvar(M) * 100
cact_var_real=principalvars(M) ./ tvar(M) * 100


# The code below plots the rest of Panel A
#descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_all_odors_different_descrs_for_prediction.jld"),"descr_names_cact")
#masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");

function shaps_on_bulb_matrices(masks::Array{Any,2},response_matrix::AbstractArray{Float64,2})
    odor_matrix=Array{Array{Float64,2},1}(undef,size(response_matrix,2))
    for l in 1:size(response_matrix,2)
       #odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
       odor_mask=fill(NaN,size(masks[1],1),size(masks[1],2)) 
       for i in 1:size(odor_mask,1)
            for j in 1:size(odor_mask,2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                         #odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l])
                         #odor_mask[i,j]=max(0,response_matrix[k,l])
                         odor_mask[i,j]=response_matrix[k,l]
                    end
                #println("$k")
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr_inside_nan(odor_mask)
        odor_mask=odor_mask[(min_y-5):(max_y+5),(min_x-5):(max_x+5)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

# masks[1][128]=Float64.(masks[1][128]);

# for mask in masks[1]
#     M = copy(mask)
#     for i=1:size(M,1)
#         mask[i,:] = M[257-i,:]
#     end
# end

Nglomeruli=[113,108];
Nglomeruli=[128,132];

this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/pom21_no_amines_shaps_on_bulb.jld"),"shap_per_glom_matrix");
this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);

M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

descr_by_gloms_transformed=MultivariateStats.transform(M,descr_by_gloms');

descr_by_gloms=copy(descr_by_gloms_transformed');

old_masks=Array{Any,1}(undef,2)
for i in 1:length(old_masks)
	old_masks[i]=Array{Any,2}(undef,1,length(masks[i]))
	old_masks[i][1,:]=masks[i]
end


#compute the chemotopy gradient matrices that are to be plotted

Ndatasets=2;
shaps_on_bulb=Array{Array{Array{Float64,2},1},1}(undef,Ndatasets);

shaps_on_bulb[1]=shaps_on_bulb_matrices(old_masks[1],descr_by_gloms[1:128,1:3])

using ColorSchemes;
Blues=ColorSchemes.:YlGn_4#Blues_4;
Reds=ColorSchemes.OrRd_4;

max_valp1=maximum(descr_by_gloms[1:128,3]);
min_valp1=minimum(descr_by_gloms[1:128,3]);
total_range=max_valp1-min_valp1;
threshold_for_0=abs(min_valp1)/total_range;
pos_range=1-threshold_for_0;

color_array=[Blues[4],Blues[3],Blues[2],:white,Reds[2],Reds[3],Reds[4]];

color_grad=cgrad(color_array,[threshold_for_0*3/10,threshold_for_0*6/10,threshold_for_0*9/10,threshold_for_0+pos_range/10,threshold_for_0+pos_range*4/10,threshold_for_0+pos_range*7/10],categorical=true);
#color_grad
p=heatmap(shaps_on_bulb[1][3],c = color_grad,rev=true,grid=false,background=:black,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "",dpi=300,aspectratio=1,right_margin = 10Plots.mm)
Plots.savefig(joinpath(homedir(),"Documents/SuppFig8_panelA_PC3.svg"))


#The code below plots panels D and E


this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/pom21_no_amines_shaps_on_bulb.jld"),"shap_per_glom_matrix");


this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);
M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

pom21_var_real=principalvars(M) ./ tvar(M) * 100
cact_var_real=principalvars(M) ./ tvar(M) * 100


grass_dist=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/$dataset_name"*"_grass_dist_resp_shuffle.jld"),"grassmanian_distances");

n_shuffles=1000;

chord_dist=Array{Array{Float64,1},1}(undef,20);
pom21_var=Array{Array{Float64,1},1}(undef,20);
cact_var=Array{Array{Float64,1},1}(undef,20);
for i in 1:length(chord_dist)
    c_dist=fill(0.0,n_shuffles)
    pom21_v=fill(0.0,n_shuffles)
    cact_v=fill(0.0,n_shuffles)
    for j in 1:n_shuffles
        c_dist[j]=grass_dist[j][3][i][2]
        pom21_v[j]=grass_dist[j][1][i]
        cact_v[j]=grass_dist[j][2][i]
    end
    chord_dist[i]=c_dist
    pom21_var[i]=pom21_v
    cact_var[i]=cact_v
end

for i in 2:20
    pom21_var[i]=pom21_var[i]+pom21_var[i-1]
    cact_var[i]=cact_var[i]+cact_var[i-1]
end

using StatsBase;


mean_chordal_shuffled=zeros(20);
lower_bounds_chordal_shuffled=zeros(20);
upper_bounds_chordal_shuffled=zeros(20);
mean_pom21_var_shuffled=zeros(20);
lower_bounds_pom21_var_shuffled=zeros(20);
upper_bounds_pom21_var_shuffled=zeros(20);
mean_cact_var_shuffled=zeros(20);
lower_bounds_cact_var_shuffled=zeros(20);
upper_bounds_cact_var_shuffled=zeros(20);
for i in 1:length(mean_chordal_shuffled)
    mean_chordal_shuffled[i]=mean(vec(chord_dist[i]))
    lower_bounds_chordal_shuffled[i]=mean(vec(chord_dist[i]))-percentile(vec(chord_dist[i]),2.5)
    upper_bounds_chordal_shuffled[i]=percentile(vec(chord_dist[i]),97.5)-mean(vec(chord_dist[i]))
    mean_pom21_var_shuffled[i]=mean(vec(pom21_var[i]))
    lower_bounds_pom21_var_shuffled[i]=mean(vec(pom21_var[i]))-percentile(vec(pom21_var[i]),2.5)
    upper_bounds_pom21_var_shuffled[i]=percentile(vec(pom21_var[i]),97.5)-mean(vec(pom21_var[i]))
    mean_cact_var_shuffled[i]=mean(vec(cact_var[i]))
    lower_bounds_cact_var_shuffled[i]=mean(vec(cact_var[i]))-percentile(vec(cact_var[i]),2.5)
    upper_bounds_cact_var_shuffled[i]=percentile(vec(cact_var[i]),97.5)-mean(vec(cact_var[i]))
end

real_chordal=zeros(20);
for i in 1:20
    real_chordal[i]=proj_frob(A[:,1:i],B[:,1:i])
end

mean_pom21_var_shuffled_cumm=zeros(20);
lower_bounds_pom21_var_shuffled_cumm=zeros(20);
upper_bounds_pom21_var_shuffled_cumm=zeros(20);
mean_cact_var_shuffled_cumm=zeros(20);
lower_bounds_cact_var_shuffled_cumm=zeros(20);
upper_bounds_cact_var_shuffled_cumm=zeros(20);
for i in 1:length(mean_pom21_var_shuffled_cumm)
    mean_pom21_var_shuffled_cumm[i]=sum(mean_pom21_var_shuffled[1:i])
    lower_bounds_pom21_var_shuffled[i]=sum(lower_bounds_pom21_var_shuffled[1:i])
    upper_bounds_pom21_var_shuffled[i]=sum(upper_bounds_pom21_var_shuffled[1:i])
    mean_cact_var_shuffled_cumm[i]=sum(mean_cact_var_shuffled[1:i])
    lower_bounds_cact_var_shuffled[i]=sum(lower_bounds_cact_var_shuffled[1:i])
    upper_bounds_cact_var_shuffled[i]=sum(upper_bounds_cact_var_shuffled[1:i])
end

pom21_var_real_cumm=fill(0.0,20);
cact_var_real_cumm=fill(0.0,20);
for i in 1:20
    pom21_var_real_cumm[i]=sum(pom21_var_real[1:i])
    cact_var_real_cumm[i]=sum(cact_var_real[1:i])
end


Plots.plot(1:20,mean_chordal_shuffled , ribbon=(lower_bounds_chordal_shuffled,upper_bounds_chordal_shuffled),fillalpha=.3, linewidth=2,color=:blue,grid=false,label="shuffled",legend=:bottomright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,real_chordal,color=:red,linewidth=2,label="real")
Plots.savefig(joinpath(homedir(),"Documents/SuppFig7_panel_D.svg"));


Plots.plot(1:20,mean_pom21_var_shuffled, ribbon=(lower_bounds_pom21_var_shuffled,upper_bounds_pom21_var_shuffled),
        fillalpha=.3, linewidth=0,color=:blue,grid=false,label="",legend=:outerright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,pom21_var_real_cumm,color=:blue,linewidth=2,label="GNN")
Plots.plot!(1:20,mean_cact_var_shuffled , ribbon=(lower_bounds_cact_var_shuffled,upper_bounds_cact_var_shuffled),fillalpha=.3, linewidth=0,color=:red,
        grid=false,label="",legend=:outerright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,cact_var_real_cumm,color=:red,linewidth=2,label="PubChem",legend=:topleft)
Plots.savefig(joinpath(homedir(),"Documents/SuppFig7_panel_E.svg"));

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shuffled_resp_cact_var_cumm.svg"));