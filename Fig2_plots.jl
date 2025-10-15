using JLD, Plots, StatsPlots;


include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Useful_functions.jl"));

dataset="all_odors";
masks=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset"*"/data_for_resp_prediction_and_clustering.jld"),"masks");
#masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"masks");

#this_descr_by_gloms=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/clustering_files/pom21no_amines_shaps_on_bulb.jld"),"shap_per_glom_matrix");
this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset"*"/clustering_files/pom21_shaps_on_bulb.jld"),"shap_per_glom_matrix");
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

A=descr_by_gloms;#cact
B=descr_by_gloms;#pom21

n_pc_features=20;
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
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig3","pearson_cor_class1.svg"));

A=hcat(class_1_idxs,descr_by_gloms);#pom
A1=A[1:113,:];
A2=A[114:221,:];
B=hcat(class_1_idxs,descr_by_gloms);#cact
B1=B[1:113,:];
B2=B[114:221,:];

using PyCall;
sci_stats=pyimport("scipy.stats");

n_pc_features=20;
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

grass_dist=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/all_odors_grass_dist.jld"),"grassmanian_distances_20_dims");
grass_dist=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/all_odors_grass_dist_resp_shuffle.jld"),"grassmanian_distances");

n_shuffles=1000;

pom21_var_real=principalvars(M) ./ tvar(M) * 100
cact_var_real=principalvars(M) ./ tvar(M) * 100

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

n_shuffles=50;

chord_dist=Array{Array{Float64,2},1}(undef,20);
for i in 1:length(chord_dist)
    matr=zeros(size(grass_dist[:,:,i]))
    for j in 1:size(grass_dist[:,:,i],1)
        for k in 1:size(grass_dist[:,:,i],2)
            matr[j,k]=grass_dist[j,k,i][2]
        end
    end
    chord_dist[i]=matr
end

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

Plots.plot(1:20,mean_pom21_var_shuffled, ribbon=(lower_bounds_pom21_var_shuffled,upper_bounds_pom21_var_shuffled),
        fillalpha=.3, linewidth=0,color=:blue,grid=false,label="",legend=:outerright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16);
Plots.plot!(1:20,pom21_var_real_cumm,color=:blue,linewidth=2,label="GNN");
Plots.plot!(1:20,mean_cact_var_shuffled , ribbon=(lower_bounds_cact_var_shuffled,upper_bounds_cact_var_shuffled),fillalpha=.3, linewidth=0,color=:red,
        grid=false,label="",legend=:outerright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16);
Plots.plot!(1:20,cact_var_real_cumm,color=:red,linewidth=2,label="PubChem",legend=:topleft)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shuffled_resp_pc_variances.svg"));

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shuffled_resp_cact_var_cumm.svg"));


Plots.plot(1:20,mean_pom21_var_shuffled , ribbon=(lower_bounds_pom21_var_shuffled,upper_bounds_pom21_var_shuffled),fillalpha=.3, linewidth=2,color=:blue,grid=false,label="shuffled",legend=:right,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,pom21_var_real_cumm,color=:red,linewidth=2,label="real")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shuffled_resp_pom21_var.svg"));

Plots.plot(1:20,mean_cact_var_shuffled , ribbon=(lower_bounds_cact_var_shuffled,upper_bounds_cact_var_shuffled),fillalpha=.3, linewidth=2,color=:blue,grid=false,label="shuffled",legend=:right,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,cact_var_real_cumm,color=:red,linewidth=2,label="real")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shuffled_resp_cact_var.svg"));

Plots.plot(1:20,mean_chordal_shuffled , ribbon=(lower_bounds_chordal_shuffled,upper_bounds_chordal_shuffled),fillalpha=.3, linewidth=2,color=:blue,grid=false,label="shuffled",legend=:bottomright,xticks=1:2:20,legendfontsize=16,ticksfontsize=16)
Plots.plot!(1:20,real_chordal,color=:red,linewidth=2,label="real")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","grass_dist_curves_shuffled_resp.svg"));


lower_bds[i]=percentile(arrs[:,i],2.5)
upper_bds[i]=percentile(arrs[:,i],97.5)
geod=fill(0.0,n_shuffles,n_shuffles);
frob=fill(0.0,n_shuffles,n_shuffles);
binet=fill(0.0,n_shuffles,n_shuffles);
fubini=fill(0.0,n_shuffles,n_shuffles);
for i in 1:n_shuffles
    for j in 1:n_shuffles
        geod[i,j]=grass_dist[i,j][1]
        frob[i,j]=grass_dist[i,j][2]
        binet[i,j]=grass_dist[i,j][3]
        fubini[i,j]=grass_dist[i,j][4]
    end
end

Plots.boxplot(vec(binet))
Plots.scatter!([1],[binet_cauchy(A,B)],markersize=12,markershape=:diamond)

Plots.boxplot(hcat(vec(geod),vec(frob),vec(binet),vec(fubini)))

Plots.boxplot(["Grassman distance" "Chordal distance"],hcat(vec(geod),vec(frob)),grid=false,ticksfontsize=14,label="")
Plots.scatter!([0.5,1.9],[geod_grass(A,B),proj_frob(A,B)],color=:green,markersize=12,markershape=:diamond,label="real data",legendfontsize=14)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","grass_dist.svg"));

Plots.scatter!(1:4,[geod_grass(A,B),proj_frob(A,B),binet_cauchy(A,B),fubini_study(A,B)],markersize=12,markershape=:diamond)

cor_matrix_copy=copy(cor_matrix);
cor_matrix_copy[:,1].=NaN;
cor_matrix_copy[[1,2,3],1].=0.01;
cor_matrix_copy[1,[1,2,3,4,6]].=0.01;
cor_matrix_copy[1,:].=NaN;

heatmap(abs.(cor_matrix_copy),c=[:blue,:pink,:salmon,:red],grid=false,yflip=true,legend=false,fontsize=12,aspectratio=:equal,xlims=(0.5,10.5),xticks=false,yticks=false)
annotate!([(j, i, Plots.text(round(abs.(cor_matrix[i,j]),digits=2), 12,"Computer Modern",:black)) for i in 1:n_pc_features for j in 1:n_pc_features if pvalue_matrix[i,j]<0.05])

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig3","pearson_cor_shap_fields.svg"));

kendal_tau_matrix=StatsBase.corspearman(A2,B2)
kendal_tau_matrix=zeros(20,20);
for i in 1:20
    for j in 1:20
        kendal_tau_matrix[i,j]=cor(A1[:,i],B1[:,j])
    end
end
C=hcat(A1,B1);
C_kendal=corkendall(C,C);
heatmap(abs.(corkendall(C,C)))

function reorder_cormat(cormatr::Array{Float64,2})
# Use correlation between variables as distance
    dd=(1 .-cormatr)/2
    hc=hclust(dd)
    cormatr=cormatr[hc.order, hc.order]
    return cormatr,hc.order
end

idxs_cact=findall(x->x in 1:20,hc.order)
idxs_pom21=findall(x->x in 21:40,hc.order)

pc_projection_vectors=Array{Any,1}(undef,3);
for i in 1:length(pc_projection_vectors);
	pc_projection_vectors[i]=abs.(projection(M)[1:140,i])#.+abs.(projection(M)[141:280,i])
end

pc_projection_important_idxs=Array{Any,1}(undef,3);


for i in 1:length(pc_projection_important_idxs)
	pc_projection_important_idxs[i]=findmax(pc_projection_vectors[i],1:5)[1]
end

# Plotting Shapley fields

descr_names=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/segmented_data_all_odors_different_descrs_for_prediction.jld"),"descr_names_cact")
masks=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");
this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/cact_all_odors_shaps_on_bulb.jld"),"shap_per_glom_matrix");
#this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/pom21_all_odors_shaps_on_bulb.jld"),"shap_per_glom_matrix");

this_descr_by_gloms_copy=this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);

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
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

#masks[1][128]=Float64.(masks[1][128]);

for mask in masks[1]
    M = copy(mask)
    for i=1:size(M,1)
        mask[i,:] = M[257-i,:]
    end
end

Nglomeruli=[113,108];
#Nglomeruli=[128,132];

this_descr_by_gloms_copy[1]=descr_by_gloms[1:Nglomeruli[1],:];
this_descr_by_gloms_copy[2]=descr_by_gloms[Nglomeruli[1]+1:end,:];

using MultivariateStats;
M=fit(PCA,descr_by_gloms';pratio=1,maxoutdim=20);

variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

descr_by_gloms_transformed=MultivariateStats.transform(M,descr_by_gloms');

descr_by_gloms=copy(descr_by_gloms_transformed');

pc_projection_vectors=Array{Any,1}(undef,3);
for i in 1:length(pc_projection_vectors);
	pc_projection_vectors[i]=abs.(projection(M)[1:140,i])#.+abs.(projection(M)[141:280,i])
end

pc_projection_important_idxs=Array{Any,1}(undef,3);


for i in 1:length(pc_projection_important_idxs)
	pc_projection_important_idxs[i]=findmax(pc_projection_vectors[i],1:5)[1]
end

this_descr_by_gloms_copy[1]=-1*this_descr_by_gloms_copy[1][:,pc_projection_important_idxs[3].+140];
this_descr_by_gloms_copy[2]=-1*this_descr_by_gloms_copy[2][:,pc_projection_important_idxs[3].+140];

this_descr_by_gloms_copy[1]=-1*this_descr_by_gloms_copy[1][:,pc_projection_important_idxs[3].+140];
this_descr_by_gloms_copy[2]=-1*this_descr_by_gloms_copy[2][:,pc_projection_important_idxs[3].+140];

old_masks=Array{Any,1}(undef,2)
for i in 1:length(old_masks)
	old_masks[i]=Array{Any,2}(undef,1,length(masks[i]))
	old_masks[i][1,:]=masks[i]
end

bulbL=combine(old_masks[1]...);
bulbR=combine(old_masks[2]...);

(min_x,max_x,min_y,max_y)=window_matr(bulbL);
bulbL=bulbL[(min_y-10):(max_y+10),(min_x-10):(max_x+10)];
(min_x,max_x,min_y,max_y)=window_matr(bulbR);
bulbR=bulbR[(min_y-10):(max_y+10),(min_x-10):(max_x+10)];

#heatmap(bulbL,legend=false,yflip=true)
#Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","bulb_L.svg"));

#heatmap(bulbR,legend=false,yflip=false)
#Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","bulb_R.svg"));

#compute the chemotopy gradient matrices that are to be plotted

Ndatasets=2;
shaps_on_bulb=Array{Array{Array{Float64,2},1},1}(undef,Ndatasets);

#shaps_on_bulb[1]=shaps_on_bulb_matrices(old_masks[1],this_descr_by_gloms_copy[1][:,1:140])
shaps_on_bulb[1]=shaps_on_bulb_matrices(old_masks[1],this_descr_by_gloms_copy[1])

#Ndatasets=2

#for i in 1:Ndatasets
#    shaps_on_bulb[i]=shaps_on_bulb_matrices(old_masks[i],this_descr_by_gloms_copy[i][:,1:140])
#end

using ColorSchemes;
Blues=ColorSchemes.:YlGn_4#Blues_4;
Reds=ColorSchemes.OrRd_4;
Greens=ColorSchemes.PuBu_4

index=200;

#pos_shap_field_indices=[5,42,69]
#neg_shap_field_indices=[64]
max_valp1=maximum(this_descr_by_gloms_copy[1][:,index]);
min_valp1=minimum(this_descr_by_gloms_copy[1][:,index]);
total_range=max_valp1-min_valp1;
threshold_for_0=abs(min_valp1)/total_range;
pos_range=1-threshold_for_0;

#color_array=[Blues[4],Blues[3],Blues[2],:white,Reds[2],Reds[3],Reds[4]];
color_array=[Blues[4],Blues[3],Blues[2],:white,Greens[2],Greens[3],Greens[4]];

color_grad=cgrad(color_array,[threshold_for_0*3/10,threshold_for_0*6/10,threshold_for_0*9/10,threshold_for_0+pos_range/10,threshold_for_0+pos_range*4/10,threshold_for_0+pos_range*7/10],categorical=true);
#color_grad
p=heatmap(shaps_on_bulb[1][index],c = color_grad,rev=true,grid=false,background=:black,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "",dpi=300,aspectratio=1,right_margin = 10Plots.mm);
p
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir","neg_shapley_fields_feat60_cact_83L.svg"));


min_valp1=-maximum(this_descr_by_gloms_copy[1][:,index]);
max_valp1=-minimum(this_descr_by_gloms_copy[1][:,index]);
total_range=max_valp1-min_valp1;
threshold_for_0=abs(max_valp1)/total_range;
neg_range=threshold_for_0-1;

color_array=[Greens[4],Greens[3],Greens[2],:white,Blues[2],Blues[3],Blues[4]];

color_grad=cgrad(color_array,[threshold_for_0+pos_range*7/10,threshold_for_0*3/10,threshold_for_0*6/10,threshold_for_0*9/10,threshold_for_0+pos_range/10,threshold_for_0+pos_range*4/10,],categorical=true);
#color_grad
p=heatmap(shaps_on_bulb[1][index],c = color_grad,rev=true,grid=false,background=:black,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "",dpi=300,aspectratio=1,right_margin = 10Plots.mm);
p
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir","neg_shapley_fields_feat45_cact_83L.svg"));

descr_names[5]
descr_names[21]
descr_names[45]
descr_names[60]