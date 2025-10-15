using JLD, Plots;

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Useful_functions.jl"));
#include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Load_the_data_from_the_files.jl"));

resp_matr=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"resp_matrices");

masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");

#Plotting Panel A, response to some stimulus on one olfactory bulb
UseDatasets=[1,2];

Ndatasets=length(UseDatasets);

#for i in 1:length(resp_matr)
#	resp_matr[i]=resp_matr[i][nonzero_rows[i],:]
#end

for mask in masks[1]
    M = copy(mask)
    for i=1:size(M,1)
        mask[i,:] = M[257-i,:]
    end
end

old_masks=Array{Any,1}(undef,2)

for i in 1:length(old_masks)
	old_masks[i]=Array{Any,2}(undef,1,length(masks[i]))
	old_masks[i][1,:]=masks[i]
end

function odor_response_matrices(masks::Array{Any,2},response_matrix::AbstractArray{Float64,2})
    odor_matrix=Array{Array{Float64,2},1}(undef,size(response_matrix,2))
    for l in 1:size(response_matrix,2)
        odor_mask=fill(NaN,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                         odor_mask[i,j]=max(0,response_matrix[k,l])
#                         odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l])
                    end
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

resp_on_bulb=Array{Array{Array{Float64,2},1},1}(undef,Ndatasets);

for i in 1:Ndatasets
    resp_on_bulb[i]=odor_response_matrices(old_masks[i],resp_matr[i])
end


gr();
#plot response to first odor. Can plot any odor if one so choses.
#[140:220,80:200]
p=heatmap(resp_on_bulb[1][1],c = :jet,grid=false,
		  	xticks=false,yticks=false,xaxis=false,yaxis=false,title = "",dpi=300,aspectratio=1);
p
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig1/PanelA_1.svg"))

#Plotting Panel B, example of features used
#schmuker cactvs and chemsense descriptors
schmuker_cact_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/cactvs_our_odors.csv");
cact_descrs, cact_descrs_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_cact_filename));

unique_descr_idx=unique(z -> cact_descrs[:,z], 1:size(cact_descrs,2));
cact_descrs=cact_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), cact_descrs, dims = 1));
cact_descrs=cact_descrs[:,zero_idxs];
colors_vec=[:white,:black];

gr()
heatmap(cact_descrs,c=cgrad(colors_vec),yflip=true,legend=false,yticks=false,xticks=false,xaxis=false,yaxis=false,aspectratio=1,dpi=300)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig1/PanelB_2.svg"))

#Plotting Panel C, Shapley values

#using PyPlot;

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));

normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"binary_responses");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");


#schmuker cactvs and chemsense descriptors
schmuker_cact_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/cactvs_our_odors.csv");
cact_descrs, cact_descrs_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_cact_filename));

unique_descr_idx=unique(z -> cact_descrs[:,z], 1:size(cact_descrs,2));
cact_descrs=cact_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), cact_descrs, dims = 1));
cact_descrs=cact_descrs[:,zero_idxs];
cact_descrs_names=cact_descrs_names[unique_descr_idx];
cact_descrs_names=cact_descrs_names[zero_idxs];
cact_descrs_names=vcat(Symbol.(cact_descrs_names),:x,:y,:A);
cact_shap_reduced_descrs=cact_descrs;

#cact_shap_reduced_descrs_names=cact_descrs_names[data_shap_idx_cact[1:30]];
#cact_shap_reduced_descrs_names=vcat(Symbol.(cact_shap_reduced_descrs_names),:x,:y,:A);

Ndatasets=2;
model_features_cact_shap_reduced=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_cact_shap_reduced=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_cact_shap_reduced)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=cact_shap_reduced_descrs[j,:]#,normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_cact_shap_reduced[k]=vec(model_feature);
    model_features_cact_shap_reduced[k]=model_feature
end

model_features=model_features_cact_shap_reduced;

#acid_id=1;
#acid_glom=2;
ester_id=12;
ester_glom=75;
#thiazoline_id=53;
#thiazoline_glom=94;
#pyrazine_id=42;
#pyrazine_glom=76;

predictor=skl_ensemble.RandomForestClassifier(n_estimators=128,max_samples=0.5,class_weight="balanced_subsample",random_state=1234);
predictor.fit(model_features[1][ester_glom,1:end.!=ester_id],binary_responses[1][ester_glom,1:end.!=ester_id]);

explainer=shap.TreeExplainer(predictor);
explainer.expected_value;

explain=model_features[1][ester_glom,ester_id];

shap_values=explainer.shap_values(explain);
#shap_values=explainer(explain)
#shap.plots.waterfall(explainer.expected_value[2],shap_values[2],predictor.predict_proba([explain])[2])
#shap_values.base_values

# #cact_descr_names=vcat(["base"],cact_shap_reduced_descrs_names,["predicted"])
# feature_names=Array{Any,1}(undef,140);#copy(cact_descrs_names);
# for i in 1:140
#     #if model_features[1][ester_glom,ester_id][i]==1.0
#     #    feature_names[i]=:rect
#     #else
#         feature_names[i]=""
#     #end
#     feature_names[i]=feature_names[i]*"="*string(model_features[1][ester_glom,ester_id][i])
# end

#feature_names=vcat(["expected"], feature_names,"predicted")

n_features=15;
feature_idxs=findmax(abs.(shap_values[2]),1:n_features)[1];
shap_values_pos=shap_values[2][feature_idxs];
shap_vals_of_interest=vcat([explainer.expected_value[2]],reverse(shap_values_pos),[predictor.predict_proba([explain])[2]]);

#shap_values_pos=shap_values[2];

#shap_vals_of_interest=vcat([explainer.expected_value[2]],shap_values_pos,[predictor.predict_proba([explain])[2]]);


colors=Array{String,1}(undef,length(shap_values_pos));
for i in 1:length(colors)
    if reverse(shap_values_pos)[i]<0
        colors[i]="red"
    else
        colors[i]="green"
    end
end
colors=vcat(["royalblue"],colors,["royalblue"]);

colors_lower=Array{String,1}(undef,n_features);
for i in 1:length(colors_lower)
    colors_lower[i]="white"
end
colors_lower=vcat("royalblue",colors_lower,"royalblue");


sums=Array{Float64,1}(undef,n_features);
shifted_sums=Array{Float64,1}(undef,n_features);
#sums[1]=explainer.expected_value[2]
shifted_sums[1]=explainer.expected_value[2]
for i in 1:length(sums)
    sums[i]=explainer.expected_value[2]+sum(reverse(shap_values_pos)[1:i])
end

for i in 2:length(shifted_sums)
    shifted_sums[i]=sums[i-1]
end

#PyPlot.svg(true)

upper_vals=max.(sums,shifted_sums);
lower_vals=min.(sums,shifted_sums);
upper_vals=vcat([explainer.expected_value[2]],upper_vals,[predictor.predict_proba([explain])[2]]);
lower_vals=vcat([explainer.expected_value[2]],lower_vals,[predictor.predict_proba([explain])[2]]);
mid=0.5*(lower_vals.+upper_vals);
mid[1]=0.5*mid[1];
mid[17]=0.5*mid[17];

# fig,ax = plt.subplots()
# #figsize=(10,10)
# #ax.set_rasterization_zorder(1) 
# fig.set_size_inches(10.5, 8.5)
# ax.barh(y=feature_names,width=upper_vals,color=colors);
# ax.barh(y=feature_names,width=lower_vals,color=colors_lower);
# ax.invert_yaxis()
# for i in 1:length(upper_vals)
#     plt.text(mid[i],i-0.8, string(round(shap_vals_of_interest[i],digits=2)))
# end
# display(fig)

#presence_of_feature_idx=findall(model_features[1][ester_glom,ester_id].==1.0);
presence_of_feature_idx=findall(model_features[1][ester_glom,ester_id][feature_idxs].==1.0);
presence_of_feature_idx=-presence_of_feature_idx.+16;
gr()
blues_idxs=findall(i->i==:royalblue,colors)
greens_idxs=findall(i->i==:green,colors)
reds_idxs=findall(i->i==:red,colors)
p=Plots.bar([1],[upper_vals[1]],c=:royalblue,orientation=:horizontal,grid=false,linecolor=:transparent,yaxis=false,legend=false);
for i in 2:17
    Plots.bar!([i],[upper_vals[i]],c=colors[i],orientation=:horizontal,grid=false,linecolor=:transparent,yaxis=false,legend=false)
    Plots.bar!([i],[lower_vals[i]],c=colors_lower[i],orientation=:horizontal,grid=false,linecolor=:transparent,yaxis=false,legend=false)
end
#p
Plots.scatter!(repeat([-0.005],15),2:16,markershape=:rect,markersize=6,color=:white,linecolor=:transparent);
Plots.scatter!(repeat([-0.005],6),presence_of_feature_idx.+1,markershape=:rect,markersize=6,color=:black);
#Plots.scatter!(repeat([-0.005],6),presence_of_feature_idx.+1,markershape=:rect,color=:black);
for i in [1,8,9,10,11,12,13,14,15,16,17]
   annotate!(mid[i],i-0.8+0.9, Plots.text(string(round(shap_vals_of_interest[i],digits=2)),:black,:right,6))
end
p
#Plots.bar!(lower_vals,orientation=:horizontal,grid=false,yticks=false,c=colors_lower,linecolor = :transparent,yaxis=false,bar_width=0.95,legend=false)

#Plots.scatter!(repeat([-0.005],6),presence_of_feature_idx.+1,markershape=:rect,color=:black);
#for i in [1,5,8,9,28,30,32]
#   annotate!(mid[i],i-0.8+0.9, Plots.text(string(round(shap_vals_of_interest[i],digits=2)),:black,:right,6))
#end
#p

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Fig1","Panel_C.svg"));

#Plots.scatter!(repeat([-0.005],140),2:141,markershape=:rect,color=:white,linecolor=:transparent);
#Plots.scatter!(repeat([-0.005],6),presence_of_feature_idx.+1,markershape=:rect,color=:black);
#for i in [1,5,8,9,28,30,32]
#   annotate!(mid[i],i-0.8+0.9, Plots.text(string(round(shap_vals_of_interest[i],digits=2)),:black,:right,6))
#end
#Plots.bar!(lower_vals,orientation=:horizontal,grid=false,yticks=false,c=colors_lower,linecolor = :transparent,yaxis=false,bar_width=0.95,legend=false)

#close(fig)

#Plotting Panel D
A=Array{Any,1}(undef,2);
A[1]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks1");
A[2]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks2");
B=Array{Any,1}(undef,2);
B[1]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values_shuffled_resp.jld"),"shap_ranks1");
B[2]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values_shuffled_resp.jld"),"shap_ranks2");

rankings=Array{Any,1}(undef,2);
for i in 1:length(rankings)
    rankings[i]=fill(0.0,140,100)
    for j in 1:size(rankings[i],2)
        for k in 1:size(rankings[i],1)
            rankings[i][k,j]=A[i][j][1][k]
        end
    end
end

rankings_shuffled=Array{Any,1}(undef,2);
for i in 1:length(rankings_shuffled)
    rankings_shuffled[i]=fill(0.0,140,100)
    for j in 1:size(rankings_shuffled[i],2)
        for k in 1:size(rankings_shuffled[i],1)
            rankings_shuffled[i][k,j]=B[i][j][1][k]
        end
    end
end

using PyCall;

sci_stats=pyimport("scipy.stats");

kendall_tau=sci_stats.pearsonr;

kendall_taus_83L=fill(0.0,100,100);
kendall_taus_83R=fill(0.0,100,100);
kendall_taus_83L_shuffle=fill(0.0,100,100);
kendall_taus_83R_shuffle=fill(0.0,100,100);
kendall_taus_83L_vs_83R=fill(0.0,100,100);
kendall_taus_83L_vs_83R_shuffle=fill(0.0,100,100);
kendall_taus_83L_vs_shuffle=fill(0.0,100,100);
kendall_taus_83R_vs_shuffle=fill(0.0,100,100);
for i in 1:100
    for j in 1:100
        kendall_taus_83L[i,j]=kendall_tau(rankings[1][:,i],rankings[1][:,j])[1]
        kendall_taus_83R[i,j]=kendall_tau(rankings[2][:,i],rankings[2][:,j])[1]
        kendall_taus_83L_shuffle[i,j]=kendall_tau(rankings_shuffled[1][:,i],rankings_shuffled[1][:,j])[1]
        kendall_taus_83R_shuffle[i,j]=kendall_tau(rankings_shuffled[2][:,i],rankings_shuffled[2][:,j])[1]
        kendall_taus_83L_vs_83R[i,j]=kendall_tau(rankings[1][:,i],rankings[2][:,j])[1]
        kendall_taus_83L_vs_83R_shuffle[i,j]=kendall_tau(rankings_shuffled[1][:,i],rankings_shuffled[2][:,j])[1]
        kendall_taus_83L_vs_shuffle[i,j]=kendall_tau(rankings[1][:,i],rankings_shuffled[1][:,j])[1]
        kendall_taus_83R_vs_shuffle[i,j]=kendall_tau(rankings[2][:,i],rankings_shuffled[2][:,j])[1]
    end
end

using StatsPlots;
StatsPlots.violin(["real"],vec(kendall_taus_83L_vs_83R),color=:red,grid=false,ylabel="Kendall Tau Correspondence",legend=false);
StatsPlots.violin!(["shuffled"],vec(kendall_taus_83L_vs_83R_shuffle),color=:blue,legend=false)
# StatsPlots.violin!(["per glom"],vec(kendall_taus_83L_per_glom),color=:red,label="");
# StatsPlots.violin!(["per glom "],vec(kendall_taus_83R_per_glom),color=:blue,label="");
# StatsPlots.violin!(["whole  "],vec(kendall_taus_83L_vs_83R),color=:green,label="1 vs 2");
# StatsPlots.violin!(["per glom  "],vec(kendall_taus_83L_vs_83R_per_glom),color=:green,label="");
# StatsPlots.violin!(["whole/per glom"],vec(kendall_taus_83L_vs_per_glom),color=:red,label="");
# StatsPlots.violin!(["whole/per glom "],vec(kendall_taus_83R_vs_per_glom),color=:blue,label="",size=(1000,500))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shap_rankings_comparison.svg"));

