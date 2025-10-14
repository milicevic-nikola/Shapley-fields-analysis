using JLD, Plots, StatsPlots;


function curves_to_plot(values_10_fold::Array{Float64,1},values_loo::Array{Float64,1},
                        values_per_glom::Array{Float64,1},
                        title::String,ylabel::String,xlabel::String,
                        label_10_fold::String,label_loo::String,label_per_glom::String
                        ,ylim::Array{Float64,1},xlims::Array{Int64,1})
    p=Plots.plot(1:length(values_10_fold),values_10_fold,xlabel=xlabel,ylabel=ylabel,title=title,
                    legend=:outerright,linewidth=2,c=:blue,label=label_10_fold,grid=false)
    p=Plots.plot!(1:length(values_loo),values_loo,xlabel=xlabel,ylabel=ylabel,
                    legend=:outerright,linewidth=2,c=:red,label=label_loo,grid=false)
    p=Plots.plot!(1:length(values_per_glom),values_per_glom,xlabel=xlabel,ylabel=ylabel,
                    legend=:outerright,linewidth=2,c=:green,label=label_per_glom,grid=false)
    return p
end

using Statistics;

function average_curves(arrs::Array{Float64,2})
    return mean(arrs, dims=1)[1,:]
end
                    
# compute standard deviation of betti numbers over arrays in arrs
# assuming arrs is an array of arrays, where each arrs[j] is the same size
                    

function std_arrs(arrs::Array{Float64,2})
    std_arrs=zeros(size(arrs,2))
    for i in 1:length(std_arrs)
        std_arrs[i]=std(arrs[:,i])
    end
    return std_arrs
end

using StatsBase;

function find_conf_interval(arrs::Array{Float64,2})
    lower_bds=zeros(size(arrs,2))
    upper_bds=zeros(size(arrs,2))
    for i in 1:length(lower_bds)
        lower_bds[i]=percentile(arrs[:,i],2.5)
        upper_bds[i]=percentile(arrs[:,i],97.5)
    end
    return (upper_bds,lower_bds)
end

function find_lower_and_upper_bds(arrs::Array{Float64,2})
    upper_bds=find_conf_interval(arrs)[1]
    lower_bds=find_conf_interval(arrs)[2]
    mean_vals=average_curves(arrs)
    upper_bds=upper_bds.-mean_vals
    lower_bds=mean_vals.-lower_bds
    return (upper_bds,lower_bds)
end

#plot average curves at values `xval`, with averages given by `means` and standard deviations given by `std`
                    
function plot_averages(xvals, means, stds; ribbon=true, label="", linestyle=:solid,linewidth=linewidth, color=:auto)
    if ribbon
       # return Plots.plot(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticksfontsize=16)
        return Plots.plot(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=[1,2,3,4,5,6,7,8,9,10,20,30,40],legendfontsize=16,ticksfontsize=16)
    else
        return Plots.plot(xvals, means, labels=label, linestyle=linestyle, c=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticskfontsize=16)
    end
end

function plot_averages!(xvals, means, stds; ribbon=true, label="", linestyle=:solid,linewidth=linewidth, color=:auto)
    if ribbon
#        return Plots.plot!(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticksfontsize=16)
        return Plots.plot!(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=[1,2,3,4,5,6,7,8,9,10,20,30,40],legendfontsize=16,ticksfontsize=16)
    else
        return Plots.plot!(xvals, means, labels=label, linestyle=linestyle, c=color,grid=false,legend=:bottomright,legendfontsize=16,xticks=1:2:32,tickfontsize=12)
    end
end

descr_name="pca_cact";#'cact";#"smarts42"; all_odors_cact_hyperparameter_search_varying_perc_data,83Lresults_10_fold
dataset_name="all_odors";

peer_pred_83L=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_peer_prediction_results.jld"),"peer_pred_2");
peer_pred_pr_auc83L=Array{Float64,1}(undef,length(peer_pred_83L));
peer_pred_roc_auc83L=Array{Float64,1}(undef,length(peer_pred_83L));
for i in 1:length(peer_pred_83L)
    peer_pred_pr_auc83L[i]=peer_pred_83L[i][2]
    peer_pred_roc_auc83L[i]=peer_pred_83L[i][4]
end

lower_bds_peer=percentile(peer_pred_roc_auc83L,2.5);
upper_bds_peer=percentile(peer_pred_roc_auc83L,97.5);
lower_bds_peer=mean(peer_pred_roc_auc83L)-lower_bds_peer;
upper_bds_peer=upper_bds_peer-mean(peer_pred_roc_auc83L);

lower_bds_peer
upper_bds_peer
#lower_bds_peer
#upper_bds_peer
#mean(peer_pred_roc_auc83L)

#pred_metrics_10_fold_83L_with_area=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/adding_area_hyperparameter_search_1.0_perc_features.jld"),"83Lresults_10_fold");
binary_responses=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
baseline_pred_pr_auc83L=sum(binary_responses[1])/(size(binary_responses[1],1)*size(binary_responses[1],2));

#n_seeds=100;
#n_features=32;
n_seeds=200;
n_features=13;
feature_idxs=[1,2,3,4,5,6,7,8,9,10,20,30,40]
#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_positions_and_identity.jld"));

#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_chemotopy_no_area.jld"),"83Lresults_10_fold");
#pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_shuffled_positions.jld"),"83Lresults_10_fold");

pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_200_random_seeds_chemotopy_analysis.jld"),"bulb_1_real_results");
pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_200_random_seeds_chemotopy_analysis.jld"),"bulb_1_control_results");
#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_200_random_seeds_chemotopy_real.jld"),"bulb_1_real_results");
#pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_200_random_seeds_chemotopy_control.jld"),"bulb_1_control_results");

#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_only_positions.jld"),"bulb_1_real_results");
#pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotuning_only_identities.jld"),"bulb_1_control_results");
#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_only_positions.jld"),"bulb_2_real_results");
#pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_shuffled_positions.jld"),"bulb_2_control_results");
#pred_metrics_10_fold_chemotopy=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_positions_and_identity.jld"),"bulb_2_real_results");
#pred_metrics_10_fold_chemotuning=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_new_chemotopy_positions_and_identity.jld"),"bulb_2_control_results");
pred_10_fold_arrs_pr_auc_chemotopy=zeros(n_seeds,n_features);
pred_10_fold_arrs_roc_auc_chemotopy=zeros(n_seeds,n_features);
pred_10_fold_arrs_pr_auc_chemotuning=zeros(n_seeds,n_features);
pred_10_fold_arrs_roc_auc_chemotuning=zeros(n_seeds,n_features);
for i in 1:n_seeds
    for j in 1:n_features
        pred_10_fold_arrs_pr_auc_chemotopy[i,j]=pred_metrics_10_fold_chemotopy[1,1,1,i,j][2]
        pred_10_fold_arrs_roc_auc_chemotopy[i,j]=pred_metrics_10_fold_chemotopy[1,1,1,i,j][4]
        pred_10_fold_arrs_pr_auc_chemotuning[i,j]=pred_metrics_10_fold_chemotuning[1,1,1,i,j][2]
        pred_10_fold_arrs_roc_auc_chemotuning[i,j]=pred_metrics_10_fold_chemotuning[1,1,1,i,j][4]
    end
end

A_pca_cact=pred_10_fold_arrs_roc_auc_chemotopy;
B_pca_cact=pred_10_fold_arrs_roc_auc_chemotuning;

A_cact=pred_10_fold_arrs_pr_auc_chemotopy;
B_cact=pred_10_fold_arrs_pr_auc_chemotuning;

A_pom21=pred_10_fold_arrs_roc_auc_chemotopy;
B_pom21=pred_10_fold_arrs_roc_auc_chemotuning;


#Figure 3 Panel A
p=plot_averages(feature_idxs,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,color=:magenta,label="Chemotopy PCA-PubChem")
plot_averages!(feature_idxs,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,color=:cyan,label="Chemotuning PCA-PubChem")
plot_averages!(feature_idxs,average_curves(A_pom21),find_lower_and_upper_bds(A_pom21),linewidth=2,color=:red,label="Chemotopy GNN")
plot_averages!(feature_idxs,average_curves(B_pom21),find_lower_and_upper_bds(B_pom21),linewidth=2,color=:blue,label="Chemotuning GNN")
plot!(feature_idxs,repeat([mean(peer_pred_roc_auc83L)],13),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
#plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],13),linewidth=2,color=:teal,label="Baseline",ylims=[0.5,0.95],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=[2,4,6,8,10,20,30,40],legendfontsize=13,size=(500,500))
plot!(feature_idxs,repeat([0.5],13),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=[2,4,6,8,10,20,30,40],legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","SuppFig2_PanelB_new.svg"));



p=plot_averages(feature_idxs,average_curves(A_cact),find_lower_and_upper_bds(A_cact),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_cact),find_lower_and_upper_bds(B_cact),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],32),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_cact_no_amines_bulb_1.svg"));

p=plot_averages(feature_idxs,average_curves(A_pom21),find_lower_and_upper_bds(A_pom21),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_pom21),find_lower_and_upper_bds(B_pom21),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],32),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_pom21_all_odors_bulb_2.svg"));

p=plot_averages(feature_idxs,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],32),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_pca_cact_all_odors_bulb_1.svg"));

using Random
A=shuffle(1:100)
typeof(A)
descr_numbers=1:32
typeof(descr_numbers)
custom_values = [1:10; 20; 30; 40]
typeof

findmax(custom_values,1:5)

A_pom21

feature_idxs=[1,2,3,4,5,6,7,8,9,10,20,30,40]
p=plot_averages(feature_idxs,average_curves(A_pom21),find_lower_and_upper_bds(A_pom21),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_pom21),find_lower_and_upper_bds(B_pom21),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],13),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],13),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=[2,4,6,8,10,20,30,40],legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_and_identities_pom21_all_odors_bulb_1.svg"));

p=plot_averages(feature_idxs,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],13),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],13),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=[2,4,6,8,10,20,30,40],legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_and_identities_pca_cact_no_amines_bulb_1.svg"));

p=plot_averages(feature_idxs,average_curves(A_cact),find_lower_and_upper_bds(A_cact),linewidth=2,color=:red,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_cact),find_lower_and_upper_bds(B_cact),linewidth=2,color=:blue,label="Chemotuning")
#plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],13),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],13),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=[2,4,6,8,10,20,30,40],legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_shuffled_positions_and_identities_cact_no_amines_bulb_2.svg"));


#Make figure 3 panel C

Plots.boxplot([0.5],A_pca_cact[:,2],color=:magenta,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2)
Plots.boxplot!([1],B_pca_cact[:,2],color=:cyan,grid=false,ticksfontsize=14,label="Chemotuning",bar_width=0.2)
#Plots.boxplot!([1.5],A_pca_cact[:,10],color=orange_yellow_rgb,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2);
#Plots.boxplot!([2],B_pca_cact[:,10],color=:green,grid=false,ticksfontsize=14,label="Chemotuning",bar_width=0.2);
Plots.boxplot!([1.5],A_pom21[:,2],color=:red,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2)
Plots.boxplot!([2],B_pom21[:,2],color=:blue,grid=false,ticksfontsize=14,label="Chemotuning",bar_width = 0.2)
Plots.boxplot!([2.5],A_pca_cact[:,13],color=:magenta,grid=false,ticksfontsize=14,label="",bar_width=0.2)
Plots.boxplot!([3],B_pca_cact[:,13],color=:cyan,grid=false,ticksfontsize=14,label="",bar_width=0.2)
#Plots.boxplot!([4.5],A_pca_cact[:,13],color=orange_yellow_rgb,grid=false,ticksfontsize=14,label="",bar_width=0.2)
#Plots.boxplot!([5],B_pca_cact[:,32],color=:green,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([3.5],A_pom21[:,13],color=:red,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([4],B_pom21[:,13],color=:blue,grid=false,ticksfontsize=14,label="",legend=:outerright,size=(1000,500),bar_width = 0.2,xticks=[1.25,3.25,5.25])
#Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures/SupFig4","Panel_C.svg"));
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir","SuppFig3_PanelB.svg"));

A_pca_cact