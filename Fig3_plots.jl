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
        return Plots.plot(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticksfontsize=16)
    else
        return Plots.plot(xvals, means, labels=label, linestyle=linestyle, c=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticskfontsize=16)
    end
end

function plot_averages!(xvals, means, stds; ribbon=true, label="", linestyle=:solid,linewidth=linewidth, color=:auto)
    if ribbon
        return Plots.plot!(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, linewidth=linewidth,color=color,grid=false,legend=:bottomright,xticks=1:2:32,legendfontsize=16,ticksfontsize=16)
    else
        return Plots.plot!(xvals, means, labels=label, linestyle=linestyle, c=color,grid=false,legend=:bottomright,legendfontsize=16,xticks=1:2:32,tickfontsize=12)
    end
end

descr_name="pom21";#'cact";#"smarts42"; all_odors_cact_hyperparameter_search_varying_perc_data,83Lresults_10_fold
dataset_name="all_odors";
#load tuples of different metric performances of different models
pred_metrics_10_fold_83L_50=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83L_100=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search_more_random_seeds.jld"),"83Lresults_10_fold");
pred_metrics_per_glom_83L_50=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83L_100=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search_more_random_seeds.jld"),"83Lresults_leave_one_out_per_glom");

#pred_metrics_10_fold_83R=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search.jld"),"83Rresults_10_fold");
#pred_metrics_per_glom_83L=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search_more_random_seeds.jld"),"83Lresults_leave_one_out_per_glom");
#pred_metrics_per_glom_83R=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_hyperparameter_search.jld"),"83Rresults_leave_one_out_per_glom");

peer_pred_83L=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_peer_prediction_results.jld"),"peer_pred_1");
peer_pred_pr_auc83L=Array{Float64,1}(undef,length(peer_pred_83L));
peer_pred_roc_auc83L=Array{Float64,1}(undef,length(peer_pred_83L));
for i in 1:length(peer_pred_83L)
    peer_pred_pr_auc83L[i]=peer_pred_83L[i][2]
    peer_pred_roc_auc83L[i]=peer_pred_83L[i][4]
end

lower_bds_peer=percentile(peer_pred_roc_auc83L,2.5);
upper_bds_peer=percentile(peer_pred_roc_auc83L,97.5);
lower_bds_peer=mean(peer_pred_pr_auc83L)-lower_bds_peer;
upper_bds_peer=upper_bds_peer-mean(peer_pred_pr_auc83L);


#pred_metrics_10_fold_83L_with_area=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/adding_area_hyperparameter_search_1.0_perc_features.jld"),"83Lresults_10_fold");
binary_responses=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
baseline_pred_pr_auc83L=sum(binary_responses[2])/(size(binary_responses[2],1)*size(binary_responses[2],2));
#baseline_pred_acc83L=1-baseline_pred_pr_auc83L;

n_seeds=50;
n_features=40;#40;#24,22,32

#Extract the performance arrays from the tuples loaded
pred_10_fold_arrs_pr_auc_50=zeros(n_seeds,n_features);
pred_10_fold_arrs_roc_auc_50=zeros(n_seeds,n_features);
#pred_loo_arrs_pr_auc=zeros(n_seeds,n_features);
#pred_loo_arrs_roc_auc=zeros(n_seeds,n_features);
pred_per_glom_arrs_pr_auc_50=zeros(n_seeds,n_features);
pred_per_glom_arrs_roc_auc_50=zeros(n_seeds,n_features);
pred_10_fold_arrs_pr_auc_100=zeros(n_seeds,n_features);
pred_10_fold_arrs_roc_auc_100=zeros(n_seeds,n_features);
pred_per_glom_arrs_pr_auc_100=zeros(n_seeds,n_features);
pred_per_glom_arrs_roc_auc_100=zeros(n_seeds,n_features);
for i in 1:n_seeds
    for j in 1:n_features
        pred_10_fold_arrs_pr_auc_50[i,j]=pred_metrics_10_fold_83L_50[1,1,1,i,j][2]
       # pred_loo_arrs_pr_auc[i,j]=pred_metrics_leave_one_out_83L[1,1,1,i,j][2]
        pred_per_glom_arrs_pr_auc_50[i,j]=pred_metrics_per_glom_83L_50[1,1,i,j][2]
        pred_10_fold_arrs_roc_auc_50[i,j]=pred_metrics_10_fold_83L_50[1,1,1,i,j][4]
       # pred_loo_arrs_roc_auc[i,j]=pred_metrics_leave_one_out_83L[1,1,1,i,j][4]
        pred_per_glom_arrs_roc_auc_50[i,j]=pred_metrics_per_glom_83L_50[1,1,i,j][4]
        pred_10_fold_arrs_pr_auc_100[i,j]=pred_metrics_10_fold_83L_100[1,1,1,i,j][2]
        pred_per_glom_arrs_pr_auc_100[i,j]=pred_metrics_per_glom_83L_100[1,1,i,j][2]
        pred_10_fold_arrs_roc_auc_100[i,j]=pred_metrics_10_fold_83L_100[1,1,1,i,j][4]
        pred_per_glom_arrs_roc_auc_100[i,j]=pred_metrics_per_glom_83L_100[1,1,i,j][4]
    end
end

#A_smarts=pred_10_fold_arrs_pr_auc;
#B_smarts=pred_per_glom_arrs_pr_auc;
A_cact=vcat(pred_10_fold_arrs_pr_auc_50,pred_10_fold_arrs_pr_auc_100);
B_cact=vcat(pred_per_glom_arrs_pr_auc_50,pred_per_glom_arrs_pr_auc_100);

A_pom21=vcat(pred_10_fold_arrs_pr_auc_50,pred_10_fold_arrs_pr_auc_100);
B_pom21=vcat(pred_per_glom_arrs_pr_auc_50,pred_per_glom_arrs_pr_auc_100);

A_pca_cact=vcat(pred_10_fold_arrs_roc_auc_50,pred_10_fold_arrs_roc_auc_100);
B_pca_cact=vcat(pred_per_glom_arrs_roc_auc_50,pred_per_glom_arrs_roc_auc_100);
#A_cact_roc=pred_10_fold_arrs_roc_auc;
#B_cact_roc=pred_per_glom_arrs_roc_auc;

A_haddad=vcat(pred_10_fold_arrs_roc_auc_50,pred_10_fold_arrs_roc_auc_100);
B_haddad=vcat(pred_per_glom_arrs_roc_auc_50,pred_per_glom_arrs_roc_auc_100);
#A_maccs=pred_10_fold_arrs_pr_auc;
#B_maccs=pred_per_glom_arrs_pr_auc;

prauc_mean=mean(B_haddad,dims=1);
prauc_std_devs=std(B_haddad,dims=1);
feature_with_largest_prauc=argmax(prauc_mean);

println(feature_with_largest_prauc)
println(prauc_mean[feature_with_largest_prauc])
println(prauc_std_devs[feature_with_largest_prauc])

#Make figure 3 panel A
using Colors;
orange_yellow_rgb = RGB(255/255, 165/255, 0/255);  # Dividing by 255 to scale between 0 and 1

feature_idxs=1:32;

n_seeds=100;
n_features=32;
pred_metrics_10_fold_no_area=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_chemotopy_no_area.jld"),"83Lresults_10_fold");
pred_10_fold_arrs_pr_auc_no_area=zeros(n_seeds,n_features);
pred_10_fold_arrs_roc_auc_no_area=zeros(n_seeds,n_features);
for i in 1:n_seeds
    for j in 1:n_features
        pred_10_fold_arrs_pr_auc_no_area[i,j]=pred_metrics_10_fold_no_area[1,1,1,i,j][2]
        pred_10_fold_arrs_roc_auc_no_area[i,j]=pred_metrics_10_fold_no_area[1,1,1,i,j][4]
    end
end

A_no_area=pred_10_fold_arrs_pr_auc_no_area
A_no_area_pom21=pred_10_fold_arrs_pr_auc_no_area

p=plot_averages(feature_idxs,average_curves(A_cact[:,feature_idxs]),find_lower_and_upper_bds(A_cact[:,feature_idxs]),linewidth=2,color=:magenta,label="Chemotopy")
plot_averages!(feature_idxs,average_curves(B_cact[:,feature_idxs]),find_lower_and_upper_bds(B_cact[:,feature_idxs]),linewidth=2,color=:cyan,label="Chemotuning")
plot_averages!(feature_idxs,average_curves(A_no_area),find_lower_and_upper_bds(A_no_area),linewidth=2,color=:red,label="Chemotopy no area")
#plot_averages!(feature_idxs,average_curves(A_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=orange_yellow_rgb,label="Chemotopy (PCA-PubChem)")
#plot_averages!(feature_idxs,average_curves(B_pca_cact[:,feature_idxs]),find_lower_and_upper_bds(B_pca_cact[:,feature_idxs]),linewidth=2,color=:green,label="Chemotuning (PCA-PubChem)")
plot!(feature_idxs,repeat([mean(peer_pred_pr_auc83L)],32),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/","Chemotopy_no_area_pom21.svg"));




#Make figure 3 panel B_cact
p=plot_averages(feature_idxs,average_curves(A_pom21[:,feature_idxs]),find_lower_and_upper_bds(A_pom21[:,feature_idxs]),linewidth=2,color=:red,label="Chemotopy (GNN)");
plot_averages!(feature_idxs,average_curves(B_pom21[:,feature_idxs]),find_lower_and_upper_bds(B_pom21[:,feature_idxs]),linewidth=2,color=:blue,label="Chemotuning (GNN)");
plot!(feature_idxs,repeat([mean(peer_pred_roc_auc83L)],32),linewidth=2,color=:black,label="Peer prediction",linestyle=:dash);
plot!(feature_idxs,repeat([0.5],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.45,0.92],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#plot!(feature_idxs,repeat([baseline_pred_pr_auc83L],32),linewidth=2,color=:teal,label="Baseline",ylims=[0.1,0.75],linestyle=:dot,legend=:bottomright,tickfontsize=12,xticks=1:2:33,legendfontsize=13,size=(500,500))
#Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures/SupFig5","Panel_B.svg"));
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/SupFig8","Panel_B.svg"));

#Make figure 3 panel C

Plots.boxplot([0.5],A_cact[:,10],color=:magenta,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2);
Plots.boxplot!([1],B_cact[:,10],color=:cyan,grid=false,ticksfontsize=14,label="Chemotuning",bar_width=0.2);
Plots.boxplot!([1.5],A_pca_cact[:,10],color=orange_yellow_rgb,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2);
Plots.boxplot!([2],B_pca_cact[:,10],color=:green,grid=false,ticksfontsize=14,label="Chemotuning",bar_width=0.2);
Plots.boxplot!([2.5],A_pom21[:,10],color=:red,grid=false,ticksfontsize=14,label="Chemotopy",bar_width=0.2);
Plots.boxplot!([3],B_pom21[:,10],color=:blue,grid=false,ticksfontsize=14,label="Chemotuning",bar_width = 0.2);
Plots.boxplot!([3.5],A_cact[:,32],color=:magenta,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([4],B_cact[:,32],color=:cyan,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([4.5],A_pca_cact[:,32],color=orange_yellow_rgb,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([5],B_pca_cact[:,32],color=:green,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([5.5],A_pom21[:,32],color=:red,grid=false,ticksfontsize=14,label="",bar_width=0.2);
Plots.boxplot!([6],B_pom21[:,32],color=:blue,grid=false,ticksfontsize=14,label="",legend=:outerright,size=(1000,500),bar_width = 0.2,xticks=[1.25,3.25,5.25])
#Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures/SupFig4","Panel_C.svg"));
Plots.savefig(joinpath(homedir(),"Documents/Nikola-Vladimir/SupFig8","Panel_C.svg"));


#p=plot_averages(1:24,average_curves(A_smarts),find_lower_and_upper_bds(A_smarts),ribbon=false,linewidth=4,color=:blue,label="Smarts");
p=plot_averages(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_cact),ribbon=false,linewidth=4,color=:blue,label="Chemotopy")
plot_averages!(1:40,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),ribbon=false,linewidth=4,color=:red,label="Chemotuning")
plot_averages!(1:40,average_curves(A_cact),find_lower_and_upper_bds(A_cact),ribbon=false,linewidth=4,color=:green,label="Chemotopy (per glom)")
plot_averages!(1:40,average_curves(B_cact),find_lower_and_upper_bds(B_cact),ribbon=false,linewidth=4,color=:yellow,label="Chemotuning (per glom)")

plot_averages!(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),ribbon=false,linewidth=4,color=:blue,label="PCA-PubChem")

p=plot_averages(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),ribbon=false,linewidth=4,color=:blue,label="PCA-PubChem")
plot_averages!(1:40,average_curves(A_cact),find_lower_and_upper_bds(A_cact),ribbon=false,linewidth=4,color=:red,label="PubChem");
plot_averages!(1:40,average_curves(A_pom21),find_lower_and_upper_bds(A_pom21),ribbon=false,linewidth=4,color=:green,label="GNN");
#plot_averages!(1:30,average_curves(A_maccs),find_lower_and_upper_bds(A_maccs),ribbon=false,linewidth=4,color=:pink,label="Maccs");
plot_averages!(1:30,average_curves(A_haddad),find_lower_and_upper_bds(A_haddad),ribbon=false,linewidth=4,color=:orange,label="Dragon32")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","chemotopy_all_descrs_83L.svg"));

#p=plot_averages(1:24,average_curves(B_smarts),find_lower_and_upper_bds(B_smarts),ribbon=false,linewidth=4,color=:blue,label="Smarts");
p=plot_averages(1:40,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),ribbon=false,linewidth=4,color=:blue,label="PCA-PubChem");
plot_averages!(1:40,average_curves(B_cact),find_lower_and_upper_bds(B_cact),ribbon=false,linewidth=4,color=:red,label="PubChem");
plot_averages!(1:40,average_curves(B_pom21),find_lower_and_upper_bds(B_pom21),ribbon=false,linewidth=4,color=:green,label="GNN");
#plot_averages!(1:30,average_curves(B_maccs),find_lower_and_upper_bds(B_maccs),ribbon=false,linewidth=4,color=:black,label="Maccs");
plot_averages!(1:30,average_curves(B_haddad),find_lower_and_upper_bds(B_haddad),ribbon=false,linewidth=4,color=:orange,label="Dragon32")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","tunotopy_all_descrs_83L.svg"));

#p=plot_averages(1:24,average_curves(A_smarts),find_lower_and_upper_bds(A_smarts),linewidth=2,color=:blue,label="Chemotopy");
#plot_averages!(1:24,average_curves(B_smarts),find_lower_and_upper_bds(B_smarts),linewidth=2,color=:red,label="Tunotopy")
#Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","smarts83L.svg"));

p=plot_averages(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,color=:blue,label="Chemotopy");
plot_averages!(1:40,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,color=:red,label="Chemotuning")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","pca_cact_83L.svg"));

p=plot_averages(1:40,average_curves(A_cact),find_lower_and_upper_bds(A_cact),linewidth=2,color=:blue,label="Chemotopy");
plot_averages!(1:40,average_curves(B_cact),find_lower_and_upper_bds(B_cact),linewidth=2,color=:red,label="Chemotuning")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","cact_83L_per_glom_order.svg"));

p=plot_averages(1:40,average_curves(A_pom21),find_lower_and_upper_bds(A_pom21),linewidth=2,color=:blue,label="Chemotopy");
plot_averages!(1:40,average_curves(B_pom21),find_lower_and_upper_bds(B_pom21),linewidth=2,color=:red,label="Chemotuning")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","pom21_83L.svg"));

#p=plot_averages(1:30,average_curves(A_maccs),find_lower_and_upper_bds(A_maccs),linewidth=2,color=:blue,label="Chemotopy");
#plot_averages!(1:30,average_curves(B_maccs),find_lower_and_upper_bds(B_maccs),linewidth=2,color=:red,label="Tunotopy")
#Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","maccs83L.svg"));

p=plot_averages(1:30,average_curves(A_haddad),find_lower_and_upper_bds(A_haddad),linewidth=2,color=:blue,label="Chemotopy");
plot_averages!(1:30,average_curves(B_haddad),find_lower_and_upper_bds(B_haddad),linewidth=2,color=:red,label="Chemotuning")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","haddad_83L.svg"));

Plots.boxplot(["PubChem"],A_cact[:,40],color=:red,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["GNN"],A_pom21[:,40],color=:green,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["Dragon32"],A_haddad[:,30],color=:orange,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["PCA-PubChem"],A_pca_cact[:,40],color=:blue,grid=false,ticksfontsize=14,label="")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","chemotopy_boxplots_83L.svg"));

Plots.boxplot(["PubChem"],B_cact[:,40],color=:red,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["GNN"],B_pom21[:,40],color=:green,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["Dragon32"],B_haddad[:,30],color=:orange,grid=false,ticksfontsize=14,label="")
Plots.boxplot!(["PCA-PubChem"],B_pca_cact[:,40],color=:blue,grid=false,ticksfontsize=14,label="")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","chemotuning_boxplots_83L.svg"));

C=average_curves(pred_10_fold_arrs_pr_auc);#pca_cact
D=average_curves(pred_per_glom_arrs_pr_auc);#pca_cact
#using Plots.
#using PyPlot;
n_features=20;
gr()

p=plot_averages(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,color=:blue,label="Chemotopy")
#plot_averages!(1:n_features,average_curves(pred_loo_arrs_pr_auc),std_arrs(pred_loo_arrs_pr_auc),linewidth=2,color=:red,label="CHEM-PA CV-LOO");
plot_averages!(1:40,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,color=:red2,label="Chemotuning")
plot_averages!(1:40,average_curves(A_cact),find_lower_and_upper_bds(A_cact),linewidth=2,color=:green,label="Chemotopy (per glom)")
plot_averages!(1:40,average_curves(B_cact),find_lower_and_upper_bds(B_cact),linewidth=2,color=:yellow,label="Chemotuning (per glom)")
plot_averages!(1:40,average_curves(A_pca_cact),find_lower_and_upper_bds(A_pca_cact),linewidth=2,linestyle=:dash,color=:green,label="Chemotopy (PCA-PubChem)");
plot_averages!(1:40,average_curves(B_pca_cact),find_lower_and_upper_bds(B_pca_cact),linewidth=2,linestyle=:dash,color=:yellow,label="Chemotuning (PCA-PubChem)")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","pred_curves_means_ordering_comparisons.svg"));



# descr_name="cact";
# shuffled_curves=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/pred_metrics/$dataset_name"*"_$descr_name"*"_shuffled_descrs.jld"),"animal1_results");
# shuffled_10_fold=fill(0.0,40);
# shuffled_10_fold_mean=fill(0.0,40);
# shuffled_per_glom=fill(0.0,40);
# shuffled_per_glom_mean=fill(0.0,40);
# for i in 1:40
#     shuffled_10_fold[i]=shuffled_curves[i,1][2][2]
#     shuffled_10_fold_mean[i]=shuffled_curves[i,1][2][3]
#     shuffled_per_glom[i]=shuffled_curves[i,1][1][2]
#     shuffled_per_glom_mean[i]=shuffled_curves[i,1][1][3]
# end


Plots.boxplot!(["Shuffled PubChem"],shuffled_10_fold,color=:blue,grid=false,ticksfontsize=14,label="");
Plots.boxplot!(["PubChem (mean)"],A_cact_mean[:,40],color=:red,grid=false,ticksfontsize=14,label="");
Plots.boxplot!(["Shuffled PubChem (mean)"],shuffled_10_fold_mean,color=:blue,grid=false,ticksfontsize=14,label="",size=(1000,500))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","chemotopy_shuffled_boxplots_83L.svg"));

Plots.boxplot(["PubChem"],B_cact[:,40],color=:red,grid=false,ticksfontsize=14,label="");
Plots.boxplot!(["Shuffled PubChem"],shuffled_per_glom,color=:blue,grid=false,ticksfontsize=14,label="");
Plots.boxplot!(["PubChem (mean)"],B_cact_mean[:,40],color=:red,grid=false,ticksfontsize=14,label="");
Plots.boxplot!(["Shuffled PubChem (mean)"],shuffled_per_glom_mean,color=:blue,grid=false,ticksfontsize=14,label="",size=(1000,500))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","chemotuning_shuffled_boxplots_83L.svg"));

#plot!(1:n_features,repeat([0.5],n_features),linewidth=2,color=:black,label="Baseline",legend=:right,ylims=[0.4,0.9])