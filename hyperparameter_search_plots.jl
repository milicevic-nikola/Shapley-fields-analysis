using JLD, Plots, StatsPlots;

descr_name="pom21";#"smarts42";
#load 10 fold predictions:
pred_metrics_10_fold_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Rresults_10_fold");
pred_metrics_per_glom_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Rresults_leave_one_out_per_glom");
pred_metrics_leave_one_out_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_$descr_name"*"_hyperparameter_search.jld"),"83Rresults_leave_one_out");

pred_metrics_10_fold_83L_with_area=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/adding_area_hyperparameter_search_1.0_perc_features.jld"),"83Lresults_10_fold");

pred_metrics_10_fold_cact83L=zeros(size(pred_metrics_10_fold_83L_with_area)[4],30);
#pred_metrics_leave_one_out_cact83L=zeros(size(pred_metrics_10_fold_83L)[4],30);
#pred_metrics_per_glom_cact83L=zeros(size(pred_metrics_10_fold_83L)[4],30);
for l in 1:size(pred_metrics_10_fold_83L_with_area)[4]
    for m in 1:30
        pred_metrics_10_fold_cact83L[l,m]=pred_metrics_10_fold_83L_with_area[1,1,1,l,:][m][2]
  #      pred_metrics_per_glom_cact83L[l,m]=pred_metrics_per_glom_83L[1,1,l,:][m][2]
   #     pred_metrics_leave_one_out_cact83L[l,m]=pred_metrics_leave_one_out_83L[1,1,1,l,:][m][2]
    end
end
pred_metrics_10_fold_cact83L=mean(pred_metrics_10_fold_cact83L, dims=1)[1,:];
pred_metrics_per_glom_cact83L=mean(pred_metrics_per_glom_cact83L, dims=1)[1,:];
pred_metrics_leave_one_out_cact83L=mean(pred_metrics_leave_one_out_cact83L, dims=1)[1,:];


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


n_featurers=24;#24


using Statistics;
pr_auc_curves=[];
for i in 1:size(pred_metrics_10_fold_83L)[1]
    for j in 1:size(pred_metrics_10_fold_83L)[2]
        for k in 1:size(pred_metrics_10_fold_83L)[3]
            pred_metrics_10_fold_cact83L=zeros(size(pred_metrics_10_fold_83L)[4],n_featurers);
            pred_metrics_leave_one_out_cact83L=zeros(size(pred_metrics_10_fold_83L)[4],n_featurers);
            pred_metrics_per_glom_cact83L=zeros(size(pred_metrics_10_fold_83L)[4],n_featurers);
            for l in 1:size(pred_metrics_10_fold_83L)[4]
                for m in 1:n_featurers
                    pred_metrics_10_fold_cact83L[l,m]=pred_metrics_10_fold_83L[i,j,k,l,:][m][2]
                    pred_metrics_per_glom_cact83L[l,m]=pred_metrics_per_glom_83L[i,j,l,:][m][2]
                    pred_metrics_leave_one_out_cact83L[l,m]=pred_metrics_leave_one_out_83L[i,j,k,l,:][m][2]
                end
            end
            pred_metrics_10_fold_cact83L=mean(pred_metrics_10_fold_cact83L, dims=1)[1,:]
            pred_metrics_per_glom_cact83L=mean(pred_metrics_per_glom_cact83L, dims=1)[1,:]
            pred_metrics_leave_one_out_cact83L=mean(pred_metrics_leave_one_out_cact83L, dims=1)[1,:]
            p=curves_to_plot(pred_metrics_10_fold_cact83L,pred_metrics_leave_one_out_cact83L,
                            pred_metrics_per_glom_cact83L,"Shapley ordered omp83L",
                            "PRC AUC", "number of features",
                            "CD+Position 10 fold","CD+Position LOO","CD LOO",[0.2,0.7],[1,24])        
            push!(pr_auc_curves,p)
        end
    end
end
Plots.plot(pr_auc_curves[1])
Plots.plot!(1:30,pred_metrics_10_fold_cact83L,label="CD+PA 10 fold",ylims=[0.2,0.7])

Plots.savefig(joinpath(homedir(),"Dropbox/figures","83L_hyperparameter_pr_auc_$descr_name"*".png"));

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","83L_hyperparameter_pr_auc_$descr_name"*".png"));

plot(pr_auc_curves[9],pr_auc_curves[10],
    pr_auc_curves[11],pr_auc_curves[12],  layout = (2,2),size=(1000,750))
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","83L_hyperparameter_mean_roc_auc.png"));

pred_metrics_10_fold_cact83L=Array{Float64,1}(undef,40);
pred_metrics_leave_one_out_cact83L=Array{Float64,1}(undef,40);
pred_metrics_per_glom_cact83L=Array{Float64,1}(undef,40);
for i in 1:length(pred_metrics_10_fold_cact83L)
    pred_metrics_10_fold_cact83L[i]=pred_metrics_10_fold_83L[1,1,1,1,:][i][2]
    pred_metrics_per_glom_cact83L[i]=pred_metrics_per_glom_83L[1,1,1,:][i][2]
    pred_metrics_leave_one_out_cact83L[i]=pred_metrics_leave_one_out_83L[1,1,1,1,:][i][2]
end

p=curves_to_plot(pred_metrics_10_fold_cact83L,pred_metrics_leave_one_out_cact83L,
                    pred_metrics_per_glom_cact83L,"Shapley ordered omp83L",
                    "PRC AUC", "number of features",
                    "CD+Position 10 fold","CD+Position LOO","CD LOO",[0.2,0.7],[1,40])        

function average_bettis(arrs; maxdim=-1)
    if size(arrs,2) > 1
        return arrs
    end
    md = maxdim
    if maxdim == -1
        md = size(arrs[1],2)
    end
    numofints = size(arrs[1],1)
    av_bet = zeros(numofints,md)                  
    for i=1:numofints
        for d=1:md
            av_bet[i,d] = mean([arrs[j][i,d] for j=1:length(arrs)])
        end
    end
    return av_bet
end
                    
                    # compute standard deviation of betti numbers over arrays in arrs
                    # assuming arrs is an array of arrays, where each arrs[j] is the same size
                    
                    function std_bettis(arrs; maxdim=-1)
                        md = maxdim
                        if maxdim == -1
                            md = size(arrs[1],2)
                        end
                        numofints = size(arrs[1],1)
                        std_bet = zeros(numofints,md)
                    
                        if size(arrs,2) > 1
                            return std_bet
                        end
                    
                        for i=1:numofints
                            for d=1:md
                                std_bet[i,d] = std([arrs[j][i,d] for j=1:length(arrs)])
                            end
                        end
                        return std_bet
                    end
                    
                    # plot average curves at values `xval`, with averages given by `means` and standard deviations given by `std`
                    
                    function plot_averages(xvals, means, stds; ribbon=true, label="", linestyle=:solid, color=:auto)
                        if ribbon
                             return plot(xvals, means, ribbon=stds,fillalpha=.3, labels=label, linestyle=linestyle, color=color)
                        else
                            return plot(xvals, means, labels=label, linestyle=linestyle, c=color)
                        end
                    end