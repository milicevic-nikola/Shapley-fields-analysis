using JLD, Plots, StatsPlots;

#load 10 fold predictions:
pred_metrics_10_fold_cact83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_hyperparameter_search.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_cact83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_hyperparameter_search.jld"),"83Lresults_10_fold");


#load 10 fold chemical feature prediction
pred_metrics_10_fold_83L_edragon=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_edragon1377_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_edragon=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_edragon1377_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_edragon=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_edragon));
pred_pr_auc_10_fold_83R_edragon=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_edragon));
for i in 1:length(pred_metrics_10_fold_83L_edragon)
    pred_pr_auc_10_fold_83L_edragon[i]=pred_metrics_10_fold_83L_edragon[i][2]
    pred_pr_auc_10_fold_83R_edragon[i]=pred_metrics_10_fold_83R_edragon[i][2]
end

pred_metrics_10_fold_83L_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_maccs_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_maccs_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_maccs=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_maccs));
pred_pr_auc_10_fold_83R_maccs=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_maccs));
for i in 1:length(pred_metrics_10_fold_83L_maccs)
    pred_pr_auc_10_fold_83L_maccs[i]=pred_metrics_10_fold_83L_maccs[i][2]
    pred_pr_auc_10_fold_83R_maccs[i]=pred_metrics_10_fold_83R_maccs[i][2]
end

pred_metrics_10_fold_83L_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_smarts42_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_smarts42_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_smarts42=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_smarts42));
pred_pr_auc_10_fold_83R_smarts42=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_smarts42));
for i in 1:length(pred_metrics_10_fold_83L_smarts42)
    pred_pr_auc_10_fold_83L_smarts42[i]=pred_metrics_10_fold_83L_smarts42[i][2]
    pred_pr_auc_10_fold_83R_smarts42[i]=pred_metrics_10_fold_83R_smarts42[i][2]
end

pred_metrics_10_fold_83L_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_haddad32_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_haddad32_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_haddad32=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_smarts42));
pred_pr_auc_10_fold_83R_haddad32=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_smarts42));
for i in 1:length(pred_metrics_10_fold_83L_haddad32)
    pred_pr_auc_10_fold_83L_haddad32[i]=pred_metrics_10_fold_83L_haddad32[i][2]
    pred_pr_auc_10_fold_83R_haddad32[i]=pred_metrics_10_fold_83R_haddad32[i][2]
end

pred_metrics_10_fold_83L_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_cact_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_cact_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_cact=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_cact));
pred_pr_auc_10_fold_83R_cact=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_cact));
for i in 1:length(pred_metrics_10_fold_83L_cact)
    pred_pr_auc_10_fold_83L_cact[i]=pred_metrics_10_fold_83L_cact[i][2]
    pred_pr_auc_10_fold_83R_cact[i]=pred_metrics_10_fold_83R_cact[i][2]
end

pred_metrics_10_fold_83L_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_pom21_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_pom21_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_pom21=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_pom21));
pred_pr_auc_10_fold_83R_pom21=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_pom21));
for i in 1:length(pred_metrics_10_fold_83L_pom21)
    pred_pr_auc_10_fold_83L_pom21[i]=pred_metrics_10_fold_83L_pom21[i][2]
    pred_pr_auc_10_fold_83R_pom21[i]=pred_metrics_10_fold_83R_pom21[i][2]
end

pred_metrics_10_fold_83L_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_pom21_shap_reduced));
pred_pr_auc_10_fold_83R_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_pom21_shap_reduced));
for i in 1:length(pred_metrics_10_fold_83L_pom21_shap_reduced)
    pred_pr_auc_10_fold_83L_pom21_shap_reduced[i]=pred_metrics_10_fold_83L_pom21_shap_reduced[i][2]
    pred_pr_auc_10_fold_83R_pom21_shap_reduced[i]=pred_metrics_10_fold_83R_pom21_shap_reduced[i][2]
end

pred_metrics_10_fold_83L_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L_pom160_shap_reduced));
pred_pr_auc_10_fold_83R_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R_pom160_shap_reduced));
for i in 1:length(pred_metrics_10_fold_83L_pom160_shap_reduced)
    pred_pr_auc_10_fold_83L_pom160_shap_reduced[i]=pred_metrics_10_fold_83L_pom160_shap_reduced[i][2]
    pred_pr_auc_10_fold_83R_pom160_shap_reduced[i]=pred_metrics_10_fold_83R_pom160_shap_reduced[i][2]
end

StatsPlots.boxplot(["edragon"],pred_pr_auc_10_fold_83L_edragon,color=:royalblue,
                    title="PRC AUC across descriptor data sets 10 fold",grid=false,
                    ylabel="",legend=false)
StatsPlots.boxplot!(["haddad"],pred_pr_auc_10_fold_83L_haddad32,color=:purple,legend=false)
StatsPlots.boxplot!(["maccs"],pred_pr_auc_10_fold_83L_maccs,color=:royalblue,legend=false)
StatsPlots.boxplot!(["cact"],pred_pr_auc_10_fold_83L_cact,color=:pink,legend=false)
StatsPlots.boxplot!(["smarts"],pred_pr_auc_10_fold_83L_smarts42,color=:green,legend=false)
StatsPlots.boxplot!(["pom"],pred_pr_auc_10_fold_83L_pom21,color=:orange,legend=false)
#StatsPlots.boxplot!(["pom160"],pred_pr_auc_10_fold_83L_pom160_shap_reduced,color=:black,legend=false)

StatsPlots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files","10_fold_descrs_comparisons.png"));

pred_metrics_per_glom_83L_edragon1377=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_edragon1377_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_edragon1377=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_edragon1377_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_edragon1377=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_edragon1377));
pred_pr_auc_per_glom_83R_edragon1377=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_edragon1377));
for i in 1:length(pred_metrics_per_glom_83L_edragon1377)
    pred_pr_auc_per_glom_83L_edragon1377[i]=pred_metrics_per_glom_83L_edragon1377[i][2]
    pred_pr_auc_per_glom_83R_edragon1377[i]=pred_metrics_per_glom_83R_edragon1377[i][2]
end

pred_metrics_per_glom_83L_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_maccs_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_maccs_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_maccs=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_maccs));
pred_pr_auc_per_glom_83R_maccs=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_maccs));
for i in 1:length(pred_metrics_per_glom_83L_maccs)
    pred_pr_auc_per_glom_83L_maccs[i]=pred_metrics_per_glom_83L_maccs[i][2]
    pred_pr_auc_per_glom_83R_maccs[i]=pred_metrics_per_glom_83R_maccs[i][2]
end

pred_metrics_per_glom_83L_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_smarts42=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_smarts42));
pred_pr_auc_per_glom_83R_smarts42=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_smarts42));
for i in 1:length(pred_metrics_per_glom_83L_smarts42)
    pred_pr_auc_per_glom_83L_smarts42[i]=pred_metrics_per_glom_83L_smarts42[i][2]
    pred_pr_auc_per_glom_83R_smarts42[i]=pred_metrics_per_glom_83R_smarts42[i][2]
end

pred_metrics_per_glom_83L_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_haddad32_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_haddad32_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_haddad32=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_haddad32));
pred_pr_auc_per_glom_83R_haddad32=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_haddad32));
for i in 1:length(pred_metrics_per_glom_83L_haddad32)
    pred_pr_auc_per_glom_83L_haddad32[i]=pred_metrics_per_glom_83L_haddad32[i][2]
    pred_pr_auc_per_glom_83R_haddad32[i]=pred_metrics_per_glom_83R_haddad32[i][2]
end

pred_metrics_per_glom_83L_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_cact=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_cact));
pred_pr_auc_per_glom_83R_cact=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_cact));
for i in 1:length(pred_metrics_per_glom_83L_cact)
    pred_pr_auc_per_glom_83L_cact[i]=pred_metrics_per_glom_83L_cact[i][2]
    pred_pr_auc_per_glom_83R_cact[i]=pred_metrics_per_glom_83R_cact[i][2]
end

pred_metrics_per_glom_83L_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_pom21=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_pom21));
pred_pr_auc_per_glom_83R_pom21=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_pom21));
for i in 1:length(pred_metrics_per_glom_83L_pom21)
    pred_pr_auc_per_glom_83L_pom21[i]=pred_metrics_per_glom_83L_pom21[i][2]
    pred_pr_auc_per_glom_83R_pom21[i]=pred_metrics_per_glom_83R_pom21[i][2]
end

pred_metrics_per_glom_83L_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_pom160_shap_reduced));
pred_pr_auc_per_glom_83R_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_pom160_shap_reduced));
for i in 1:length(pred_metrics_per_glom_83L_pom160_shap_reduced)
    pred_pr_auc_per_glom_83L_pom160_shap_reduced[i]=pred_metrics_per_glom_83L_pom160_shap_reduced[i][2]
    pred_pr_auc_per_glom_83R_pom160_shap_reduced[i]=pred_metrics_per_glom_83R_pom160_shap_reduced[i][2]
end

pred_metrics_per_glom_83L_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L_pom21_shap_reduced));
pred_pr_auc_per_glom_83R_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R_pom21_shap_reduced));
for i in 1:length(pred_metrics_per_glom_83L_pom21_shap_reduced)
    pred_pr_auc_per_glom_83L_pom21_shap_reduced[i]=pred_metrics_per_glom_83L_pom21_shap_reduced[i][2]
    pred_pr_auc_per_glom_83R_pom21_shap_reduced[i]=pred_metrics_per_glom_83R_pom21_shap_reduced[i][2]
end


StatsPlots.boxplot(["edragon"],pred_pr_auc_per_glom_83L_edragon1377,color=:red,title="PRC AUC across descriptor data sets per glom",grid=false,ylabel="",legend=false)
StatsPlots.boxplot!(["haddad"],pred_pr_auc_per_glom_83L_haddad32,color=:purple,grid=false,legend=false)
StatsPlots.boxplot!(["maccs"],pred_pr_auc_per_glom_83L_maccs,color=:royalblue,grid=false,legend=false)
StatsPlots.boxplot!(["cact"],pred_pr_auc_per_glom_83L_cact,color=:pink,grid=false,legend=false)
StatsPlots.boxplot!(["smarts"],pred_pr_auc_per_glom_83L_smarts42,color=:green,grid=false,legend=false)
StatsPlots.boxplot!(["pom"],pred_pr_auc_per_glom_83L_pom21_shap_reduced,color=:orange,grid=false,legend=false)
StatsPlots.violin!(["pom160"],pred_pr_auc_per_glom_83L_pom160_shap_reduced,color=:black,grid=false,legend=false)

StatsPlots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files","per_glom_descrs_comparisons.png"));

pred_metrics_leave_one_out_83L_edragon1377=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_edragon1377_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_edragon1377=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_edragon1377_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_edragon1377=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_edragon1377));
pred_pr_auc_leave_one_out_83R_edragon1377=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_edragon1377));
for i in 1:length(pred_metrics_leave_one_out_83L_edragon1377)
    pred_pr_auc_leave_one_out_83L_edragon1377[i]=pred_metrics_leave_one_out_83L_edragon1377[i][2]
    pred_pr_auc_leave_one_out_83R_edragon1377[i]=pred_metrics_leave_one_out_83R_edragon1377[i][2]
end

pred_metrics_leave_one_out_83L_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_maccs_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_maccs=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_maccs_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_maccs=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_maccs));
pred_pr_auc_leave_one_out_83R_maccs=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_maccs));
for i in 1:length(pred_metrics_leave_one_out_83L_maccs)
    pred_pr_auc_leave_one_out_83L_maccs[i]=pred_metrics_leave_one_out_83L_maccs[i][2]
    pred_pr_auc_leave_one_out_83R_maccs[i]=pred_metrics_leave_one_out_83R_maccs[i][2]
end

pred_metrics_leave_one_out_83L_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_smarts42_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_smarts42=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_smarts42_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_smarts42=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_smarts42));
pred_pr_auc_leave_one_out_83R_smarts42=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_smarts42));
for i in 1:length(pred_metrics_leave_one_out_83L_smarts42)
    pred_pr_auc_leave_one_out_83L_smarts42[i]=pred_metrics_leave_one_out_83L_smarts42[i][2]
    pred_pr_auc_leave_one_out_83R_smarts42[i]=pred_metrics_leave_one_out_83R_smarts42[i][2]
end

pred_metrics_leave_one_out_83L_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_haddad32_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_haddad32=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_haddad32_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_haddad32=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_haddad32));
pred_pr_auc_leave_one_out_83R_haddad32=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_haddad32));
for i in 1:length(pred_metrics_leave_one_out_83L_haddad32)
    pred_pr_auc_leave_one_out_83L_haddad32[i]=pred_metrics_leave_one_out_83L_haddad32[i][2]
    pred_pr_auc_leave_one_out_83R_haddad32[i]=pred_metrics_leave_one_out_83R_haddad32[i][2]
end

pred_metrics_leave_one_out_83L_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_cact_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_cact=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_cact_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_cact=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_cact));
pred_pr_auc_leave_one_out_83R_cact=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_cact));
for i in 1:length(pred_metrics_leave_one_out_83L_cact)
    pred_pr_auc_leave_one_out_83L_cact[i]=pred_metrics_leave_one_out_83L_cact[i][2]
    pred_pr_auc_leave_one_out_83R_cact[i]=pred_metrics_leave_one_out_83R_cact[i][2]
end

pred_metrics_leave_one_out_83L_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_pom21_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_pom21=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_correct_pom21_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_pom21=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_pom21));
pred_pr_auc_leave_one_out_83R_pom21=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_pom21));
for i in 1:length(pred_metrics_leave_one_out_83L_pom21)
    pred_pr_auc_leave_one_out_83L_pom21[i]=pred_metrics_leave_one_out_83L_pom21[i][2]
    pred_pr_auc_leave_one_out_83R_pom21[i]=pred_metrics_leave_one_out_83R_pom21[i][2]
end

pred_metrics_leave_one_out_83L_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_pom160_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom160_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_pom160_shap_reduced));
pred_pr_auc_leave_one_out_83R_pom160_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_pom160_shap_reduced));
for i in 1:length(pred_metrics_leave_one_out_83L_pom160_shap_reduced)
    pred_pr_auc_leave_one_out_83L_pom160_shap_reduced[i]=pred_metrics_leave_one_out_83L_pom160_shap_reduced[i][2]
    pred_pr_auc_leave_one_out_83R_pom160_shap_reduced[i]=pred_metrics_leave_one_out_83R_pom160_shap_reduced[i][2]
end

pred_metrics_leave_one_out_83L_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R_pom21_shap_reduced=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_pom21_shap_reduced_descrs_segmented_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L_pom21_shap_reduced));
pred_pr_auc_leave_one_out_83R_pom21_shap_reduced=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R_pom21_shap_reduced));
for i in 1:length(pred_metrics_leave_one_out_83L_pom21_shap_reduced)
    pred_pr_auc_leave_one_out_83L_pom21_shap_reduced[i]=pred_metrics_leave_one_out_83L_pom21_shap_reduced[i][2]
    pred_pr_auc_leave_one_out_83R_pom21_shap_reduced[i]=pred_metrics_leave_one_out_83R_pom21_shap_reduced[i][2]
end

#10 fold plots
StatsPlots.boxplot(["edragon"],pred_pr_auc_10_fold_83L_edragon,color=:royalblue,
                    title="PRC AUC across descriptor data sets",grid=false,
                    ylabel="",legend=:outerright,label="CD + PA 10 fold")
StatsPlots.boxplot!(["haddad"],pred_pr_auc_10_fold_83L_haddad32,color=:royalblue,legend=:outerright,label="")
StatsPlots.boxplot!(["maccs"],pred_pr_auc_10_fold_83L_maccs,color=:royalblue,legend=:outerright,label="")
StatsPlots.boxplot!(["cactus"],pred_pr_auc_10_fold_83L_cact,color=:royalblue,legend=:outerright,label="")
StatsPlots.boxplot!(["smarts"],pred_pr_auc_10_fold_83L_smarts42,color=:royalblue,legend=:outerright,label="")
StatsPlots.boxplot!(["pom"],pred_pr_auc_10_fold_83L_pom21,color=:royalblue,legend=:outerright,label="")

#leave one out plots
StatsPlots.boxplot!(["edragon"],pred_pr_auc_leave_one_out_83L_edragon1377,color=:red,grid=false,
                    ylabel="",legend=:outerright,label="CD + PA L.O.O")
StatsPlots.boxplot!(["haddad"],pred_pr_auc_leave_one_out_83L_haddad32,color=:red,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["maccs"],pred_pr_auc_leave_one_out_83L_maccs,color=:red,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["cactus"],pred_pr_auc_leave_one_out_83L_cact,color=:red,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["smarts"],pred_pr_auc_leave_one_out_83L_smarts42,color=:red,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["pom"],pred_pr_auc_leave_one_out_83L_pom21_shap_reduced,color=:red,grid=false,legend=:outerright,label="")

#leave one out per glom plots
StatsPlots.boxplot!(["edragon"],pred_pr_auc_per_glom_83L_edragon1377,color=:green,
                    grid=false,ylabel="",legend=:outerright,label="CD L.O.O")
StatsPlots.boxplot!(["haddad"],pred_pr_auc_per_glom_83L_haddad32,color=:green,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["maccs"],pred_pr_auc_per_glom_83L_maccs,color=:green,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["cactus"],pred_pr_auc_per_glom_83L_cact,color=:green,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["smarts"],pred_pr_auc_per_glom_83L_smarts42,color=:green,grid=false,legend=:outerright,label="")
StatsPlots.boxplot!(["pom"],pred_pr_auc_per_glom_83L_pom21_shap_reduced,color=:green,grid=false,legend=:outerright,label="")


StatsPlots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files","all_descrs_comparisons.png"));

StatsPlots.violin(["smarts42 loo"],pred_pr_auc_leave_one_out_83L_smarts42,color=:green,grid=false,legend=false)
StatsPlots.violin!(["pom160 loo"],pred_pr_auc_leave_one_out_83L_pom160_shap_reduced,color=:orange,grid=false,legend=false)
StatsPlots.violin!(["smarts42 per glom"],pred_pr_auc_per_glom_83L_smarts42,color=:green,grid=false,legend=false)
StatsPlots.violin!(["pom160 per glom"],pred_pr_auc_per_glom_83L_pom160_shap_reduced,color=:orange,grid=false,legend=false)


PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","Rankings_kendall_tau.eps"));


Ndatasets=2;

#load the data for plotting
#load peer prediction
#Below plots the results for the sets with no amines
#load peer prediction

#load 10 fold chemical feature prediction
pred_metrics_10_fold_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_varied_descrs_number_pred_metrics.jld"),"83Lresults_10_fold");
pred_metrics_10_fold_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_cact_varied_descrs_number_pred_metrics.jld"),"83Rresults_10_fold");
pred_pr_auc_10_fold_83L=Array{Float64,1}(undef,length(pred_metrics_10_fold_83L));
pred_pr_auc_10_fold_83R=Array{Float64,1}(undef,length(pred_metrics_10_fold_83R));
for i in 1:length(pred_metrics_10_fold_83L)
    pred_pr_auc_10_fold_83L[i]=pred_metrics_10_fold_83L[i][2]
    pred_pr_auc_10_fold_83R[i]=pred_metrics_10_fold_83R[i][2]
end

#load per glom chemical feature prediction
pred_metrics_per_glom_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_varied_descrs_number_pred_metrics.jld"),"83Lresults_leave_one_out_per_glom");
pred_metrics_per_glom_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_varied_descrs_number_pred_metrics.jld"),"83Rresults_leave_one_out_per_glom");
pred_pr_auc_per_glom_83L=Array{Float64,1}(undef,length(pred_metrics_per_glom_83L));
pred_pr_auc_per_glom_83R=Array{Float64,1}(undef,length(pred_metrics_per_glom_83R));
for i in 1:length(pred_metrics_per_glom_83L)
    pred_pr_auc_per_glom_83L[i]=pred_metrics_per_glom_83L[i][2]
    pred_pr_auc_per_glom_83R[i]=pred_metrics_per_glom_83R[i][2]
end

#load leave one out chemical feature prediction
pred_metrics_leave_one_out_83L=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_varied_descrs_number_pred_metrics.jld"),"83Lresults_leave_one_out");
pred_metrics_leave_one_out_83R=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/pred_metrics/all_odors_smarts42_varied_descrs_number_pred_metrics.jld"),"83Rresults_leave_one_out");
pred_pr_auc_leave_one_out_83L=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83L));
pred_pr_auc_leave_one_out_83R=Array{Float64,1}(undef,length(pred_metrics_leave_one_out_83R));
for i in 1:length(pred_metrics_leave_one_out_83L)
    pred_pr_auc_leave_one_out_83L[i]=pred_metrics_leave_one_out_83L[i][2]
    pred_pr_auc_leave_one_out_83R[i]=pred_metrics_leave_one_out_83R[i][2]
end

baseline_pred_pr_auc83L=sum(binary_responses[1])/(size(binary_responses[1],1)*size(binary_responses[1],2));
baseline_pred_acc83L=1-baseline_pred_pr_auc83L;
#Plot omp84
pyplot()
p1=Plots.plot(1:24,pred_acc_10_fold_83L,xlabel="number of features",ylabel="accuracy",grid=false,label="CDPA10",ylim=[0.6,0.92],xlims=[1,25],title="omp83L Shapley ordered features",linewidth=2,c=:blue,legend=:outerright)
Plots.plot!(1:25,pred_acc_shuffled_features_10_fold_83L,label="SCDPA10",linewidth=2,c=:lightblue)
Plots.plot!(1:25,pred_acc_shuffled_resp_10_fold_83L,label="SRCDPA10",linewidth=2,c=:aqua)
Plots.plot!(1:25,pred_acc_leave_one_out_83L,label="CDPALOO",linewidth=2,c=:red)
Plots.plot!(1:25,pred_acc_shuffled_features_leave_one_out_83L,label="SCDPALOO",linewidth=2,c=:orange)
Plots.plot!(1:25,pred_acc_shuffled_resp_leave_one_out_83L,label="SRCDPALOO",linewidth=2,c=:pink)
Plots.plot!(1:25,pred_acc_per_glom_83L,label="CDLOO",linewidth=2,c=:green)
Plots.plot!(1:25,pred_acc_shuffled_features_per_glom_83L,label="SCDLOO",linewidth=2,c=:greenyellow)
Plots.plot!(1:25,pred_acc_shuffled_resp_per_glom_83L,label="SRCDLOO",linewidth=2,c=:yellow)
Plots.plot!(1:25,repeat([0.89],25),label="peer pred",linewidth=2,c=:purple)
#Plots.plot!(1:4:55,repeat([peer_preds84_ac],39),label="peer pred",linewidth=2,c=:purple)
Plots.plot!(1:25,repeat([baseline_pred_acc83L],25),label="baseline",linewidth=2,c="black")

#Plots.plot!(1:4:155,repeat([baseline_pred_acc84],39),label="baseline",linewidth=2,c="black")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","84segmented_acc.png"));

p1=Plots.plot(1:24,pred_pr_auc_10_fold_83L[1:24],xlabel="number of features",ylabel="pr auc",grid=false,label="CD+PA 10 fold",ylim=[0.12,0.75],xlims=[1,24],title="omp83L Shapley ordered features",linewidth=2,c=:blue,legend=:outerright)
Plots.plot!(1:24,pred_pr_auc_leave_one_out_83L[1:24],label="CD+PA L.O.O",linewidth=2,c=:red)
Plots.plot!(1:24,pred_pr_auc_per_glom_83L[1:24],label="CD L.O.O",linewidth=2,c=:green)
Plots.plot!(1:24,repeat([0.69],24),label="peer pred",linewidth=2,c=:purple)
Plots.plot!(1:24,repeat([baseline_pred_pr_auc83L],24),label="baseline",linewidth=2,c="black")
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","83L_smarts42_pr_auc.png"));
pred_pr_auc_10_fold_83L
