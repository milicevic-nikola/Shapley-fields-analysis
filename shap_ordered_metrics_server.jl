using Distributed,JLD;

addprocs(40)

@everywhere include("Response_prediction_pipeline/Useful_functions.jl");
@everywhere include("Response_prediction_pipeline/python_shap_functions_new.jl")
@everywhere using Random;
#@everywhere Ndatasets=3;

#load the necessary data
#println("shuffled analysis")
#@everywhere shap_10_fold=load(joinpath(homedir(),"shap_data_schmuker_descr_no_amines.jld"),"data_shap_idx");
#@everywhere shap_10_fold=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_schmuker_descrs_shuffled_descrs_and_resps_no_amines.jld"),"data_shap_idx");
@everywhere shap_10_fold=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"data_shap_idx_smarts42");
#@everywhere shap_10_fold_idx=shap_10_fold[2];
@everywhere shap_10_fold_idx=shap_10_fold;
#@everywhere odor_descr_subset_names=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_learned_shuffled_descrs_and_resps.jld"),"descr_names");
@everywhere Data_frames=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_smarts42");
@everywhere X=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_smarts42");
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"bin_resp");
#@everywhere binary_responses=load(joinpath(homedir(),"original_data_new_descrs_for_prediction.jld"),"bin_resp");
#@everywhere binary_responses=shuffle.(binary_responses)

@everywhere Random.seed!(1234);
@everywhere ntrees=256;
@everywhere sample_perc=0.7;
@everywhere random_seed=1234;
@everywhere nfolds=10;

println("data all odors smarts42")

#@everywhere  f(number)=chemotopy_ordered_metrics_leave_one_out(Data,binary_responses[1],gloms_to_consider_idx[1],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
@everywhere  f1(number)=chemotopy_ordered_metrics_leave_one_out_per_glom(X[1],binary_responses[1],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
@everywhere  f2(number)=chemotopy_ordered_metrics_leave_one_out_per_glom(X[2],binary_responses[2],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
#@everywhere  f(number)=chemotopy_ordered_metrics_leave_one_out_per_glom(X[1][1],binary_responses[1],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
runs=1:24#:72#length(odor_descr_subset_names)
omp83L_results_leave_one_out_per_glom=pmap(f1,runs);
omp83R_results_leave_one_out_per_glom=pmap(f2,runs);

println("per glom finished")

#a
@everywhere  g1(number)=chemotopy_ordered_metrics_old(Matrix(Data_frames[1]),binary_responses[1],shap_10_fold_idx,nfolds,ntrees,sample_perc,random_seed,number)
@everywhere  g2(number)=chemotopy_ordered_metrics_old(Matrix(Data_frames[2]),binary_responses[2],shap_10_fold_idx,nfolds,ntrees,sample_perc,random_seed,number)
#@everywhere  g(number)=chemotopy_ordered_metrics(Matrix(Data_frames[1][1]),binary_responses[1],gloms_to_consider_idx[1],shap_10_fold_idx,nfolds,ntrees,sample_perc,random_seed,number)
#runs=1:2
runs=1:24#:72#length(odor_descr_subset_names)
omp83L_results_10_fold=pmap(g1,runs);
omp83R_results_10_fold=pmap(g2,runs);

println("10 fold finished")

@everywhere  h1(number)=chemotopy_ordered_metrics_leave_one_out_old(X[1],binary_responses[1],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
@everywhere  h2(number)=chemotopy_ordered_metrics_leave_one_out_old(X[2],binary_responses[2],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
#@everywhere  h(number)=chemotopy_ordered_metrics_leave_one_out(X[1][1],binary_responses[1],gloms_to_consider_idx[1],shap_10_fold_idx,ntrees,sample_perc,random_seed,number)
#runs=1:2
runs=1:24#:72#length(odor_descr_subset_names)
omp83L_results_leave_one_out=pmap(h1,runs);
omp83R_results_leave_one_out=pmap(h2,runs);

println("leave one out finished")

save(joinpath(homedir(),"correct_chemotopy_results/all_odors_smarts42_varied_descrs_number_pred_metrics.jld"),
                                                    "83Lresults_leave_one_out_per_glom",omp83L_results_leave_one_out_per_glom,
                                                    "83Rresults_leave_one_out_per_glom",omp83R_results_leave_one_out_per_glom,
                                                    "83Lresults_10_fold", omp83L_results_10_fold,
                                                    "83Rresults_10_fold", omp83R_results_10_fold,
                                                    "83Lresults_leave_one_out",omp83L_results_leave_one_out,
                                                    "83Rresults_leave_one_out",omp83R_results_leave_one_out);
