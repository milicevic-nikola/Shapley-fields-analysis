using Distributed,JLD;

addprocs(40)

@everywhere include("Response_prediction_pipeline/Useful_functions.jl");
@everywhere include("Response_prediction_pipeline/python_shap_functions_new.jl")
@everywhere using Random;
#@everywhere Ndatasets=3;

#load the necessary data
#descr names
#@everywhere descr_names_maccs=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_maccs");
#@everywhere descr_names_smarts42=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_smarts42");
#@everywhere descr_names_edragon1377=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_edragon1377");
#@everywhere descr_names_haddad32=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_haddad32");
#@everywhere descr_names_cact=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_cact");
#@everywhere descr_names_pom21=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"descr_names_pom21");

#data frames for learning
#@everywhere Data_frames_maccs=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_maccs");
#@everywhere Data_frames_smarts42=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_smarts42");
#@everywhere Data_frames_edragon1377=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_edragon1377");
#@everywhere Data_frames_haddad32=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_haddad32");
#@everywhere Data_frames_cact=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_cact");
#@everywhere Data_frames_pom21=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_pom21");
@everywhere Data_frames=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_frame_edragon1377");

#matrices for learning
#@everywhere X_maccs=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_maccs");
#@everywhere X_smarts42=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_smarts42");
#@everywhere X_edragon1377=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_edragon1377");
#@everywhere X_haddad32=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_haddad32");
#@everywhere X_cact=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_cact");
#@everywhere X_pom21=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_pom21");
@everywhere X=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_different_descrs_for_prediction.jld"),"Data_edragon1377");

#binary response matrices
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_learned_shuffled_descrs_and_resps.jld"),"bin_resp");
#@everywhere binary_responses=shuffle.(binary_responses)

@everywhere Random.seed!(0);
@everywhere ntrees=256;
@everywhere sample_perc=0.7;
@everywhere random_seeds=sample(1:10000,120,replace=false);
@everywhere nfolds=10;

#println("all odors pom21 shap reduced descrs leave one out per glom")
#@everywhere  f1(number)=chem_descr_classification_leave_one_out_per_glom(X_pom21_shap_reduced[1],binary_responses[1],ntrees,sample_perc,random_seeds[number]);
#@everywhere  f2(number)=chem_descr_classification_leave_one_out_per_glom(X_pom21_shap_reduced[2],binary_responses[2],ntrees,sample_perc,random_seeds[number]);
#runs=1:100;
#omp83L_results_leave_one_out_per_glom=pmap(f1,runs);
#omp83R_results_leave_one_out_per_glom=pmap(f2,runs);
#println("per glom finished")

println("all odors edragon1377 descrs 10 fold")
@everywhere  g1(number)=chem_descr_and_pos_classification_old(Matrix(Data_frames[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seeds[number]);
@everywhere  g2(number)=chem_descr_and_pos_classification_old(Matrix(Data_frames[2]),binary_responses[2],nfolds,ntrees,sample_perc,random_seeds[number]);
runs=1:120;
omp83L_results_10_fold=pmap(g1,runs);
omp83R_results_10_fold=pmap(g2,runs);
println("10 fold finished")

println("all odors edragon1377 descrs leave one out")
@everywhere  h1(number)=chem_descr_and_pos_classification_leave_one_out_old(X[1],binary_responses[1],ntrees,sample_perc,random_seeds[number]);
@everywhere  h2(number)=chem_descr_and_pos_classification_leave_one_out_old(X[2],binary_responses[2],ntrees,sample_perc,random_seeds[number]);
runs=1:120;
omp83L_results_leave_one_out=pmap(h1,runs);
omp83R_results_leave_one_out=pmap(h2,runs);
println("leave one out finished")

save(joinpath(homedir(),"correct_chemotopy_results/all_odors_correct_edragon1377_descrs_segmented_pred_metrics.jld"),
                                                    "83Lresults_10_fold", omp83L_results_10_fold,
                                                    "83Rresults_10_fold", omp83R_results_10_fold,
                                                    "83Lresults_leave_one_out",omp83L_results_leave_one_out,
                                                    "83Rresults_leave_one_out",omp83R_results_leave_one_out);
 # "83Lresults_leave_one_out_per_glom",omp83L_results_leave_one_out_per_glom,
 # "83Rresults_leave_one_out_per_glom",omp83R_results_leave_one_out_per_glom,