using Distributed,JLD, IterTools;

addprocs(40)

@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Useful_functions.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/python_shap_functions_new.jl"));
@everywhere using Random;

#choose descriptor set name to run the analysis on. Choices are "smarts42", "pom21", "cact", "haddad32", "maccs","pca_cact".
@everywhere descr_set_name="cact";
@everywhere dataset_name="all_odors";#"no_amines"
#choose dataset set name to run the analysis on. Choices are "data", "data_no_amines".
#@everywhere dataset_name="pom21";

println("data $dataset_name"*" $descr_set_name")

#load the necessary data
@everywhere Data_frames=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_frame_$descr_set_name");
@everywhere X=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_$descr_set_name");
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
@everywhere descr_names=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"descr_names_$descr_set_name");


#@everywhere binary_responses=shuffle.(binary_responses)

#the hyperparameters for a grid search along with 
@everywhere ntrees_array=[256];#[32,64,128]; #number of trees the random forest has
@everywhere ntrees_array_per_glom=[128];#[32,64,128]; #number of trees the random forest has
@everywhere perc_features_array=[1.0];#[0.7,0.9]; #percentage of features used by each tree, not used in the per glom models                                
@everywhere sample_perc_array_per_glom=[0.5];#[0.5,0.7]; #percentage of data each tree uses
@everywhere sample_perc_array=[0.7];#[0.5,0.7]; #percentage of data each tree uses
#@everywhere Random.seed!(1234);  #make sure every run uses the same random seeds
@everywhere Random.seed!(1337);  #make sure every run uses the same random seeds
@everywhere random_seeds_array=rand(1:10000,50); #generate 10 random seeds
@everywhere nfolds=10; #number of folds to use in one of the models
@everywhere number_of_descrs=length(descr_names);
@everywhere descr_numbers=1:32;#min(30,number_of_descrs); #number of features to add, from 1 to 24 for smarts42, 1:40 for pom21. 1:40 for cactus, 1:40 for cact_pca, 32 for haddad32

#println("data $dataset_name"*" descrs $descr_set_name")

println("Computing shapley orderings")

@everywhere k1(number)=find_shap_ordering(Matrix(Data_frames[1]),descr_names,binary_responses[1],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);
@everywhere k2(number)=find_shap_ordering(Matrix(Data_frames[2]),descr_names,binary_responses[2],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);

#@everywhere k1(number)=find_shap_ordering_per_glom(X[1],binary_responses[1],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],number);
#@everywhere k2(number)=find_shap_ordering_per_glom(X[2],binary_responses[2],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],number);


@time shap_idx_array1=pmap(k1,random_seeds_array);
@time shap_idx_array2=pmap(k2,random_seeds_array);

@everywhere begin
    function f1(params)
        ntrees, sample_perc, shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_leave_one_out_per_glom(X[1],binary_responses[1],shap_idx[2:end],ntrees,sample_perc,shap_idx[1],n_descrs)
    end

    function f2(params)
        ntrees, sample_perc, shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_leave_one_out_per_glom(X[2],binary_responses[2],shap_idx[2:end],ntrees,sample_perc,shap_idx[1],n_descrs)
    end 
end 

println("Computing per glom $descr_set_name")

@time omp83L_results_leave_one_out_per_glom=pmap(f1,Iterators.product(ntrees_array_per_glom,sample_perc_array_per_glom,shap_idx_array1,descr_numbers));
@time omp83R_results_leave_one_out_per_glom=pmap(f2,Iterators.product(ntrees_array_per_glom,sample_perc_array_per_glom,shap_idx_array2,descr_numbers));

println("per glom finished $descr_set_name")

println("saving per glom metrics $descr_set_name")


# save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search_per_glom_order.jld"),
#                                                     "83Lresults_leave_one_out_per_glom",omp83L_results_leave_one_out_per_glom);#,
#                                                    "83Rresults_leave_one_out_per_glom",omp83R_results_leave_one_out_per_glom);

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search_more_random_seeds.jld"),
                                                    "83Lresults_leave_one_out_per_glom",omp83L_results_leave_one_out_per_glom,
                                                    "83Rresults_leave_one_out_per_glom",omp83R_results_leave_one_out_per_glom);


@everywhere begin
    function g1(params)
        ntrees, sample_perc,perc_features, shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_old(Data_frames[1],binary_responses[1],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end

    function g2(params)
        ntrees, sample_perc, perc_features,shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_old(Data_frames[2],binary_responses[2],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end 
end

println("Computing 10 fold $descr_set_name")

@time omp83L_results_10_fold=pmap(g1,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array1,descr_numbers));
@time omp83R_results_10_fold=pmap(g2,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array2,descr_numbers));

println("10 fold finished $descr_set_name")

println("saving 10 fold metrics $descr_set_name")

# save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search_varying_perc_data.jld"),
#                                                     "83Lresults_10_fold",omp83L_results_10_fold,
#                                                     "83Rresults_10_fold",omp83R_results_10_fold);

file = jldopen(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search_more_random_seeds.jld"), "r+");
#file = jldopen(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search.jld"), "r+");
write(file,"83Lresults_10_fold", omp83L_results_10_fold);
write(file,"83Rresults_10_fold", omp83R_results_10_fold);
close(file)

# @everywhere begin
#     function h1(params)
#         ntrees, sample_perc, perc_features, shap_idx, n_descrs=params
#         return chemotopy_ordered_metrics_leave_one_out_old(X[1],binary_responses[1],shap_idx[2:end],ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
#     end

#     function h2(params)
#         ntrees, sample_perc, perc_features, shap_idx, n_descrs=params
#         return chemotopy_ordered_metrics_leave_one_out_old(X[2],binary_responses[2],shap_idx[2:end],ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
#     end 
# end

# println("Computing leave one out $descr_set_name")

# @time omp83L_results_leave_one_out=pmap(h1,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array1,descr_numbers));
# @time omp83R_results_leave_one_out=pmap(h2,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array2,descr_numbers));

# println("leave one out finished $descr_set_name")

# println("saving leave one out metrics")

# file = jldopen(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_hyperparameter_search.jld"), "r+");
# write(file,"83Lresults_leave_one_out",omp83L_results_leave_one_out);
# write(file,"83Rresults_leave_one_out",omp83R_results_leave_one_out);
# close(file)

println("$dataset_name"*" hyperparameter search for $descr_set_name"*" more random seeds finished")
#println("$dataset_name"*" hyperparameter search for $descr_set_name"*" finished")



