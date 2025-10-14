using Distributed,JLD, IterTools;

addprocs(40)

@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Useful_functions.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/python_shap_functions_new.jl"));
@everywhere using Random;

#choose descriptor set name to run the analysis on. Choices are "smarts42", "pom21", "cact", "haddad32", "maccs","pca_cact".
@everywhere descr_set_name="pom21";
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
@everywhere random_seeds_array=rand(1:10000,100); #generate 10 random seeds
@everywhere nfolds=10; #number of folds to use in one of the models
@everywhere number_of_descrs=length(descr_names);
@everywhere descr_numbers=1:32;#min(30,number_of_descrs); #number of features to add, from 1 to 24 for smarts42, 1:40 for pom21. 1:40 for cactus, 1:40 for cact_pca, 32 for haddad32

#println("data $dataset_name"*" descrs $descr_set_name")

println("Computing shapley orderings")

@everywhere begin
    new_matrix=Array{Any,1}(undef,2)
    # Remove the last column
    for i in 1:2
        modified_matrix = Data_frames[i][:, 1:end-1]
    
    #Calculate k and create the new column
        num_rows = size(modified_matrix, 1)
        k = div(num_rows, 59)  # Assuming num_rows is exactly divisible by 59 as mentioned
    #    k = div(num_rows, 52)  # Assuming num_rows is exactly divisible by 59 as mentioned      
         # Generate a single shuffled set of indices for k rows
        shuffle_indices = shuffle(1:k)
     # Apply the same shuffled indices to each block of k rows for the last two columns
     #   for block in 1:52
        for block in 1:59
            start_index = (block - 1) * k + 1
            end_index = block * k
            # Apply shuffled indices to the last two columns in each block
            modified_matrix[start_index:end_index, end-1:end] = modified_matrix[start_index:end_index, end-1:end][shuffle_indices, :]
        end
        new_matrix[i] = modified_matrix
    end
end

@everywhere k1(number)=find_shap_ordering(new_matrix[1],descr_names,binary_responses[1],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);
@everywhere k2(number)=find_shap_ordering(new_matrix[2],descr_names,binary_responses[2],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);

@time shap_idx_array1_real=pmap(k1,random_seeds_array);
@time shap_idx_array2_real=pmap(k2,random_seeds_array);


@everywhere begin
    function f1(params)
        ntrees, sample_perc,perc_features, shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_old_2(new_matrix[1],binary_responses[1],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
        #return chemotopy_ordered_metrics_old(modified_matrix,binary_responses[1],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end

    function f2(params)
        ntrees, sample_perc, perc_features,shap_idx,n_descrs=params
        return chemotopy_ordered_metrics_old_2(new_matrix[2],binary_responses[2],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end 
end

println("Computing 10 fold control $descr_set_name")

@time bulb_1_control_results=pmap(f1,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array1_real,descr_numbers));
@time bulb_2_control_results=pmap(f2,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array2_real,descr_numbers));

println("Control 10 fold finished $descr_set_name")

println("saving control 10 fold metrics $descr_set_name")

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_new_chemotopy_shuffled_positions.jld"),
                                                    "bulb_1_control_results",bulb_1_control_results,
                                                    "bulb_2_control_results",bulb_2_control_results);


println("$dataset_name"*" chemotopy, shuffled positions for $descr_set_name"*" finished")
