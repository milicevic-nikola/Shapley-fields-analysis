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
@everywhere Random.seed!(1337);  #make sure every run uses the same random seeds
# Generate seeds on the master process
random_seeds_array = rand(1:10000, 200)
# Broadcast these seeds to all workers
@everywhere begin
    global random_seeds_array = $random_seeds_array
end

@everywhere nfolds=10; #number of folds to use in one of the models
@everywhere number_of_descrs=length(descr_names);
@everywhere descr_numbers=[1,2,3,4,5,6,7,8,9,10,20,30,40]

println("Computing shapley orderings")

@everywhere begin
    new_matrix=Array{Any,1}(undef,2)
    # Remove the last column
    for i in 1:2
        modified_matrix = Data_frames[i][:, 1:end-1]
        new_matrix[i] = modified_matrix
    end
end

@everywhere begin
    integer_identities=Array{Any,1}(undef,2)
    for i in 1:length(integer_identities)
        num_rows=size(Data_frames[i],1)
        k=div(num_rows,59)
        #k=div(num_rows,52)
        repeated_vectors=Array{Any,1}(undef,200)
        for j in 1:length(repeated_vectors)
            shuffled_vector=shuffle(1:k)
            repeated_vectors[j]=repeat(shuffled_vector, outer=59)
            #repeated_vectors[j]=repeat(shuffled_vector, outer=52)
        end
    integer_identities[i]=repeated_vectors
    end
end

@everywhere begin
    shuffled_indices=Array{Any,1}(undef,2)
    for i in 1:length(integer_identities)
        num_rows=size(Data_frames[i],1)
        k=div(num_rows,59)
        #k=div(num_rows,52)
        shuffle_indices=Array{Any,1}(undef,200)
        for j in 1:length(shuffle_indices)
            shuffle_indices[j]=shuffle(1:k)
           # repeated_vectors[j]=repeat(shuffled_vector, outer=59)
        end
    shuffled_indices[i]=shuffle_indices
    end
end

@everywhere function k1(identity, seed)
    find_shap_ordering_with_shuffling(new_matrix[1], identity, descr_names, binary_responses[1], ntrees_array[1], sample_perc_array[1], perc_features_array[1], seed)
end

@everywhere function k2(identity, seed)
    find_shap_ordering_with_shuffling(new_matrix[2], identity, descr_names, binary_responses[2], ntrees_array[1], sample_perc_array[1], perc_features_array[1], seed)
end

@time shap_idx_array1_real = pmap((args) -> k1(args...), zip(integer_identities[1], random_seeds_array));
@time shap_idx_array2_real = pmap((args) -> k2(args...), zip(integer_identities[2], random_seeds_array));

@everywhere begin
    function f1(params)
        ntrees, sample_perc,perc_features, shap_idx, n_descrs=params
        number=findfirst(==(shap_idx[1]), random_seeds_array)
        return chemotopy_ordered_metrics_old_with_shuffling(new_matrix[1], integer_identities[1][number], binary_responses[1],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end

    function f2(params)
        ntrees, sample_perc,perc_features, shap_idx, n_descrs=params
        number=findfirst(==(shap_idx[1]), random_seeds_array)
        return chemotopy_ordered_metrics_old_with_shuffling(new_matrix[2], integer_identities[2][number], binary_responses[2],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end
end

println("Computing 10 fold real $descr_set_name")

@time bulb_1_real_results=pmap(f1,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array1_real,descr_numbers));
@time bulb_2_real_results=pmap(f2,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array2_real,descr_numbers));

println("Real 10 fold finished $descr_set_name")

println("saving real 10 fold metrics $descr_set_name")

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_200_random_seeds_chemotopy_analysis.jld"),
                                                    "bulb_1_real_results",bulb_1_real_results,
                                                    "bulb_2_real_results",bulb_2_real_results);


@everywhere function g1(identity, shuffle_idx, seed)
    find_shap_ordering_with_shuffling_v2(new_matrix[1], identity, shuffle_idx ,descr_names, binary_responses[1], ntrees_array[1], sample_perc_array[1], perc_features_array[1], seed)
end
                                                    
@everywhere function g2(identity, shuffle_idx, seed)
    find_shap_ordering_with_shuffling_v2(new_matrix[2], identity, shuffle_idx,descr_names, binary_responses[2], ntrees_array[1], sample_perc_array[1], perc_features_array[1], seed)
end
                                                    
@time shap_idx_array1_control = pmap((args) -> g1(args...), zip(integer_identities[1],shuffled_indices[1], random_seeds_array));
@time shap_idx_array2_control = pmap((args) -> g2(args...), zip(integer_identities[2],shuffled_indices[2], random_seeds_array));

println("$dataset_name"*" chemotopy, real positions and identities for $descr_set_name"*" finished")

@everywhere begin
    function h1(params)
        ntrees, sample_perc,perc_features, shap_idx, n_descrs=params
        number=findfirst(==(shap_idx[1]), random_seeds_array)
        return chemotopy_ordered_metrics_old_with_shuffling_v2(new_matrix[1], integer_identities[1][number], shuffled_indices[1][number], binary_responses[1],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end

    function h2(params)
        ntrees, sample_perc,perc_features, shap_idx, n_descrs=params
        number=findfirst(==(shap_idx[1]), random_seeds_array)
        return chemotopy_ordered_metrics_old_with_shuffling_v2(new_matrix[2], integer_identities[2][number], shuffled_indices[2][number], binary_responses[2],shap_idx[2:end],nfolds,ntrees,sample_perc,perc_features,shap_idx[1],n_descrs)
    end
end

println("Computing 10 fold control $descr_set_name")

@time bulb_1_control_results=pmap(h1,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array1_control,descr_numbers));
@time bulb_2_control_results=pmap(h2,Iterators.product(ntrees_array,sample_perc_array,perc_features_array,shap_idx_array2_control,descr_numbers));

println("Control 10 fold finished $descr_set_name")

println("saving control 10 fold metrics $descr_set_name")


file = jldopen(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_200_random_seeds_chemotopy_analysis.jld"), "r+");
write(file,"bulb_1_control_results",bulb_1_control_results);
write(file,"bulb_2_control_results",bulb_2_control_results);
close(file)


println("$dataset_name"*" chemotopy, control positions and identities for $descr_set_name"*" finished")
