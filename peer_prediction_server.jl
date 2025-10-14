using Distributed,JLD;

addprocs(40)

@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Useful_functions.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/python_shap_functions_new.jl"));
@everywhere using Random;

@everywhere dataset_name="no_amines";#"all_odors";

#println("data all odors $descr_set_name")

#load the necessary data
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
@everywhere resp_matr=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"resp_matr");


#@everywhere binary_responses=shuffle.(binary_responses)

#the hyperparameters for a grid search along with 
@everywhere ntrees=128;#number of trees the random forest has
@everywhere sample_perc=0.5;#percentage of data each tree uses
@everywhere Random.seed!(1234);  #make sure every run uses the same random seeds
@everywhere random_seeds_array=rand(1:10000,100); #generate 100 random seeds

println("$dataset_name"*" peer prediction")

@everywhere f1(number)=peer_prediction(copy(resp_matr[1]),binary_responses[1],ntrees,sample_perc,random_seeds_array[number])
@everywhere f2(number)=peer_prediction(copy(resp_matr[2]),binary_responses[2],ntrees,sample_perc,random_seeds_array[number])


println("Computing peer prediction performance.")

@everywhere runs=1:100;
@time peer_pred_1=pmap(f1,runs);
@time peer_pred_2=pmap(f2,runs);

println("peer prediction all odors finished.")

println("saving peer prediction performance.")

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_peer_prediction_results.jld"),
                                                    "peer_pred_1",peer_pred_1,
                                                    "peer_pred_2",peer_pred_2);

println("peer prediction $dataset_name"*" finished.")


