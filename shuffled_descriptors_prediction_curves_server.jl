using Distributed;

addprocs(40)

@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Useful_functions.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/python_shap_functions_new.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Initialize_analyses.jl"));

@everywhere dataset_name="all_odors";
#load the necessary data
#@everywhere X=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_$descr_set_name");
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_all_odors_different_descrs_for_prediction.jld"),"bin_resp");
#@everywhere combined_bin_resp=vcat(binary_responses[1],binary_responses[2]);
@everywhere glomeruli_positions=load(joinpath(homedir(),"data_to_use_for_predictions/data_for_resp_prediction_and_clustering.jld"),"gloms_positions");
@everywhere glomeruli_areas=load(joinpath(homedir(),"data_to_use_for_predictions/data_for_resp_prediction_and_clustering.jld"),"gloms_areas");

@everywhere using DelimitedFiles;

# @everywhere pom21_descrs=readdlm(joinpath(homedir(),"data_to_use_for_predictions/learned_descrs_21odor_labels.csv"), ',', Float64);#[setdiff(1:59,26:32),:];
# @everywhere descr_set_name="pom21";#pom21
# @everywhere unique_descr_idx=unique(z -> pom21_descrs[:,z], 1:size(pom21_descrs,2));
# @everywhere pom21_descrs=pom21_descrs[:,unique_descr_idx];
# @everywhere zero_idxs=vec(mapslices(col -> any(col .!= 0), pom21_descrs, dims = 1));
# @everywhere pom21_descrs=pom21_descrs[:,zero_idxs];

@everywhere schmuker_cact_filename=joinpath(homedir(),"data_to_use_for_predictions/cactvs_our_odors.csv");
@everywhere cact_descrs, cact_descrs_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_cact_filename));
@everywhere unique_descr_idx=unique(z -> cact_descrs[:,z], 1:size(cact_descrs,2));
@everywhere cact_descrs=cact_descrs[:,unique_descr_idx];
@everywhere zero_idxs=vec(mapslices(col -> any(col .!= 0), cact_descrs, dims = 1));
@everywhere cact_descrs=cact_descrs[:,zero_idxs];

@everywhere begin
    function shuffle_descrs_for_prediction_curves(descrs::Matrix{Float64},bin_resp::BitMatrix,pos::Matrix{Float64},area::Vector{Float64})
        shuffled_descrs=shuffle(descrs)
        m,n=size(bin_resp)
        model_features=Array{Any,2}(undef,size(bin_resp))
        model_features_copy=Array{Any,2}(undef,size(bin_resp))
        model_features_data_frame=zeros(m*n,size(descrs,2)+3)
        for i in 1:size(model_features,1)
            for j in 1:size(model_features,2)
                model_features[i,j]=shuffled_descrs[j,:]
            end
        end
        for i in 1:size(model_features_copy,1)
            for j in 1:size(model_features_copy,2)
                model_features_copy[i,j]=vcat(shuffled_descrs[j,:],pos[i,:],area[i])
            end
        end
        for i in 1:size(model_features_data_frame,1)
            model_features_data_frame[i,:]=vec(model_features_copy)[i]
        end
        return model_features,model_features_data_frame
    end
end

@everywhere ntrees=256;
@everywhere ntrees_per_glom=128;

@everywhere sample_perc=0.7;
@everywhere sample_perc_per_glom=0.5;

@everywhere perc_features=1.0;                                

@everywhere Random.seed!(1234);
@everywhere random_seed=rand(1:10000,50)[1];
@everywhere dimensions=[40];

@everywhere descr_names=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"descr_names_cact");


@everywhere begin
    function compute_shap_orders_and_prediction_curves(descrs1::Matrix{Float64},bin_resp::BitMatrix,pos::Matrix{Float64},area::Vector{Float64},number1::Int64,dim::Int64)
        #descrs1, descrs2, bin_resp, number1,number2=params
        features_per_glom, features_10_fold=shuffle_descrs_for_prediction_curves(descrs1,bin_resp,pos,area)
        println("Computing shapley orderings")
        shap_idxs=find_shap_ordering(features_10_fold,descr_names,bin_resp,ntrees,sample_perc,perc_features,random_seed);
        println("Computing per glom curves")
        pred_curves_per_glom=chemotopy_ordered_metrics_leave_one_out_per_glom(features_per_glom,bin_resp,shap_idxs[2:end],ntrees_per_glom,sample_perc_per_glom,shap_idxs[1],dim)
        println("Computing 10 fold curves")
        pred_curves_10_fold=chemotopy_ordered_metrics_old(features_10_fold,bin_resp,shap_idxs[2:end],10,ntrees,sample_perc,perc_features,shap_idxs[1],dim)
        println(number1)
        return pred_curves_per_glom, pred_curves_10_fold
    end
    
    function f1(params)
        n1,d=params
        return compute_shap_orders_and_prediction_curves(cact_descrs,binary_responses[1],glomeruli_positions[1],glomeruli_areas[1],n1,d)
    end

    function f2(params)
        n1,d=params
        return compute_shap_orders_and_prediction_curves(cact_descrs,binary_responses[2],glomeruli_positions[2],glomeruli_areas[2],n1,d)
    end
end

@everywhere random_shuffles=1:50;

println("Computing shuffled prediction curves")

@time animal1_results=pmap(f1,Iterators.product(random_shuffles,dimensions));

@time animal2_results=pmap(f2,Iterators.product(random_shuffles,dimensions));

println("Saving data")

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_cact_shuffled_descrs.jld"),"animal1_results",animal1_results,
                                                                                                "animal2_results",animal2_results);
