using Distributed,JLD, IterTools;

addprocs(40)

@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/Useful_functions.jl"));
@everywhere include(joinpath(homedir(),"Response_prediction_pipeline/python_shap_functions_new.jl"));
@everywhere using Random;

@everywhere descr_set_name="cact";
@everywhere dataset_name="all_odors";

@everywhere X=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_$descr_set_name");
@everywhere binary_responses=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
@everywhere Data_frames=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_frame_$descr_set_name");
@everywhere descr_names=load(joinpath(homedir(),"data_to_use_for_predictions/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"descr_names_$descr_set_name");

#the hyperparameters for a grid search along with 
@everywhere ntrees_array=[256];#[32,64,128]; #number of trees the random forest has
@everywhere ntrees_array_per_glom=[128];#[32,64,128]; #number of trees the random forest has
@everywhere perc_features_array=[1.0];#[0.7,0.9]; #percentage of features used by each tree, not used in the per glom models                                
@everywhere sample_perc_array_per_glom=[0.5];#[0.5,0.7]; #percentage of data each tree uses
@everywhere sample_perc_array=[0.7];#[0.5,0.7]; #percentage of data each tree uses
#@everywhere Random.seed!(1234);  #make sure every run uses the same random seeds
@everywhere Random.seed!(1337);  #make sure every run uses the same random seeds
#@everywhere random_seeds_array=rand(1:10000,100); #generate 100 random seeds
@everywhere nfolds=10; #number of folds to use in one of the models
@everywhere number_of_descrs=length(descr_names);

#@everywhere random_seeds_array=repeat([rand(1:10000,100)[1]],100); #generate 100 random seeds
@everywhere random_seeds_array=rand(1:10000,100)[1]; #generate 100 random seeds

@everywhere binary_responses_shuffled=Array{Any,1}(undef,2);
@everywhere begin
    for i in 1:2
        bin_resp=Array{Any,1}(undef,100)
        for j in 1:100
            bin_resp[j]=shuffle(binary_responses[i])
        end
        binary_responses_shuffled[i]=bin_resp
    end
end

println("data $dataset_name"*" descrs $descr_set_name")

println("Computing shapley orderings")

#@everywhere k1(number)=find_shap_ordering(Data_frames[1],descr_names,binary_responses[1],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);
#@everywhere k2(number)=find_shap_ordering(Data_frames[2],descr_names,binary_responses[2],ntrees_array[1],sample_perc_array[1],perc_features_array[1],number);

@everywhere k1(number)=find_shap_ordering(Data_frames[1],descr_names,binary_responses_shuffled[1][number],ntrees_array[1],sample_perc_array[1],perc_features_array[1],random_seeds_array);
@everywhere k2(number)=find_shap_ordering(Data_frames[2],descr_names,binary_responses_shuffled[2][number],ntrees_array[1],sample_perc_array[1],perc_features_array[1],random_seeds_array);

# @time shap_idx_array1=pmap(k1,random_seeds_array);
# @time shap_idx_array2=pmap(k2,random_seeds_array);

@time shap_idx_array1=pmap(k1,1:100);
@time shap_idx_array2=pmap(k2,1:100);


#@everywhere f1(number)=find_shap_ordering_per_glom(X[1],binary_responses[1],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],number);
#@everywhere f2(number)=find_shap_ordering_per_glom(X[2],binary_responses[2],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],number);

@everywhere f1(number)=find_shap_ordering_per_glom(X[1],binary_responses_shuffled[1][number],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],928);
@everywhere f2(number)=find_shap_ordering_per_glom(X[2],binary_responses_shuffled[2][number],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],random_seeds_array);

@everywhere f1(number)=chem_descr_shap_values_leave_one_out_per_glom(X[1],binary_responses_shuffled[1][number],ntrees_array_per_glom[1],sample_perc_array_per_glom[1],928)

#@time shap_idx_array1_per_glom=pmap(f1,random_seeds_array);
#@time shap_idx_array2_per_glom=pmap(f2,random_seeds_array);

@time shap_idx_array1_per_glom=pmap(f1,1:100);
@time shap_idx_array2_per_glom=pmap(f2,1:100);

save(joinpath(homedir(),"correct_chemotopy_results/$dataset_name"*"_$descr_set_name"*"_shapley_rankings_and_absolute_mean_values_shuffled_resp.jld"),
                                                    "shap_ranks1",shap_idx_array1,
                                                    "shap_ranks2",shap_idx_array2,
                                                    "shap_ranks1_per_glom",shap_idx_array1_per_glom,
                                                    "shap_ranks2_per_glom",shap_idx_array2_per_glom);



include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Useful_functions.jl"));

descr_set_name="smarts42";
dataset_name="all_odors";

X=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_$descr_set_name");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");
Data_frames=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"Data_frame_$descr_set_name");
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"descr_names_$descr_set_name");

A=Array{Any,1}(undef,2);
A[1]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks1");
A[2]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks2");
B=Array{Any,1}(undef,2);
B[1]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks1_per_glom");
B[2]=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/all_odors_cact_shapley_rankings_and_absolute_mean_values.jld"),"shap_ranks2_per_glom");

rankings=Array{Any,1}(undef,2);
for i in 1:length(rankings)
    rankings[i]=fill(0.0,140,100)
    for j in 1:size(rankings[i],2)
        for k in 1:size(rankings[i],1)
            rankings[i][k,j]=A[i][j][1][k]
        end
    end
end

rankings_per_glom=Array{Any,1}(undef,2);
for i in 1:length(rankings_per_glom)
    rankings_per_glom[i]=fill(0.0,140,100)
    for j in 1:size(rankings_per_glom[i],2)
        for k in 1:size(rankings_per_glom[i],1)
            rankings_per_glom[i][k,j]=B[i][j][1][k]
        end
    end
end

kendall_taus_83L=fill(0.0,100,100);
kendall_taus_83R=fill(0.0,100,100);
kendall_taus_83L_per_glom=fill(0.0,100,100);
kendall_taus_83R_per_glom=fill(0.0,100,100);
kendall_taus_83L_vs_83R=fill(0.0,100,100);
kendall_taus_83L_vs_83R_per_glom=fill(0.0,100,100);
kendall_taus_83L_vs_per_glom=fill(0.0,100,100);
kendall_taus_83R_vs_per_glom=fill(0.0,100,100);
for i in 1:100
    for j in 1:100
        kendall_taus_83L[i,j]=kendall_tau(rankings[1][:,i],rankings[1][:,j])[1]
        kendall_taus_83R[i,j]=kendall_tau(rankings[2][:,i],rankings[2][:,j])[1]
        kendall_taus_83L_per_glom[i,j]=kendall_tau(rankings_per_glom[1][:,i],rankings_per_glom[1][:,j])[1]
        kendall_taus_83R_per_glom[i,j]=kendall_tau(rankings_per_glom[2][:,i],rankings_per_glom[2][:,j])[1]
        kendall_taus_83L_vs_83R[i,j]=kendall_tau(rankings[1][:,i],rankings[2][:,j])[1]
        kendall_taus_83L_vs_83R_per_glom[i,j]=kendall_tau(rankings_per_glom[1][:,i],rankings_per_glom[2][:,j])[1]
        kendall_taus_83L_vs_per_glom[i,j]=kendall_tau(rankings[1][:,i],rankings_per_glom[1][:,j])[1]
        kendall_taus_83R_vs_per_glom[i,j]=kendall_tau(rankings[2][:,i],rankings_per_glom[2][:,j])[1]
    end
end

using Plots;
# gr()

# using StatsPlots;
# StatsPlots.violin(["whole"],vec(kendall_taus_83L),color=:red,grid=false,ylabel="Kendall Tau Correspondence",label="animal 1");
# StatsPlots.violin!(["whole "],vec(kendall_taus_83R),color=:blue,label="animal 2");
# StatsPlots.violin!(["per glom"],vec(kendall_taus_83L_per_glom),color=:red,label="");
# StatsPlots.violin!(["per glom "],vec(kendall_taus_83R_per_glom),color=:blue,label="");
# StatsPlots.violin!(["whole  "],vec(kendall_taus_83L_vs_83R),color=:green,label="1 vs 2");
# StatsPlots.violin!(["per glom  "],vec(kendall_taus_83L_vs_83R_per_glom),color=:green,label="");
# StatsPlots.violin!(["whole/per glom"],vec(kendall_taus_83L_vs_per_glom),color=:red,label="");
# StatsPlots.violin!(["whole/per glom "],vec(kendall_taus_83R_vs_per_glom),color=:blue,label="",size=(1000,500))
# Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","shap_rankings_comparison.svg"));


# Plots.boxplot(kendall_taus_83L)





# rankings_per_glom=Array{Any,1}(undef,2)
# for i in 1:length(rankings_per_glom)
#     rankings_per_glom[i]=fill(0.0,140,100)
#     for j in 1:size(rankings_per_gloms[i],2)

# rankings_matrices=Array{Any,1}(undef,2);
# for i in 1:length(rankings_matrices)
#     rankings_matrices[i]=fill(0.0,140,100)
#     for j in 1:size(rankings_matrices[i],2)
#         #rankings_matrices[i][:,j]=rankings[i][j][2]
#         for k in 1:size(rankings_matrices[i],1)
#             rankings_matrices[i][k,j]=findall(x->x==k,rankings[i][j][2])[1]
#         end
#     end
# end


# a=find_shap_ordering_per_glom(X[1],binary_responses[1],128,0.5,0)
# c=find_shap_ordering_per_glom(X[1][1:1,:],binary_responses[1][1:1,:],128,0.5,123)

# b=find_shap_ordering(Data_frames[1],descr_names,binary_responses[1],256,0.7,1.0,123)
# d=find_shap_ordering(Data_frames[1],descr_names,binary_responses[1],256,0.7,1.0,0)

# scipy= pyimport("scipy");

# kendall_tau=scipy.stats.kendalltau

# kendall_tau(a[2:end],c[2:end],method="exact")
# kendall_tau(a[2:end],b[2:end],method="exact")

# kendall_tau(b[2:11],d[2:11],method="exact")

# Plots.plot(1:165,rankings_matrices[1],label="omp83",c=:red,legend=false)
# Plots.plot!(1:165,rankings_matrices[2],label="omp85",c=:blue,legend=false)
# Plots.histogram(abs.(vec(rankings_matrices[1]).-vec(rankings_matrices[2])),legend=false,grid=false,xlabel="change in rank", ylabel="count")
# Plots.annotate!(40, 750, "mean change in rank="*string(round(mean_rank_diff,digits=2)))
# PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","shapley_rank_hist.eps"));


# using Distributions, StatsPlots

# pois=Distributions.Poisson(mean(vec(abs.(rankings_matrices[1].-rankings_matrices[2]))))
# pois=Distributions.Poisson(2)

# StatsPlots.plot(pois)