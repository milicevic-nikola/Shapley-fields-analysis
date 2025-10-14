using PyCall,JLD, DataFrames, LinearAlgebra, Statistics,Random, MLBase;

#ENV["PYTHON"]="/home/nqm5625/anaconda3/bin/python";
#Pkg.build("PyCall");

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));


normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"binary_responses");
descr_vals_norm=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"descr_vals_norm");
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"descr_names");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/schmuker_descrs_data_segmented.jld"),"masks");

#normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chem_data_segmented.jld"),"gloms_positions");
normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"binary_responses");
descr_vals_norm=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"descr_vals_norm");
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/schmuker_descrs_data_segmented.jld"),"descr_names");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/learned_descrs_data_segmented.jld"),"masks");

normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"binary_responses");
descr_vals_norm=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"descr_vals_norm");
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"descr_names");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"masks");


using DelimitedFiles;
descr_vals_norm = matopen(joinpath(homedir(),"Dropbox/MattData/descriptorsets_omp8x_itskovordering.mat"));
maccs_descrs=read(descr_vals_norm)["maccs_omp8xitskov"]
smarts42_descrs=read(descr_vals_norm)["smarts42_omp8xitskov"]
edragon1377_descrs=read(descr_vals_norm)["edragon1377_omp8xitskov"]
haddad32_descrs=read(descr_vals_norm)["haddad32_omp8xitskov"]



Nshuffles=1;

descr_vals_norm_shuffled=Array{Any,1}(undef,Nshuffles);
for i in 1:length(descr_vals_norm_shuffled)
    descr_vals_norm_shuffled[i]=shuffle(descr_vals_norm)
end

#m,n=size(descr_vals_norm);
#descr_vals_norm=fill(0.0,m,n)
#for i in 1:m
#    descr_vals_norm[i,:]=descr_vals_norm_shuffled[i][i,:]
#end
descr_names=1:1:256
good_descr_names=vcat(Symbol.(descr_names),:x,:y,:A);

Ndatasets=length(binary_responses);

model_features=Array{Any,1}(undef,Ndatasets);
model_features_data_frame=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(descr_vals_norm[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
            #model_feature[i,j]=descr_vals_norm[j,:]#,normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame[k]=vec(model_feature);
    model_features[k]=model_feature
end
#model_features=vec(model_features);

model_features_shuffled=Array{Any,1}(undef,Ndatasets);
model_features_shuffled_data_frame=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features)
    model_feature=Array{Any,1}(undef,Nshuffles)
    model_feature_data_frame=Array{Any,1}(undef,Nshuffles)
    for i in 1:length(model_feature)
        model_feature[i]=Array{Any,2}(undef,size(binary_responses[k]))
        for j in 1:size(model_feature[i],1)
            for l in 1:size(model_feature[i],2)
                model_feature[i][j,l]=vcat(descr_vals_norm_shuffled[i][l,:],normalized_glomeruli_positions[k][j,:],glomeruli_areas[k][j])
            end
        end
        model_feature_data_frame[i]=vec(model_feature[i])
    end
    model_features_shuffled[k]=model_feature
    model_features_shuffled_data_frame[k]=model_feature_data_frame
end

y=Array{Any,1}(undef,Ndatasets);
for i in 1:length(y)
    y[i]=vec(binary_responses[i]);
end

features=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features)
    feature=fill(0.0,length(y[k]),length(model_features[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame[k][i]
    end
    features[k]=feature
end

features_shuffled=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_shuffled)
    features_shuf=Array{Any,1}(undef,Nshuffles)
    for i in 1:length(features_shuf)
        feature=fill(0.0,length(y[k]),length(model_features_shuffled[k][i][1]))
        for j in 1:size(feature,1)
            feature[j,:]=model_features_shuffled_data_frame[k][i][j]
        end
        features_shuf[i]=feature
    end
    features_shuffled[k]=features_shuf
end

using DataFrames
X=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X)
    X[i]=DataFrame(features[i],good_descr_names,makeunique=true);
end

X_shuffled=Array{Any,1}(undef,Ndatasets);
for k in 1:length(X_shuffled)
    X_frames=Array{Any,1}(undef,Nshuffles)
    for i in 1:length(X_frames)
        X_frames[i]=DataFrame(features_shuffled[k][i],good_descr_names,makeunique=true)
        #println("$k,$i")
    end
    X_shuffled[k]=X_frames
end


