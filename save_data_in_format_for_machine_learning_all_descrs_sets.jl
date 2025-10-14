using PyCall,JLD, DataFrames, LinearAlgebra, Statistics,Random, MLBase;


#ENV["PYTHON"]="/home/nqm5625/anaconda3/bin/python";
#Pkg.build("PyCall");

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));
#pred_metrics_10_fold_pom160=chem_descr_and_pos_classification_old(Matrix(X_pom160[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_pom21=chem_descr_and_pos_classification_old(Matrix(X_pom21[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_cact=chem_descr_and_pos_classification_old(Matrix(X_cact[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_smarts42=chem_descr_and_pos_classification_old(Matrix(X_smarts42[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_haddad32=chem_descr_and_pos_classification_old(Matrix(X_haddad32[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_maccs=chem_descr_and_pos_classification_old(Matrix(X_maccs[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);
#pred_metrics_10_fold_edragon1377=chem_descr_and_pos_classification_old(Matrix(X_edragon1377[1]),binary_responses[1],nfolds,ntrees,sample_perc,random_seed);

#normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chem_data_segmented.jld"),"gloms_positions");
normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"binary_responses");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering_all_omp8x.jld"),"masks");


#normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chem_data_segmented.jld"),"gloms_positions");
normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"binary_responses");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/data_for_resp_prediction_and_clustering.jld"),"masks");


#normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chem_data_segmented.jld"),"gloms_positions");
normalized_glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"gloms_positions");
glomeruli_areas=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"gloms_areas");
resp_matrices=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"resp_matrices");
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"binary_responses");
masks=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/data_for_resp_prediction_and_clustering.jld"),"masks");

histogram(glomeruli_areas[2],bins=25,grid=false,xtickfontsize=16,ytickfontsize=16,legend=false)
Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/Manuscript_figures_final/Supplemental Figures","OB_areas_4.svg"));

#uncomment [setdiff(1:59,26:32),:] if considering the no-amines dataset;
using DelimitedFiles;
descr_vals = matopen(joinpath(homedir(),"Dropbox/MattData/descriptorsets_omp8x_itskovordering.mat"));
maccs_descrs=read(descr_vals)["maccs_omp8xitskov"][setdiff(1:59,26:32),:];
smarts42_descrs=read(descr_vals)["smarts42_omp8xitskov"];#[setdiff(1:59,26:32),:];
#edragon1377_descrs=read(descr_vals)["edragon1377_omp8xitskov"][setdiff(1:59,26:32),:];
haddad32_descrs=read(descr_vals)["haddad32_omp8xitskov"][setdiff(1:59,26:32),:];

pom21_descrs=readdlm(joinpath(homedir(),"Dropbox/Nikola-Vladimir/learned_descrs_21odor_labels.csv"), ',', Float64)[setdiff(1:59,26:32),:];

#M=fit(PCA,pom21_descrs';pratio=1,maxoutdim=20);
#variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

#pom21_pca_descrs=MultivariateStats.transform(M,pom21_descrs');
#pom21_pca_descrs=copy(pom21_pca_descrs');

maccs_descrs_names=Array{String,1}(undef,size(maccs_descrs,2));
for i in 1:length(maccs_descrs_names)
    maccs_descrs_names[i]="feat$i"
end

smarts42_descrs_names=Array{String,1}(undef,size(smarts42_descrs,2));
for i in 1:length(smarts42_descrs_names)
    smarts42_descrs_names[i]="feat$i"
end

#edragon1377_descrs_names=Array{String,1}(undef,size(edragon1377_descrs,2));
#for i in 1:length(edragon1377_descrs_names)
#    edragon1377_descrs_names[i]="feat$i"
#end

haddad32_descrs_names=Array{String,1}(undef,size(haddad32_descrs,2));
for i in 1:length(haddad32_descrs_names)
    haddad32_descrs_names[i]="feat$i"
end

pom21_descrs_names=Array{String,1}(undef,size(pom21_descrs,2));
for i in 1:length(pom21_descrs_names)
    pom21_descrs_names[i]="feat$i"
end

unique_descr_idx=unique(z -> maccs_descrs[:,z], 1:size(maccs_descrs,2));
maccs_descrs=maccs_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), maccs_descrs, dims = 1));
maccs_descrs=maccs_descrs[:,zero_idxs];
maccs_descrs_names=maccs_descrs_names[unique_descr_idx];
maccs_descrs_names=maccs_descrs_names[zero_idxs];
maccs_descrs_names=vcat(Symbol.(maccs_descrs_names),:x,:y,:A);

unique_descr_idx=unique(z -> smarts42_descrs[:,z], 1:size(smarts42_descrs,2));
smarts42_descrs=smarts42_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), smarts42_descrs, dims = 1));
smarts42_descrs=smarts42_descrs[:,zero_idxs];
smarts42_descrs_names=smarts42_descrs_names[unique_descr_idx];
smarts42_descrs_names=smarts42_descrs_names[zero_idxs];
smarts42_descrs_names=vcat(Symbol.(smarts42_descrs_names),:x,:y,:A);

#unique_descr_idx=unique(z -> edragon1377_descrs[:,z], 1:size(edragon1377_descrs,2));
#edragon1377_descrs=edragon1377_descrs[:,unique_descr_idx];
#zero_idxs=vec(mapslices(col -> any(col .!= 0), edragon1377_descrs, dims = 1));
#edragon1377_descrs=edragon1377_descrs[:,zero_idxs];
#edragon1377_descrs_names=edragon1377_descrs_names[unique_descr_idx];
#edragon1377_descrs_names=edragon1377_descrs_names[zero_idxs];
#edragon1377_descrs_names=vcat(Symbol.(edragon1377_descrs_names),:x,:y,:A);

unique_descr_idx=unique(z -> haddad32_descrs[:,z], 1:size(haddad32_descrs,2));
haddad32_descrs=haddad32_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), haddad32_descrs, dims = 1));
haddad32_descrs=haddad32_descrs[:,zero_idxs];
haddad32_descrs_names=haddad32_descrs_names[unique_descr_idx];
haddad32_descrs_names=haddad32_descrs_names[zero_idxs];
haddad32_descrs_names=vcat(Symbol.(haddad32_descrs_names),:x,:y,:A);

unique_descr_idx=unique(z -> pom21_descrs[:,z], 1:size(pom21_descrs,2));
pom21_descrs=pom21_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), pom21_descrs, dims = 1));
pom21_descrs=pom21_descrs[:,zero_idxs];
pom21_descrs_names=pom21_descrs_names[unique_descr_idx];
pom21_descrs_names=pom21_descrs_names[zero_idxs];
pom21_descrs_names=vcat(Symbol.(pom21_descrs_names),:x,:y,:A);

#schmuker cactvs and chemsense descriptors
schmuker_cact_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/cactvs_our_odors.csv");
cact_descrs, cact_descrs_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_cact_filename));

unique_descr_idx=unique(z -> cact_descrs[:,z], 1:size(cact_descrs,2));
cact_descrs=cact_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), cact_descrs, dims = 1));
cact_descrs=cact_descrs[:,zero_idxs];
cact_descrs_names=cact_descrs_names[unique_descr_idx];
cact_descrs_names=cact_descrs_names[zero_idxs];
cact_descrs_names=vcat(Symbol.(cact_descrs_names),:x,:y,:A);

using MultivariateStats;

M=fit(PCA,cact_descrs';pratio=1,maxoutdim=40);
variance_explained=sum(principalvars(M) ./ tvar(M) * 100)

pca_cact_descrs=MultivariateStats.transform(M,cact_descrs');
pca_cact_descrs=copy(pca_cact_descrs');

# using Plots;
# p1=Plots.scatter(pca_cact_descrs[1:5,1],pca_cact_descrs[1:5,2],label="acids",legend=:outerright,grid=false,c=:blue,markershape=:rect,title="2 dim PCA of CACTUS");
# Plots.scatter!(pca_cact_descrs[6:11,1],pca_cact_descrs[6:11,2],label="aldehydes",legend=:outerright,grid=false,c=:red,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[12:16,1],pca_cact_descrs[12:16,2],label="esters",legend=:outerright,grid=false,c=:green,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[17:18,1],pca_cact_descrs[17:18,2],label="furans",legend=:outerright,grid=false,c=:orange,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[19:24,1],pca_cact_descrs[19:23,2],label="ketones",legend=:outerright,grid=false,c=:purple,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[24:25,1],pca_cact_descrs[24:25,2],label="alcohols",legend=:outerright,grid=false,c=:black,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[26:32,1],pca_cact_descrs[26:32,2],label="amines",legend=:outerright,grid=false,c=:pink,markershape=:rect);
# Plots.scatter!(pca_cact_descrs[[33,36,37,38,39,40,41,42],1],pca_cact_descrs[[33,36,37,38,39,40,41,42],2],label="pyrazines",legend=:outerright,grid=false,c=:blue,markershape=:diamond);
# Plots.scatter!(pca_cact_descrs[34:35,1],pca_cact_descrs[34:35,2],label="pyridines",legend=:outerright,grid=false,c=:red,markershape=:diamond);
# Plots.scatter!(pca_cact_descrs[[43,47],1],pca_cact_descrs[[43,47],2],label="phenols",legend=:outerright,grid=false,c=:green,markershape=:diamond);
# Plots.scatter!(pca_cact_descrs[[44,45,46,48,50,51,52],1],pca_cact_descrs[[44,45,46,48,50,51,52],2],label="aromatics",legend=:outerright,grid=false,c=:orange,markershape=:diamond);
# Plots.scatter!([pca_cact_descrs[49,1]],[pca_cact_descrs[49,2]],label="mixed",legend=:outerright,grid=false,c=:purple,markershape=:diamond);
# Plots.scatter!(pca_cact_descrs[53:59,1],pca_cact_descrs[53:59,2],label="thiazolines",legend=:outerright,grid=false,c=:black,markershape=:diamond);
# Plots.plot(p1)
# Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files","cactus_pca.png"))


pca_cact_descrs_names=Array{String,1}(undef,size(pca_cact_descrs,2));
for i in 1:length(pca_cact_descrs_names)
    pca_cact_descrs_names[i]="feat$i"
end

pca_cact_descrs_names=vcat(Symbol.(pca_cact_descrs_names),:x,:y,:A);


Ndatasets=length(binary_responses);

model_features_maccs=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_maccs=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_maccs)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(maccs_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_maccs[k]=vec(model_feature);
    model_features_maccs[k]=model_feature
end

model_features_smarts42=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_smarts42=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_smarts42)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(smarts42_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_smarts42[k]=vec(model_feature);
    model_features_smarts42[k]=model_feature
end

# model_features_edragon1377=Array{Any,1}(undef,Ndatasets);
# model_features_data_frame_edragon1377=Array{Any,1}(undef,Ndatasets);
# for k in 1:length(model_features_edragon1377)
#     model_feature=Array{Any,2}(undef,size(binary_responses[k]));
#     for i in 1:size(model_feature,1)
#         for j in 1:size(model_feature,2)
#             model_feature[i,j]=vcat(edragon1377_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
#         end
#     end
#     model_features_data_frame_edragon1377[k]=vec(model_feature);
#     model_features_edragon1377[k]=model_feature
# end

model_features_haddad32=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_haddad32=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_haddad32)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(haddad32_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_haddad32[k]=vec(model_feature);
    model_features_haddad32[k]=model_feature
end

model_features_cact=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_cact=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_cact)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(cact_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_cact[k]=vec(model_feature);
    model_features_cact[k]=model_feature
end


model_features_pca_cact=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_pca_cact=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_pca_cact)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(pca_cact_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_pca_cact[k]=vec(model_feature);
    model_features_pca_cact[k]=model_feature
end


model_features_pom21=Array{Any,1}(undef,Ndatasets);
model_features_data_frame_pom21=Array{Any,1}(undef,Ndatasets);
for k in 1:length(model_features_pom21)
    model_feature=Array{Any,2}(undef,size(binary_responses[k]));
    for i in 1:size(model_feature,1)
        for j in 1:size(model_feature,2)
            model_feature[i,j]=vcat(pom21_descrs[j,:],normalized_glomeruli_positions[k][i,:],glomeruli_areas[k][i])
        end
    end
    model_features_data_frame_pom21[k]=vec(model_feature);
    model_features_pom21[k]=model_feature
end

y=Array{Any,1}(undef,Ndatasets);
for i in 1:length(y)
    y[i]=vec(binary_responses[i]);
end

using DataFrames;

features_maccs=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_maccs)
    feature=fill(0.0,length(y[k]),length(model_features_maccs[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_maccs[k][i]
    end
    features_maccs[k]=feature
end

X_maccs=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_maccs)
    X_maccs[i]=Matrix(DataFrame(features_maccs[i],maccs_descrs_names,makeunique=true));
end

features_smarts42=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_smarts42)
    feature=fill(0.0,length(y[k]),length(model_features_smarts42[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_smarts42[k][i]
    end
    features_smarts42[k]=feature
end

X_smarts42=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_smarts42)
    X_smarts42[i]=Matrix(DataFrame(features_smarts42[i],smarts42_descrs_names,makeunique=true));
end

features_haddad32=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_haddad32)
    feature=fill(0.0,length(y[k]),length(model_features_haddad32[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_haddad32[k][i]
    end
    features_haddad32[k]=feature
end

X_haddad32=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_haddad32)
    X_haddad32[i]=Matrix(DataFrame(features_haddad32[i],haddad32_descrs_names,makeunique=true));
end

features_pom21=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_pom21)
    feature=fill(0.0,length(y[k]),length(model_features_pom21[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_pom21[k][i]
    end
    features_pom21[k]=feature
end

X_pom21=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_pom21)
    X_pom21[i]=Matrix(DataFrame(features_pom21[i],pom21_descrs_names,makeunique=true));
end

features_cact=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_cact)
    feature=fill(0.0,length(y[k]),length(model_features_cact[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_cact[k][i]
    end
    features_cact[k]=feature
end

X_cact=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_cact)
    X_cact[i]=Matrix(DataFrame(features_cact[i],cact_descrs_names,makeunique=true));
end

features_pca_cact=Array{Any,1}(undef,Ndatasets);
for k in 1:length(features_pca_cact)
    feature=fill(0.0,length(y[k]),length(model_features_pca_cact[k][1]));
    for i in 1:size(feature,1)
        feature[i,:]=model_features_data_frame_pca_cact[k][i]
    end
    features_pca_cact[k]=feature
end

X_pca_cact=Array{Any,1}(undef,Ndatasets);
for i in 1:length(X_pca_cact)
    X_pca_cact[i]=Matrix(DataFrame(features_pca_cact[i],pca_cact_descrs_names,makeunique=true));
end

save(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_no_amines_different_descrs_for_prediction_all_omp8x.jld"),
                    "descr_names_maccs",maccs_descrs_names,"descr_names_smarts42",smarts42_descrs_names,
                    "descr_names_haddad32",haddad32_descrs_names,"descr_names_pom21",pom21_descrs_names,
                    "descr_names_cact",cact_descrs_names,"descr_names_pca_cact",pca_cact_descrs_names,
                    "Data_maccs",model_features_maccs,"Data_smarts42",model_features_smarts42,
                    "Data_haddad32",model_features_haddad32,"Data_pom21",model_features_pom21,
                    "Data_cact",model_features_cact,"Data_pca_cact",model_features_pca_cact,
                    "Data_frame_maccs",X_maccs,"Data_frame_smarts42",X_smarts42,
                    "Data_frame_haddad32",X_haddad32,"Data_frame_pom21",X_pom21,
                    "Data_frame_cact",X_cact,"Data_frame_pca_cact",X_pca_cact,"bin_resp",binary_responses,"resp_matr",resp_matrices);#,
using JLD;
A=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/segmented_data_no_amines_different_descrs_for_prediction.jld"),"descr_names_cact") 