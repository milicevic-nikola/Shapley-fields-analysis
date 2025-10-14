include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
#include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Load_the_data_from_the_files_high_and_low_conc.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Load_the_data_from_the_files_high_and_low_conc_no_amines.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/to_share/More_functions.jl"));
using JLD;


#schmuker cactvs and chemsense descriptors
schmuker_descr_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/cactvs_our_odors.csv");
descr_cact, descr_cact_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_descr_filename));

schmuker_descr_filename=joinpath(homedir(),"Dropbox/Nikola-Vladimir/chemsense_our_odors.csv");
descr_chemsense, descr_chemsense_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_descr_filename));


descr_vals_norm=hcat(descr_chemsense,descr_cact)#,descr_MACCS);
descr_names=vcat(descr_chemsense_names,descr_cact_names);

descr_vals_norm=descr_vals_norm[setdiff(1:59,26:32),:];

unique_descr_idx=unique(z -> descr_vals_norm[:,z], 1:size(descr_vals_norm,2));
descr_vals_norm=descr_vals_norm[:,unique_descr_idx];
descr_names=descr_names[unique_descr_idx];


cor_feat=cor(descr_vals_norm);
for i in 1:size(cor_feat,1)
	cor_feat[i,i]=0
end
high_cor_feat = cor_feat.>0.9
high_cor_feat=sum(high_cor_feat, dims=1)[1,:]
high_cor_idx=findall(high_cor_feat.!=0)
#high_cor_idx=high_cor_idx[8:100]
descr_vals_norm=descr_vals_norm[:,setdiff(1:end,high_cor_idx)]
descr_names=descr_names[setdiff(1:end,high_cor_idx)]



#=
outfile = joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/chemsense_and_cactvs_descrs.txt")
open(outfile, "w") do f
	for descr in descr_names
		println(f, descr)
	end
end
=#
UseDatasets=[1,2,3,4];
Ndatasets=length(UseDatasets);

Nglomeruli= zeros(Int,length(UseDatasets));

animal_gloms=Array{Any,1}(undef,2);
for i in 1:length(animal_gloms)
	animal_gloms[i]=intersect(nonzero_rows[i],nonzero_rows[i+2])
end

resp_matr[1]=resp_matr[1][animal_gloms[1],:];
resp_matr[2]=resp_matr[2][animal_gloms[2],:];
resp_matr[3]=resp_matr[3][animal_gloms[1],:];
resp_matr[4]=resp_matr[4][animal_gloms[2],:];
masks[1]=masks[1][animal_gloms[1]];
masks[2]=masks[2][animal_gloms[2]];
masks[3]=masks[3][animal_gloms[1]];
masks[4]=masks[4][animal_gloms[2]];


for i in 1:Ndatasets
    resp_matr[UseDatasets[i]]=normalize_matrix_by_max(resp_matr[UseDatasets[i]],dim=1)
end

centroids=Array{Array{Float64,2},1}(undef,Ndatasets);

for i in 1:Ndatasets
    centers=zeros(Float64,length(masks[i]),2)
    for j in 1:length(masks[i])
        (min_x,max_x,min_y,max_y)=window_matr(masks[i][j])
        x=ceil(min_x-64+(max_x-min_x)/2)
        y=ceil(min_y-64+(max_y-min_y)/2)
        centers[j,1]=x
        centers[j,2]=y
    end
                            #centroids[i]=centers
    centroids[i]=convert(Array{Float64},centers)
end


glomeruli_positions=Array{Array{Float64,2},1}(undef,Ndatasets);
for i in 1:Ndatasets
    glomeruli_positions[i]=centroids[i][:,1:2]
end

normalized_glomeruli_positions=Array{Array{Float64,2},1}(undef,Ndatasets);
for i in 1:Ndatasets
    normalized_glomeruli_positions[i]=normalize_matrix_zscore(centroids[i][:,1:2]);
end

glomeruli_areas=Array{Any,1}(undef,Ndatasets);
for i in 1:Ndatasets
    #glomeruli_areas[i]=fill(0.0,size(masks[i],2))
	glomeruli_areas[i]=fill(0.0,length(masks[i]))
    for j in 1:length(glomeruli_areas[i])
        glomeruli_areas[i][j]=sum(masks[i][j])
    end
    glomeruli_areas[i]=glomeruli_areas[i]./findmax(glomeruli_areas[i])[1];
end

binary_responses=Array{BitArray{2},1}(undef,Ndatasets);
for i in 1:length(binary_responses)
    binary_responses[i]=resp_matr[UseDatasets[i]].!=0.0
end

save(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"binary_responses",
										binary_responses,"gloms_positions",normalized_glomeruli_positions,
                                        "gloms_areas",glomeruli_areas,"descr_vals_norm",descr_vals_norm,
										"resp_matrices",resp_matr,"descr_names",
										descr_names,"masks",masks,"glom_pos",glomeruli_positions);
