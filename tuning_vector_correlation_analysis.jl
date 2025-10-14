include(joinpath(homedir(),"Documents/Nikola-Vladimir/to_share/Useful_functions.jl"));
include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));


schmuker_cact_filename=joinpath(homedir(),"Documents/Nikola-Vladimir/cactvs_our_odors.csv");
cact_descrs, cact_descrs_names = load_descriptors_schmuker(joinpath(homedir(),schmuker_cact_filename));

unique_descr_idx=unique(z -> cact_descrs[:,z], 1:size(cact_descrs,2));
cact_descrs=cact_descrs[:,unique_descr_idx];
zero_idxs=vec(mapslices(col -> any(col .!= 0), cact_descrs, dims = 1));
cact_descrs=cact_descrs[:,zero_idxs];

pom21_descrs=readdlm(joinpath(homedir(),"Dropbox/Nikola-Vladimir/learned_descrs_21odor_labels.csv"), ',', Float64);#[setdiff(1:59,26:32),:];


# Class indices corrected to use arrays
acid_idxs = 1:5
aldehyde_idxs = 6:11
ester_idxs = 12:16
#furan_idxs = 17:18
ketone_idxs = 19:23
#alcohol_idxs = 24:25
amine_idxs = 26:32
pyrazine_idxs = [33, 36, 37, 38, 39, 40, 41, 42]
#pyridine_idxs = 34:35
#phenolic_idxs = [43, 47]
aromatic_idxs = [44, 45, 46, 48, 50, 51, 52]
#mixed_idxs = [49]
thiazoline_idxs = [53, 54, 55, 56, 57, 58, 59]


using MultivariateStats   # For PCA
using Plots               # For plotting
using UMAP

# Perform PCA and project to 2 dimensions
pca_model = fit(PCA, pom21_descrs'; maxoutdim=2)
X_projected = MultivariateStats.transform(pca_model, pom21_descrs')

umap_result = umap(cact_descrs', n_neighbors=15, min_dist=0.1, n_epochs=200)  # n_components is implicit in the output size


# Prepare colors and labels for each class
color_palette = [:red, :blue, :green, :purple, :orange, :cyan, :brown, :pink, :yellow, :black, :magenta, :lightblue, :limegreen]
class_labels = ["Acid", "Aldehyde", "Ester", "Furan", "Ketone", "Alcohol", "Amine", "Pyrazine", "Pyridine", "Phenolic", "Aromatic", "Mixed", "Thiazoline"]
class_idxs = [acid_idxs, aldehyde_idxs, ester_idxs, furan_idxs, ketone_idxs, alcohol_idxs, amine_idxs, pyrazine_idxs, pyridine_idxs, phenolic_idxs, aromatic_idxs, mixed_idxs, thiazoline_idxs]

# Initialize an empty plot
plt = Plots.plot(grid=false,title="PCA GNN",legend=:outerright)

# Plot each class separately
for (i, idxs) in enumerate(class_idxs)
    Plots.scatter!(plt, X_projected[1, idxs],X_projected[2, idxs], color=color_palette[i], label=class_labels[i])
end

display(plt)

Plots.savefig(plt, "Documents/PCA_Projection_Plot_GNN.svg") 