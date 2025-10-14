using JLD, Plots;

include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/Useful_functions.jl"));

# Load data
dataset_name = "all_odors";
masks = load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/data_for_resp_prediction_and_clustering.jld"),"masks");
binary_responses = load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/segmented_data_$dataset_name"*"_different_descrs_for_prediction.jld"),"bin_resp");

# Only keep classes with ≥2 odorants. These are:
# acids: columns 1:5 (5), aldehydes: 6:11 (6), esters: 12:16 (5),
# ketones: 19:23 (5), amines: 26:32 (7), phenolics: [43,47] (2),
# aromatics: [44,45,46,48,50,51,52] (7) and thiazolines: 53:59 (7).

using Statistics;

# Standardization function
function standardize(x)
    return (x .- mean(x)) ./ std(x)
end;

# Compute response vectors for the kept odorant classes
# Acids
acids_idx1 = vec(sum(binary_responses[1][:, 1:5], dims=2));
acids_idx2 = vec(sum(binary_responses[2][:, 1:5], dims=2));
acids_resp_vector = vcat(acids_idx1, acids_idx2);

# Aldehydes
aldehydes_idxs1 = vec(sum(binary_responses[1][:, 6:11], dims=2));
aldehydes_idxs2 = vec(sum(binary_responses[2][:, 6:11], dims=2));
aldehydes_resp_vector = vcat(aldehydes_idxs1, aldehydes_idxs2);

# Esters
esters_idxs1 = vec(sum(binary_responses[1][:, 12:16], dims=2));
esters_idxs2 = vec(sum(binary_responses[2][:, 12:16], dims=2));
esters_resp_vector = vcat(esters_idxs1, esters_idxs2);

# Furans
furans_idxs1 = vec(sum(binary_responses[1][:, 17:18], dims=2));
furans_idxs2 = vec(sum(binary_responses[2][:, 17:18], dims=2));
furans_resp_vector = vcat(furans_idxs1, furans_idxs2);

# Ketones
ketones_idxs1 = vec(sum(binary_responses[1][:, 19:23], dims=2));
ketones_idxs2 = vec(sum(binary_responses[2][:, 19:23], dims=2));
ketones_resp_vector = vcat(ketones_idxs1, ketones_idxs2);

# Alcohols
alcohols_idxs1 = vec(sum(binary_responses[1][:, 24:25], dims=2));
alcohols_idxs2 = vec(sum(binary_responses[2][:, 24:25], dims=2));
alcohols_resp_vector = vcat(alcohols_idxs1, alcohols_idxs2);

# Amines
amines_idxs1 = vec(sum(binary_responses[1][:, 26:32], dims=2));
amines_idxs2 = vec(sum(binary_responses[2][:, 26:32], dims=2));
amines_resp_vector = vcat(amines_idxs1, amines_idxs2);

# Pyrazines
pyrazines_idxs1 = vec(sum(binary_responses[1][:, [33,36,37,38,39,40,41,42]], dims=2));
pyrazines_idxs2 = vec(sum(binary_responses[2][:, [33,36,37,38,39,40,41,42]], dims=2));
pyrazines_resp_vector = vcat(pyrazines_idxs1, pyrazines_idxs2);

# Pyridines
pyridines_idxs1 = vec(sum(binary_responses[1][:, 34:35], dims=2));
pyridines_idxs2 = vec(sum(binary_responses[2][:, 34:35], dims=2));
pyridines_resp_vector = vcat(pyridines_idxs1, pyridines_idxs2);

# Phenolics
phenolics_idxs1 = vec(sum(binary_responses[1][:, [43,47]], dims=2));
phenolics_idxs2 = vec(sum(binary_responses[2][:, [43,47]], dims=2));
phenolics_resp_vector = vcat(phenolics_idxs1, phenolics_idxs2);

# Aromatics
aromatics_idxs1 = vec(sum(binary_responses[1][:, [44,45,46,48,50,51,52]], dims=2));
aromatics_idxs2 = vec(sum(binary_responses[2][:, [44,45,46,48,50,51,52]], dims=2));
aromatics_resp_vector = vcat(aromatics_idxs1, aromatics_idxs2);

# Thiazolines
thiazolines_idxs1 = vec(sum(binary_responses[1][:, 53:59], dims=2));
thiazolines_idxs2 = vec(sum(binary_responses[2][:, 53:59], dims=2));
thiazolines_resp_vector = vcat(thiazolines_idxs1, thiazolines_idxs2);

# Standardize all the kept vectors
acids_resp_std      = standardize(acids_resp_vector);
aldehydes_resp_std  = standardize(aldehydes_resp_vector);
esters_resp_std     = standardize(esters_resp_vector);
furans_resp_std     = standardize(furans_resp_vector);
ketones_resp_std    = standardize(ketones_resp_vector);
alcohols_resp_std    = standardize(alcohols_resp_vector);
amines_resp_std     = standardize(amines_resp_vector);
pyrazines_resp_std     = standardize(pyrazines_resp_vector);
pyridines_resp_std     = standardize(pyridines_resp_vector);
phenolics_resp_std  = standardize(phenolics_resp_vector);
aromatics_resp_std  = standardize(aromatics_resp_vector);
thiazolines_resp_std = standardize(thiazolines_resp_vector);

# Build U from only the kept odorant classes
U = hcat(acids_resp_std, aldehydes_resp_std, esters_resp_std, furans_resp_std, ketones_resp_std,
         alcohols_resp_std, amines_resp_std, pyrazines_resp_std, pyridines_resp_std, phenolics_resp_std, aromatics_resp_std, thiazolines_resp_std);

# Update vector_sizes to reflect the kept classes:
# [acids (5), aldehydes (6), esters (5), ketones (5), amines (7), phenolics (2), aromatics (7), thiazolines (7)]
vector_sizes = [5, 6, 5, 2, 5, 2, 7, 8, 2, 2, 7, 7];

# Define the allowed column indices from binary_responses for the kept classes
allowed_columns = vcat(1:5, 6:11, 12:16, 17:18, 19:23, 24:25, 26:32, [33,36,37,38,39,40,41,42], 34:35, [43,47], [44,45,46,48,50,51,52], 53:59);

################################################################################
# The following section (PCA on descriptors, correlation analyses, permutation
# tests, and plotting) remains largely unchanged except for the updates in the
# permutation loop that now uses only the allowed columns.
################################################################################
this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/cact_all_odors_shaps_on_bulb.jld"),"shap_per_glom_matrix");
this_descr_by_gloms=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/pom21_all_odors_shaps_on_bulb.jld"),"shap_per_glom_matrix");

#this_descr_by_gloms = load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/$dataset_name"*"/clustering_files/cact_all_odors_shaps_on_bulb_no_mixed_odorant_classes.jld"),"shap_per_glom_matrix");
#this_descr_by_gloms = load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/resp_data/all_odors/clustering_files/pom21_all_odors_shaps_on_bulb_no_mixed_odorant_classes.jld"),"shap_per_glom_matrix");
this_descr_by_gloms_copy = this_descr_by_gloms;
descr_by_gloms = vcat(this_descr_by_gloms[1], this_descr_by_gloms[2]);

using MultivariateStats
M = fit(PCA, descr_by_gloms'; pratio=1, maxoutdim=20);
variance_explained = sum(principalvars(M) ./ tvar(M) * 100)
descr_by_gloms_transformed = MultivariateStats.transform(M, descr_by_gloms');
descr_by_gloms = copy(descr_by_gloms_transformed');

A = descr_by_gloms;   # cact
B = descr_by_gloms;   # pom21

using PyCall;
sci_stats = pyimport("scipy.stats");

r_values = zeros(2, 10);  # Store Pearson r values
p_values = zeros(2, 10);  # Store p-values

# Correlation loops (first 10 columns of B and A)
for i in 1:10
    v_std_B = standardize(B[:, i])
    alpha_real_B = inv(transpose(U) * U) * transpose(U) * v_std_B
    w_B = U * alpha_real_B
    r_B, p_B = sci_stats.pearsonr(v_std_B, w_B)
    r_values[1, i] = r_B
    p_values[1, i] = p_B
end

for i in 1:10
    v_std_A = standardize(A[:, i])
    alpha_real_A = inv(transpose(U) * U) * transpose(U) * v_std_A
    w_A = U * alpha_real_A
    r_A, p_A = sci_stats.pearsonr(v_std_A, w_A)
    r_values[2, i] = r_A
    p_values[2, i] = p_A
end

using Printf;
using Plots;

# Create a blank white grid with fixed limits for 10 columns and 2 rows
heatmap(1:10, 1:2, ones(2,10), color=:white, 
    title="Pearson r and p-values", xlabel="Column Index", ylabel="Matrix (1=B, 2=A)", 
    colorbar=false, size=(1600, 400), aspect_ratio=1.5,
    xlims=(0.5, 10.5), ylims=(0.5, 2.5))

# Draw vertical lines to outline each column
for x in 0.5:1:10.5
    plot!([x, x], [0.5, 2.5], color=:black, lw=1)
end

# Draw horizontal lines to outline each row
for y in 0.5:1:2.5
    plot!([0.5, 10.5], [y, y], color=:black, lw=1)
end

# Add text annotations for r and p values in each cell
for i in 1:10
    p_val_B = p_values[1, i] < 1e-5 ? @sprintf("p=%.1e", p_values[1, i]) : @sprintf("p=%.3f", p_values[1, i])
    p_val_A = p_values[2, i] < 1e-5 ? @sprintf("p=%.1e", p_values[2, i]) : @sprintf("p=%.3f", p_values[2, i])
    annotate!(i, 1, "$(round(r_values[1,i], digits=2))\n$p_val_B"; textcolor=:black, halign=:center, valign=:middle, fontsize=12)
    annotate!(i, 2, "$(round(r_values[2,i], digits=2))\n$p_val_A"; textcolor=:black, halign=:center, valign=:middle, fontsize=12)
end

plot!()
Plots.savefig(joinpath(homedir(),"Documents/","correlations_with_odorant_classes.svg"));

using Random
# Permutation test with updated allowed columns
num_columns = 10
num_trials = 10000
significant_weights = Dict()
correlations = Vector{Tuple{Float64, Float64}}(undef, num_columns)
p_values_results = Vector{Vector{Float64}}(undef, num_columns)

for i in 1:num_columns
    v_std = standardize(A[:, i])
    alpha_real = inv(transpose(U) * U) * transpose(U) * v_std
    w = U * alpha_real
    corr_result = sci_stats.pearsonr(standardize(A[:, i]), w)
    correlation = corr_result[1]  # Correlation coefficient
    p_value = corr_result[2]      # P-value for the correlation
    correlations[i] = (correlation, p_value)

    alphas = Array{Float64}(undef, length(vector_sizes), num_trials)
    for trial in 1:num_trials
        # Permute only over the allowed odorant indices
        numbers = shuffle(allowed_columns)
        vectors = []
        start_index = 1
        for size in vector_sizes
            end_index = start_index + size - 1
            push!(vectors, numbers[start_index:end_index])
            start_index = end_index + 1
        end

        random_odorant_resp_vectors = [standardize(vcat(sum(binary_responses[1][:, vectors[j]], dims=2),
                                                        sum(binary_responses[2][:, vectors[j]], dims=2)))
                                       for j in 1:length(vector_sizes)]
        U_random = hcat(random_odorant_resp_vectors...)
        alpha = inv(transpose(U_random) * U_random) * transpose(U_random) * v_std
        alphas[:, trial] = alpha
    end

    # Compute p-values for each weight via the permutation test
    p_vals = [sum(abs.(alphas[k, :]) .>= abs(alpha_real[k])) / num_trials for k in 1:length(alpha_real)]
    p_values_results[i] = p_vals

    # Save indices where the weight is significant
    significant_indices = findall(p -> p < 0.05, p_vals)
    if !isempty(significant_indices)
        significant_weights[i] = (indices = significant_indices, weights = alpha_real[significant_indices])
    end
end

println("Correlations and p-values for each vector A[:,i]: ", correlations)
println("P-values from randomization test: ", p_values_results)
println("Significant weights and their indices: ", significant_weights)

using ColorSchemes

n_pc_features = 10
# The weight matrix now has one column per kept class plus an extra for correlation
weight_matrix = fill(NaN, n_pc_features, length(vector_sizes) + 1)
length_vector_sizes = length(vector_sizes)


for i in 1:n_pc_features
    sig_weights = get(significant_weights, i, nothing)
    if sig_weights !== nothing
        for (index, weight) in zip(sig_weights.indices, sig_weights.weights)
            weight_matrix[i, index] = abs(weight)
        end
    end
    # Add correlation value in the last column
    weight_matrix[i, end] = correlations[i][1]
end

# Remove the last column (correlations) from the matrix
weight_matrix_without_corr = weight_matrix[:, 1:length(vector_sizes)]
n_pc_features = size(weight_matrix_without_corr, 1)

weight_matrix_without_corr=weight_matrix_without_corr[:,[5,7,12,8,1,2,4,3,6,9,10,11]] #pom21
#weight_matrix_without_corr=weight_matrix_without_corr[:,[5,7,1,12,8,2,4,3,9,10,11]] #cact
# Define the odorant class names to be shown on the x-axis


odorant_class_names = ["Acids", "Aldehydes", "Esters", "Furans", "Ketones", "Alcohols",
                         "Amines", "Pyrazines", "Pyridines", "Phenolics", "Aromatics", "Thiazolines"]

# Plot the heatmap: use a white background for cells with a green scale,
# no y-axis labels, and x-axis labels as odorant class names.
heatmap(
    abs.(weight_matrix_without_corr),
    color=:greens,
    clim=(0, 1),
    grid=false,
    yflip=true,
    legend=false,
    fontsize=12,
    aspect_ratio=:equal,
    xlims=(0.5, size(weight_matrix_without_corr, 2) + 0.5),
    xticks=(1:size(weight_matrix_without_corr, 2), odorant_class_names),
    yticks=([], [])  # removes y-axis labels
)

# Annotate each cell with the rounded value (if desired)
annotate!([
    (j, i, Plots.text(round(weight_matrix_without_corr[i, j], digits=2), 12, "Computer Modern", :black))
    for i in 1:n_pc_features, j in 1:size(weight_matrix_without_corr, 2)
    if !isnan(weight_matrix_without_corr[i, j])
])

Plots.savefig(joinpath(homedir(),"Documents","PubChem_odorant_classes_correlation.svg"))

# A helper function to diagonalize a matrix (if needed)
function diagonalize_matrix(matrix)
    n_rows, n_cols = size(matrix)
    used_columns = falses(n_cols)  # Track which columns have been used
    new_order = Int[]              # New order of columns

    for i in 1:n_rows
        row = copy(matrix[i, :])
        for attempt in 1:n_cols
            best_fit_col = argmax(row)
            if !used_columns[best_fit_col]
                push!(new_order, best_fit_col)
                used_columns[best_fit_col] = true
                break
            else
                row[best_fit_col] = -Inf
            end
        end
    end

    # Append any columns not yet used
    for j in 1:n_cols
        if !used_columns[j]
            push!(new_order, j)
        end
    end

    return matrix[:, new_order]
end

