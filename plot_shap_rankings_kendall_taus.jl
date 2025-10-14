using JLD,Plots;

shap_ranks_1=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/no_amines_cact_shapley_rankings_april2024.jld"),"shap_ranks1");
shap_ranks_2=load(joinpath(homedir(),"Documents/Nikola-Vladimir/new_code_files/no_amines_cact_shapley_rankings_april2024.jld"),"shap_ranks2");

n_features=132;
n_seeds=100;
ranking_matrices_1=fill(0.0,n_features,n_seeds);
ranking_matrices_2=fill(0.0,n_features,n_seeds);
for i in 1:n_seeds
    for j in 1:n_features
    ranking_matrices_1[j,i]=findall(x->x==j, shap_ranks_1[i][2:end])[1]
    ranking_matrices_2[j,i]=findall(x->x==j, shap_ranks_2[i][2:end])[1]
  #  ranking_matrices_2[:,i]=shap_ranks_2[i][2:end]
    end
end

#rankings_matrices[i][k,j]=findall(x->x==k,rankings[i][j][2])[1]

using PyCall;

scipy= pyimport("scipy");

kendall_tau=scipy.stats.kendalltau;
kendall_tau(ranking_matrices_1[:,1],ranking_matrices_1[:,3])

kendall_tau_matrix_real=fill(0.0,100,100);
for i in 1:size(kendall_tau_matrix_real,1)
    for j in 1:size(kendall_tau_matrix_real,2)
        kendall_tau_matrix_real[i,j]=kendall_tau(ranking_matrices_1[:,i],ranking_matrices_2[:,j])[1]
    end
end

A=kendall_tau_matrix_real;# 83L and 83L

B=kendall_tau_matrix_real;# 83L and 83R

C=kendall_tau_matrix_real;# 83L and 84L


kendall_tau_matrix_random_test=fill(0.0,100,100);
for i in 1:size(kendall_tau_matrix_random_test,1)
    for j in 1:size(kendall_tau_matrix_random_test,2)
        kendall_tau_matrix_random_test[i,j]=kendall_tau(rankings_matrices[1][:,i],shuffle(rankings_matrices[1][:,i]))[1]
    end
end



StatsPlots.violin(["omp83L"],vec(A),color=:red,grid=false,ylabel="Kendall Tau Correspondence",legend=false)
StatsPlots.violin!(["omp83L & omp83R"],vec(B),color=:blue,grid=false,ylabel="Kendall Tau Correspondence",legend=false)
StatsPlots.violin!(["omp84L & omp85L"],vec(C),color=:green,grid=false,ylabel="Kendall Tau Correspondence",legend=false)
Plots.savefig(joinpath(homedir(),"Documents/SuppFig8_panelA.svg"));

StatsPlots.violin!(["Random"],vec(kendall_tau_matrix_random_test),color=:royalblue,grid=false,ylabel="Kendall Tau Correspondence",legend=false)


feature_names=descr_names[Int64.(rankings_matrices[1][:,1][1:15])];
data_shap_083=hcat(shap_matrices[1][:,1][Int64.(rankings_matrices[1][:,1][1:15])],feature_names);
data_shap_083=DataFrame(data_shap_083,[:shapley_value,:feature_name]);
data_plot_sort_083 = sort(data_shap_083, order(:shapley_value),rev=true);#, rev = true))
p1=Plots.bar(data_plot_sort_083.feature_name,data_plot_sort_083.shapley_value,legend=false,yticks=:all,orientation=:horizontal,grid=false,title="Mean Absolute Shapley Value",
                                    xlabel="Shapley effect units",c=:royalblue)
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","Mean_absolute_shap_value.eps"));

#


using Plots;

# Plotting the heatmap
heatmap(1:100, 1:140, ranking_matrices_1,
        xlabel="Ranking Number",
        ylabel="Feature Index",
        title="Heatmap of Feature Rankings",
        color=:blues,
        colorbar_title="Rank Position",
        yflip=true)