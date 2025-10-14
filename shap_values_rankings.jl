using Distributed#,JLD;

addprocs(40)

@everywhere include("Response_prediction_pipeline/Useful_functions.jl");
@everywhere include("Response_prediction_pipeline/python_shap_functions_new.jl")
@everywhere using Random;
#@everywhere Ndatasets=3;

#load the necessary data
#@everywhere using JLD

@everywhere odor_descr_subset_names=load(joinpath(homedir(),"segmented_83_and_85_data_schmuker_descrs_shuffled_for_prediction_no_noise.jld"),"descr_names");

@everywhere odor_descr_subset_names=load(joinpath(homedir(),"segmented_83_and_85_data_schmuker_descrs_shuffled_for_prediction_no_noise.jld"),"descr_names");

@everywhere X=load(joinpath(homedir(),"segmented_83_and_85_data_schmuker_descrs_shuffled_for_prediction_no_noise.jld"),"Data");

@everywhere binary_responses=load(joinpath(homedir(),"segmented_83_and_85_data_schmuker_descrs_shuffled_for_prediction_no_noise.jld"),"bin_resp");

@everywhere Random.seed!(0);
@everywhere ntrees=128;
@everywhere sample_perc=0.7;
@everywhere random_seed=rand(1:10000,100);

@everywhere shap_rankings=Array{Any,1}(undef,length(binary_responses));

@everywhere  f(number)=chem_descr_shap_values_leave_one_out_per_glom(X[1],binary_responses[1],ntrees,sample_perc,random_seed[number]);
#@everywhere  f(number)=compute_shap_values_signed_positive_class(X[1][number,:],binary_responses[1][number,:],ntrees,sample_perc,random_seed);
#@everywhere  f(number)=compute_shap_values(X[1][number,:],binary_responses[1][number,:],ntrees,sample_perc,random_seed);
runs=1:length(random_seed);
shap_rankings[1]=pmap(f,runs);

@everywhere  h(number)=chem_descr_shap_values_leave_one_out_per_glom(X[2],binary_responses[2],ntrees,sample_perc,random_seed[number]);
#@everywhere  h(number)=compute_shap_values_signed_positive_class(X[2][number,:],binary_responses[2][number,:],ntrees,sample_perc,random_seed);
#@everywhere  h(number)=compute_shap_values(X[2][number,:],binary_responses[2][number,:],ntrees,sample_perc,random_seed);
runs=1:length(random_seed);
shap_rankings[2]=pmap(h,runs);



save(joinpath(homedir(),"new_chemotopy_results/83_and_85_schmuker_descrs_shap_rankings.jld"),"shap_rankings",shap_rankings);
#save(joinpath(homedir(),"new_chemotopy_results/83_and_85_schmuker_descrs_on_bulb.jld"),"shap_per_glom_matrix",shap_per_glom_matrix);


using JLD,Plots;
rankings=load(joinpath(homedir(),"C:/Users/nikol/Dropbox (Personal)/Nikola-Vladimir/83_and_85_schmuker_descrs_shap_rankings.jld"),"shap_rankings");

rankings_matrices=Array{Any,1}(undef,2);
for i in 1:length(rankings_matrices)
    rankings_matrices[i]=fill(0.0,165,100)
    for j in 1:size(rankings_matrices[i],2)
        #rankings_matrices[i][:,j]=rankings[i][j][2]
        for k in 1:size(rankings_matrices[i],1)
            rankings_matrices[i][k,j]=findall(x->x==k,rankings[i][j][2])[1]
        end
    end
end
rankings_mean=Array{Any,1}(undef,2);
rankings_min=Array{Any,1}(undef,2);
rankings_max=Array{Any,1}(undef,2);
for i in 1:length(rankings_mean)
    rankings_mean[i]=vec(mean(rankings_matrices[i],dims=2))
    rankings_max[i]=vec(findmax(rankings_matrices[i],dims=2)[1])
    rankings_min[i]=vec(findmin(rankings_matrices[i],dims=2)[1])
end

StatsPlots.plot(rankings_mean[1], ribbon=(rankings_min[1],rankings_max[1]))
StatsPlots.plot(rankings_mean[2], ribbon=(rankings_min[2],rankings_max[2]))

pyplot()
Plots.plot(1:165,rankings_mean[1],label="omp83",c=:red,linewidth=2,grid=false,xlabel="feature",ylabel="Average Shapley Ranking")
Plots.plot!(1:165,rankings_mean[2],label="omp85",c=:royalblue,linewith=2)
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","average_shapley_ranks.eps"));


Plots.plot(1:165,rankings_matrices[1],label="omp83",c=:red,legend=false)
Plots.plot!(1:165,rankings_matrices[2],label="omp85",c=:blue,legend=false)
Plots.histogram(abs.(vec(rankings_matrices[1]).-vec(rankings_matrices[2])),legend=false,grid=false,xlabel="change in rank", ylabel="count")
Plots.annotate!(40, 750, "mean change in rank="*string(round(mean_rank_diff,digits=2)))
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","shapley_rank_hist.eps"));

Plots.histogram(abs.(rankings_matrices[1].-rankings_matrices[2]),legend=false)


shap_matrices=Array{Any,1}(undef,2);
for i in 1:length(shap_matrices)
    shap_matrices[i]=fill(0.0,165,100)
    for j in 1:size(shap_matrices[i],2)
        shap_matrices[i][:,j]=rankings[i][j][1]
    end
end

shap_mean=Array{Any,1}(undef,2);
for i in 1:length(shap_mean)
    shap_mean[i]=vec(mean(shap_matrices[i],dims=2))
end

Plots.plot(1:165,shap_mean[1],label="omp83",c=:red,linewidth=2,grid=false,xlabel="feature",ylabel="Shapley Global Value")
Plots.plot!(1:165,shap_mean[2],label="omp85",c=:royalblue,linewidth=2,grid=false,xlabel="feature",ylabel="Shapley Global Value")
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","Shapley_mean_values.eps"));

Plots.plot(1:165,shap_mean[1],label="omp83",c=:red,legend=false)
Plots.plot!(1:165,shap_mean[2],label="omp85",c=:royalblue,legend=false)

using StatsBase

hist=fit(Histogram, vec(abs.(rankings_matrices[1].-rankings_matrices[2])))
hist_norm=StatsBase.normalize(hist; mode=:pdf)
StatsBase.norm(hist_norm)
mean_rank_diff=mean(vec(abs.(rankings_matrices[1].-rankings_matrices[2])))

using Distributions, StatsPlots

pois=Distributions.Poisson(mean(vec(abs.(rankings_matrices[1].-rankings_matrices[2]))))
pois=Distributions.Poisson(2)

StatsPlots.plot(pois)


#using Conda;
#Conda.add("shap");
#pyimport_conda("shap",PKG)
#using PyCall;
#run(`$(PyCall.python) -m pip install shap`;)
#run(`$(PyCall.python) -m pip install xgboost`;)
using PyPlot;

predictor=skl_ensemble.RandomForestClassifier(n_estimators=256,max_samples=0.7,class_weight="balanced_subsample",random_state=0);
predictor.fit(model_features[1][1,1:end.!=1],binary_responses[1][1,1:end.!=1])

explainer=shap.TreeExplainer(predictor)
explainer.expected_value

explain=model_features[1][1,1]

shap_values=explainer.shap_values(explain)

colors=Array{String,1}(undef,length(shap_values[2][1:15]))
for i in 1:length(colors)
    if shap_values[2][i]<0
        colors[i]="red"
    else
        colors[i]="green"
    end
end
colors=vcat(["royalblue"],colors,["royalblue"])

colors_lower=Array{String,1}(undef,15)
for i in 1:length(colors_lower)
    colors_lower[i]="white"
end
colors_lower=vcat("royalblue",colors_lower,"royalblue")

descr_names=vcat(["base"],descr_names,["predicted"])
feature_names=descr_names;
for i in 1:length(feature_names)
    feature_names[i]=feature_names[i]*"="*string(model_features[1][1,1][i])
end
feature_names=vcat(["expected"], feature_names,"predicted")
shap_vals_of_interest=vcat([explainer.expected_value[2]],shap_values_pos,[predictor.predict_proba([explain])[2]]);

sums=Array{Float64,1}(undef,15)
shifted_sums=Array{Float64,1}(undef,15)
#sums[1]=explainer.expected_value[2]
shifted_sums[1]=explainer.expected_value[2]
for i in 1:length(sums)
    sums[i]=explainer.expected_value[2]+sum(shap_values_pos[1:i])
end

for i in 2:length(shifted_sums)
    shifted_sums[i]=sums[i-1]
end

upper_vals=max.(sums,shifted_sums)
lower_vals=min.(sums,shifted_sums)
upper_vals=vcat([explainer.expected_value[2]],upper_vals,[predictor.predict_proba([explain])[2]])
lower_vals=vcat([explainer.expected_value[2]],lower_vals,[predictor.predict_proba([explain])[2]])
mid=0.5*(lower_vals.+upper_vals)
mid[1]=0.5*mid[1]
mid[17]=0.5*mid[17]
fig,ax = plt.subplots()
#figsize=(10,10)
fig.set_size_inches(10.5, 8.5)
ax.barh(y=feature_names,width=upper_vals,color=colors)
ax.barh(y=feature_names,width=lower_vals,color=colors_lower)
ax.invert_yaxis()
for i in 1:length(upper_vals)
    plt.text(mid[i],i-0.8, string(round(shap_vals_of_interest[i],digits=2)))
end
display(fig)
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/meeting_10_05","local_shapls.eps"));
close(fig)
shap_values_pos=shap_values[2][1:15]


using Conda;
Conda.add("scipy")
Conda.add("shap");
#pyimport_conda("shap",PKG)
using PyCall;
#run(`$(PyCall.python) -m pip install shap`;)
#run(`$(PyCall.python) -m pip install xgboost`;)
run(`$(PyCall.python) -m pip install scipy.stats`;)

scipy= pyimport("scipy");

kendall_tau=scipy.stats.kendalltau


kendall_tau_matrix_real=fill(0.0,100,100)
kendall_tau(rankings_matrices[1][:,1],rankings_matrices[2][:,1])
for i in 1:size(kendall_tau_matrix_real,1)
    for j in 1:size(kendall_tau_matrix_real,2)
        kendall_tau_matrix_real[i,j]=kendall_tau(rankings_matrices[1][:,i],rankings_matrices[2][:,j])[1]
    end
end
findmin(kendall_tau_matrix_real)
findmax(kendall_tau_matrix_real)

kendall_tau_matrix_random_test=fill(0.0,100,100)
for i in 1:size(kendall_tau_matrix_random_test,1)
    for j in 1:size(kendall_tau_matrix_random_test,2)
        kendall_tau_matrix_random_test[i,j]=kendall_tau(rankings_matrices[1][:,i],shuffle(rankings_matrices[1][:,i]))[1]
    end
end
StatsPlots.violin(["Real"],vec(kendall_tau_matrix_real),color=:red,grid=false,ylabel="Kendall Tau Correspondence",legend=false)
StatsPlots.violin!(["Random"],vec(kendall_tau_matrix_random_test),color=:royalblue,grid=false,ylabel="Kendall Tau Correspondence",legend=false)
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","Rankings_kendall_tau.eps"));


feature_names=descr_names[Int64.(rankings_matrices[1][:,1][1:15])];
data_shap_083=hcat(shap_matrices[1][:,1][Int64.(rankings_matrices[1][:,1][1:15])],feature_names);
data_shap_083=DataFrame(data_shap_083,[:shapley_value,:feature_name]);
data_plot_sort_083 = sort(data_shap_083, order(:shapley_value),rev=true);#, rev = true))
p1=Plots.bar(data_plot_sort_083.feature_name,data_plot_sort_083.shapley_value,legend=false,yticks=:all,orientation=:horizontal,grid=false,title="Mean Absolute Shapley Value",
                                    xlabel="Shapley effect units",c=:royalblue)
PyPlot.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_manuscript_figures/Shapley_Figure","Mean_absolute_shap_value.eps"));

#
