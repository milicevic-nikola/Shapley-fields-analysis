using PyCall,JLD, LinearAlgebra, Statistics,Random, MLBase, DataFrames, Random;

#run(`$(PyCall.python) -m pip install shap`;);
#shap=pyimport("shap");
skl_ensemble = pyimport("sklearn.ensemble");
np=pyimport("numpy");
metrics=pyimport("sklearn.metrics");
precision_recall_curve=metrics.precision_recall_curve;
roc_auc_score=metrics.roc_auc_score
f1_score=metrics.f1_score;
auc=metrics.auc;

#generate random binary matrix
dowker_matrix = rand(10, 60) .< 0.3;

#assign label 1 to columns with more than 4 ones
labels=sum(dowker_matrix,dims=1).>4;

#calculate percentage of the positive class_1_idxs
baseline=sum(labels)/60;

#function for cross validating model in leave-one-out fashion
#computes accuracy, auprc and auroc metrics, returns as a 3-tuple in that order
function random_forest_model(features::Matrix{Float64},lab::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    #predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    n=length(lab)
    binary_preds=fill(0.0,n)
    pred_probs=fill(0.0,n)
    X=features'
    y=lab
    for j in 1:n
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
        Xtrain=X[1:end.!=j,:]
        ytrain=y[1:end.!=j]
        predictor.fit(Xtrain,ytrain)
        Xtest=X[j,:]
        binary_preds[j] = predictor.predict([Xtest])[1]
        pred_proba=predictor.predict_proba([Xtest])
        if size(pred_proba)==(1,1)
            pred_probs[j]=0.0
        else
            pred_probs[j]=pred_proba[1,2]
        end
    end
    precision, recall, _ = precision_recall_curve(lab, pred_probs)
    pr_auc=auc(recall,precision)
    roc_auc = roc_auc_score(lab, pred_probs)
    return sum(binary_preds.==lab)/(n),pr_auc,roc_auc 
end

#compute leave one out cross validated auprc metric
auprc=random_forest_model(Float64.(dowker_matrix),vec(labels),128,0.5,0)[2];
println(auprc)

#let us shuffle the dowker matrix a few times and see what happens
n_shuffles=10;
shuffled_auprc=fill(0.0,n_shuffles);
for i in 1:n_shuffles
    shuffled_auprc[i]=random_forest_model(Float64.(shuffle(dowker_matrix)),vec(labels),128,0.5,0)[2]
end
#print and plot to compare with the real auprc and the baseline
println(shuffled_auprc)
Plots.boxplot(["shuffled"],shuffled_auprc,color=:blue,grid=false,ticksfontsize=14,label="");
Plots.scatter!([0.5],[baseline],color=:green,label="baseline",markershape=:diamond,markersize=14);
Plots.scatter!([0.5],[auprc],color=:red,label=:"real",markershape=:diamond,markersize=14)

#Optional, let us shuffle the labels a few times and see what happens
n_shuffles=10;
shuffled_resp_auprc=fill(0.0,n_shuffles);
for i in 1:n_shuffles
    shuffled_resp_auprc[i]=random_forest_model(Float64.(dowker_matrix),vec(shuffle(labels)),128,0.5,0)[2]
end
println(shuffled_resp_auprc)

