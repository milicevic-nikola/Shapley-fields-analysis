using PyCall,JLD, LinearAlgebra, Statistics,Random, MLBase, DataFrames, Random;

#run(`$(PyCall.python) -m pip install shap`;);
shap=pyimport("shap");
skl_ensemble = pyimport("sklearn.ensemble");
np=pyimport("numpy");
metrics=pyimport("sklearn.metrics");
precision_recall_curve=metrics.precision_recall_curve;
roc_auc_score=metrics.roc_auc_score
f1_score=metrics.f1_score;
auc=metrics.auc;

function find_shap_ordering(Data::Array{Float64,2},descr_names::Vector{Symbol},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64)
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,max_features=feature_perc,class_weight="balanced_subsample",random_state=random_seed);
    predictor.fit(Data,vec(bin_resp));
    explainer=shap.TreeExplainer(predictor);
    explain=Data;
    shap_values=explainer.shap_values(explain);
    shap_vals=shap_values[2];
    data_shap=mean(abs.(shap_vals),dims=1);
    data_shap_idx=findmax(data_shap[1,1:(length(descr_names)-3)],1:(length(descr_names)-3))[1];
   # data_shap_idx=findmax(data_shap[1,1:(length(descr_names)-3)],1:(length(descr_names)-3))[1];
    #return data_shap,vcat(random_seed,data_shap_idx)
    return vcat(random_seed,data_shap_idx)
end

function find_shap_ordering_with_shuffling(Data::Array{Float64,2},integer_identities::Array{Int64,1},descr_names::Vector{Symbol},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64)
    new_data=[Data integer_identities]
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,max_features=feature_perc,class_weight="balanced_subsample",random_state=random_seed);
    predictor.fit(new_data,vec(bin_resp));
    explainer=shap.TreeExplainer(predictor);
    explain=new_data;
    shap_values=explainer.shap_values(explain);
    shap_vals=shap_values[2];
    data_shap=mean(abs.(shap_vals),dims=1);
    data_shap_idx=findmax(data_shap[1,1:(length(descr_names)-3)],1:(length(descr_names)-3))[1];
    # Find the index with the maximum SHAP value, including all descriptors
   # data_shap_idx_all = findmax(data_shap[1, 1:end],1:size(data_shap,2))[1];
    #return (vcat(random_seed,data_shap_idx), vcat(random_seed, data_shap_idx_all))
    return vcat(random_seed,data_shap_idx)
end

function find_shap_ordering_with_shuffling_v2(Data::Array{Float64,2},integer_identities::Array{Int64,1},shuffled_idx::Array{Int64,1},descr_names::Vector{Symbol},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64)
    num_rows = size(Data, 1)
    k = div(num_rows, 59)
    #k = div(num_rows, 52)
    shuffle_indices=shuffled_idx
    for block in 1:59
    #for block in 1:52
        start_index = (block - 1) * k + 1
        end_index = block * k
        # Apply shuffled indices to the last two columns in each block
        Data[start_index:end_index, end-1:end] = Data[start_index:end_index, end-1:end][shuffle_indices, :]
    end
    new_data=[Data integer_identities]
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,max_features=feature_perc,class_weight="balanced_subsample",random_state=random_seed);
    predictor.fit(new_data,vec(bin_resp));
    explainer=shap.TreeExplainer(predictor);
    explain=new_data;
    shap_values=explainer.shap_values(explain);
    shap_vals=shap_values[2];
    data_shap=mean(abs.(shap_vals),dims=1);
    data_shap_idx=findmax(data_shap[1,1:(length(descr_names)-3)],1:(length(descr_names)-3))[1];
    # Find the index with the maximum SHAP value, including all descriptors
   # data_shap_idx_all = findmax(data_shap[1, 1:end],1:size(data_shap,2))[1];
    #return (vcat(random_seed,data_shap_idx), vcat(random_seed, data_shap_idx_all))
    return vcat(random_seed,data_shap_idx)
end

function find_shap_ordering_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    n,m=size(bin_resp)
    l=length(Data[1,1])
    #descr_names=descr_names[1:l-3]
    shap_values_matrix=fill(0.0,n,l-3)
    for i in 1:n
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
        Xtrain=fill(0.0,m,l-3)
        for j in 1:size(Xtrain,1)
            Xtrain[j,:]=Data[i,j][1:l-3]
        end
        ytrain=bin_resp[i,:]
        predictor.fit(Xtrain,ytrain)
        explainer=shap.TreeExplainer(predictor)
        explain=Xtrain
        shap_values=explainer.shap_values(explain)
        shap_vals=shap_values[2]
        if shap_vals==0.0
            shap_values_matrix[i,:].=0.0
        else
            shap_values_matrix[i,:]=mean(abs.(shap_vals),dims=1);
        end
    end
    data_shap=mean(shap_values_matrix,dims=1)
    data_shap_idx=findmax(vec(data_shap),1:l-3)[1];
    return data_shap,vcat(random_seed,data_shap_idx)
    #return vcat(random_seed,data_shap_idx)
end


function chem_descr_and_pos_classification_old(Data::Array{Float64,2},bin_resp::BitMatrix,nfolds::Int64,ntrees::Int64,
                                            sample_perc::Float64,feature_perc::Float64,random_seed::Int64)
    labels=vec(bin_resp);
    #set random seed to make sure we always pick same folds
    Random.seed!(1337)
    folds=collect(Kfold(length(labels), nfolds));
    pred_acc=Array{Float64,1}(undef,nfolds)
    pr_auc=Array{Float64,1}(undef,nfolds)
    m,n=size(bin_resp)
    binary_probs=fill(0.0,m,n)
    pred_probs=fill(0.0,m,n)
    y_real=[];
    y_proba=[];
    pr_aucs=Array{Float64,1}(undef,length(folds))
    roc_aucs=Array{Float64,1}(undef,length(folds))
    for k in 1:length(folds)
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,max_features=feature_perc,class_weight="balanced_subsample",random_state=random_seed);
        train_index, test_index=folds[k], setdiff(1:length(labels),folds[k])
        Xtrain=Data[train_index,:]
        ytrain=labels[train_index]
        Xtest=Data[test_index,:]
        ytest=labels[test_index]
        append!(y_real,ytest)
        predictor.fit(Xtrain,ytrain)
        preds = predictor.predict(Xtest)
        pred_proba=predictor.predict_proba(Xtest)[:,2]
        append!(y_proba,pred_proba)
        pred_acc[k]=sum(preds.==ytest)/length(ytest)
        precision,recall, _=precision_recall_curve(ytest,pred_proba)
        pr_aucs[k]=auc(recall,precision)
        roc_aucs[k]=roc_auc_score(ytest,pred_proba)
        #println("$k")
    end
    precision, recall, _ = precision_recall_curve(y_real, y_proba);
    pr_auc=auc(recall,precision);
    roc_auc = roc_auc_score(y_real, y_proba);
    return mean(pred_acc), pr_auc, mean(pr_aucs), roc_auc, mean(roc_aucs)
end

function chemotopy_ordered_metrics_old(Data::Array{Float64,2},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
                                    nfolds::Int64,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    n_descrs=size(Data,2)
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-2,n_descrs-1,n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cactus
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts42
    pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_and_pos_classification_old(Data[:,idxs],bin_resp,nfolds,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end

function chemotopy_ordered_metrics_old_with_shuffling(Data::Array{Float64,2},integer_identities::Array{Int64,1},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
    nfolds::Int64,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    new_Data=[Data integer_identities]
    n_descrs=size(new_Data,2)
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-2,n_descrs-1,n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cactus
#idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom
#idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
#idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts42
    pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_and_pos_classification_old(new_Data[:,idxs],bin_resp,nfolds,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end

function chemotopy_ordered_metrics_old_with_shuffling_v2(Data::Array{Float64,2},integer_identities::Array{Int64,1},shuffled_idx::Array{Int64,1},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
    nfolds::Int64,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    num_rows = size(Data, 1)
    k = div(num_rows, 59)
    #k = div(num_rows, 52)
    shuffle_indices=shuffled_idx
    for block in 1:59
    #for block in 1:52
        start_index = (block - 1) * k + 1
        end_index = block * k
        # Apply shuffled indices to the last two columns in each block
        Data[start_index:end_index, end-1:end] = Data[start_index:end_index, end-1:end][shuffle_indices, :]
    end
    new_Data=[Data integer_identities]
    n_descrs=size(new_Data,2)
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-2,n_descrs-1,n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
#idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cactus
#idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom
#idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
#idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts42
    pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_and_pos_classification_old(new_Data[:,idxs],bin_resp,nfolds,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end

function chemotopy_ordered_metrics_old_1(Data::Array{Float64,2},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
                                    nfolds::Int64,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    n_descrs=size(Data,2)
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cactus
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts42
    pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_and_pos_classification_old(Data[:,idxs],bin_resp,nfolds,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end

function chemotopy_ordered_metrics_old_2(Data::Array{Float64,2},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
                                    nfolds::Int64,ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    n_descrs=size(Data,2)
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-1,n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs])#general formula
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cactus
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts42
    pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_and_pos_classification_old(Data[:,idxs],bin_resp,nfolds,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end


function chem_descr_and_pos_classification_leave_one_out_old(Data::Array{Any,2},bin_resp::BitMatrix,
                                            ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64)
    m,n=size(bin_resp)
    binary_preds=fill(0.0,m,n)
    pred_probs=fill(0.0,m,n)
    pr_aucs=Array{Float64,1}(undef,n)
    roc_aucs=Array{Float64,1}(undef,n)
    for i in 1:n
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,max_features=feature_perc,class_weight="balanced_subsample",random_state=random_seed);
        Xtrain=vec(Data[:,1:end.!=i])
        ytrain=vec(bin_resp[:,1:end.!=i])
        predictor.fit(Xtrain,ytrain)
        Xtest=Data[:,i]
        binary_preds[:,i]=predictor.predict(Xtest)
        pred_probs[:,i]=predictor.predict_proba(Xtest)[:,2]
        precisions, recalls, _= precision_recall_curve(bin_resp[:,i], pred_probs[:,i]);
        pr_aucs[i]=auc(recalls,precisions)
        roc_aucs[i]=roc_auc_score(bin_resp[:,i], pred_probs[:,i])
       # println("$i")
    end
    precision, recall, _ = precision_recall_curve(vec(bin_resp), vec(pred_probs));
    pr_auc=auc(recall,precision);
    roc_auc=roc_auc_score(vec(bin_resp), vec(pred_probs));
    return sum(binary_preds.==bin_resp)/(m*n),pr_auc,mean(pr_aucs),roc_auc,mean(roc_aucs)
end

function chemotopy_ordered_metrics_leave_one_out_old(Data::Array{Any,2},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
                                    ntrees::Int64,sample_perc::Float64,feature_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    n_descrs=length(Data[1,1])
    idxs=vcat(shap_ord_idx[1:descr_numbers],[n_descrs-2,n_descrs-1,n_descrs])#general
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[73,74,75])
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[144,145,146])
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[21,22,23])#pca_cact
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[257,258,259])#pom21 has 256 descrs
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[133,134,135])#cact has 140/132 descrs
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[156,157,158])
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[166,167,168])
    #idxs=vcat(shap_ord_idx[1:descr_numbers],[23,24,25])#smarts has 24/22 decrs
    m,n=size(Data)
    new_Data=Array{Any,2}(undef,size(Data))
    for i in 1:m
        for j in 1:n
            new_Data[i,j]=Data[i,j][idxs]
        end
    end
    pred_acc,pr_auc, mean_pr_auc, roc_auc, mean_roc_auc=chem_descr_and_pos_classification_leave_one_out_old(new_Data,bin_resp,ntrees,sample_perc,feature_perc,random_seed)
    return pred_acc,pr_auc, mean_pr_auc, roc_auc, mean_roc_auc
end


function chem_descr_classification_leave_one_out_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    m,n=size(bin_resp)
    binary_preds=fill(0.0,m,n)
    pred_probs=fill(0.0,m,n)
    for i in 1:m
        X=Data[i,:]
        y=bin_resp[i,:]
        for j in 1:n
            predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
            Xtrain=X[1:end.!=j]
            ytrain=y[1:end.!=j]
            predictor.fit(Xtrain,ytrain)
            Xtest=X[j]
            binary_preds[i,j] = predictor.predict([Xtest])[1]
            pred_proba=predictor.predict_proba([Xtest])
            if size(pred_proba)==(1,1)
                pred_probs[i,j]=0.0
            else
                pred_probs[i,j]=pred_proba[1,2]
            end
        end
        #println("$i")
    end
    precision, recall, _ = precision_recall_curve(vec(bin_resp), vec(pred_probs));
    pr_auc=auc(recall,precision)
    roc_auc = roc_auc_score(vec(bin_resp), vec(pred_probs));
    pr_aucs=Array{Float64,1}(undef,size(bin_resp,1))
    roc_aucs=Array{Float64,1}(undef,size(bin_resp,1))
    for i in 1:size(bin_resp,1)
        precision, recall, _ = precision_recall_curve(bin_resp[i,:], pred_probs[i,:]);
        pr_aucs[i]=auc(recall,precision)
        roc_aucs[i]=roc_auc_score(bin_resp[i,:], pred_probs[i,:])
    end
    return sum(binary_preds.==bin_resp)/(m*n),pr_auc, mean(pr_aucs),roc_auc, mean(roc_aucs)
end


function chemotopy_ordered_metrics_leave_one_out_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,shap_ord_idx::Vector{Int64},
                                    ntrees::Int64,sample_perc::Float64,random_seed::Int64,descr_numbers::Int64)
    idxs=vcat(shap_ord_idx[1:descr_numbers])
    m,n=size(Data)
    new_Data=Array{Any,2}(undef,size(Data))
    for i in 1:m
        for j in 1:n
            new_Data[i,j]=Data[i,j][idxs]
        end
    end
    pred_acc,pr_auc, mean_pr_auc,roc_auc,mean_roc_auc=chem_descr_classification_leave_one_out_per_glom(new_Data,bin_resp,ntrees,sample_perc,random_seed)
    return pred_acc,pr_auc,mean_pr_auc,roc_auc,mean_roc_auc
end

function peer_prediction(resp_matr::Array{Float64,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    #predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    m,n=size(bin_resp)
    binary_preds=fill(0.0,m,n)
    pred_probs=fill(0.0,m,n)
    for i in 1:m
        X=resp_matr[1:end.!=i,:]'
        y=bin_resp[i,:]
        for j in 1:n
            predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
            Xtrain=X[1:end.!=j,:]
            ytrain=y[1:end.!=j]
            Xtest=X[j,:]
            predictor.fit(Xtrain, ytrain)
            pred_proba = predictor.predict_proba([Xtest])
            binary_preds[i,j]=predictor.predict([Xtest])[1]
            #, $j")
            if size(pred_proba)==(1,1)
                pred_probs[i,j]=0.0
            else
                pred_probs[i,j]=pred_proba[1,2]#,[true,false])[1]
            end
        end
      #  println("$i")
    end
    precision, recall, _ = precision_recall_curve(vec(bin_resp), vec(pred_probs));
    pr_auc=auc(recall,precision)
    roc_auc = roc_auc_score(vec(bin_resp), vec(pred_probs));
    pr_aucs=Array{Float64,1}(undef,size(bin_resp,1))
    roc_aucs=Array{Float64,1}(undef,size(bin_resp,1))
    for i in 1:size(bin_resp,1)
        precision, recall, _ = precision_recall_curve(bin_resp[i,:], pred_probs[i,:]);
        pr_aucs[i]=auc(recall,precision)
        roc_aucs[i]=roc_auc_score(bin_resp[i,:], pred_probs[i,:])
    end
    return sum(binary_preds.==bin_resp)/(m*n), pr_auc, mean(pr_aucs),roc_auc, mean(roc_aucs)
end


function peer_prediction_fixed(resp_matr::Array{Float64,2},bin_resp::BitMatrix,gloms_to_consider::Vector{Vector{Int64}},ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    #predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    m,n=size(bin_resp)
    binary_preds=fill(0.0,m,n)
    pred_probs=fill(0.0,m,n)
    for i in 1:m
        X=resp_matr[setdiff(1:m,gloms_to_consider[i]),:]'
        y=bin_resp[i,:]
        for j in 1:n
            predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
            Xtrain=X[1:end.!=j,:]
            ytrain=y[1:end.!=j]
            Xtest=X[j,:]
            predictor.fit(Xtrain,ytrain)
            pred_proba=predictor.predict_proba([Xtest])
            binary_preds[i,j]=predictor.predict([Xtest])[1]
            #println("$i, $j")
            if size(pred_proba)==(1,1)
                pred_probs[i,j]=0.0
            else
                pred_probs[i,j]=pred_proba[1,2]#,[true,false])[1]
            end
        end
        println("$i")
    end
    precision, recall, _ = precision_recall_curve(vec(bin_resp), vec(pred_probs));
    peer_pred_pr_auc=auc(recall,precision)
    peer_pred_acc=sum(binary_preds.==bin_resp)/(m*n)
    return peer_pred_acc,peer_pred_pr_auc
end

#function compute_peer_prediction(resp_matr::Array{Float64},labels::BitVector,)

function compute_shap_values(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    n=length(Data[1])
    m=length(labels)
    shap_values=fill(0.0,m,n)
    for i in 1:length(labels)
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
        Xtrain=Data[1:end.!=i]
        ytrain=labels[1:end.!=i]
        predictor.fit(Xtrain,ytrain)
        explainer=shap.TreeExplainer(predictor)
        explain=Data[i]
        shap_val=explainer.shap_values(explain)
        if length(shap_val)==n
            shap_values[i,:]=fill(0.0,n)
        else
            shap_values[i,:]=abs.(shap_val[2])
        end
        #shap_values=Array{Float64,1}(undef,length(shap_val[1]))
        #shap_vals=Array{Float64,1}(undef,length(explain))
        #if predictor.predict_proba([explain])[1,1]>predictor.predict_proba([explain])[1,2]
    #        shap_vals=shap_val[1]
    #    else
    #        shap_vals=shap_val[2]
    #    end
    #    shap_values[i,:]=shap_vals
        #println("$i")
    end
    return vec(mean(abs.(shap_values),dims=1))
end

function chem_descr_shap_values_leave_one_out_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    m,n=size(Data)
    shap_vector_length=length(Data[1,1])-3
    shap_values_matrix=fill(0.0,m,shap_vector_length)
    for i in 1:m
        shap_values_matrix[i,:]=compute_shap_values(Data[i,:],bin_resp[i,:],ntrees,sample_perc,random_seed)[1:shap_vector_length]
    end
    shap_vals=vec(mean(shap_values_matrix,dims=1))
    return shap_vals,findmax(shap_vals,1:length(shap_vals))[1]
end

function compute_shap_values_signed(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    n=length(Data[1])
    m=length(labels)
    shap_values=fill(0.0,m,n)
    for i in 1:length(labels)
        Xtrain=Data[1:end.!=i]
        ytrain=labels[1:end.!=i]
        predictor.fit(Xtrain,ytrain)
        explainer=shap.TreeExplainer(predictor)
        explain=Data[i]
        shap_val=explainer.shap_values(explain)
        if length(shap_val)==n
            shap_values[i,:]=fill(0.0,n)
        else
            if predictor.predict_proba([explain])[1,1]>predictor.predict_proba([explain])[1,2]
                shap_values[i,:]=shap_val[1]
            else
                shap_values[i,:]=shap_val[2]
            end
        end
        #shap_values=Array{Float64,1}(undef,length(shap_val[1]))
        #shap_vals=Array{Float64,1}(undef,length(explain))
        #if predictor.predict_proba([explain])[1,1]>predictor.predict_proba([explain])[1,2]
    #        shap_vals=shap_val[1]
    #    else
    #        shap_vals=shap_val[2]
    #    end
    #    shap_values[i,:]=shap_vals
        #println("$i")
    end
    return vec(mean(shap_values,dims=1))
end

function compute_shap_values_signed_positive_class(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    n=length(Data[1])
    m=length(labels)
    shap_values=fill(0.0,m,n)
    for i in 1:length(labels)
        predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
        Xtrain=Data[1:end.!=i]
        ytrain=labels[1:end.!=i]
        predictor.fit(Xtrain,ytrain)
        explainer=shap.TreeExplainer(predictor)
        explain=Data[i]
        shap_val=explainer.shap_values(explain)
        if length(shap_val)==n
            shap_values[i,:]=fill(0.0,n)
        else
            shap_values[i,:]=shap_val[2]
        end
        #shap_values=Array{Float64,1}(undef,length(shap_val[1]))
        #shap_vals=Array{Float64,1}(undef,length(explain))
        #if predictor.predict_proba([explain])[1,1]>predictor.predict_proba([explain])[1,2]
    #        shap_vals=shap_val[1]
    #    else
    #        shap_vals=shap_val[2]
    #    end
    #    shap_values[i,:]=shap_vals
        #println("$i")
    end
    return vec(mean(shap_values,dims=1))
end


function compute_shap_values_only_positive(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    n=length(Data[1])
    m=length(labels)
    shap_values=fill(0.0,n-3)
    Xtrain=fill(0.0,m,n)
    for i in 1:m
        Xtrain[i,:]=Data[i]
    end
    #Xtrain=Data
    ytrain=labels
    predictor.fit(Xtrain,ytrain)
    explainer=shap.TreeExplainer(predictor)
    explain=Xtrain
    shap_val=explainer.shap_values(explain)[2]
    shap_val=shap_val[:,1:(n-3)]
    for i in 1:length(shap_values)
        x=shap_val[:,i]
        pos_idx=findall(x.>0.0)
        shap_values[i]=sum(x[pos_idx])
    end
    return shap_values
end

function chem_descr_shap_values_only_positive_leave_one_out_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int32)
    m,n=size(Data)
    shap_vector_length=length(Data[1,1])-3
    shap_values_matrix=fill(0.0,m,shap_vector_length)
    for i in 1:m
        shap_values_matrix[i,:]=compute_shap_values_only_positive(Data[i,:],bin_resp[i,:],ntrees,sample_perc,random_seed)[1:shap_vector_length]
    end
    shap_vals=vec(mean(shap_values_matrix,dims=1))
    return shap_vals,findmax(shap_vals,1:length(shap_vals))[1]
end

function compute_shap_values_separate_positive_and_negative(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    n=length(Data[1])
    m=length(labels)
    shap_values_pos=fill(0.0,n-3)
    shap_values_neg=fill(0.0,n-3)
    Xtrain=fill(0.0,m,n)
    for i in 1:m
        Xtrain[i,:]=Data[i]
    end
    #Xtrain=Data
    ytrain=labels
    predictor.fit(Xtrain,ytrain)
    explainer=shap.TreeExplainer(predictor)
    explain=Xtrain
    shap_val=explainer.shap_values(explain)[2]
    if shap_val==0.0
        return vcat(shap_values_pos,shap_values_neg)
    else
        shap_val=shap_val[:,1:(n-3)]
        for i in 1:length(shap_values_pos)
            x=shap_val[:,i]
            pos_idx=findall(x.>0.0)
            neg_idx=findall(x.<0.0)
#           shap_values_pos[i]=sum(x[pos_idx])
#           shap_values_neg[i]=-1*sum(x[neg_idx])
            if length(pos_idx)==0
                shap_values_pos[i]=0
            else
                shap_values_pos[i]=mean(x[pos_idx])
            end
            if length(neg_idx)==0
                shap_values_neg[i]=0
            else
                shap_values_neg[i]=-1*mean(x[neg_idx])
            end
        end
    end
    return vcat(shap_values_pos,shap_values_neg)
end

function compute_shap_values_separate_positive_and_negative_matrix(Data::Array{Any,2},labels::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    m=length(Data[1,1])-3
    shap_vals=fill(0.0,size(Data,1),2*m)
    for i in 1:size(shap_vals,1)
        shap_vals[i,:]=compute_shap_values_separate_positive_and_negative(Data[i,:],labels[i,:],ntrees,sample_perc,random_seed)
    end
    return shap_vals
end

function compute_shap_values_sum(Data::Array{Any,1},labels::BitVector,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    predictor=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);
    n=length(Data[1])
    m=length(labels)
    shap_values=fill(0.0,n-3)
    Xtrain=fill(0.0,m,n)
    for i in 1:m
        Xtrain[i,:]=Data[i]
    end
    #Xtrain=Data
    ytrain=labels
    predictor.fit(Xtrain,ytrain)
    explainer=shap.TreeExplainer(predictor)
    explain=Xtrain
    shap_val=explainer.shap_values(explain)[2]
    shap_val=shap_val[:,1:(n-3)]
    for i in 1:length(shap_values)
        x=shap_val[:,i]
        shap_values[i]=sum(abs.(x))
    end
    return shap_values
end


function compute_shap_values_absolute_value_sum_matrix(Data::Array{Any,2},labels::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int64)
    m=length(Data[1,1])-3
    shap_vals=fill(0.0,size(Data,1),m)
    for i in 1:size(shap_vals,1)
        shap_vals[i,:]=compute_shap_values_sum(Data[i,:],labels[i,:],ntrees,sample_perc,random_seed)
    end
    return shap_vals
end

function chem_descr_shap_values_separate_positive_and_negative_leave_one_out_per_glom(Data::Array{Any,2},bin_resp::BitMatrix,ntrees::Int64,sample_perc::Float64,random_seed::Int32)
    m,n=size(Data)
    shap_vector_length=2*(length(Data[1,1])-3)
    shap_values_matrix=fill(0.0,m,shap_vector_length)
    for i in 1:m
        shap_values_matrix[i,:]=compute_shap_values_separate_positive_and_negative(Data[i,:],bin_resp[i,:],ntrees,sample_perc,random_seed)[1:shap_vector_length]
    end
    shap_vals=vec(mean(shap_values_matrix,dims=1))
    return shap_vals,findmax(shap_vals,1:length(shap_vals))[1]
end
