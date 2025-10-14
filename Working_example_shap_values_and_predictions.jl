"""
    The following should install the necessary python libraries to run the code.
"""

using PyCall;
run(`$(PyCall.python) -m pip install shap`;);
run(`$(PyCall.python) -m pip install -U scikit-learn`;);

"""
    The followind loads the necessary functions I wrote to run the code.
"""

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files//Useful_functions.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Initialize_analyses.jl"));
include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/python_shap_functions_new.jl"));

"""
    Use one of these files to load the necessary data to run the code.
    The first file contains data with all odors, the second data without the amines.
    Pick one by commenting out the other.
"""

data_file="segmented_data_different_descrs_for_prediction.jld";
descr_name="pom21";
#data_file="segmented_data_schmuker_descrs_shuffled_descrs_and_resps_no_amines.jld";

"""
    This loads the descriptor names.
"""

using JLD;
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/$data_file"),"descr_names_$descr_name");

"""
    The following loads the data for building random forest models of glomerular activation and/or computing Shapley fields on the olfactory bulb.
    Data_frame and Data contain the same information, but in different formats. Both are arrays, each element of the array corresponds
    to one of the olfactory bulbs we have. For example, if we picked data sets with amines, the first array is omp83L, second array is omp83R.
    If we picked the data sets without amines, the first array is omp84, second array is omp85.

    Data is a list of matrices of type Array{Any,2}. Rows are indexed by glomeruli, columns by odorants.
    Each entry is a vector of chemical features of the corresponding odorant and position and area
    of the corresponding glomerulus (normalized, in the last 3 coordinates).

    Data frame is a list of data frames. You can get Data_frame[i] from Data[i] by doing vec(Data[i]) which flattens the matrix by columns,
    and collecting the resulting array into a Data frame object.
    The rows in each data frame correspond to a glomerulus-odorant pair, total number of rows is (number of gloms) x (number of odorants).
    The columns are the chemical features of the odorant
    and the position and area (normalized, in the last 3 columns) of the glomerulus.

    binary_responses is an array of Boolean matrices, recording the response pattern of the given response matrices for a given olfactory bulb.
    These are the labels that are to be used for training the models.
"""

Data_frame=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/$data_file"),"Data_frame_$descr_name");
Data=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/$data_file"),"Data_$descr_name");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/$data_file"),"bin_resp");

data_shap_idx_new=find_shap_ordering(Data_frame[1],descr_names,binary_responses[1],ntrees,sample_perc,feature_perc,random_seed)

"""
    The following are the hyperparameters we can choose for building our random forest models.

    ntrees are the number of decision trees to be built in the model. Each tree sees only \sqrt(n) randomly chosen
    features for training where n is the total number of features. This is the default, but it can be changed if we so choose,
    although so far I've just been sticking with the default.

    sample_perc is the percentage of training data each tree is allowed to see for training.

    random_seed is the random_state specified for the model.

    skl_ensemble.RandomForestClassifier builds a python random forest model from the scikit-learn library, with the specified hyperparameters.

    class_weight is set to balanced_subsample, which balances the weights of the data on each individual tree, separately. It can also be set
    to balanced, which will do weights for the data before data is passed to a tree. The default is None, which will not attempt to assign
    weights to different classes, in the case of imbalance of the data set.

    After the model is built, then it can be trained given some data and labels. After training, give it a test data set and run predictions.
    I wrote functions that do all this automatically given one of our data sets.
"""

ntrees=256;
sample_perc=0.7;
random_seed=0;
feature_perc=0.9;#1.0;
predictor_example=skl_ensemble.RandomForestClassifier(n_estimators=ntrees,max_samples=sample_perc,class_weight="balanced_subsample",random_state=random_seed);

"""
    chem_descr_and_pos_classification_old from python_shap_functions_new.jl will run 10 fold cross-validation and compute the average prediction accuracy and pr-auc
    across folds, returning an array with first entry prediction accuracy, second entry pr-auc value.
    We first specify the number of folds to be used.

    This is fast, feel free to run on a local machine.

    As the function runs, it will print out the fold it is currently computing. Feel free to comment this out in the function if you wish.
"""

nfolds=10;
@time pred_metrics_10_fold_2=chem_descr_and_pos_classification_old(Data_frame[1],binary_responses[1],nfolds,ntrees,sample_perc,feature_perc,0);

"""
    chem_descr_and_pos_classification_leave_one_out_old from python_shap_functions_new.jl will run leave-one-odor out cross-validation and compute the average prediction accuracy and pr-auc
    across folds, returning an array with first entry prediction accuracy, second entry pr-auc value.

    This is still relatively fast, feel free to run on a local machine.

    As the function runs, it will print out index of the odor being currently used as a testing data set. Feel free to comment this out in the function if you wish.
"""

pred_metrics_leave_one_out=chem_descr_and_pos_classification_leave_one_out_old(Data[1],binary_responses[1],ntrees,sample_perc,random_seed);

"""
    chem_descr_classification_leave_one_out_per_glom from python_shap_functions_new.jl will run leave-one-odor out cross-validation for each glomerulus separately (building a different random forest for each
    glomerulus, thus not using position and area information as features anymore) and compute the average prediction accuracy and pr-auc
    across folds and glomeruli, returning an array with first entry prediction accuracy, second entry pr-auc value.

    THIS IS NO LONGER FAST ENOUGH TO RUN ON A LOCAL MACHINE FOR THE WHOLE DATASET!!!!!!!
    I would suggest selecting a subset of glomeruli to run this on. Pick a subset of rows of Data[1] and the same subset of rows of binary_responses and
    pass these to the function.

    As the function runs, it will print out index of the glomeruli on which a random forest is being cross-validated across the odorant set in a leave-one out fashion.
    Feel free to comment this out in the function if you wish.

    We first construct new_Data, which we get from Data[i], where i=1,2, by just removing the position and area features of each glomerulus.

    Then a subset of glomeruli is selected, in order to run this on a local machine. I chose the first 5.
"""

descr_number=length(descr_names);
m,n=size(Data[1]);
new_Data=Array{Any,2}(undef,m,n);
for i in 1:m
    for j in 1:n
        new_Data[i,j]=Data[1][i,j][1:descr_number]
    end
end

subset_of_glomeruli=1:5;

pred_metrics_leave_one_out_per_glom=chem_descr_classification_leave_one_out_per_glom(new_Data,binary_responses[1],ntrees,sample_perc,random_seed);

pred_metrics_leave_one_out_per_glom=chem_descr_classification_leave_one_out_per_glom(new_Data[subset_of_glomeruli,:],binary_responses[1][subset_of_glomeruli,:],ntrees,sample_perc,random_seed);


pred_metrics_peer_prediction=peer_prediction(copy(resp_matr[1]),binary_responses[1],ntrees,sample_perc,random_seed)

"""
    The following is an example where Shapley values are computed.

    We first construct a model called predictor.
"""

predictor=skl_ensemble.RandomForestClassifier(n_estimators=128,max_samples=0.7,class_weight="balanced_subsample",random_state=0);

"""
    We then fit the model on the first glomerulus in our data set.

    We pick a training set and testing set indices. In this example we set the test set to be the first odor, some acid.

    X_train,y_train, X_test, y_test are the training/testing sets and labels.
"""

train_idx=2:59; #exclude the first odorant from training
test_idx=1; #use the first odorant in testing
X_train=Data[1][1,train_idx];
y_train=binary_responses[1][1,train_idx];
X_test=Data[1][1,test_idx];
predictor.fit(X_train,y_train)

"""
    We then run our prediction on the testing set. Note that predict_proba expects an Array{Float64,1}, so a single Float64 as X_test above will
    cause an Error. Make it into an Array via [X_test] before passing to the function. If our testing set was an Array to begin with, remove the square brackets.

    There are two functions that make a prediction. predict_proba is the honest output of the model. In an n-class clasification problem predict_proba will output
    an n-dimensional vector, each element of the vector is the probability of the testing data being in one of the given classes. Since we are doing 2-class classification
    we get a vector of length 2. First entry is the probability of having label 0 (meaning no response), second entry is the probability of having label 1 (meaning response).

    There is also a function called predict, which will output the actual class the model is supposed to predict, based on the computed probabilities.
"""

#println(predictor.predict_proba(X_test)) #this causes an error
println(predictor.predict_proba([X_test]));
println(predictor.predict([X_test]));

"""
    Now we buld a shap.TreeExplainer object. This requires as input our model.

    expected_value are the expected probabilities the model will give to any data point,
    without using any feature information! Basically the model looks at the training set and
    outputs the average probabilities over the training set. Since we used class_weight="balanced_subsample"
    when we defined our model, even though our data set is heavily imbalanced, expected_value should be a
    vector of two numbers, both being close to 0.5. If class_weight="None", then the expected_value should be
    two numbers representing the actual percentages of the two classes.
"""

explainer=shap.TreeExplainer(predictor);
println(explainer.expected_value);

"""
    Now explain feature contribution of our testing set prediction, but really you
    could also explain any data point, even coming from the training set.
    explainer.shap_values will compute Shapley values of each feature towards the model prediction.

    shap_values returns tuple of Arrays. shap_values[1] are the Shapley values contributing towards predict_proba([X_test])[1]
    and shap_values[2] are the Shapley values contributing towards predict_proba([X_test])[2].

    In an n-class classification, shap_values would return an n-dimensional array of arrays, one for each class.

    What we are saying is that predict_proba([X_test])[i]=sum(shap_values[i])+explainer.expected_value[i].

    Note that we are using X_train=Data[1][1,train_idx] as features, which also contains position and area as features in the last 3 columns.
    Since we are using the same glomerulus throughout, these remain unchanged and should have no contribution towards any model prediction.
    Indeed, their Shapley values are identical to 0.
"""

explain=X_test;
shap_values=explainer.shap_values(explain);
println(shap_values[1][setdiff(1:end,1:length(descr_names))].==0.0;)
