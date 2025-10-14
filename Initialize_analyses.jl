println("  Initializing...");
using PyPlot, MAT,   MultivariateStats, TSne, Measures, Distances, Clustering, Plots, DelimitedFiles, Statistics, StatsPlots, LinearAlgebra

#pyplot()

#clibrary() is no longer a function supported in Plots
#clibrary(:colorcet)

#new color pallete function
#cgrad(:tab10)
#palette(:tab10)
"""
    load_data(filename:::String; thresh_neg = true, delete_zero=true)

    Load the response data. The function assumes that the data is in the
    file "filename.mat", in the variable "filename"

    Use "thresh_neg = true" to threshold the negative values (they become zeros)
    Use "delete_zero = true" to remove all-zero _rows_.
    Note that this changes the index of the remaining nonzero rows,
    so other files relying on these indices have to be accessed with care

    When "delete_zero = true", the function returns a pair (response_matrix,
    nonzero_rows), containing the response matrix and the indices of nonzero rows
    When "delete_zero = false", the function returns a pair (response_matrix, 1:number_of_rows)
"""
#fname= joinpath(response_path,response_filenames[1]*".mat")
#vars = matread(fname)
#close(vars)
#response=vars["responses"]
#response=filename["df_respmatrix_z6"]
#close(vars)

function load_data_b(filename, my_path::String="";   thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    vars=matread(FNAME)
    #odors_to_omit=Int.(vars["olfr1377_omitodorindices"])
    #file = matopen(FNAME)
    responses = vars["ORdata_rawdF_from_maps_maporder"]["RespMatrix"]
    #indices_to_omit=fill(true,size(responses)[2])
    #for i in 1:length(odors_to_omit)
    #    indices_to_omit[odors_to_omit[i]]=false
    #end
    #responses= responses[:,indices_to_omit]
    #close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end


function load_data_c_2(filename, my_path::String="";   thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    vars=matread(FNAME)
    #odors_to_omit=Int.(vars["olfr1377_omitodorindices"])
    #file = matopen(FNAME)
    responses = vars["ORdata_rawdF_from_maps_maporder"]["RespMatrix"]
    #indices_to_omit=fill(true,size(responses)[2])
    #for i in 1:length(odors_to_omit)
    #    indices_to_omit[odors_to_omit[i]]=false
    #end
    #responses= responses[:,indices_to_omit]
    #close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end


function load_data_c(filename, my_path::String="";   thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    vars=matread(FNAME)
    odors_to_omit=Int.(vars["olfr1377_omitodorindices"])
    #file = matopen(FNAME)
    responses = vars["ORdata_rawdF_from_maps_maporder"]["RespMatrix"]
    indices_to_omit=fill(true,size(responses)[2])
    for i in 1:length(odors_to_omit)
        indices_to_omit[odors_to_omit[i]]=false
    end
    responses= responses[:,indices_to_omit]
    #close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end



function load_data_with_amines(filename,my_path::String="", thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    file = matopen(FNAME)
    relevant_odors= 1:59;
    #relevant_odors=setdiff(1:59,26:32);
    #animal_name=collect(keys(read(file)))[animal_name_idx]
    #responses = read(file,filename)
    #responses = read(file,"omp85_respmat_noamines")["respmat"][2]
    #responses = read(file)[animal_name]["RespMatrix"]
    #responses = read(file)[animal_name]["RespMatrix"]
    responses = read(file)["ORdata"]["RespMatrix"]
    responses=responses[:,relevant_odors]
    close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0.0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0.0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
#        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end


function load_data_no_amines_new(filename,my_path::String="", thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    file = matopen(FNAME)
    #relevant_odors= vcat(1:26,34:59);
    relevant_odors=setdiff(1:59,26:32);
    #animal_name=collect(keys(read(file)))[animal_name_idx]
    #responses = read(file,filename)
    #responses = read(file,"omp85_respmat_noamines")["respmat"][2]
    #responses = read(file)[animal_name]["RespMatrix"]
    #responses = read(file)[animal_name]["RespMatrix"]
    responses = read(file)["ORdata"]["RespMatrix"]
    responses=responses[:,relevant_odors]
    close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0.0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0.0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end


function load_data_no_amines(filename, animal_name_idx,my_path::String="", thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    file = matopen(FNAME)
    #relevant_odors= vcat(1:26,34:59);
    relevant_odors=vcat(1:59);
    animal_name=collect(keys(read(file)))[animal_name_idx]
    #responses = read(file,filename)
    #responses = read(file,"omp85_respmat_noamines")["respmat"][2]
    responses = read(file)[animal_name]["respmat"][2]
    responses=responses[:,relevant_odors]
    close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0.0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0.0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end

function load_data(filename, my_path::String="";   thresh_neg = true, delete_zero=true)
    FNAME= isempty(my_path) ? joinpath(pwd(),filename*".mat") : joinpath(my_path,filename*".mat")
    file = matopen(FNAME)
    responses = read(file,filename)
    no_amines_idx=setdiff(1:61,26:32)
    responses=responses[:,no_amines_idx]
    close(file)
    for resp in eachindex(responses)
        if isnan(responses[resp])
            responses[resp] = 0.0
        elseif thresh_neg && responses[resp] < 0.0
            responses[resp] = 0.0
        end
    end
    if delete_zero
        nonzero_rows=[]
        for i=1:size(responses,1)
            if VERSION>= v"0.7.0"
                if findfirst(x->x!=0,responses[i,:]) !=  nothing
                    push!(nonzero_rows, i)
                end
            else
                if findfirst(x->x!=0,responses[i,:]) > 0
                    push!(nonzero_rows, i)
                end
            end
        end
        responses = responses[nonzero_rows,:]
        return (responses, nonzero_rows)
    end

    return (responses, 1:size(responses,1))
end

"""
    load_mask(filename:::String)

    Load the glomeruli masks.
    The masks are in the file "filename.mat", recorded as follows:
    The variable "filename" contains a dictionary. The mask is contained in the
     entry of this dictionary under "mask"
    Returns an array of matrices/images, where the k'th matrix has 1's in the
    spots for the k'th glomerulis, and 0's otherwise

"""
function load_mask(filename::String,kkey::String)
    println("Loading file ",filename); println
    file = matopen(filename)
    data = read(file)
    close(file)
    return data[kkey]["mask"]
end


"""
    load_descriptors(file::String replace_missing = 0.0, only_positive = false,
    only_nonunique=true)

    Load the descriptor data from "file". The data should be in a ".csv" file,
    where columns are descriptors and rows are odors.
    Returns a tuple (matrix, names_array), containing
    a matrix of the data -- rows are odors, cols are descriptors --
    and the array with names of descriptors.

    Use "replace_missing = value" to replace missing (i.e. empty string) values
     by "value". Default "replace_missing=0.0"
    Use "only_positive = true" to discard descriptors with negative values.
    Default "only_positive =false".
    Use "only_nonunique = true" to only use descriptors that have at least two
    different values within one column. Default "only_nonunique = true"
"""

function load_descriptors_c_2(file; replace_missing = 0.0, only_positive = true, only_nonunique = true)
    response_path= joinpath(homedir(),"C:\\Users\\nikol\\Dropbox\\Olfactory_Datasets\\Data_raw\\item_c");
    response_filenames="1377_007_ORdata_useforanalysis"
    fname=joinpath(response_path,response_filenames*".mat")
    vars=matread(fname)
    #odors_to_omit=Int.(vars["olfr1377_omitodorindices"])
    s = readdlm(file,',',header=true);
    #smiles = s[2:end,1];
    descr = String.(s[2][9:end]);
    vals = s[1][1:end, 9:end];
    #indices_to_omit=fill(true,size(vals)[1])
    #for i in 1:length(odors_to_omit)
    #    indices_to_omit[odors_to_omit[i]]=false
    #end
    #vals=vals[indices_to_omit,:]
    vals[findall(x->x==" ", vals)] .= replace_missing;
    vals[findall(x->x=="", vals)] .= replace_missing;
    vals[findall(x->isnan(x), vals)] .= replace_missing;
    rowcount = size(vals,1);
    colcount = size(vals,2);
    useful_descr = fill(true, colcount)
    if only_nonunique || only_positive
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i]))!=nothing)
               useful_descr[i] = false
            end
        end
    end
    useful_descr_names = descr[useful_descr];
    useful_vals= vals[:,useful_descr];
    useful_vals = Float64.(useful_vals);
#    useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    return useful_vals, useful_descr_names
end


function load_descriptors_c(file; replace_missing = 0.0, only_positive = true, only_nonunique = true)
    response_path= joinpath(homedir(),raw"C:\Users\nikol\Dropbox (UFL)\Olfactory_Datasets\Data_raw\item_c");
    response_filenames="1377_007_ORdata_useforanalysis"
    fname=joinpath(response_path,response_filenames*".mat")
    vars=matread(fname)
    odors_to_omit=Int.(vars["olfr1377_omitodorindices"])
    s = readdlm(file,',',header=true);
    #smiles = s[2:end,1];
    descr = String.(s[2][9:end]);
    vals = s[1][1:end, 9:end];
    indices_to_omit=fill(true,size(vals)[1])
    for i in 1:length(odors_to_omit)
        indices_to_omit[odors_to_omit[i]]=false
    end
    vals=vals[indices_to_omit,:]
    vals[findall(x->x==" ", vals)] .= replace_missing;
    vals[findall(x->x=="", vals)] .= replace_missing;
    vals[findall(x->isnan(x), vals)] .= replace_missing;
    rowcount = size(vals,1);
    colcount = size(vals,2);
    useful_descr = fill(true, colcount)
    if only_nonunique || only_positive
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i]))!=nothing)
               useful_descr[i] = false
            end
        end
    end
    useful_vals= vals[:,useful_descr];
    useful_descr_names = descr[useful_descr];
    useful_vals = Float64.(useful_vals);
# useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    return useful_vals, useful_descr_names
end

function load_descriptors_schmuker(file; replace_missing = 0.0, only_positive = true, only_nonunique = true)
    s = readdlm(file,',',header=true);
    #smiles = s[2:end,1];
    descr = String.(s[2][4:end]);
    s[1][findall(x->x==" ", s[1])] .= replace_missing;
    s[1][findall(x->x=="", s[1])] .= replace_missing;
    vals = s[1][1:end, 4:end];
    #vals[isnan.(vals)].=replace_missing
    #relevant_odors=vcat(1:26,34:59)
    relevant_odors=vcat(1:59)
    vals=vals[relevant_odors,:]
    vals[findall(x->isnan(x), vals)] .= replace_missing;
    rowcount = size(vals,1);
    colcount = size(vals,2);
    useful_descr = fill(true, colcount)
    if only_nonunique || only_positive
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i]))!=nothing)
               useful_descr[i] = false
            end
        end
    end
    useful_descr_names = descr[useful_descr];
    useful_vals= vals[:,useful_descr];
    useful_vals = Float64.(useful_vals);
#    useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    return useful_vals, useful_descr_names
end

function load_descriptors_no_amines(file,relevant_odors; replace_missing = 0.0, only_positive = true, only_nonunique = true)
    s = readdlm(file,',',header=true);
    #smiles = s[2:end,1];
    descr = String.(s[2][2:end]);
    s[1][findall(x->x==" ", s[1])] .= replace_missing;
    s[1][findall(x->x=="", s[1])] .= replace_missing;
    vals = s[1][relevant_odors, 2:end];
    #vals[isnan.(vals)].=replace_missing
    vals[findall(x->isnan(x), vals)] .= replace_missing;
    rowcount = size(vals,1);
    colcount = size(vals,2);
    useful_descr = fill(true, colcount)
    if only_nonunique || only_positive
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i]))!=nothing)
               useful_descr[i] = false
            end
        end
    end
    useful_descr_names = descr[useful_descr];
    useful_vals= vals[:,useful_descr];
    norm_vals=normalize_matrix(useful_vals)
    bad_idxs=findall(sum.(eachcol(norm_vals)).==1)
    useful_vals = Float64.(useful_vals[:,setdiff(1:end,bad_idxs)]);
    useful_descr_names=useful_descr_names[setdiff(1:end,bad_idxs)]
#    useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    return useful_vals, useful_descr_names
end


function load_descriptors(file; replace_missing = 0.0, only_positive = true, only_nonunique = true)
    s = readdlm(file,',',header=true);
    #smiles = s[2:end,1];
    descr = String.(s[2][2:end]);
    s[1][findall(x->x==" ", s[1])] .= replace_missing;
    s[1][findall(x->x=="", s[1])] .= replace_missing;
    vals = s[1][1:end, 2:end];
    #vals[isnan.(vals)].=replace_missing
    vals[findall(x->isnan(x), vals)] .= replace_missing;
    rowcount = size(vals,1);
    colcount = size(vals,2);
    useful_descr = fill(true, colcount)
    if only_nonunique || only_positive
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i]))!=nothing)
               useful_descr[i] = false
            end
        end
    end
    useful_descr_names = descr[useful_descr];
    useful_vals= vals[:,useful_descr];
    useful_vals = Float64.(useful_vals);
#    useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    return useful_vals, useful_descr_names
end

#=
function load_descriptors(file; replace_missing = 0.0, only_positive = false, only_nonunique=true)
    if VERSION>= v"0.7.0"

        s = readdlm(file,',',header=true);
        descr =String.(readdlm(IOBuffer(s[2][1,1]),',')[2:end]);
        vals=zeros( length(s[1]) , length(descr));
        for i=1: length(s[1])
              the_array=(readdlm(IOBuffer(s[1][i]),',')[2:end]);
              the_array[findall(x->x==" ", the_array)] .= replace_missing;
              the_array[findall(x->x=="", the_array)] .= replace_missing;
              vals[i,:]=  Float64.(the_array) ;
    end
        smiles= [String(readdlm(IOBuffer(s[1][i]),',')[1]) for i=1: length(s[1])]
    else # this is the original version written by Alexandra. Unfortunately this only works in Julia0.6
        s = readcsv(file);
        smiles = s[2:end,1];
        descr = String.(s[1,2:end]);
        s[find(x->x==" ", s)] .= replace_missing;
        s[find(x->x=="", s)] .= replace_missing;
        vals = s[2:end, 2:end];
        vals[find(x->isnan(x), vals)] .= replace_missing;
    end

    rowcount = size(vals,1);
    colcount = size(vals,2);
    if only_nonunique || only_positive
        useful_descr = fill(true, colcount)
        for i=1:colcount
            if (only_nonunique && length(unique(vals[:,i])) == 1) || (only_positive && (findfirst(x->x<0.0, vals[:,i])>0))
               useful_descr[i] = false
            end
        end
    end

    useful_decr_names = descr[useful_descr];
    useful_vals= vals[:,useful_descr];
    useful_vals = Float64.(useful_vals);
    if VERSION>= v"0.7.0"
        useful_vals[findall(x->x=="", useful_vals)] .= replace_missing;
    else
       useful_vals[find(x->x=="", useful_vals)] .= replace_missing;
    end
    return (useful_vals, useful_decr_names)
end
=#

"""
    normalize_matrix(matr::Array{Float62,2}; dim=2)

    Normalize matrix along the dimension "dim" ("dim=1" -- rows, "dim=2" -- columns)
     to vector norm 1. Default "dim=2".
"""
function normalize_matrix(matr; dim=2)
    if dim == 2
        matr_norm = copy(matr)
    elseif dim == 1
        matr_norm = copy(transpose(matr))
    end

    cols = size(matr_norm,2)
    for i=1:cols
        matr_norm[:,i] = normalize(matr_norm[:,i])
    end
    if dim == 2
        return matr_norm
    elseif dim == 1
        return transpose(matr_norm)
    end
end

"""
    matrix_zscore(matr::Array{Float64,2}; dim=2)

    Normalize matrix  along the dimension "dim" ("dim=1" -- rows, "dim=2" -- columns)
     to mean 0, variance 1. Default "dim=2".
"""
function normalize_matrix_zscore(matr; dim=2)

    if dim == 2
        matr_norm = copy(matr)
    elseif dim == 1
        matr_norm = copy(tranpose(matr))
    end

    cols = size(matr_norm,2)
    for i=1:cols
        matr_norm[:,i] = (matr_norm[:,i] .- mean(matr_norm[:,i])) ./ std(matr_norm[:,i])
    end
    if dim == 2
        return matr_norm
    elseif dim == 1
        return transpose(matr_norm)
    end
end


"""
normalize_matrix_by_max normalizes a given matrix; each
row or column is normalized by the maximum of said row or
column
"""
function normalize_matrix_by_max(matr; dim=2)

    if dim == 2
        matr_norm = copy(matr)
    elseif dim == 1
        matr_norm = copy(transpose(matr))
    end

    cols = size(matr_norm,2)
    for i=1:cols
        matr_norm[:,i] = (matr_norm[:,i]) ./ maximum(matr_norm[:,i])
    end
    if dim == 2
        return matr_norm
    elseif dim == 1
        return transpose(matr_norm)
    end
end


"""
normalize_matrix_stdscore normalizes each row or column
by scaling by the variance of said row or column
"""


function normalize_matrix_stdscore(matr; dim=2)

    if dim == 2
        matr_norm = copy(matr)
    elseif dim == 1
        matr_norm = copy(transpose(matr))
    end

    cols = size(matr_norm,2)
    for i=1:cols
        matr_norm[:,i] = (matr_norm[:,i]) ./ std(matr_norm[:,i])
    end
    if dim == 2
        return matr_norm
    elseif dim == 1
        return transpose(matr_norm)
    end
end

"""
    window(mask::Array{Array{Float64,2},1})

    Find the pixel range that contains the glomeruli (for plotting).
    Takes an array with glomeruli masks (one for each glomerulus) and returns
    a quadruple (min_x,mx_x, min_y, max_y)
"""

function window_matr(matr)
    xs = size(matr,2)
    ys = size(matr,1)
    min_x = 1
    for x=1:xs
        if findfirst(a->a!=0.0, matr[:,x]) != nothing
            break
        else
            min_x=x
        end
    end

    max_x=xs
    for x=1:xs
        if findfirst(a->a!=0.0, matr[:,xs-x]) != nothing
            break
        else
            max_x=xs-x
        end
    end

    min_y = 1
    for y=1:ys
        if findfirst(a->a!=0.0, matr[y,:]) != nothing
            break
        else
            min_y=y
        end
    end

    max_y=ys
    for y=1:ys
        if findfirst(a->a!=0.0, matr[ys-y,:]) != nothing
            break
        else
            max_y=ys-y
        end
    end
    return (min_x,max_x,min_y,max_y)
end

function window_matr_inside_nan(matr::Matrix{Float64})
    xs = size(matr, 2)
    ys = size(matr, 1)
    min_x = 1
    for x = 1:xs
        if findfirst(a -> !isnan(a), matr[:, x]) != nothing
            break
        else
            min_x = x
        end
    end

    max_x = xs
    for x = 1:xs
        if findfirst(a -> !isnan(a), matr[:, xs-x+1]) != nothing
            break
        else
            max_x = xs-x+1
        end
    end

    min_y = 1
    for y = 1:ys
        if findfirst(a -> !isnan(a), matr[y, :]) != nothing
            break
        else
            min_y = y
        end
    end

    max_y = ys
    for y = 1:ys
        if findfirst(a -> !isnan(a), matr[ys-y+1, :]) != nothing
            break
        else
            max_y = ys-y+1
        end
    end

    # Adjusting the indices to ensure they are within bounds after the margin is applied
    min_x = max(1, min_x - 10)
    max_x = min(xs, max_x + 10)
    min_y = max(1, min_y - 10)
    max_y = min(ys, max_y + 10)

    return (min_x, max_x, min_y, max_y)
end

# function window_matr_inside_nan(matr)
#     xs = size(matr,2)
#     ys = size(matr,1)
#     min_x = 1
#     for x=1:xs
#         if findfirst(a->!isnan(a), matr[:,x]) != nothing
#             break
#         else
#             min_x=x
#         end
#     end

#     max_x=xs
#     for x=1:xs
#         if findfirst(a->!isnan(a), matr[:,xs-x]) != nothing
#             break
#         else
#             max_x=xs-x
#         end
#     end

#     min_y = 1
#     for y=1:ys
#         if findfirst(a->!isnan(a), matr[y,:]) != nothing
#             break
#         else
#             min_y=y
#         end
#     end

#     max_y=ys
#     for y=1:ys
#         if findfirst(a->!isnan(a), matr[ys-y,:]) != nothing
#             break
#         else
#             max_y=ys-y
#         end
#     end
#     return (min_x,max_x,min_y,max_y)
# end




function window(mask)
    im_width=size(mask[1],1)
    im_height=size(mask[1],2)
    mask_matr=zeros(im_height,im_width)
    for i in 1:length(mask)
        mask_matr+=mask[i]
    end
    return window_matr(mask_matr)
end

function window_new(mask)
    im_width=size(mask[1],1)
    im_height=size(mask[1],2)
    mask_matr=zeros(im_height,im_width)
    for i in 1:length(mask)
        mask_matr+=mask[i]
    end
    return window_matr_inside_nan(mask_matr)
end


####################   Clustering  ##################

"""
    distance_matrix(matr; distance = Distances.euclidean, dim=2)

    Returns the distance matrix of vectors of "matr" alons the dimension "dim"
    (dim=1 -- distance between rows, dim=2 -- distance between columns, default -- columns)
    according to the distance function "distance"

"""

function distance_matrix(matr, distance = Distances.euclidean; dim=2)
    if dim==1
        M = transpose(matr)
    elseif dim==2
        M = matr
    end
    n_trials = size(M,2)
    dist_matr = fill(0.0, n_trials, n_trials)
    for i=1:n_trials
        for j=i+1:n_trials
            dist_matr[i,j] = dist_matr[j,i] = distance(M[:,i], M[:,j])
        end
    end
    return dist_matr
end


"""
    pick_clusters(h::Hclust, height; min_cluster=4)

    Pick clusters from the hierarchical clustering object H based on height.
    Returns an array of clusters, where each cluster represented by an array of node numbers.

    Use "min_cluster" to set the threshold for the minimal cluster size (clusters of size below "min_cluster" are discarded)

"""

function pick_clusters(h::Clustering.Hclust, height::Union{Float64,Int}; min_cluster=4)#::Vector{Vector{Int}}
    clust_nums = pick_cluster_inds(h, height)
    unique_clust_nums = unique(clust_nums)
    clusts = Vector{Vector{Int}}([])
    for i in unique_clust_nums
        cluster = VERSION>= v"0.7.0" ? findall(x->x==i,clust_nums) : find(x->x==i,clust_nums);
        if (length(cluster) >= min_cluster) && i != 0
            push!(clusts, cluster)
        end
    end
    return clusts
end

"""
    plot_dendrogram(H::Hclust;
                        names=false,
                        rotation=90,
                        tickfontsize=10)

    Plot the dendrogram of the hierarhical clustering object H. Returns a plot object.
    Use "names" to specify the names of the nodes (by default they will be numbers of nodes in the original matrix).
    Use "rotation" to adjust the angle of the node labels.
    Use "tickfontsize" to adjust the font size of the node labels.

"""
function plot_dendrogram(H::Hclust; names=false, rotation=90, tickfontsize=10)
    #pyplot()
    if names != false
        p = Plots.plot(H,
            xrotation=90,
            tickfontsize=10,
             xticks =(1:length(H.order), names[H.order]))
    else
        p = Plots.plot(H,
            xrotation=90,
            tickfontsize=10,
             xticks =(1:length(H.order), H.order))

    end
    return p
end

####################    Plotting   ##################

"""
    plot_dendrogram_and_matrix(h::Hclust,
                                matr;
                                title=""
                                names=false,
                                row_names=false,
                                rotation=90,
                                tickfontsize=10,
                                color=:rainbow )

    Plot the dendrogram of the hierarhical clustering object H together with the matrix "matr".
    The COLUMNS of "matr" are aligned with the nodes of the clustering object.
    Returns a plot object.

    Use "names" to specify the names of the nodes (by default they will be numbers of nodes in the original matrix).
    Use "row_names" to specify the names of the rows of the matrix (non by default)
    Use "rotation" to adjust the angle of the node labels.
    Use "tickfontsize" to adjust the font size of the node labels.
    Use "color" to specify the colorscheme of the matrix.

"""
function plot_dendrogram_and_matrix(h::Hclust, matr;
        names=false, row_names=false, rotation=90, tickfontsize=10, color=:rainbow, title="")
    n_cols = size(matr,2)
    if names != false
        labels = names[h.order]
    else
        labels = 1:n_cols
    end

    p = Plots.plot(h,
            xticks = [],
            rotation=rotation,
            tickfontsize=tickfontsize, title=title)
    #h1 = heatmap(matr[:,h.order],
    h1 = heatmap(transpose(matr[h.order,:]),
        c=:jet1,
        colorbar=false,
        yticks=(1:n_cols, labels),
        xticks = row_names,
        rotation=rotation,
        tickfontsize=tickfontsize
    )


    return Plots.plot(p,h1, layout = Plots.grid(2,1, heights=[0.4, 0.6]))
end

"""
    plot_clusters_tsne(matr, clusters;
                        d=2,
                        pca_dims=0,
                        perp=20,
                        n_iter=1000,
                        distance=Distances.euclidean,
                        names=false,
                        fontsize=10,
                        markersize=5
                        )
    Perform t-SNE in dimension "d" on the COLUMNS of "matr" according to distance "distance", and plot, coloring the points according to clusters in "cluster".
    Returns a plot object.

    "clusters" should be an array of arrays (clusters).Each cluster contains the numbers of nodes (columns) belonging to that cluster.

    The colors are assigned to clusters according to the "distingishable_colors" procedure (see Colors.jl). Points that are not included in "clusters" are colored gray.

    Use "names" to specify point labels.
    Use "d", "pca_dims", "perp", "n_iter" to specify the parameters of t-SNE (see TSne.jl)
    Use "fontsize" and "markersize" to adjust the corresponding parameters of the plot.
"""
function plot_clusters_tsne(matr, clusters; d=2, pca_dims=0, perp=20, n_iter=1000,  distance=Distances.euclidean, names=false, fontsize=10, markersize=5)
    @assert (d==2) || (d==3)

    Y = tsne(matr', d, pca_dims,n_iter, perp, progress=false, distance=distance)
    not_clustered = setdiff(1:size(matr,2), union(clusters...))
    if d==2
        tsne_plot = plot_clusters(Y, clusters, not_clustered, names=names, fontsize=fontsize, markersize=markersize)
    elseif d==3
        tsne_plot = plot_clusters3d(Y, clusters, not_clustered, names=names, fontsize=fontsize, markersize=markersize)
    end
    return tsne_plot
end




"""
The following function is a shortened version of plot_clusters_tsne
    All it does is computing the 2-dimensional tsne ..

"""
function clusters_tsne(matr, clusters;    pca_dims=0, perp=20, n_iter=1000,distance=Distances.euclidean)
Y = tsne(matr', d, pca_dims,n_iter, perp, progress=false, distance=distance);
return Y
end




"""
    plot_clusters(Y, clusters;
                        names=false,
                        fontsize=10,
                        markersize=5
                        )
    Plot the points in "Y" in 2d, coloring the points according to clusters in "cluster", and coloring the point in "not_clustered" gray.
    "Y" should be a matrix with two columns, so that the k'th row containing the coordinates of

    "clusters" should be an array of arrays (clusters). Each cluster contains the numbers of nodes (columns) belonging to that cluster

    The colors are assigned to clusters according to the "distingishable_colors" procedure (see Colors.jl)
    Use "names" to specify point labels.

    Use "fontsize" and "markersize" to adjust the corresponding parameters of the plot.

"""


function plot_clusters(Y, clusters, not_clustered; names=false, fontsize=7,markersize=5)
    p = plot()
    cols = distinguishable_colors(length(clusters)+1,Color(RGB(1,1,1)))[2:end]
    for (i, clust) in enumerate(clusters)
        point_array = Y[clust,:]
        p = scatter!(point_array[:,1],
            point_array[:,2],
            color=cols[i],
            markerstrokewidth=0,
            markersize=markersize,
            label="Cluster $i",
            grid=false,
            series_annotations = names!=false ? text.(names[clust], fontsize, :bottom, cols[i]) : fill("", length(clust))
        )
    end
    p = scatter!(Y[not_clustered,1],Y[not_clustered,2],
        color=:grey90,
        markerstrokewidth=0,
        markersize=markersize,
        label="Not clustered",
        legend=:best,
        series_annotations = names!=false ? text.(names[not_clustered], fontsize, :bottom, :grey85) : fill("", length(not_clustered))
    )
    return p
end


"""
    plot_clusters_3d(Y, clusters;
                        names=false,
                        fontsize=10,
                        markersize=5
                        )
    Plot the points in "Y" in 3d, coloring the points according to clusters in "cluster", and coloring the point in "not_clustered" gray.
    "Y" should be a matrix with 3 columns, so that the k'th row containing the coordinates of

    "clusters" should be an array of arrays (clusters). Each cluster contains the numbers of nodes (columns) belonging to that cluster

    The colors are assigned to clusters according to the "distingishable_colors" procedure (see Colors.jl)
    Use "names" to specify point labels.

    Use "fontsize" and "markersize" to adjust the corresponding parameters of the plot.

"""


function plot_clusters3d(Y, clusters, not_clustered; names=false, fontsize=7,markersize=2)
    p = plot()
    cols = distinguishable_colors(length(clusters)+1,Color(RGB(1,1,1)))[2:end]
    for (i, clust) in enumerate(clusters)
        point_array = Y[clust,:,:]
        p = scatter!(point_array[:,1],
            point_array[:,2],
            point_array[:,3],
            color=cols[i],
            markerstrokewidth=0,
            markersize=markersize,
            label="Cluster $i",
            grid=false,
            series_annotations = names!=false ? text.(names[clust], fontsize, :bottom, cols[i]) : fill("", length(clust))
        )
    end
    p = scatter!(Y[not_clustered,1],Y[not_clustered,2], Y[not_clustered,3],
        color=:grey90,
        markerstrokewidth=0,
        markersize=markersize,
        label="Not clustered",
        legend=:best,
        series_annotations = names!=false ? text.(names[not_clustered], fontsize, :bottom, :grey85) : fill("", length(not_clustered))
    )
    return p
end
"""
find_glomeruli_gradient_matrices combines the individual glomeruli descriptor features in the order
that the array of indices idx dictates, into one matrix that can be used to plot a heatmap
of the responses of all the glomeruli to the descriptors
"""
#function find_glomeruli_gradient_matrices(masks::Array{Any,2},idx::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true},glom_vs_descr_matrix::Array{Float64,2})
function find_glomeruli_gradient_matrices(masks::Array{Any,2},idx::Vector{Int64},glom_vs_descr_matrix::Array{Float64,2})

    animal_1_glomeruli_gradients=Array{Array{Float64,2},1}(undef,length(idx))
    for l in 1:length(idx)
        animal_1_glom=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                        #animal_1_glom[i,j]=max(animal_1_glom[i,j],glom_vs_descr_matrix[k,idx[l]],0.0000001)
                        animal_1_glom[i,j]=max(animal_1_glom[i,j],glom_vs_descr_matrix[k,idx[l]]-0.05*findmax(glom_vs_descr_matrix[:,idx[l]])[1],0.005)
#                        animal_1_glom[i,j]=max(animal_1_glom[i,j],glom_vs_descr_matrix[k,idx[l]]-2*std(glom_vs_descr_matrix[:,idx[l]]),0.0000001)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(animal_1_glom)
        animal_1_glom=animal_1_glom[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        animal_1_glomeruli_gradients[l]=animal_1_glom
    end
    #need to find min windrow of matrix first
    return animal_1_glomeruli_gradients
end

"""
classification_of_glomeruli, classifies glomeruli into class 1 or 2, makes masks to plot
the classes
"""

function glomeruli_accuracies_to_plot(masks::Array{Any,2},accuracies::Array{Float64,1})
    accuracies_mask=fill(0.0,size(masks[1]))
    for i in 1:size(accuracies_mask,1)
        for j in 1:size(accuracies_mask,2)
            for k in 1:length(accuracies)
                img=masks[k]
                if img[i,j]!=0.0
                    accuracies_mask[i,j]=max(accuracies_mask[i,j],accuracies[k])
                end
            end
        end
    end
    (min_x,max_x,min_y,max_y)=window_matr(accuracies_mask)
    accuracies_mask=accuracies_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
    #odor_matrix[l]=odor_mask
    return accuracies_mask
end



function class_2_of_glomeruli_gradients(masks::Array{Any,2},idx::Vector{Int64},class_1_gloms::Vector{Int64},class_2_gloms::Vector{Int64},response_matrix::AbstractArray{Float64,2},thresh::Float64)
    odor_matrix=Array{Array{Float64,2},1}(undef,length(idx))
    for l in 1:length(idx)
        odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(class_2_gloms)
                    img=masks[class_2_gloms[k]]
                    if img[i,j]!=0.0
                        #odor_mask[i,j]=max(odor_mask[i,j],2.0)
                         odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,idx[l]]-thresh*findmax(response_matrix[:,idx[l]])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
                for k in class_1_gloms
                    img=masks[k]
                    if img[i,j]!=0.0
                        odor_mask[i,j]=max(0.005,odor_mask[i,j])#max(1.0,odor_mask[i,j])
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*findmax(response_matrix[:,l])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(odor_mask)
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end


function class_1_of_glomeruli_gradients(masks::Array{Any,2},idx::Vector{Int64},class_1_gloms::Vector{Int64},class_2_gloms::Vector{Int64},response_matrix::AbstractArray{Float64,2},thresh::Float64)
    odor_matrix=Array{Array{Float64,2},1}(undef,length(idx))
    for l in 1:length(idx)
        odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(class_1_gloms)
                    img=masks[class_1_gloms[k]]
                    if img[i,j]!=0.0
                        #odor_mask[i,j]=max(odor_mask[i,j],2.0)
                         odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,idx[l]]-thresh*findmax(response_matrix[:,idx[l]])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
                for k in class_2_gloms
                    img=masks[k]
                    if img[i,j]!=0.0
                        odor_mask[i,j]=max(0.005,odor_mask[i,j])#max(1.0,odor_mask[i,j])
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*findmax(response_matrix[:,l])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(odor_mask)
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

function classification_of_glomeruli(masks::Array{Any,2},class_1_gloms::Vector{Int64},class_2_gloms::Vector{Int64},response_matrix::AbstractArray{Float64,2})
    odor_matrix=Array{Array{Float64,2},1}(undef,size(response_matrix,2))
    for l in 1:size(response_matrix,2)
        odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in class_1_gloms
                    img=masks[k]
                    if img[i,j]!=0.0
                        odor_mask[i,j]=max(odor_mask[i,j],2.0,0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*findmax(response_matrix[:,l])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
                for k in class_2_gloms
                    img=masks[k]
                    if img[i,j]!=0.0
                        odor_mask[i,j]=max(1.0,odor_mask[i,j],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*findmax(response_matrix[:,l])[1],0.005)
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[odors,l]),0.005)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(odor_mask)
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

"""
odor_response_matrices, makes masks of the given response
matrix to be plotted.
"""

function odor_response_matrices(masks::Array{Any,2},response_matrix::AbstractArray{Float64,2},thresh::Float64)
    odor_matrix=Array{Array{Float64,2},1}(undef,size(response_matrix,2))
    for l in 1:size(response_matrix,2)
        odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(masks[1],1)
            for j in 1:size(masks[1],2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                        # odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*findmax(response_matrix[:,l])[1],0.005)
                         odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l]-thresh*std(response_matrix[:,l]),0.005)
                    end
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(odor_mask)
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

function shaps_on_bulb_matrices(masks::Array{Any,2},response_matrix::AbstractArray{Float64,2})
    odor_matrix=Array{Array{Float64,2},1}(undef,size(response_matrix,2))
    for l in 1:size(response_matrix,2)
        odor_mask=fill(0.0,size(masks[1],1),size(masks[1],2))
        for i in 1:size(odor_mask,1)
            for j in 1:size(odor_mask,2)
                for k in 1:length(masks)
                    img=masks[k]
                    if img[i,j]!=0.0
                         #odor_mask[i,j]=max(odor_mask[i,j],response_matrix[k,l])
                         odor_mask[i,j]=response_matrix[k,l]
                    end
                #println("$k")
                end
            end
        end
        (min_x,max_x,min_y,max_y)=window_matr(odor_mask)
        odor_mask=odor_mask[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
        odor_matrix[l]=odor_mask
    end
    #need to find min windrow of matrix first
    return odor_matrix
end

function high_vs_low_quadrants_on_bulb_matrices(masks::Array{Any,1},quadrants::Array{Int64,1})
    quadrants_matrix=fill(0.0,size(masks[1]))
    for i in 1:size(quadrants_matrix,1)
        for j in 1:size(quadrants_matrix,2)
            for k in 1:length(masks)
                img=masks[k]
                if img[i,j]!=0.0
                    quadrants_matrix[i,j]=quadrants[k]
                end
            end
        end
    end
    (min_x,max_x,min_y,max_y)=window_matr(quadrants_matrix)
    quadrants_matrix=quadrants_matrix[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
    return quadrants_matrix
end


"""
    plot_glomeruli_clusters(mask, clusters)

        Plot glomeruli according to "mask" and color them according to "clusters".

        "Mask" should be an array containing an image/matrix for each glomeruli,
        such that the matrix has 1's in the pixel positions of that glomerulus.
        "clusters" should be an array of arrays (clusters).
        Each cluster contains the numbers of nodes (columns) belonging to that cluster

        The colors are assigned to clusters according to the "distingishable_colors" procedure (see Colors.jl)
"""
#[RGB(1,1,1),RGB(0,0,0)]
function plot_glomeruli_clusters(mask, clusters::Array{Array{Int64,1},1})
    lim_offset=10;
    #Ms=map(x->combine(mask[x]...),clusters)
    Ms=map(x->combine(mask[x]...,imsize=256),clusters)
    not_clustered=setdiff(1:length(mask),union(clusters...))
    #custom_cgrad=[:white,:red,:green,:yellow,:purple,:blue,:cyan,:orange,:pink,:green2]#pom21
    custom_cgrad=[:red,:green,:yellow,:purple,:blue,:cyan,:pink,:white]#cact
   # custom_cgrad=[:red,:green,:yellow,:purple,:blue,:cyan,:pink,:white,:orange,:brown,:lightsalmon]#cact,no_amines
    #custom_cgrad = distinguishable_colors(length(clusters)+1)#,[RGB(1,1,1), RGB(0,0,0)])#[1:end]
    #push!(custom_cgrad, RGB(0.9, 0.9, 0.9))
    #push!(custom_cgrad, :white)
    M_new = fill(NaN,size(Ms[1],1), size(Ms[2],2))
    M_not_cl = combine(mask[not_clustered]...,imsize=256)
    M_new[findall(x->x==1.0,M_not_cl)] .= length(clusters)+1
    for (j,M) in enumerate(Ms)
        M_new[findall(x->x==1.0,M)] .= j
        #M_new[findall(x->x==0.0,M)].= NaN
    end
    #mask_matr = zeros(256,256)
    (min_x,max_x,min_y,max_y) = window(mask)
    M_new=M_new[(min_y-10):(max_y+10),(min_x-10):(max_x+10)]
    p1 = heatmap(M_new, color=palette(custom_cgrad,length(custom_cgrad)),background=:black,
        #title="Glom clusters",
        #xlim=(min_x-lim_offset, max_x+lim_offset),
        #ylim=(min_y-lim_offset, max_y+lim_offset),
        axis=false,
        yaxis=false,
        xticks=[],
        yticks=[],
        yflip=false,
        colorbar=false,aspectratio=1,right_margin = 10Plots.mm)
    return p1
end

#=
function plot_glomeruli_clusters(mask, clusters::Array{Array{Int64,1},1})
    lim_offset=10;
    Ms = map(x->combine(mask[x]...), clusters)
    not_clustered = setdiff(1:length(mask), union(clusters...))
    custom_cgrad = distinguishable_colors(length(clusters)+1, Color(RGB(1,1,1)))[1:end]
    push!(custom_cgrad, RGB(0.9, 0.9, 0.9))
    M_new = fill(0.0, size(Ms[1],1), size(Ms[2],2))
    M_not_cl = combine(mask[not_clustered]...)
    M_new[find(x->x==1.0,M_not_cl)] = length(clusters)+1
    for (j,M) in enumerate(Ms)
        M_new[find(x->x==1.0,M)] .= j
    end
    mask_matr = zeros(256,256)
    (min_x,max_x,min_y,max_y) = window(mask)
    p1 = heatmap(M_new, c=ColorGradient(custom_cgrad),
        xlim= (min_x-lim_offset, max_x+lim_offset),
        ylim=(min_y-lim_offset, max_y+lim_offset),
        yflip=true,
        colorbar=true)
    return p1
end
=#
######################### Helper functions #######################

function pca(matr, d; transposed=true)
    if transposed
        ready_for_pca = transpose(matr)
    else
        ready_for_pca = matr
    end
    M = fit(PCA, ready_for_pca, maxoutdim=d, pratio=1.0)
    comps = transform(M, ready_for_pca)
    return comps
end

function total_heights(hc)
    merges = VERSION>= v"0.7.0" ? hc.merges : hc.merge
    heights = VERSION>= v"0.7.0" ? hc.heights : hc.height

    total_height = zeros(size(merges,1))
    for n=1:size(merges,1)
        if (merges[n,1] < 0) && (merges[n,2] < 0)
            total_height[n] = heights[n]
        elseif (merges[n,1] > 0) && (merges[n,2] > 0)
            total_height[n] = max(total_height[merges[n,1]],total_height[merges[n,2]]) + heights[n]
        else
            pos = max(merges[n,1], merges[n,2])
            total_height[n] = total_height[pos]  + heights[n]
        end
    end
    return total_height
end

function pick_cluster_inds(hc::Hclust, thresh::Float64)
    total_height = total_heights(hc)
    if VERSION>= v"0.7.0"
        cutoff_inds = findall(x->x<=thresh, total_height)
        cluster_trees = fill(0,size(hc.merges,1))
    else
        cutoff_inds = find(x->x<=thresh, total_height)
        cluster_trees = fill(0,size(hc.merge,1))
    end
    cluster_count = 0
    clusters = zero(hc.order)
    for i in cutoff_inds
        if VERSION>= v"0.7.0"
            i1=hc.merges[i,1]
            i2=hc.merges[i,2]
        else
            i1=hc.merge[i,1]
            i2=hc.merge[i,2]
        end
        if (i1 < 0) && (i2 < 0)
            cluster_count += 1
            cluster_trees[i]= cluster_count
            clusters[abs(i1)] = cluster_count
            clusters[abs(i2)] = cluster_count
        elseif (i1 > 0) && (i2 > 0)
            cluster_ind = min(cluster_trees[i1], cluster_trees[i2])
            cluster_repl = max(cluster_trees[i1], cluster_trees[i2])
            if VERSION>= v"0.7.0"
                clusters[findall(x->x==cluster_repl, clusters)] .= cluster_ind
                cluster_trees[findall(x->x==cluster_repl, cluster_trees)] .= cluster_ind
            else
                clusters[find(x->x==cluster_repl, clusters)] .= cluster_ind
                cluster_trees[find(x->x==cluster_repl, cluster_trees)] .= cluster_ind
            end
            cluster_trees[i] = cluster_ind
        else
            pos = max(i1,i2)
            neg = min(i1, i2)
            clusters[abs(neg)] = cluster_trees[pos]
            cluster_trees[i] = cluster_trees[pos]
        end
    end
    return clusters
end

function plot_regions(matrs...; imsize=256, colorscheme=:inferno, symm=false)
    num_of_matrs = length(matrs)
    matr = fill(NaN, imsize, imsize)
    for i=1:imsize
        for j=1:imsize
            for m=1:num_of_matrs
                img = matrs[m]
                if img[i,j] != 0.0
                    if symm
                        color = 10.0/num_of_matrs*(num_of_matrs-cld(m,2)+1)
                        matr[i,j] = iseven(m) ? color : -color
                    else
                        matr[i,j] = m+0.5
                    end
                end

            end
        end
    end
    return heatmap(matr, c=colorscheme,
        legend=false,
        colorbar=true,
        grid=false,
        xticks=false,
        yticks=false,
        clim=(0,num_of_matrs)
    )
end

#function plot_regions(mask, clusts; imsize=256, colorscheme=:rainbow, symm=false)
#    Ms = map(x->(mask[x]...), clusts)
#    return plot_regions(Ms..., colorscheme=colorscheme, symm=symm, imsize=imsize)

#end

"""
matr=combine(matrs...; fill_value=1.0, imsize=256)
this takes a list of matrices/images  of size imsize x imsize and then
goes through them all and places the value of fill_value to the appropriate pixel of matr
for every non-zero  pixel  in any of the received matrices
"""

function combine_1(matrs...; fill_value=1.0, imsize=512)
    num_of_matrs = length(matrs)
    imsize_x=size(matrs[1],1)
    imsize_y=size(matrs[1],2)
    matr = fill(0.0, imsize_x, imsize_y)
    for i=1:imsize_x
        for j=1:imsize_y
            for m=1:num_of_matrs
                img = matrs[m]
                if img[i,j] != 0.0
                    if img[i,j]==1.0
                        matr[i,j] = 1.0
                    else
                        matr[i,j]=2.0
                    end
                end
            end
        end
    end
    return matr
end


function combine(matrs...; fill_value=1.0, imsize=256)
    num_of_matrs = length(matrs)
    matr = fill(0.0, imsize, imsize)
    for i=1:imsize
        for j=1:imsize
            for m=1:num_of_matrs
                img = matrs[m]
                if img[i,j] != 0.0
                    matr[i,j] = fill_value
                end
            end
        end
    end
    return matr
end

function k_nearest_neighbors(node, dist_matr; k=1)
    dist_to_nodes = dist_matr[node,:]
    neighbors = sortperm(dist_to_nodes)
    all_but = neighbors[setdiff(1:size(dist_matr,1),node)]
    return all_but[1:k]
end

function knn_average_distance(node, dist_matr; k=1)
    return mean(dist_matr[node, k_nearest_neighbors(node, dist_matr; k=k)])
end

function average_distance(nodes, dist_matr)
    submatr = dist_matr[nodes, nodes]
    n = length(nodes)
    return sum(submatr)/(n*(n-1))
end

function average_distance(node, nodes, dist_matr)
    return mean(dist_matr[node, setdiff(nodes, node)])
end


"""
    select_descriptors(clusters, descr_matr)

Select a positive descriptor with the highest variance in the cluster, or discard the cluster if it doesn't contain positive descriptors
Returns an array of descriptor indices

"""
function select_descriptors(clusters::Array{Array{Int64,1},1}, descr_matr::Array{Float64,2})
    good_feats=[]
    for clust in clusters
        max_var = 0
        max_var_ind = 0
        for c in clust
            v =VERSION>= v"0.7.0" ? Statistics.var(descr_matr[:, c])   : var(descr_matr[:, c])  ;
            if (v >= max_var) && all(descr_matr[:, c].>=0)
                max_var = v
                max_var_ind = c
            end
        end
        if max_var_ind > 0
            push!(good_feats,max_var_ind)
        end
    end
    return good_feats
end # function select_descriptors

"""
make_descr_by_gloms_matrix(responses, descriptors)

    Creates a matrix of sensitivity of a glomerulus to an odor.
    "responses" should contain the response matrix with rows being glomeruli, and columns being stimuli
    "descriptors" should containg the descriptor matrix with rows being odors, and columns being descriptors

    Returns a matrix with rows -- glomeruli, columns -- descriptors

"""
function make_descr_by_gloms_matrix(responses, descriptors)
    descr_by_gloms = fill(0.0, size(responses,1), size(descriptors,2))
    for descr = 1:size(descriptors,2)
        for glom = 1:size(responses, 1)
                descr_by_gloms[glom, descr] =  descr_value(descr,descriptors,glom, responses)
            end
        end
    return descr_by_gloms
end # make_descr_by_gloms_matrix

#########################

function resp_odor_profile(x,y,resp_matr)
    return resp_matr[y,x,:]
end

function resp_odor_profile(x,y,mask, resp_matr)
    gloms = select_glom(x,y, mask)
    return [mean(resp_matr[gloms, i]) for i=1:size(resp_matr,2)]
end

function resp_odor_profile(gloms, resp_matr)
    return [mean(resp_matr[gloms, i]) for i=1:size(resp_matr,2)]
end

function stochastify(matr)
    M = copy(matr)
    for col=1:size(M,2)
        if sum(M[:,col]) != 0
            M[:,col] = M[:,col] / sum(M[:,col])
        else
            M[:,col] = 0
        end
    end
    return M
end

function descr_value(descr, descr_matr, resp_prof)
    val = 0.0
#    descr_matr_s = stochastify(descr_matr)
    val = dot(descr_matr[:, descr], resp_prof)

    return val
end

function descr_value(descr,descr_matr,gloms, resp_matr)
    resp_prof = resp_odor_profile(gloms, resp_matr)
    return descr_value(descr, descr_matr, resp_prof)
end

"""
The following function chooses "meaningful" descriptors
good_descr_vals, good_descr_names=SelectSubsetOfDescriptors(descr_vals_norm);

"""
function SelectSubsetOfDescriptors(descr_vals_norm::Matrix{Float64}):: Tuple{Array{Float64,2},Array{String,1}}
    #specify distance
    descr_distance_metric = Distances.euclidean
    # compute descr distances
    # Note that the rows of descr_vals_norm are the odors, and the colums are the normalized values of descriptors
    descr_distances= distance_matrix(descr_vals_norm, descr_distance_metric)
    #specify linkage criterion, i.e. how to compute distance between clusters
    #:average - average distance between all pts, :complete -- max distance between pts in two clusters, :single - min distance
    descr_linkage = :average
    #cluster
    descr_clustering_obj = hclust(descr_distances, descr_linkage)
    #plot the resulting clusters
    #descr_dend = plot_dendrogram_and_matrix(descr_clustering_obj, descr_vals_norm, tickfontsize=7,title= "Descriptor clustering")
    #make plot big
    #descr_dend = plot!(; size=(1000,500))
    #display(descr_dend);

    #################  This chooses descriptors according to some criterium, based on clustering
    descr_height =  22.50 # 20.0
    descr_min_cluster = 1
    descr_clusters = pick_clusters(descr_clustering_obj, descr_height, min_cluster=descr_min_cluster);
    #select the descriptor indices
    good_descriptors = select_descriptors(descr_clusters, descr_vals_norm);
    N_good_descriptors=length(good_descriptors);
    # make a new matrix containing only the good descriptors
    good_descr_vals = descr_vals_norm[:, good_descriptors];
    good_descr_names=String.(descr_names[good_descriptors])
    println("The $N_good_descriptors chosen descriptors are"); println.(good_descr_names);
    ##########################
    return good_descr_vals,good_descr_names
end

println("done initializing");
