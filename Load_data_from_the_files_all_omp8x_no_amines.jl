println("Loading data  ...");

"""
Specify the files
"""
########################
#filenames for files contating the data, WITHOUT the ".mat" extension
my_path= joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines");
filenames= ["omp83Lhigh_ORdata_dFregistered_checked_aminswitch112622","omp84high_ORdata_dFregistered_checked_noamines112622","omp85high_ORdata_mwsel_dFregistered_checked_noamines112622"];

#relevant odors. Currently the datasets contain 61 odors, where the last two are controls
#exclude the amines, which are odors 27-33
#relevant_odors = vcat(1:26,34:59);
#relevant_odors= vcat(1:59);
#filename for the file with odor names
odors_filename = "master_odorlist_omp83-85subset.txt";

#relevant strings in the odor name file
relevant_odor_names = vcat(2:27,33:60);
#relevant_odor_names=vcat(2:60);

#filenames for files with glomeruli masks
#mask_filenames = ["omp83_ROI","omp85_ROI"];
#mask_filenames=[""];
mask_filenames= ["omp83L_high_MapsData_mwsel_aminswitch","omp84highMapsData_mwsels_1_2sec_100222","omp85highMapsData1_2sec_mwsel_163_112022"];
#filename for the descriptor files
#descr_filename = "results_0402.csv";


#############################################

"""
Load the data from the files
"""

#array contating the response matrices
resp_matr =[];# These are the response matrices for the data

#array contating the indices of nonzero rows in the original datasets.
#Notice that in the response matrices these rows are removed
nonzero_rows = [];# these are the indices of the non-zero rows (glomeruli)

#animal_name_indices=[8,3];
#animal_name_indices=[1];
#load data
for i in 1:length(filenames)
#for file in filenames
    # this was added last moment:
    #   my_path= "/Users/vui1/Dropbox/to_share/"
    #
    # (resp, nz_rows) = load_data(file) #
    #(resp, nz_rows) = load_data_no_amines(filenames[i],animal_name_indices[i],my_path)
    (resp, nz_rows) = load_data_no_amines_new(filenames[i],my_path)

    #resp = resp[:, relevant_odors]
    push!(resp_matr, resp)
    push!(nonzero_rows, nz_rows)
end



#array containing the masks for glomeruli
# each element is a 256x256 mask
masks=[];
for i in 1:length(filenames)
    file_name=joinpath(my_path,mask_filenames[i])*".mat"
    file = matopen(file_name)
    data=read(file)["MapsData"]["roi"]["mask"]
    close(file)
    data=data[nonzero_rows[i]]
    push!(masks,data)
end


#odor_names = readdlm(joinpath(my_path,odors_filename))
#odor_names_str = ["$(odor_names[i,2]) $(odor_names[i,3]) $(odor_names[i,4]) $(odor_names[i,5])" for i in relevant_odor_names]
#odors = odor_names[relevant_odor_names,2];

#number of odors
#n_odors = length(relevant_odors)
#array of the number of glomeruli in each dataset
#n_gloms = [size(resp,1) for resp in resp_matr];

println("done loading ");
#####################################################