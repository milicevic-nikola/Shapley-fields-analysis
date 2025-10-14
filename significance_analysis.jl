using JLD, Simplicial,Plots;
Consistent_sig=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/decorr09_descrs_on_bulb_significant_dowker_complexes_absolute_sum.jld"),"const");

Increasing_sig=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/decorr09_descrs_on_bulb_significant_dowker_complexes_absolute_sum.jld"),"increasing");

Decreasing_sig=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/decorr09_descrs_on_bulb_significant_dowker_complexes_absolute_sum.jld"),"decreasing");

Sig_high=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/decorr09_descrs_on_bulb_significant_dowker_complexes_absolute_sum.jld"),"sig_high");

Sig_low=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/decorr09_descrs_on_bulb_significant_dowker_complexes_absolute_sum.jld"),"sig_low");


using CSV, DataFrames;

descr_file=joinpath(homedir(),"Dropbox/Nikola-Vladimir/chemsense_our_odors.csv");
CSV.File(descr_file);
df = DataFrame(CSV.File(descr_file));
df.SMILES

descr_vals_norm=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"descr_vals_norm");
descr_names=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"descr_names");

#=
omp83L_sig_high_pos=Sig_high[1]#[:,1:99]
omp83L_sig_low_pos=Sig_low[1]#[:,1:99]
combined_omp83L_sig_pos=hcat(omp83L_sig_high_pos,omp83L_sig_low_pos)
combined_cor_omp83L_sig_pos=cor(combined_omp83L_sig_pos)
heatmap(combined_cor_omp83L_sig_pos[1:85,86:170].>0.9)
histogram(vec(combined_cor_omp83L_sig_pos[1:85,86:170]),grid=false,legend=false)
idx=findall(combined_cor_omp83L_sig_pos[1:85,86:170].>0.7)
=#
glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"glom_pos");
#glomeruli_positions=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines/decorr09_descrs_segmented_high_and_low_conc.jld"),"glom_pos");
binary_responses=load(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors/decorr09_descrs_segmented_high_and_low_conc.jld"),"binary_responses");

const_dowker_omp83L=SimplicialComplex(Consistent_sig[1]);#');
const_dowker_omp83R=SimplicialComplex(Consistent_sig[2]);#');

inc_dowker_omp83L=SimplicialComplex(Increasing_sig[1]);#');
inc_dowker_omp83R=SimplicialComplex(Increasing_sig[2]);#');

#dec_dowker_omp83L=SimplicialComplex(Decreasing_sig[1]);#');
#dec_dowker_omp83R=SimplicialComplex(Decreasing_sig[2]);#');

const_omp83L_facets=collect.(const_dowker_omp83L.facets[length.(const_dowker_omp83L.facets).>3])
const_omp83R_facets=collect.(const_dowker_omp83R.facets[length.(const_dowker_omp83R.facets).>4])

inc_omp83L_facets=collect.(inc_dowker_omp83L.facets[length.(inc_dowker_omp83L.facets).>10])
inc_omp83R_facets=collect.(inc_dowker_omp83R.facets[length.(inc_dowker_omp83R.facets).>11])

#dec_omp83L_facets=collect.(dec_dowker_omp83L.facets[length.(dec_dowker_omp83L.facets).>6])
#dec_omp83R_facets=collect.(dec_dowker_omp83R.facets[length.(dec_dowker_omp83R.facets).>7])

include(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/Useful_functions.jl"));
#const_featuresL=findmax(sum(Consistent_sig[1],dims=1)[1,:],1:4)[1]

const_glomsL=findmax(sum(Consistent_sig[1],dims=2)[:,1],1:13)[1]

gloms_max_resps_idxs=findmax(sum(binary_responses[1],dims=2)[:,1],1:10)[1];

#const_featuresL=findmax(sum(Consistent_sig[1],dims=2)[:,1],1:7)[1]
const_featuresR=findmax(sum(Consistent_sig[2],dims=1)[1,:],1:4)[1]

incr_featuresL=findmax(sum(Increasing_sig[1],dims=1)[1,:],1:6)[1]
incr_featuresR=findmax(sum(Increasing_sig[2],dims=1)[1,:],1:6)[1]

decr_featuresL=findmax(sum(Decreasing_sig[1],dims=1)[1,:],1:5)[1]
decr_featuresR=findmax(sum(Decreasing_sig[2],dims=1)[1,:],1:5)[1]

function find_response_idxs(gloms::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true},bin_resp::BitMatrix)
    idxs=Array{Any,1}(undef,length(gloms))
    for i in 1:length(idxs)
        idxs[i]=findall(bin_resp[gloms[i],:].==1)
    end
    return idxs
end

odor_resp_idxs_high=find_response_idxs(const_glomsL,binary_responses[1])
odor_resp_idxs_low=find_response_idxs(const_glomsL,binary_responses[3])

#const_glomsL
#const_omp83L_facets

function make_gloms_analysis_data_frame(smiles::Vector{String},odor_resp_idx_high::Array{Int64,1},odor_resp_idx_low::Array{Int64,1},descr_vals_norm::Array{Float64,2},dowker_facets::Vector{Int16})
    response_idxs_high=fill(0,size(descr_vals_norm,1))
    response_idxs_low=fill(0,size(descr_vals_norm,1))
    response_idxs_high[odor_resp_idx_high].=1
    response_idxs_low[odor_resp_idx_low].=1
    features_values=descr_vals_norm[:,dowker_facets]
    data_frame=DataFrame(hcat(smiles,response_idxs_high,response_idxs_low, features_values),:auto)
end

gloms_to_send=[6,10,11,12];

omp83_df_glom1=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[1],odor_resp_idxs_low[1],descr_vals_norm,const_omp83L_facets[13])
rename!(omp83_df_glom1,[:SMILES,:RespHigh,:RespLow,:ring,:CO,:nitrogen,:AnyRingSize6,:oxygen,:NCCCC,:CHHH])
#fieldnames(typeof(omp83_df))
omp83_df_glom2=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[2],odor_resp_idxs_low[2],descr_vals_norm,const_omp83L_facets[12])
rename!(omp83_df_glom2,[:SMILES,:RespHigh,:RespLow,:NCCC,:SaturatedOrAromaticCarbonOnlyRingSize5,:CCCCC,:NCCCC,:AmbCCCCC,:NCCCCC])

omp83_df_glom3=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[3],odor_resp_idxs_low[3],descr_vals_norm,const_omp83L_facets[11]);
rename!(omp83_df_glom3,[:SMILES,:RespHigh,:RespLow,:CCCO,:ring,:CO,:aldehyde,:nitrogen,:OCCC]);

omp83_df_glom4=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[4],odor_resp_idxs_low[4],descr_vals_norm,const_omp83L_facets[10]);
rename!(omp83_df_glom4,[:SMILES,:RespHigh,:RespLow,:OCCC,:OCCCCCCC,:NCCC,:OCCCN,:CNCC]);

omp83_df_glom5=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[5],odor_resp_idxs_low[5],descr_vals_norm,const_omp83L_facets[9]);
rename!(omp83_df_glom5,[:SMILES,:RespHigh,:RespLow,:nitrogen,:oxygen,:NCCCC,:NCCCCC,:CHHH]);

omp83_df_glom6=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[6],odor_resp_idxs_low[6],descr_vals_norm,const_omp83L_facets[8]);
rename!(omp83_df_glom6,[:SMILES,:RespHigh,:RespLow,:TertiaryAmine,:nitrogen,:oxygen,Symbol("N-C-C-C-C"),Symbol("N-C-C-C-C-C")]);

omp83_df_glom7=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[7],odor_resp_idxs_low[7],descr_vals_norm,const_omp83L_facets[7]);
rename!(omp83_df_glom7,[:SMILES,:RespHigh,:RespLow,:furanoid,:FourC,:CCCHO,:OH,:TwoO]);

omp83_df_glom8=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[8],odor_resp_idxs_low[8],descr_vals_norm,const_omp83L_facets[6]);
rename!(omp83_df_glom8,[:SMILES,:RespHigh,:RespLow,:OrthoSubsRings,:nitrogen,:RingSize6,:NCCCC]);

omp83_df_glom9=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[9],odor_resp_idxs_low[9],descr_vals_norm,const_omp83L_facets[5]);
rename!(omp83_df_glom9,[:SMILES,:RespHigh,:RespLow,:CCCO,:OCCCO,:OCCCN,:ParaSubNotFusedRing]);

omp83_df_glom10=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[10],odor_resp_idxs_low[10],descr_vals_norm,const_omp83L_facets[4]);
rename!(omp83_df_glom10,[:SMILES,:RespHigh,:RespLow,Symbol("C(~O)(~O)"),Symbol("C-C=C-C"),Symbol("O=C-C-C"),:CarboxylicAcid]);

omp83_df_glom11=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[11],odor_resp_idxs_low[11],descr_vals_norm,const_omp83L_facets[3]);
rename!(omp83_df_glom11,[:SMILES,:RespHigh,:RespLow,Symbol("C(~O)(~O)"),Symbol("C=O"),:nitrogen,Symbol("O=C-C-C")]);

omp83_df_glom12=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[12],odor_resp_idxs_low[12],descr_vals_norm,const_omp83L_facets[2]);
rename!(omp83_df_glom12,[:SMILES,:RespHigh,:RespLow,:ring,:nitrogen,Symbol("O=C-C-C"),:CarboxylicAcid]);

omp83_df_glom13=make_gloms_analysis_data_frame(df.SMILES,odor_resp_idxs_high[13],odor_resp_idxs_low[13],descr_vals_norm,const_omp83L_facets[1]);
rename!(omp83_df_glom13,[:SMILES,:RespHigh,:RespLow,:NCCC,:DifferentNCCC,:CCCCC,:pyridine]);

CSV.write(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/omp83L_glom1.csv"), data = omp83_df_glom1);

Plots.scatter(glomeruli_positions[1][:,1],glomeruli_positions[1][:,2],
                yflip=true,grid=false,title="omp83L",ticks=false,axis=false,label=false)

Plots.scatter!([glomeruli_positions[1][const_glomsL[6],1]],
                [glomeruli_positions[1][const_glomsL[6],2]],
                yflip=true,legend=true,label="glom6",grid=false, markershape=:rect,markersize=15)#,

Plots.scatter!([glomeruli_positions[1][const_glomsL[10],1]],
                [glomeruli_positions[1][const_glomsL[10],2]],
                yflip=true,legend=true,label="glom10",grid=false, markershape=:rect,markersize=15)#,


Plots.scatter!([glomeruli_positions[1][const_glomsL[11],1]],
                [glomeruli_positions[1][const_glomsL[11],2]],
                yflip=true,legend=true,label="glom11",grid=false, markershape=:rect,markersize=15)#,


Plots.scatter!([glomeruli_positions[1][const_glomsL[12],1]],
                [glomeruli_positions[1][const_glomsL[12],2]],
                yflip=true,legend=true,label="glom12",grid=false, markershape=:rect,markersize=15)#,

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","glom_examples.png"));


Plots.scatter!([glomeruli_positions[1][const_glomsL[1],1]],
                [glomeruli_positions[1][const_glomsL[1],2]],
                yflip=true,legend=true,label="glom1",grid=false, markershape=:rect,markersize=15)#,

heatmap(binary_responses[1])



function find_response_rates_for_glom_feature_pair(gloms::Vector{Int16},feat_number::Int64,bin_resp::BitMatrix,descrs::Matrix{Float64})
    n_gloms=length(gloms)
    perc_of_stim_having_feature=Array{Float64,1}(undef,n_gloms)
    for i in 1:n_gloms
        activating_stimuli=findall(bin_resp[gloms[i],:].!=0)
        perc_of_stim_having_feature[i]=sum(descrs[activating_stimuli,feat_number])/length(activating_stimuli)
    end
    return perc_of_stim_having_feature
end

A=round.(find_response_rates_for_glom_feature_pair(inc_omp83L_facets[3],39,binary_responses[1],descr_vals_norm),digits=2)
B=round.(find_response_rates_for_glom_feature_pair(inc_omp83L_facets[3],39,binary_responses[3],descr_vals_norm),digits=2)

resp_A_idx=findall(A.>0.5)
resp_B_idx=findall(B.>0.5)


text_A=vcat(string.(A))
text_B=vcat(string.(B))
for i in 1:length(text_A)
    if text_A[i]=="1.0"
        text_A[i]="1"
    elseif text_A[i]=="0.0"
        text_A[i]="0"
    end
    if text_B[i]=="1.0"
        text_B[i]="1"
    elseif text_B[i]=="0.0"
        text_B[i]="0"
    end
end

text=string.(text_A,"/",text_B);

Plots.scatter(glomeruli_positions[1][:,1],glomeruli_positions[1][:,2],
                yflip=false,grid=false,title="omp84 C(~O)(~O)",ticks=false,axis=false,label=false)

Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[3][resp_A_idx]],
                glomeruli_positions[1][:,2][inc_omp83L_facets[3][resp_A_idx]],
                yflip=false,legend=true,label="presence",grid=false, markershape=:rect,markersize=15)#,

Plots.annotate!(glomeruli_positions[1][:,1][inc_omp83L_facets[3][resp_A_idx]],
                glomeruli_positions[1][:,2][inc_omp83L_facets[3][resp_A_idx]],
                yflip=false,text[resp_A_idx],annotationfontsize=8)

Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[3][setdiff(1:end,resp_A_idx)]],
                glomeruli_positions[1][:,2][inc_omp83L_facets[3][setdiff(1:end,resp_A_idx)]],
                yflip=false,legend=true,label="absence",grid=false, markershape=:rect,markersize=15)#,

Plots.annotate!(glomeruli_positions[1][:,1][inc_omp83L_facets[3]][setdiff(1:end,resp_A_idx)],
                glomeruli_positions[1][:,2][inc_omp83L_facets[3]][setdiff(1:end,resp_A_idx)],
                yflip=false,text[setdiff(1:end,resp_A_idx)],annotationfontsize=8)


#Plots.scatter!(glomeruli_positions[1][:,1][const_omp83L_facets[4]],glomeruli_positions[1][:,2][const_omp83L_facets[4]],yflip=false,grid=false,label= "O=C-C-C", markershape= :octagon,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_inc_omp84_COO.png"));



C=round.(find_response_rates_for_glom_feature_pair(inc_omp83R_facets[6],39,binary_responses[2],descr_vals_norm),digits=2)
D=round.(find_response_rates_for_glom_feature_pair(inc_omp83R_facets[6],39,binary_responses[4],descr_vals_norm),digits=2)

resp_C_idx=findall(C.>0.5)
resp_D_idx=findall(D.>0.5)


text_C=vcat(string.(C))
text_D=vcat(string.(D))
for i in 1:length(text_C)
    if text_C[i]=="1.0"
        text_C[i]="1"
    elseif text_C[i]=="0.0"
        text_C[i]="0"
    end
    if text_D[i]=="1.0"
        text_D[i]="1"
    elseif text_D[i]=="0.0"
        text_D[i]="0"
    end
end

text=string.(text_C,"/",text_D)


Plots.scatter(glomeruli_positions[2][:,1],glomeruli_positions[2][:,2],
                yflip=false,grid=false,title="omp85 C(~O)(~O)",ticks=false,axis=false,label=false)

Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[6][resp_C_idx]],
                glomeruli_positions[2][:,2][inc_omp83R_facets[6][resp_C_idx]],
                yflip=false,legend=true,label="presence",grid=false, markershape=:rect,markersize=15)#,

Plots.annotate!(glomeruli_positions[2][:,1][inc_omp83R_facets[6][resp_C_idx]],
                glomeruli_positions[2][:,2][inc_omp83R_facets[6][resp_C_idx]],
                yflip=false,text[resp_C_idx],annotationfontsize=8)

Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[6][setdiff(1:end,resp_C_idx)]],
                glomeruli_positions[2][:,2][inc_omp83R_facets[6][setdiff(1:end,resp_C_idx)]],
                yflip=false,legend=true,label="absence",grid=false, markershape=:rect,markersize=15)#,

Plots.annotate!(glomeruli_positions[2][:,1][inc_omp83R_facets[6]][setdiff(1:end,resp_C_idx)],
                glomeruli_positions[2][:,2][inc_omp83R_facets[6]][setdiff(1:end,resp_C_idx)],
                 yflip=false,text[setdiff(1:end,resp_C_idx)],annotationfontsize=8)


Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_inc_omp85_COO.png"));



Plots.scatter(glomeruli_positions[2][:,1],glomeruli_positions[2][:,2],grid=false,legend=:outerright,title="omp85",yflip=false,ticks=false,axis=false,label=false)
Plots.scatter!(glomeruli_positions[2][:,1][const_omp83R_facets[5]],glomeruli_positions[2][:,2][const_omp83R_facets[5]],grid=false,yflip=false,label= "ring", markershape= :rect,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][const_omp83R_facets[7]],glomeruli_positions[2][:,2][const_omp83R_facets[7]],grid=false,yflip=false,label= "O=C-C-C", markershape=:octagon,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][const_omp83R_facets[4]],glomeruli_positions[2][:,2][const_omp83R_facets[4]],grid=false,yflip=false,label="C(~C)(~C)(~O)", markershape= :diamond,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][const_omp83R_facets[6]],glomeruli_positions[2][:,2][const_omp83R_facets[6]],grid=false,yflip=false,label="O=C-C-C", markershape= :star4,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_const_omp85_absolute_sum.png"));

Plots.scatter(glomeruli_positions[1][:,1],glomeruli_positions[1][:,2],yflip=true,grid=false,title="omp83L",ticks=false,axis=false,label=false)
Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[5]],glomeruli_positions[1][:,2][inc_omp83L_facets[5]],
                                yflip=true,grid=false,label= "nitrogen", markershape= :rect,markersize=15)
Plots.annotate!(glomeruli_positions[1][:,1][inc_omp83L_facets[5]],glomeruli_positions[1][:,2][inc_omp83L_facets[5]],yflip=true,text,annotationfontsize=8)

Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[5]],glomeruli_positions[1][:,2][inc_omp83L_facets[5]],yflip=false,grid=false,label= ">=2O", markershape=:octagon,markersize=15)
Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[4]],glomeruli_positions[1][:,2][inc_omp83L_facets[4]],yflip=false,grid=false,label= "O=C-C-C", markershape= :diamond,markersize=15)
Plots.scatter!(glomeruli_positions[1][:,1][inc_omp83L_facets[3]],glomeruli_positions[1][:,2][inc_omp83L_facets[3]],yflip=false,grid=false,label= "C(~O)(~O)", markershape=:star4  ,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/all_odors","decorr09_dowker_incr_omp83L_nitrogen.png"));

Plots.scatter(glomeruli_positions[2][:,1],glomeruli_positions[2][:,2],yflip=false,grid=false,legend=:outerright,title="omp85",ticks=false,axis=false,label=false)
Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[5]],glomeruli_positions[2][:,2][inc_omp83R_facets[5]],yflip=false,grid=false,label= "ring", markershape= :rect,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[7]],glomeruli_positions[2][:,2][inc_omp83R_facets[7]],yflip=false,grid=false,label= ">=2O", markershape=:octagon,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[2]],glomeruli_positions[2][:,2][inc_omp83R_facets[2]],yflip=false,grid=false,label= "O=C-C-C", markershape= :diamond,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][inc_omp83R_facets[6]],glomeruli_positions[2][:,2][inc_omp83R_facets[6]],yflip=false,grid=false,label= "C(~O)(~O)", markershape=:star4  ,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_incr_omp85_absolute_sum.png"));

Plots.scatter(glomeruli_positions[1][:,1],glomeruli_positions[1][:,2],yflip=true,grid=false,legend=:outerright,title="omp83 L",ticks=false,axis=false,label=false)
Plots.scatter!(glomeruli_positions[1][:,1][dec_omp83L_facets[7]],glomeruli_positions[1][:,2][dec_omp83L_facets[7]],yflip=true,grid=false,label= "alkene", markershape= :rect,markersize=15)
Plots.scatter!(glomeruli_positions[1][:,1][dec_omp83L_facets[6]],glomeruli_positions[1][:,2][dec_omp83L_facets[6]],yflip=true,grid=false,label= "aldehyde", markershape=:octagon,markersize=15)
Plots.scatter!(glomeruli_positions[1][:,1][dec_omp83L_facets[5]],glomeruli_positions[1][:,2][dec_omp83L_facets[5]],yflip=true,grid=false,label= "furan", markershape= :diamond,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_decr_omp83L_absolute_sum.png"));

Plots.scatter(glomeruli_positions[2][:,1],glomeruli_positions[2][:,2],yflip=false,grid=false,legend=:outerright,title="omp83 R",ticks=false,axis=false,label=false)
Plots.scatter!(glomeruli_positions[2][:,1][dec_omp83R_facets[8]],glomeruli_positions[2][:,2][dec_omp83R_facets[8]],yflip=false,grid=false,label= "alkene", markershape= :rect,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][dec_omp83R_facets[10]],glomeruli_positions[2][:,2][dec_omp83R_facets[10]],yflip=false,grid=false,label= "aldehyde", markershape=:octagon,markersize=15)
Plots.scatter!(glomeruli_positions[2][:,1][dec_omp83R_facets[4]],glomeruli_positions[2][:,2][dec_omp83R_facets[4]],yflip=false,grid=false,label= "S:C:C-[#1]", markershape= :diamond,markersize=15)

Plots.savefig(joinpath(homedir(),"Dropbox/Nikola-Vladimir/new_code_files/resp_data/no_amines","decorr09_dowker_decr_omp83R_absolute_sum.png"));
