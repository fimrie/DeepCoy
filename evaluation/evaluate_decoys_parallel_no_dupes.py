#!/usr/bin/env python

import os, csv

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import decoy_utils

import sascorer

from joblib import Parallel, delayed

### Settings ###
#dataset = "dekois"
#num_cand_dec_per_act = 100
#num_dec_per_act = 30
#n_cores = 15

#dataset = "ALL"
dataset = "dude_orig"
num_cand_dec_per_act = 100
num_dec_per_act = 50
n_cores = 15
min_active_size = 10

###   End    ###

# Data source
#file_loc = '/data/pegasus/imrie/decoy-analysis/DeepCoy-DEKOIS-raw/' # DEKOIS FINAL
#output_loc = '/data/pegasus/imrie/decoy-analysis/DeepCoy-DEKOIS-output-div-test-multi/' # DEKOIS FINAL
file_loc = '/data/pegasus/imrie/decoy-analysis/DeepCoy-DUDE-raw/' # DUDE FINAL
#output_loc = '/data/pegasus/imrie/decoy-analysis/DeepCoy-DUDE-output_no_dupes/' # DUDE FINAL
output_loc = '/data/pegasus/imrie/decoy-analysis/DeepCoy-DUDE-output_no_dupes_dude_orig_props/' # DUDE orig props-
#output_loc = '/data/pegasus/imrie/decoy-analysis/dummy_dir/' # DUDE FINAL

# Data files
res_files = [f for f in os.listdir(file_loc)]
res_files = sorted(res_files)
#res_files = ['dude-target-1081-generated.txt'] # TODO

# Declare metric variables
columns = ['File name', 'Orig num actives', 'Num actives', 'Num generated mols', 'Num unique gen mols', '% macrocycles', 
           'Actives % < 5 SA score', 'Actives % < 4 SA score', 'Actives % < 3 SA score',
           'Decoys % < 5 SA score', 'Decoys % < 4 SA score', 'Decoys % < 3 SA score',
           'Actives mean SA score', 'Actives std dev SA score', 'Decoys mean SA score', 'Decoys std dev SA score',
           'Max Doppleganger score', 'DG score % > 0.25', 'DG score % > 0.30',
           'Inital LADS v2 mean', 'Initial LADS v2 std',
           'Num with successes', 'Avg num successes (unique)', 'Max num successes (unique)', 'Min num successes (unique)',
           'Num actives with success', 'Num decoys', 'Num decoys per active', 'Num conf fails', 'Num dupes wanted',
           'Avg prop diff (all)', 'Std prop diff (all)', 'Min prop diff (all)', 'Max prop diff (all)',
           'Avg prop diff (sel)', 'Std prop diff (sel)', 'Min prop diff (sel)', 'Max prop diff (sel)',
           'Avg active-active sim', 'Avg active-decoy sim', 'Avg decoy-decoy sim',
           'Morgan FPs: 1nn', 'Morgan FPs: RF', 'MACCS FPs: 1nn', 'MACCS FPs: RF',
           'MUV Props: 1nn', 'MUV Props: RF', 'DEKOIS Props: 1nn', 'DEKOIS Props: RF', 
           'DUD-E Props: 1nn', 'DUD-E Props: RF', 'All Props: 1nn', 'All Props: RF',
           'DOE score', 'DOE score DUD-E', 'LADs v1 mean', 'LADs v1 std', 'LADs v2 mean', 'LADs v2 std', 'Doppelganger score mean', 'Doppelganger score std', 'Doppelganger score max',
           'Actives QED mean', 'Actives QED std', 'Decoys QED mean', 'Decoys QED std',
           'Actives SA mean', 'Actives SA std', 'Decoys SA mean', 'Decoys SA std',]

# Output to CSV file
with open(output_loc+'results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)

# Worker function
def get_decoys_from_input_file(f, file_loc=file_loc, output_loc=output_loc, dataset=dataset, num_cand_dec_per_act=num_cand_dec_per_act, num_dec_per_act=num_dec_per_act):
    print(f)
    dec_results = [f]
    # Load data
    if dataset == "ALL" or dataset == "dude_orig":
        data = decoy_utils.read_paired_dude_file(file_loc+f)
    else:
        data = decoy_utils.read_paired_file(file_loc+f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    seen = set()
    data = [d for d in data if Chem.MolFromSmiles(d[0]).GetNumHeavyAtoms()>min_active_size]
    unique_data = [x for x in data if not (tuple(x) in seen or seen.add(tuple(x)))]
    #TODO
    in_mols = [d[0] for d in data]
    gen_mols = [d[1] for d in data]
    #in_mols = [d[0] for d in unique_data]
    #gen_mols = [d[1] for d in unique_data]
    dec_results.extend([len(set(in_mols)), len(data), len(unique_data)])
    print(f, len(set(in_mols)), len(data), len(unique_data))

    # Calculate number of macrocycles generated
    macros = [decoy_utils.num_macro(smi) for smi in gen_mols]
    frac_macros = len([m for m in macros if m > 0]) / len(macros)*100
    dec_results.append(frac_macros)

    # Calculate properties of in_mols and gen_mols
    used = set([])
    in_mols_set = [x for x in in_mols if x not in used and (used.add(x) or True)]

    if dataset == "dude":
        in_props_temp = decoy_utils.calc_dataset_extended_props(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_extended_props(gen_mols, verbose=True)
    elif dataset == "dekois":
        in_props_temp = decoy_utils.calc_dataset_dekois_props(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_dekois_props(gen_mols, verbose=True)
    elif dataset == "MUV":
        in_props_temp = decoy_utils.calc_dataset_MUV_props(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_MUV_props(gen_mols, verbose=True)
    elif dataset == "ALL":
        in_props_temp = decoy_utils.calc_dataset_all_props(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_all_props(gen_mols, verbose=True)
    elif dataset == "dude_orig":
        in_props_temp = decoy_utils.calc_dataset_props(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props(gen_mols, verbose=True)
    else:
        print("Incorrect dataset")
        exit()
    in_mols_temp = list(in_mols_set) # copy
    in_props = []
    for i, smi in enumerate(in_mols):
        in_props.append(in_props_temp[in_mols_temp.index(smi)])

    in_basic_temp = decoy_utils.calc_basic_dataset_props(in_mols_set, verbose=True)
    in_mols_temp = list(in_mols_set) # copy
    in_basic = []
    for i, smi in enumerate(in_mols):
        in_basic.append(in_basic_temp[in_mols_temp.index(smi)])

    gen_basic_props = decoy_utils.calc_basic_dataset_props(gen_mols, verbose=True)

    # Scale properties based on in_mols props
    active_props_scaled_all = []
    decoy_props_scaled_all = []

    active_min_all = []
    active_max_all = []
    active_scale_all = []

    active_props = in_props_temp
    # Exclude errors from min/max calc
    act_prop = np.array(active_props)

    active_maxes = np.amax(act_prop, axis=0)
    active_mins = np.amin(act_prop, axis=0)

    active_max_all.append(active_maxes)
    active_min_all.append(active_mins)

    scale = []
    for (a_max, a_min) in zip(active_maxes,active_mins):
        if a_max != a_min:
            scale.append(a_max - a_min)
        else:
            scale.append(a_min)
    scale = np.array(scale)
    scale[scale == 0.0] = 1.0
    active_scale_all.append(scale)
    active_props_scaled = (active_props - active_mins) / scale
    active_props_scaled_all.append(active_props_scaled)

    # Calc SA scores
    in_sa_temp = [sascorer.calculateScore(Chem.MolFromSmiles(mol)) for mol in set(in_mols)]
    in_mols_temp = list(set(in_mols))
    in_sa = []
    for i, smi in enumerate(in_mols):
        in_sa.append(in_sa_temp[in_mols_temp.index(smi)])
    gen_sa_props = [sascorer.calculateScore(Chem.MolFromSmiles(mol)) for mol in gen_mols]

    # Add to results
    dec_results.append(len([sa for sa in in_sa if sa < 5]) / len(in_sa))
    dec_results.append(len([sa for sa in in_sa if sa < 4]) / len(in_sa))
    dec_results.append(len([sa for sa in in_sa if sa < 3]) / len(in_sa))
    dec_results.append(len([sa for sa in gen_sa_props if sa < 5]) / len(gen_sa_props))
    dec_results.append(len([sa for sa in gen_sa_props if sa < 4]) / len(gen_sa_props))
    dec_results.append(len([sa for sa in gen_sa_props if sa < 3]) / len(gen_sa_props))

    dec_results.extend([np.mean(in_sa), np.std(in_sa)])
    dec_results.extend([np.mean(gen_sa_props), np.std(gen_sa_props)])

    in_fps = []
    for i, mol in enumerate(in_mols):
        in_fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol),2,nBits=1024))
    gen_fps = []
    for i, mol in enumerate(gen_mols):
        gen_fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol),2,nBits=1024))

    # Calc DG scores
    dg_scores, dg_ids = decoy_utils.dg_score_rev(set(in_mols), gen_mols)
    dec_results.append(max(dg_scores))
    dec_results.append(len([dg for dg in dg_scores if dg>0.25])/len(dg_scores)*100)
    dec_results.append(len([dg for dg in dg_scores if dg>0.30])/len(dg_scores)*100)

    # Calc LADS scores
    lads_scores = decoy_utils.lads_score_v2(set(in_mols), gen_mols)
    dec_results.extend([np.mean(lads_scores), np.std(lads_scores)])
    #TODO
    #print(list(set(in_mols)))
    #exit()
    # Calculate dictionary of results
    results_dict = {}
    for i in range(len(in_mols)):
        # Get scaling
        in_props_scaled = (in_props[i] - active_min_all) / active_scale_all
        gen_props_scaled = (gen_props[i] - active_min_all) / active_scale_all
        prop_diff = np.linalg.norm(np.array(in_props_scaled)-np.array(gen_props_scaled))

        # Look at basic props
        basic_diff = np.sum(abs(np.array(in_basic[i])-np.array(gen_basic_props[i])))

        if in_mols[i] in results_dict:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            #if sim == 1:
            #    sim = 0 # To stop identical mols
            results_dict[in_mols[i]].append([in_mols[i], gen_mols[i], in_props[i], gen_props[i], prop_diff, sim, basic_diff, abs(gen_sa_props[i]-in_sa[i]), dg_scores[i], lads_scores[i]])
        else:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            #if sim == 1:
            #    sim = 0 # To stop identical mols
            results_dict[in_mols[i]] = [ [in_mols[i], gen_mols[i], in_props[i], gen_props[i], prop_diff, sim, basic_diff, abs(gen_sa_props[i]-in_sa[i]), dg_scores[i], lads_scores[i]] ]

    # Get decoy matches
    max_idx_cmpd = 1000
    results = []
    results_success_only = []
    sorted_mols_success = []
    for key in results_dict:
        # Set initial criteria
        threshold = 1.0 # Dice 0.6 = 0.4286 Tanimoto
        prop_max_diff = 5
        max_basic_diff = 3#0.9#1.1
        max_sa_diff = 1.51#10.51
        max_dg_score = 0.35
        max_lads_score = 5 # Set to equal prop_max_diff
        while True:
            count_success = sum([i[5]<threshold and i[4]<prop_max_diff and i[6]<max_basic_diff and i[7]<max_sa_diff and i[8]<max_dg_score and i[9]<max_lads_score for i in results_dict[key][0:max_idx_cmpd]])
            # Adjust criteria if not enough successes
            if count_success < num_cand_dec_per_act:
                print("Widening search", count_success)
                #threshold *= 0.9
                prop_max_diff *= 1.1
                max_basic_diff += 1
                max_sa_diff *= 1.1
                max_dg_score *= 1.1
                max_lads_score *= 1.1
            else:
                print("Reached threshold", count_success)
                # NEW - COMBO OF LADS AND PROP SIM
                sorted_mols_success.append([(i[0], i[1], i[4], i[9], i[4]+i[9]) for i in sorted(results_dict[key][0:max_idx_cmpd], key=lambda i: i[4]+i[9], reverse=False)   
                    if i[5]<threshold and i[4]<prop_max_diff and i[6]<max_basic_diff and i[7]<max_sa_diff and i[8]<max_dg_score and i[9]<max_lads_score]) # Changed div score to match is_success
                assert count_success == len(sorted_mols_success[-1]) # TODO
                #print(sorted_mols_success[0][:20])
                #return None # TODO
                # OLD - ONLY PROP SIM
                #sorted_mols_success.append([(i[0], i[1]) for i in sorted(results_dict[key][0:max_idx_cmpd], key=lambda i: i[4], reverse=False) 
                #    if i[5]>threshold and i[4]<prop_max_diff and i[6]<max_basic_diff and i[7]<max_sa_diff and i[8]<max_dg_score] and i[9]<max_lads_score ) # Changed div score to match is_success
                break
    # Append results
    dec_results.append(len(sorted_mols_success))
    dec_results.append(np.mean([len(set(i)) for i in sorted_mols_success]))
    dec_results.append(max([len(set(i)) for i in sorted_mols_success]))
    dec_results.append(min([len(set(i)) for i in sorted_mols_success]))

    # Choose decoys
    active_mols_gen = []
    decoy_mols_gen = []

    embed_fails = 0
    dupes_wanted = 0
    for act_res in sorted_mols_success:
        count = 0
        # GREEDY CHOOSING
        for ent in act_res:
            # Check can gen conformer
            m = Chem.AddHs(Chem.MolFromSmiles(ent[1]))
            if AllChem.EmbedMolecule(m, randomSeed=42) != -1 and ent[1] not in decoy_mols_gen: # Check conf and not a decoy for another ligand
                decoy_mols_gen.append(ent[1])
                count +=1
                if count >= num_dec_per_act:
                    break
            elif ent[1] in decoy_mols_gen:
                dupes_wanted +=1
            else:
                embed_fails += 1
        #decoy_mols_gen.extend([ent[1] for ent in act_res[0:num_dec_per_act]]) # GREDDY CHOOSING
        # DIVERSITY CHOOSING USING MAXMIN ON TANIMOTO SIM OF FCFP_6
        #cand_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi[1]),3,useFeatures=True, nBits=1024) for smi in act_res[:num_cand_dec_per_act]] # Roughly FCFP_6
        #def distij(i,j,fps=cand_fps):
        #    return 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        #picker = MaxMinPicker()
        #pickIndices = picker.LazyPick(distij,len(cand_fps),num_dec_per_act,seed=42)
        #decoy_mols_gen.extend([act_res[idx][1] for idx in pickIndices])
        active_mols_gen.append(act_res[0][0])
    #print(decoy_mols_gen) # TODO
    #return None # TODO
    dec_results.extend([len(active_mols_gen), len(decoy_mols_gen), len(decoy_mols_gen)/num_dec_per_act, embed_fails, dupes_wanted])

    # Calc props for chosen decoys
    if dataset == "dude":
        actives_feat = decoy_utils.calc_dataset_extended_props(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_extended_props(decoy_mols_gen, verbose=True)
    elif dataset == "dekois":
        actives_feat = decoy_utils.calc_dataset_dekois_props(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_dekois_props(decoy_mols_gen, verbose=True)
    elif dataset == "MUV":
        actives_feat = decoy_utils.calc_dataset_MUV_props(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_MUV_props(decoy_mols_gen, verbose=True)
    elif dataset == "ALL":
        actives_feat = decoy_utils.calc_dataset_all_props(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_all_props(decoy_mols_gen, verbose=True)
    elif dataset == "dude_orig":
        actives_feat = decoy_utils.calc_dataset_props(active_mols_gen)
        decoys_feat = decoy_utils.calc_dataset_props(decoy_mols_gen)
    else:
        print("Incorrect dataset")
        exit()

    in_props_scaled = [(active_feat - active_min_all) / active_scale_all for active_feat in actives_feat]
    gen_props_scaled = [(gen_feat - active_min_all) / active_scale_all for gen_feat in decoys_feat]
    prop_diffs = []
    for i in range(len(gen_props_scaled)):
        prop_diffs.append(np.linalg.norm(gen_props_scaled[i]-in_props_scaled[i//num_dec_per_act]))
    
    all_props_diffs = []
    for key in results_dict:
        all_props_diffs.extend([i[4] for i in results_dict[key]])

    dec_results.extend([np.average(all_props_diffs), np.std(all_props_diffs), min(all_props_diffs), max(all_props_diffs)])
    dec_results.extend([np.average(prop_diffs), np.std(prop_diffs), min(prop_diffs), max(prop_diffs)])

    # Assess active-decoy set
    
    # Mol similarity
    # Active-active
    actives_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=1024) for smi in active_mols_gen]
    sims = []
    for i in range(len(actives_fps)):
        for j in range(i+1, len(actives_fps)):
            sims.append(DataStructs.TanimotoSimilarity(actives_fps[i], actives_fps[j]))
    dec_results.append(np.mean(sims))
    # Active-decoy
    decoys_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=1024) for smi in decoy_mols_gen]
    sims = []
    for i in range(len(actives_fps)):
        for j in range(i*num_dec_per_act, (i+1)*num_dec_per_act):
            sims.append(DataStructs.TanimotoSimilarity(actives_fps[i], decoys_fps[j]))
    dec_results.append(np.mean(sims))
    # Decoy-decoy
    sims = []
    for i in range(0, len(decoys_fps), num_dec_per_act):
        for j in range(i+1, i+num_dec_per_act):
            sims.append(DataStructs.TanimotoSimilarity(decoys_fps[i], decoys_fps[j]))
    dec_results.append(np.mean(sims))
    
    # Property matching performance
    dec_results.extend(decoy_utils.calc_perf_morgan_fps(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))
    dec_results.extend(decoy_utils.calc_perf_maccs_fps(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))
    dec_results.extend(decoy_utils.calc_perf_muv_props(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))
    dec_results.extend(decoy_utils.calc_perf_dekois_props(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))
    dec_results.extend(decoy_utils.calc_perf_dude_props(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))
    dec_results.extend(decoy_utils.calc_perf_all_props(active_mols_gen, decoy_mols_gen, split=0.8, num_dec_per_act=num_dec_per_act))

    # For DUD-E props DOE score
    actives_feat_dude = decoy_utils.calc_dataset_props(active_mols_gen)
    decoys_feat_dude = decoy_utils.calc_dataset_props(decoy_mols_gen)

    # DEKOIS paper metrics (LADS, DOE, Doppelganger score)
    dec_results.append(decoy_utils.doe_score(actives_feat, decoys_feat))
    dec_results.append(decoy_utils.doe_score(actives_feat_dude, decoys_feat_dude))
    lads_scores = decoy_utils.lads_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(lads_scores), np.std(lads_scores)])
    lads_scores = decoy_utils.lads_score_v2(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(lads_scores), np.std(lads_scores)])
    print("MEAN LADS v2", np.mean(lads_scores))
    dg_scores, dg_ids = decoy_utils.dg_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(dg_scores), np.std(dg_scores), max(dg_scores)])
    print("MEAN DG", np.mean(dg_scores))
    # QED property results
    actives_qed = []
    decoys_qed = []
    for smi in active_mols_gen:
        actives_qed.append(Chem.QED.qed(Chem.MolFromSmiles(smi)))
    for smi in decoy_mols_gen:
        decoys_qed.append(Chem.QED.qed(Chem.MolFromSmiles(smi)))

    dec_results.extend([np.mean(actives_qed), np.std(actives_qed)])
    dec_results.extend([np.mean(decoys_qed), np.std(decoys_qed)])

    # SA score results
    actives_sa = []
    decoys_sa = []
    for smi in active_mols_gen:
        actives_sa.append(sascorer.calculateScore(Chem.MolFromSmiles(smi)))
    for smi in decoy_mols_gen:
        decoys_sa.append(sascorer.calculateScore(Chem.MolFromSmiles(smi)))

    dec_results.extend([np.mean(actives_sa), np.std(actives_sa)])
    dec_results.extend([np.mean(decoys_sa), np.std(decoys_sa)])

    # Save intermediate performance results in unique file
    with open(output_loc+'results_'+f+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(dec_results)

    # Save decoy mols
    with open(output_loc+f, 'w') as outfile:
        for i, mol in enumerate(decoy_mols_gen):
            outfile.write(mol + ' ' + active_mols_gen[i//num_dec_per_act] + '\n')

    return dec_results

    
# Iterate over targets
with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
    results = parallel(delayed(get_decoys_from_input_file)(f) for f in res_files)

for dec_results in results:
    # Write performance results
    with open(output_loc+'results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(dec_results)
    
