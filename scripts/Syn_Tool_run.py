import argparse
import re
import subprocess
import pandas as pd
import numpy as np
from keras import models
from tokenizers import Tokenizer
from Transformer_layers import TransformerEncoder, PositionalEmbedding
import os
from glob import glob
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from scipy.spatial.distance import pdist, cosine
from Bio import Phylo
from io import StringIO
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
import pickle
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


# Run Mothur's RDP classification
# Input: input_fasta
# Output: RDP_pick.fasta, RDP_pick.taxonomy
# Return: None

def run_mothur_NB(input_fasta):
    
    '''
    Define a list of Mothur commands to execute. 
    These commands classify sequences using the reference database (Picocyanobacteria_ITS_database.fasta and Picocyanobacteria_ITS_database.tax), 
    remove sequences classified as 'unknown,' and rename the output files with the prefix 'RDP_pick.'
    '''
    commands = [
        f"mothur \"#classify.seqs(fasta={input_fasta}, reference=Picocyanobacteria_ITS_database.fasta, taxonomy=Picocyanobacteria_ITS_database.tax, cutoff=0);"
        f"remove.lineage(fasta={input_fasta}, taxonomy=current, taxon=unknown);"
        f"rename.file(fasta=current, taxonomy=current, prefix=RDP_pick)\""
    ]


    '''
    Execute each command in the list. 
    Use subprocess.run to execute the command and capture both the output and error messages.
    '''
    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        ''' 
        Check if the command executed successfully. 
        If successful, print a success message and the output; if failed, print the error message and exit the loop.
        '''
        if result.returncode == 0:
            print(f"Mothur command '{cmd}' ran successfully")
            print(result.stdout)
        else:
            print(f"Mothur command '{cmd}' failed to run")
            print(result.stderr)
            break


# Locate the RDP_pick.fasta file in the current directory
# Input: None
# Output: None
# Return: Path to RDP_pick.fasta

def find_fasta_file():
    
    fasta_files = glob("RDP_pick.fasta")

    ''' 
    Raise a FileNotFoundError if no RDP_pick.fasta file is found.
    '''
    if not fasta_files:
        raise FileNotFoundError("No RDP_pick.fasta file found after running Mothur.")

    ''' 
    Raise a RuntimeError if multiple RDP_pick.fasta files are found.
    '''
    if len(fasta_files) > 1:
        raise RuntimeError("Multiple RDP_pick.fasta files found. Ensure there is only one in the directory.")

    ''' 
    Print and return the path of the located RDP_pick.fasta file.
    '''
    print(f"Located fasta file: {fasta_files[0]}")
    return fasta_files[0]


# Convert a fasta-format file to a txt-format file
# Input: fasta-format file
# Output: None
# Return: txt-format data as a dictionary

def fasta_to_txt(input_file):
    data_txt = {"Sequence": [], "Seq_id": []}

    try:
        with open(input_file, 'r') as f_in:
            ''' 
            Read the fasta file line by line. 
            Lines starting with '>' are sequence IDs, and the following line contains the sequence. 
            Store sequences and IDs in the data_txt dictionary.
            '''
            for line in f_in:
                if line.startswith('>'):
                    label = line.split()[0][1:]  # Extract sequence ID
                    sequence = next(f_in).strip()  # Get the next line as the sequence
                    data_txt["Sequence"].append(sequence)  
                    data_txt["Seq_id"].append(label)  
        
        return data_txt

    except FileNotFoundError:
        ''' 
        Print an error message and raise an exception if the fasta file is not found.
        '''
        print(f"Fasta file not found: {input_file}")
        raise


# Classify sequences using a Transformer model
# Input: fasta-format file, tokenizer, model
# Output: Syn.fasta
# Return: DataFrame (columns: Seq_id, Sequence)

def transformer_classify_sequences(input_fasta, tokenizer, model):
    
    data = fasta_to_txt(input_fasta)
    data_sequences = data["Sequence"]
    sequence_id = data["Seq_id"]

    try:
        ''' 
        Initialize the tokenizer and tokenize the sequences.
        Load the tokenizer configuration from Transformer_BPE_tokenizer.1024.json.
        Catch exceptions if the tokenizer file is not found and return an error message.
        '''
        tokenizer = Tokenizer.from_file(tokenizer)
        tokenizer.enable_truncation(max_length=150)  
        tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=150)  
        data_sequences_tokens = tokenizer.encode_batch(data_sequences)  
        data_ids = np.array([i.ids for i in data_sequences_tokens])  
    except FileNotFoundError:
        print(f"Transformer tokenizer file not found: {tokenizer}")
        return

    try:
        ''' 
        Load the pre-trained Transformer model transformer_model.h5.
        Catch exceptions if the model file is not found and return an error message.
        '''
        loaded_model = models.load_model(model, custom_objects={"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding})
    except FileNotFoundError:
        print(f"Transformer model file not found: {model}")
        return

    ''' 
    Perform predictions using the loaded model. 
    Classify predictions greater than 0.5 as 1, otherwise classify as 0.
    '''
    predictions = loaded_model.predict(data_ids)
    predicted_labels = (predictions > 0.5).astype('int32').ravel()

    ''' 
    Combine prediction results with original sequences and sequence IDs.
    '''
    result_df = pd.DataFrame({'Seq_id': sequence_id, 'Sequence': data_sequences, 'Predicted Label': predicted_labels})

    ''' 
    Filter sequences with a predicted label of 0 and generate a FASTA file named Syn.fasta.
    '''
    Syn_df = result_df[result_df['Predicted Label'] == 0]
    Syn_df = Syn_df[['Seq_id', 'Sequence']]

    with open('Syn.fasta', 'w') as fasta_file:
        for seq_id, seq in zip(Syn_df['Seq_id'], Syn_df['Sequence']):
            fasta_file.write(f">{seq_id}\n{seq}\n")
    print("FASTA file 'Syn.fasta' has been generated, containing sequences with Predicted Label 0.")

    return Syn_df


def getKmers(sequence, size=6):
    ''' 
    Extract k-mer subsequences from a given sequence.
    
    Parameters:
        sequence: The input string sequence.
        size: The size of the k-mer, default is 6.
        
    Returns:
        A list of k-mer subsequences in lowercase.
    '''
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]



# Process sequence data using a CNN model for classification and feature extraction
# Input: DataFrame returned by transformer_classify_sequences (columns: Seq_id, Sequence), tokenizer, model, label_encoder
# Output: None
# Return: DataFrame (columns: 512 features, Clade, Sequence_ID, Source, Predicted_Probability)

def CNN_process(edata, tokenizer, model, label_encoder):
    
    ''' 
    Combine sequences from the database and edata, and create a label list.
    Label 0 represents database sequences, and label 1 represents edata sequences.
    '''
    database = pd.read_table("Syn_ITS_database.txt")
    combined_sequence = pd.concat([database['Sequence'], edata['Sequence']], axis=0)
    labels = [0] * len(database) + [1] * len(edata)

    ''' 
    Convert sequences to k-mer representations, tokenize sequences using the tokenizer, 
    and pad or truncate them to a fixed maximum length.
    '''
    try:
        with open(tokenizer, 'rb') as tokenizer_file:
            loaded_tokenizer = pickle.load(tokenizer_file)
    except FileNotFoundError:
        print(f"CNN tokenizer file not found: {tokenizer}")
        return
    
    max_length = 1024
    combined_sequence_word_2 = [' '.join(getKmers(seq)) for seq in combined_sequence]
    combine_sequence_token = loaded_tokenizer.texts_to_sequences(combined_sequence_word_2)
    pad_combine_sequence_token = pad_sequences(combine_sequence_token, maxlen=max_length, padding='post', truncating='post')

    ''' 
    Build an intermediate model to output both classification results and globally pooled features.
    Use the intermediate model to predict on the padded sequences.
    '''
    try:
        loaded_model = load_model(model)
    except FileNotFoundError:
        print(f"CNN model file not found: {model}")
        return
    
    intermediate_model = Model(inputs=loaded_model.input, outputs=[loaded_model.output, loaded_model.get_layer('global_max_pooling1d').output])
    classification_output, pooling_output = intermediate_model.predict(pad_combine_sequence_token)

    ''' 
    Decode the classification results by mapping model outputs to labels 
    and extract the highest prediction probability for each sequence.
    '''
    try:
        loaded_label_encoder = joblib.load(label_encoder)
    except FileNotFoundError:
        print(f"CNN label encoder file not found: {label_encoder}")
        return
    
    predicted_labels = np.argmax(classification_output, axis=1)
    predicted_labels = loaded_label_encoder.inverse_transform(predicted_labels)
    predicted_probabilities = np.max(classification_output, axis=1)

    ''' 
    Create a DataFrame to store pooled features, predicted labels, sequence IDs, 
    data sources, and prediction probabilities.
    '''
    combined_df = pd.DataFrame(pooling_output)
    combined_df['Clade'] = predicted_labels.tolist()
    combined_df['Sequence_ID'] = list(database['Accession_number']) + list(edata['Seq_id'])
    combined_df['Source'] = ['database'] * len(database) + ['edata'] * len(edata)
    combined_df['Predicted_Probability'] = predicted_probabilities.tolist()  # Add prediction probabilities

    return combined_df



def find_minimum_clade(tree, known_leaf_names):
    
    ''' 
    Find the smallest clade in the tree that contains all the specified known leaf names.
    
    Parameters:
        tree: A tree object, typically from the Biopython library.
        known_leaf_names: A list of leaf names that need to be contained within the clade.
        
    Returns:
        smallest_clade: The smallest clade object containing all known leaf names.
    '''
    clades_with_known = [clade for clade in tree.find_clades() if any(leaf.name in known_leaf_names for leaf in clade.get_terminals())]
    

    smallest_clade = min(
        (clade for clade in clades_with_known if all(name in [leaf.name for leaf in clade.get_terminals()] for name in known_leaf_names)), 
        key=lambda clade: len(clade.get_terminals()),  # Select the clade with the fewest terminals
        default=None  # Return None if no such clade is found
    )
    
    return smallest_clade, smallest_clade


def extract_sequences_from_clade(clade):
    ''' 
    Extract the names of all terminal (leaf) nodes in a given clade.
    
    Parameters:
        clade: A clade object representing a branch of the tree.
        
    Returns:
        A list containing the names of all terminal nodes in the clade.
    '''
    return [leaf.name for leaf in clade.get_terminals()]


def get_newick(node, newick, parentdist, leaf_names):
    ''' 
    Convert a tree structure into a Newick formatted string representation.
    
    Parameters:
        node: The current tree node being processed.
        newick: The partial Newick string constructed so far.
        parentdist: The distance from the parent node to the current node.
        leaf_names: A dictionary mapping node IDs to their corresponding leaf names.
        
    Returns:
        A complete Newick formatted string representing the tree structure.
    '''
    if node.is_leaf():
        return "%s:%f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        
        newick = get_newick(node.get_left(), newick, node.dist, leaf_names)
        newick = get_newick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        
        newick = "(%s" % (newick)
        return newick


def UPMGA_delineation_in_clade(group):
    
    features = group.drop(columns=['Clade', 'Sequence_ID', 'Source', 'Predicted_Probability']).values
    dist_matrix = pdist(features, metric='cosine')
    Z = linkage(dist_matrix, method='average')
    tree = to_tree(Z, rd=False)
    leaf_names = group['Sequence_ID'].tolist()
    newick = get_newick(tree, "", tree.dist, leaf_names).replace(");", ");\n")
    tree = Phylo.read(StringIO(newick), "newick")
    known_leaf_names = group.loc[group['Source'] == 'database', 'Sequence_ID'].tolist()
    minimum_clade, minimum_clade_node = find_minimum_clade(tree, known_leaf_names)
    known_sequences = extract_sequences_from_clade(minimum_clade)
    max_height = max(tree.distance(minimum_clade_node, leaf) for leaf in minimum_clade.get_terminals())
    clusters = fcluster(Z, t=max_height + 0.0001, criterion='distance')
    group['Cluster'] = clusters
    cluster_features_df = group.drop(columns=['Clade', 'Sequence_ID', 'Source', 'Predicted_Probability'])
    cluster_centroids = cluster_features_df.groupby('Cluster').mean().values
    min_clade_features = group[group['Sequence_ID'].isin(known_sequences)].drop(columns=['Clade', 'Sequence_ID', 'Source', 'Cluster', 'Predicted_Probability']).values
    min_clade_centroid = min_clade_features.mean(axis=0)
    cosine_distances = [cosine(min_clade_centroid, centroid) for centroid in cluster_centroids]
    
    return group, clusters, cosine_distances


def mark_group(group):
    
    group['Mark'] = 0 if any(source == 'database' for source in group['Source']) else 1
    
    return group


def classify_known_identify_unknown(combined_df, threshold=0.15):
    
    """
    Classifies known sequences and identifies potential novel sequences based on cosine distance 
    and clustering results within each clade.

    Parameters:
        combined_df: A pandas DataFrame containing sequence data, including columns such as 'Clade',
                     'Cluster', 'Sequence_ID', 'Source', 'Predicted_Probability', and cosine distances.
        threshold: A float value specifying the cosine distance threshold to identify novel sequences.

    Returns:
        potential_novel_sequences: A pandas Series containing the IDs of sequences identified as potential novel sequences.
        unclassified_df: A pandas DataFrame containing details of sequences identified as potential novel sequences.
        classified_df: A pandas DataFrame containing the classified sequences, including their clade and predicted probability.
    """
    clade_result_df = pd.DataFrame(columns=['Clade', 'Cluster', 'Cosine_Distance'])
    sequence_cluster_df = pd.DataFrame(columns=['Clade', 'Cluster', 'Sequence_ID', 'Source'])

    clade_groups = combined_df.groupby('Clade')
    
    for clade, group in clade_groups:
        # Process each clade and perform clustering using UPGMA
        # The function 'UPMGA_delineation_in_clade' is assumed to return:
        # - the processed group DataFrame
        # - a list of cluster assignments
        # - a list of cosine distances for each cluster
        group, clusters, cosine_distances = UPMGA_delineation_in_clade(group)
        
        for cluster_id, dist in enumerate(cosine_distances, 1):
            clade_result_df = pd.concat([clade_result_df, pd.DataFrame({
                'Clade': [clade],
                'Cluster': [cluster_id],
                'Cosine_Distance': [dist]
            })], ignore_index=True)
        
        temp_df = group.drop(columns=['Clade', 'Cluster', 'Sequence_ID', 'Source', 'Predicted_Probability'])
        temp_df['Clade'] = clade
        temp_df['Cluster'] = clusters
        temp_df['Sequence_ID'] = group['Sequence_ID'].values
        temp_df['Source'] = group['Source'].values
        sequence_cluster_df = pd.concat([sequence_cluster_df, temp_df], ignore_index=True)

    merge_columns = ['Clade', 'Cluster']
    merged_df = pd.merge(sequence_cluster_df, clade_result_df, on=merge_columns)
    merged_df = merged_df.groupby(merge_columns).apply(mark_group)
    
    filtered_df = merged_df[(merged_df['Mark'] == 1) & (merged_df['Cosine_Distance'] > threshold)]


    classified_df = merged_df[~((merged_df['Mark'] == 1) & 
                                 (merged_df['Cosine_Distance'] > threshold))]
    classified_df = classified_df[classified_df['Source'] == 'edata']
    classified_df = classified_df[['Sequence_ID', 'Clade']]
    

    probability_df = combined_df[['Sequence_ID', 'Predicted_Probability']]
    classified_df = pd.merge(classified_df, probability_df, on='Sequence_ID', how='inner')


    potential_novel_sequences = filtered_df['Sequence_ID']
    unclassified_df = filtered_df.drop(columns=['Clade', 'Cluster', 'Source', 'Cosine_Distance', 'Mark'])
    
    return potential_novel_sequences, unclassified_df, classified_df


def cluster_sequences(abundance_file, unclassified_df, eps_range=(0.075, 0.1, 0.005)):
    
    """
    Performs clustering on unclassified sequences using abundance data and a range of epsilon (eps) values 
    for the DBSCAN algorithm. Returns a DataFrame with clustering results.

    Parameters:
        abundance_file (str): Path to the file containing sequence abundance information.
                              Expected to be a tab-delimited file with columns ['Sequence_ID', 'abundance'].
        unclassified_df (DataFrame): A DataFrame of unclassified sequences to be clustered.
        eps_range (tuple): A range for the epsilon parameter of DBSCAN in the format (start, stop, step).

    Returns:
        result (DataFrame): A DataFrame containing sequence IDs, their assigned clusters (as clades), 
                            and a 'Predicted_Probability' column indicating "Novel".
    """
    abundance_data = pd.read_csv(abundance_file, sep='\t', header=None, names=['Sequence_ID', 'abundance'])
    df = pd.merge(unclassified_df, abundance_data, on='Sequence_ID', how='left')
    expanded_df = df.loc[df.index.repeat(df['abundance'])]
    dist_matrix = cosine_distances(expanded_df.drop(columns=['Sequence_ID', 'abundance'], errors='ignore'))
    best_eps, best_labels = None, None
    max_clusters = 0

    for eps in np.arange(*eps_range):
        labels = DBSCAN(eps=eps, min_samples=100, metric='precomputed').fit_predict(dist_matrix)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > max_clusters or (num_clusters == max_clusters and (best_eps is None or eps < best_eps)):
            max_clusters, best_eps, best_labels = num_clusters, eps, labels

    print(f'Max clusters: {max_clusters} at eps: {best_eps}')
    
    expanded_df['Cluster'] = best_labels
    result = expanded_df[['Cluster', 'Sequence_ID']].drop_duplicates(subset='Sequence_ID')
    result = result[result['Cluster'] != -1]
    result['Cluster'] = 'Novel_' + result['Cluster'].astype(str)
    result.columns = ['Clade' if col == 'Cluster' else col for col in result.columns]
    result['Predicted_Probability'] = 'Novel'
    
    return result


def main():
    
    """
    Main function to process Mothur outputs, classify sequences using Transformer and CNN models,
    identify novel sequences, and cluster unclassified sequences.

    Steps:
    1. Parse command-line arguments for input files.
    2. Run Mothur to preprocess sequence data.
    3. Classify sequences using a Transformer model.
    4. Use a CNN model to process classified sequences and combine results.
    5. Identify potential novel sequences and cluster unclassified sequences.
    6. Save all intermediate and final results to files.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Mothur commands and classify sequences using Transformer and CNN models")
    parser.add_argument('-fa', '--fasta', required=True, help='Fasta file output from Mothur processing')
    parser.add_argument('-a', '--abundance', required=True, help='Abundance file in txt format for novel cluster delineation')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Retrieve input file paths
    input_fasta = args.fasta
    input_abundance = args.abundance

    # Step 1: Run Mothur-based processing
    run_mothur_NB(input_fasta)
    
    # Step 2: Locate the resulting FASTA file after Mothur processing
    try:
        fasta_file = find_fasta_file()
    except FileNotFoundError as e:
        print(e)
        return  # Exit the program if the file is not found

    # Step 3: Classify sequences using the Transformer model
    transformer_tokenizer = 'Transformer_BPE_tokenizer.1024.json'
    transformer_model = 'Transformer_model.h5'
    Syn_df = transformer_classify_sequences(fasta_file, transformer_tokenizer, transformer_model)
    Syn_df.to_csv('Syn_df.csv', index=False, header=False)  # Save intermediate results

    # Step 4: Further process the classified sequences using a CNN model
    CNN_tokenizer = 'CNN_kmer_tokenizer.pkl'
    CNN_model = 'CNN_model.h5'
    CNN_label_encoder = 'CNN_label_encoder.pkl'
    combined_df = CNN_process(Syn_df, CNN_tokenizer, CNN_model, CNN_label_encoder)
    combined_df.to_csv('combined_df.csv', index=False)  # Save combined results

    # Step 5: Identify potential novel sequences and filter/classify sequences
    potential_novel_sequences, unclassified_df, classified_df = classify_known_identify_unknown(
        combined_df, threshold=0.15
    )

    # Step 6: Cluster unclassified sequences into novel groups
    result = cluster_sequences(input_abundance, unclassified_df)
    result.to_csv('result.csv', index=False)  # Save clustering results

    # Step 7: Combine classified and clustered results
    final_df = pd.concat([classified_df, result], ignore_index=True)
    final_df.to_csv('Syn_Tool_final_result.txt', sep='\t', index=False)  # Save final output


if __name__ == "__main__":
    main()
