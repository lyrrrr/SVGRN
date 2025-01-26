import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

def read_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df.values

# Function to perform clustering
def cluster_matrices(folder_path, n_clusters):
    filename_list = []
    all_GRN_array = []

    for file_name in os.listdir(folder_path):
        #print(file_name)
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            matrix = read_matrix(file_path)

            # Flatten the matrix for clustering
            matrix_flattened = matrix.flatten().reshape(1, -1)
            #print(np.squeeze(matrix_flattened).shape)
            all_GRN_array.append(np.squeeze(matrix_flattened))
            filename_list.append(file_name[:-4])

    all_GRN_array = np.array(all_GRN_array)
    print(all_GRN_array.shape)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(all_GRN_array)
    cluster_ids = kmeans.predict(all_GRN_array)
    # print(filename_list)
    # print(cluster_ids)

    cluster_dict = dict(zip(filename_list, cluster_ids))
    #print(cluster_dict)

    return cluster_dict


folder_name = "g110_c6k_n1"
gene_number = 110

############################
# Folder containing the CSV files
folder_path = f'in_sim/{folder_name}/cell_specific_GRN'  # Update this path
# Number of clusters
n_clusters = 4

# Perform clustering and save the results
cluster_dict = cluster_matrices(folder_path, n_clusters)

# Save the cluster IDs to a CSV file
output_path = f'in_sim/{folder_name}/GRN_cluster_ids.csv'  # Update this path
grn_cluster_ids_df = pd.DataFrame.from_dict(cluster_dict, orient='index', columns=['ClusterID']).to_csv(output_path)
# print(grn_cluster_ids_df.head(5))

# Load the dataframes
cell_loc_path = f'in_sim/{folder_name}/cell_loc.csv'   # cell 2k
grn_cluster_ids_path = f'in_sim/{folder_name}/GRN_cluster_ids.csv'

cell_loc_df = pd.read_csv(cell_loc_path, index_col=0)
grn_cluster_ids_df = pd.read_csv(grn_cluster_ids_path, index_col=0)

# Combine the dataframes using the index value
combined_df = cell_loc_df.join(grn_cluster_ids_df)

# Save the combined dataframe to a new CSV file
output_path = f'in_sim/{folder_name}/cell_loc_GRNcluster{n_clusters}.csv'
combined_df.to_csv(output_path)

# Display the combined dataframe
print(combined_df.head())



#########################################################
# get the average of all GRNs as GRN for the whole tissue
import os
import pandas as pd
import numpy as np

# Folder containing the CSV files

folder_path = f'in_sim/{folder_name}/cell_specific_GRN'

# Function to read a matrix from a CSV file
def read_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df.values

# Get list of all CSV files in the folder
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize a variable to store the sum of all matrices
sum_matrix = None
count = 0

# Iterate through each file and accumulate the sum
for file_path in file_list:
    matrix = read_matrix(file_path)
    
    if sum_matrix is None:
        sum_matrix = np.zeros_like(matrix)
    
    sum_matrix += matrix
    count += 1

print("count number: ", count)
# Calculate the average matrix
average_matrix = sum_matrix / count

example_csv = pd.read_csv(f"in_sim/{folder_name}/cell_specific_GRN/cell1.csv", index_col=0)
# Convert the average matrix back to a DataFrame
str_col = [str(c) for c in example_csv.columns]
str_index = [str(i) for i in example_csv.index]
average_df = pd.DataFrame(average_matrix, columns=str_col, index=str_index)

# Save the average matrix to a new CSV file
output_path = os.path.join(f"in_sim/{folder_name}", 'allcells_GRN.csv')
average_df.to_csv(output_path)

print(f"Average matrix saved to {output_path}")


#############################################################
# get all the gene pairs that have interaction strength != 0 #
#############################################################

# Load the CSV file
file_path = f'in_sim/{folder_name}/allcells_GRN.csv'
df = pd.read_csv(file_path, index_col=0)

# Find all non-zero interactions
non_zero_interactions = df.stack().reset_index()

# Rename the columns to 'gene1', 'gene2', and 'interaction_strength'
non_zero_interactions.columns = ['gene1', 'gene2', 'interaction_strength']

# Filter out rows where interaction_strength is zero
non_zero_interactions = non_zero_interactions[non_zero_interactions['interaction_strength'] != 0]

# Drop the interaction_strength column to only keep gene pairs
non_zero_gene_pairs = non_zero_interactions[['gene1', 'gene2']]
non_zero_gene_pairs = non_zero_gene_pairs.astype(str)

# Save the non-zero gene pairs to a new CSV file TODO
output_file_path = f'in_sim/{folder_name}/allcells_gt_gene_pairs.csv'  # Replace with your desired output file path
non_zero_gene_pairs.to_csv(output_file_path, index=False)

print(f'Non-zero gene pairs saved to {output_file_path}')


#########################################
# get GT pairs for each cell-specifc GRNs

folder_path = f'in_sim/{folder_name}/cell_specific_GRN'  # Update this path
save_path = f'in_sim/{folder_name}/cell_specific_gt_gene_pair'

try:
    os.mkdir(save_path)
except:
    print(f'{save_path} exist')

# Get list of all CSV files in the folder
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize a variable to store the sum of all matrices
sum_matrix = None
count = 0

print(len(file_list))

# Iterate through each file and accumulate the sum
for file_path in file_list:
    df = pd.read_csv(file_path, index_col=0)

    # Find all non-zero interactions
    non_zero_interactions = df.stack().reset_index()

    # Rename the columns to 'gene1', 'gene2', and 'interaction_strength'
    non_zero_interactions.columns = ['gene1', 'gene2', 'interaction_strength']

    # Filter out rows where interaction_strength is zero
    non_zero_interactions = non_zero_interactions[non_zero_interactions['interaction_strength'] != 0]

    # Drop the interaction_strength column to only keep gene pairs
    non_zero_gene_pairs = non_zero_interactions[['gene1', 'gene2']]
    non_zero_gene_pairs = non_zero_gene_pairs.astype(str)

    file_name = file_path.split("/")[-1].split(".")[0]

    # Save the non-zero gene pairs to a new CSV file
    output_file_path = os.path.join(save_path, f"{file_name}_gt_gene_pairs.csv")  # Replace with your desired output file path
    non_zero_gene_pairs.to_csv(output_file_path, index=False)


###################################
# raw count sc data normalization #
###################################

import pandas as pd
import numpy as np

# Load the single-cell expression count CSV file
input_file_path = f'in_sim/{folder_name}/raw_count.csv'  # Update this path
output_file_path = f'in_sim/{folder_name}/normalized_count.csv'  # Update this path

# Read the data into a DataFrame
df = pd.read_csv(input_file_path, index_col=0)

# Define the normalization function
def normalize_counts(df, scale_factor=10000, pseudo_count=1):
    # Calculate the total counts for each cell
    total_counts = df.sum(axis=1)
    print(total_counts.shape)
    
    # Normalize counts
    normalized_df = df.div(total_counts, axis=0) * scale_factor
    print(normalized_df.shape)
    
    # Add a pseudo-count and log transform
    normalized_df = np.log1p(normalized_df + pseudo_count)
    print(normalized_df.shape)
    
    return normalized_df

# Apply the normalization function
normalized_df = normalize_counts(df)

# Save the normalized DataFrame to a new CSV file
normalized_df.columns = [str(i) for i in range(1, gene_number+1)]
normalized_df.to_csv(output_file_path)

print(f"Normalized expression counts saved to {output_file_path}")

#############################################
# combine gene expression, x, y, clusterid

# Load the dataframes
gene_expression_path = f'in_sim/{folder_name}/normalized_count.csv'
loc_cluster_path = f'in_sim/{folder_name}/cell_loc_GRNcluster4.csv'

gene_exp_df = pd.read_csv(gene_expression_path, index_col=0)
loc_cluster_df = pd.read_csv(loc_cluster_path, index_col=0)

# Combine the dataframes using the index value
combined_df = gene_exp_df.join(loc_cluster_df)

# Save the combined dataframe to a new CSV file
output_path = f'in_sim/{folder_name}/expression_loc_cluster_wlayout.csv'
combined_df.to_csv(output_path)

# Display the combined dataframe
print(combined_df.head())