import os
import pickle
import numpy as np
import pandas as pd
from abagen.mouse import io, mouse, utils

'''
First associated the downloaded gene ids with corresponding acronyms
'''
dir_list=os.listdir('allendata')
acronyms={}
all_gene_ids = io.fetch_allenref_genes()
for gene_id in dir_list:
    idx=np.where(np.asarray(all_gene_ids['id'])==int(gene_id))[0][0]
    acronym=all_gene_ids['acronym'][idx]
    acronyms[acronym]=gene_id

'''
Make a list of the amigo files with associated GO terms
'''

path = os.getcwd()+'/AmiGO_import'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))

GO_data={}
for filename in files:
    df=pd.read_csv(filename, sep='\t', lineterminator='\n', header=None)
    gene_acronyms=df[2]
    GO_terms=df[4]
    for acronym, GO_term in zip(gene_acronyms, GO_terms):
        try:
            gene_id=acronyms[acronym]
            try:
                if not GO_term in GO_data[gene_id]: #include only GO terms once per gene id
                    GO_data[gene_id].append(GO_term)
            except:
                GO_data[gene_id]=[GO_term]
            
        except:
            continue

filename='GO_terms.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(GO_data, handle)
        