import os
import os.path as op
from abagen.mouse import io, mouse, utils
import requests

OUT_DIR = op.abspath('./allendata')

all_gene_ids = io.fetch_allenref_genes()
gene_ids = io.fetch_allenref_genes('id')

i=0
for gene in gene_ids:
    try:
        experiments = mouse._get_experiments_from_gene(id=gene, slicing_direction='sagittal')
    except ValueError:
        print(f'Gene {gene} does not appear to have any associated experiments/images. Skipping.')

    gene_dir = op.join(str(OUT_DIR), str(gene))
    for experiment in experiments:
        images = utils._make_api_query('SectionImage', criteria=f'[data_set_id$in{experiment}]')

        #select the most medial section in the experiment
        section=0
        for img in images:
            if img['section_number']>section:
                section=img['section_number']
                selected_img=img

        img=selected_img
        img_id = img['id']
        url = f'http://api.brain-map.org/api/v2/image_download/{img_id}?downsample=5&view=expression'
        img_data = requests.get(url).content

        filename = op.join(str(gene_dir), str(img_id)+'_section'+str(img['section_number'])+'.jpg')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as dest:
            dest.write(img_data)
    if i%1000==0:
        print(i)
    i+=1
