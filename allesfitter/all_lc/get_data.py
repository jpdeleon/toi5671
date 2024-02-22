import sys
sys.path.insert(0, '../..')
import numpy as np
from toi5671 import read_all_phot

cols = ['BJD_TDB','Flux','Err']
phot_dir = '~/github/research/project/toi5671/data/photometry'
out_dir = '~/github/research/project/toi5671/allesfitter/all_lc'
data = read_all_phot(phot_dir, sort_by='inst')
for inst in data:
    for band in data[inst]:
        for i,d in enumerate(data[inst][band]):
            fp = inst if inst==band else f'{inst}_{band}'
            if len(data[inst][band])>1:
                fp+=f'_{i+1}.csv'
            else:
                fp+='.csv'
            d['Flux'] = d['Flux']/np.median(d['Flux'])
            if inst=='keplercam':
                d['Flux'] -= 0.05
            d[cols].sort_values(by='BJD_TDB').to_csv(fp, index=False, header=False)
            print('Saved: ', fp)