#!/usr/bin/env python
import pandas as pd
from pathlib import Path

datadir = Path('/ut3/jerome/github/research/project/toi5671/tfop')
#import pdb; pdb.set_trace()
files = list(datadir.glob('*.csv'))
#print(files)

for f in files:
    df = pd.read_csv(f)
    band = f.name.split('_')[2]
    fp = f'{band}.csv'
    df[['BJD_TDB','Flux','Err']].to_csv(fp, index=False, header=False)
    print('Saved:', fp)
