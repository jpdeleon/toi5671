import pandas as pd

fin = '~/github/research/project/toi5671/rv/rvout_TOI5671.dat'
fout = '~/github/research/project/toi5671/allesfitter/rv/circular/ird.csv'
rv = pd.read_csv(fin, header=None, delim_whitespace=True)
rv = rv.drop(rv.index[[5,12,16]])
rv2 = rv[[0,1,2]].copy()
rv2[[1,2]] = rv2[[1,2]].apply(lambda x: x/1e3)
rv2.to_csv(fout, index=False, header=False)
print("Saved: ", fout)
