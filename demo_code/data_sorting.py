import numpy as np 
import rlxnix as rlx 
from IPython import embed

dset = rlx.Dataset("data/2022-10-20-ab-invivo-1.nix")

def sort_reodfs(data):
    """Sorting of the relative EODs of chirps data.

    Parameters
    ----------
    data : rlx.Dataset nix file 
        Dataset with different EODs for chirp data
    Returns
    -------
    dic
        Dictionary with the relative EODs as keys, Items are the name of the trace  
    """
    r_eodf = []
    for chirp in data.repro_runs('Chirps'):
        r_eodf.append(chirp.relative_eodf)

    r_eodf_arr = np.array(r_eodf)
    r_eodf_arr_uniq = np.unique(r_eodf_arr)

    r_eodf_dict = {}

    for unique_r_eodf in r_eodf_arr_uniq:
        r_eodf_dict[f"{unique_r_eodf}"] = []
        for r in range(len(r_eodf)):
            chirps = data.repro_runs('Chirps')[r]
            if unique_r_eodf == r_eodf[r]:
                r_eodf_dict[f"{unique_r_eodf}"].append(chirps.name)
    
    return r_eodf_dict

dict = sort_reodfs(dset)
print(dict)