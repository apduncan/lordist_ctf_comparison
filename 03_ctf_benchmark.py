from pathlib import Path

from biom import Table
from gemelli.ctf import ctf
import numpy as np
import pandas as pd
from skbio.stats.distance import permanova

simulation_dir: Path = Path("data") / "simulated"
simulation_metadata: pd.DataFrame = pd.read_csv(
    simulation_dir / "metadata.tsv",
    sep = "\t",
    index_col = 0
)
sample_metadata: pd.DataFrame = pd.read_csv(
    Path("data") / "derived" / "lordist_sim_meta.tsv",
    sep = "\t"
)

# Test single dataframe
pth_data: Path = Path("data") / "simulated" / "sparsity_0" / "1.tsv"
data: pd.DataFrame = pd.read_csv(pth_data, sep="\t")
pth_tmp: Path = Path("tmp.txt")
# np.savetxt(pth_tmp, data.values, delimiter="\t")
# data.to_csv(pth_tmp, sep="\t")
with pth_tmp.open("rt") as f:
    data_table: Table = Table.from_tsv(
        f, None, None, lambda x: x
    )
ctf_results = ctf(
    data_table,
    sample_metadata.set_index("SampleID"),
    "SubjectID",
    "Timepoint"
)
# Note that the CTF distance matrix is not in the input order, it returns
# in what appears to be a random order
# TODO: Reduce md to one entry per subject, then reorder, extract
pres = permanova(
    ctf_results[2],
    grouping=sample_metadata.loc[ctf_results[2].ids, 'Label']
)
print(ctf_results)