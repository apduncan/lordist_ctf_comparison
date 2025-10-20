from argparse import ArgumentParser, FileType
from pathlib import Path
from typing import List, NamedTuple

from biom import Table
from gemelli.ctf import ctf
import numpy as np
import pandas as pd
from skbio import OrdinationResults
from skbio.stats.distance import permanova, DistanceMatrix
from sklearn.metrics.pairwise import pairwise_distances
import logging

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch: logging.Handler = logging.StreamHandler()
fm: logging.Formatter = logging.Formatter(
    fmt='%(levelname)s [%(name)s] [%(asctime)s]: %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
)
ch.setFormatter(fm)
logger.addHandler(ch)



class CTFBenchResults(NamedTuple):
    subject_biplot: OrdinationResults
    state_biplot: OrdinationResults
    distance_matrix: DistanceMatrix
    state_subject_ordination: pd.DataFrame
    state_feature_orindation: pd.DataFrame
    permanova_distance_matrix: pd.Series
    permanova_subject_distance_matrix: pd.Series
    input_file: Path

    def results_series(self) -> pd.Series:
        # Make a series with the key evaluation results
        return pd.Series(
            dict(
                sample_f = self.permanova_distance_matrix['test statistic'],
                sample_p = self.permanova_distance_matrix['p-value'],
                subject_f = self.permanova_subject_distance_matrix['test statistic'],
                subject_p = self.permanova_subject_distance_matrix['p-value'],
                matrix = str(self.input_file)
            ),
            name = str(self.input_file)
        )
    
    @staticmethod
    def results_table(results) -> pd.DataFrame:
        # Join multiple evaluation results into a single table
        return (
            pd.concat(
                [x.results_series() for x in results],
                axis=1
            ).T
        )

def ctf_evaluate(
        pth_mat: Path,
        sample_metadata: pd.DataFrame,
        seed: int,
        biom_dir: str
    ) -> CTFBenchResults:
    logger.info("Evaluate | Matrix: %s, Seed: %s", pth_mat, seed)
    # Output not quite in correct format for loading into BIOM
    # Load with pandas and save as correct format
    logger.debug("Load and convert to BIOM friendly format")
    df: pd.DataFrame = pd.read_csv(
        pth_mat, sep="\t"
    )
    df.index.name = "Sample"
    biom_parts = list(pth_mat.parts)
    pth_biom: Path = Path(biom_dir) / biom_parts[-1]
    logger.debug("BIOM path: %s", pth_biom)
    pth_biom.parent.mkdir(parents=True, exist_ok=True)

    # The sparsification may have set some samples to all 0
    # We need to remove these before passing to CTF
    drop: pd.Series = df.sum() == 0
    drop_names: List[str] = list(drop.index[drop])
    if len(drop_names) > 0:
        logger.warning(
            "Dropping %s samples due to 0 counts: %s",
            len(drop_names),
            list(drop_names)
        )
    df = df.loc[:, ~drop]
    sample_metadata = sample_metadata.loc[
        ~sample_metadata['SampleID'].isin(drop_names), :
    ]
    df.to_csv(pth_biom, sep="\t")

    # Neither sklearn permanova nor CTF expose a seed or random generator
    # by parameter, so will try to make reproducible with random.seed
    np.random.seed(seed)

    with pth_biom.open("rt") as f:
        data_table: Table = Table.from_tsv(
            f, None, None, lambda x: x
        )
    logger.info("Run CTF")
    ctf_results = ctf(
        data_table,
        sample_metadata.set_index("SampleID"),
        "SubjectID",
        "Timepoint"
    )

    # CTF produces both a sample-level distance matrix, and a subject level
    # ordination in 3 dimensions. The subject level ordination is maybe most
    # directly comparable to LorDist which produces a subject level distance,
    # so we will test both with PERMANOVA. The sample distances are tested directly,
    # and subject using Euclidean distance based on ordination.
    logger.debug("Format CTF distance matrices")
    ctf_dist = ctf_results[2]
    ctf_subj_bp = ctf_results[0]
    ctf_subj_dist = DistanceMatrix(
        pairwise_distances(
            ctf_subj_bp.samples,
            metric='euclidean'
        ),
        ids=ctf_subj_bp.samples.index,
        validate=False
    )
    # For permanova, underlying array must be C-contiguous, so make a new object 
    # meeting these criteria
    ctf_cdist = DistanceMatrix(
        data=ctf_dist.data.copy(order="C"),
        ids=ctf_dist.ids,
        validate=True
    )

    # Note that the CTF distance matrix is not in the input order, it returns
    # in what appears to be a random order, so metadata needs to be adjusted to
    # match
    logger.debug("Run PERMANOVA")
    pres_dist = permanova(
        ctf_cdist,
        grouping=(
            sample_metadata
            .set_index('SampleID')
            .loc[
                list(ctf_cdist.ids), 'Label'
            ]
            .values
        ),
        permutations=1000
    )
    pres_subj_dist = permanova(
        ctf_subj_dist,
        grouping=(
            sample_metadata
            .drop_duplicates(subset=['SubjectID'])
            .set_index('SubjectID')
            .loc[
                list(ctf_subj_dist.ids), 'Label'
            ]
            .values
        ),
        permutations=1000
    )
    sample_f: float = pres_dist['test statistic']
    sample_p: float = pres_dist['p-value']
    subj_f: float = pres_subj_dist['test statistic']
    subj_p: float = pres_subj_dist['p-value']
    logger.info(
        "PERMANOVA resuts | Sample - p: %s, F: %s | Subject - p: %s, F: %s",
        sample_p, sample_f, subj_p, subj_f
    )

    return CTFBenchResults(*ctf_results, pres_dist, pres_subj_dist, pth_mat)

def run_benchmarks(
        sim_md: Path,
        sample_md: Path,
        biom_dir: Path,
        output_loc: Path
):
    logger.info("Loading simulation metadata")
    # Simulation meteadata (sparsity etc)
    simulation_metadata: pd.DataFrame = pd.read_csv(
        sim_md,
        sep = "\t",
        index_col = 0
    )
    logger.debug("Assigning seed to each iteration")
    # Give each a fixed seed to try and make reproducible across multiprocessing
    np.random.seed(4298)
    simulation_metadata['seed'] = np.random.randint(
        low=0, high=2**16,
        size=simulation_metadata.shape[0]
    )

    # Sample metadata
    logger.info("Loading sample metadata")
    sample_metadata: pd.DataFrame = pd.read_csv(
        sample_md,
        sep = "\t"
    )

    # Evaluate for every dataframe
    args = list(
        dict(
            pth_mat=Path(x[1]['output']),
            sample_metadata=sample_metadata,
            seed=x[1]['seed'],
            biom_dir=biom_dir
        ) for x in 
        simulation_metadata[['output', 'seed']].iterrows()
    )

    logger.info("Starting runs for each matrix")
    res = [ctf_evaluate(**x) for x in args]

    # Append results to simulation metadata table and save
    tbl: pd.DataFrame = CTFBenchResults.results_table(res)
    tbl['matrix'] = "./" + tbl['matrix']
    pth_results: Path = output_loc
    pth_results.parent.mkdir(parents=True, exist_ok=True)
    full_tbl: pd.DataFrame = pd.merge(
        tbl,
        simulation_metadata,
        left_on="matrix", right_on="output"
    )
    full_tbl.to_csv(output_loc, sep = "\t")

if __name__ == "__main__":
    # Argument parsing
    parser: ArgumentParser = ArgumentParser(
        description = "Run CTF simulated data benchmarks"
    )
    parser.add_argument(
        "--simulation-metadata",
        dest="sim_md",
        help="Metadata giving details of each simulated dataset",
        required=True
    )
    parser.add_argument(
        "--sample-metadata",
        dest="sample_md",
        help=("Metadata about each sample in all simulation, giving time and "
        "phenotype"),
        required=True
    )
    parser.add_argument(
        "--biom-dir",
        dest="biom_dir",
        help=("Directory to store BIOM compatible tsv in"),
        required=True
    )
    parser.add_argument(
        "--output-loc",
        dest="output_loc",
        help="File to write results to",
        required=True
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logging messages"
    )
    args = parser.parse_args()
    sim_md: Path = Path(args.sim_md)
    sample_md: Path = Path(args.sample_md)
    output_loc: Path = Path(args.output_loc)
    biom_dir: Path = Path(args.biom_dir)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    run_benchmarks(sim_md, sample_md, biom_dir, output_loc)