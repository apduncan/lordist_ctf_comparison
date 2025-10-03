#' Run LorDist for simulated data with vary levels of sparsity.
#' The simulated data should already have been produced and saved as TSV
library(readr)
library(vegan)
library(tibble)
library(logger)

sparsity_dir <- file.path("data", "simulated")
simulation_metadata <- read_delim(
  file.path(sparsity_dir, "metadata.tsv")
)
sample_metadata <- read.table(
  file.path("data", "derived", "lordist_sim_meta.tsv"),
  sep = "\t",
  header = TRUE
)
subject_metadata <- sample_metadata |>
  group_by(SubjectID) |>
  sample_n(size=1) |>
  column_to_rownames("SubjectID")

run_lordist <- function(...) {
  row <- tibble(...)
  mat <- read.table(row$output, sep = "\t", header = TRUE)
  log_info("Run LorDist | Requested Sparsity: {row$requested_sparsity}, Actual Sparsity: {row$achieved_sparsity}, i: {row$i}")
  dist <- LorDist(
    mat,
    sample_metadata,
    SampleID = "SampleID",
    SubjectID = "SubjectID",
    Timepoint="Timepoint"
  )
  # PERMANOVA with adonis2
  res <- adonis2(
    dist ~ Label,
    subject_metadata[attr(dist, "Labels"), ],
    permutations = 1000
  )
  row$pval <- res$`Pr(>F)`[1]
  row$fval <- res$F[1]
  log_info("PERMANOVA results | p : {row$pval}, F: {row$fval}")
  return(row)
}

set.seed(4298)
# Run for all simulated matrices
sim_res <- simulation_metadata |>
  # All 0 sparsity tables are the same, no need to test iterations
  filter(requested_sparsity > 0) |>
  pmap(run_lordist) |>
  bind_rows()
sim_res |> 
  write_delim("results/simulated/lordist.tsv", delim = "\t")
