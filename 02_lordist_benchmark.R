#' Run LorDist for simulated data with vary levels of sparsity.
#' The simulated data should already have been produced and saved as TSV
library(readr)
library(vegan)
library(tibble)
library(logger)
library(tidyverse)
library(LorDist)
library(logger)

show_usage <- function() {
  log_info("Run LorDist benchmarks for simulated data.")
  log_info("Usage:")
  log_info("02_lordist_benchmark.R path/to/simulation_metdata.tsv path/to/subject_metadata.tsv path/to_output.tsv")
}

args <- commandArgs(trailingOnly = TRUE)

if(args[1] %in% c("-h", "h", "--help", "help")) {
  show_usage()
  quit()
}
if(length(args) != 3) {
  show_usage()
  log_fatal("All arguments must be provided")
  quit()
}

simulation_metadata <- args[[1]]
subject_metadata <- args[[2]]
output_location <- args[[3]]

simulation_metadata <- read_delim(simulation_metadata)
sample_metadata <- read.table(
  subject_metadata,
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
  pmap(run_lordist) |>
  bind_rows()
sim_res |> 
  write_delim(output_location, delim = "\t")
