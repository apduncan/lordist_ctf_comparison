# Can the subjects be separated by just using the mean between participant
# distance?

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
  log_info("Run Mean Bray-Curtis benchmarks for simulated data.")
  log_info("Usage:")
  log_info("02b_bc_benchmark.R path/to/simulation_metdata.tsv path/to/subject_metadata.tsv path/to_output.tsv")
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
  log_info("Run Mean B-C | Requested Sparsity: {row$requested_sparsity}, Actual Sparsity: {row$achieved_sparsity}, i: {row$i}")
  filt_md <- sample_metadata

  # Warn abot empty samples
  empty_samples <- (mat |> colSums()) > 0
  if(sum(!empty_samples) > 0) {
    log_warn("Empty samples dues to sparsity: {sum(!empty_samples)}")
    log_warn("These samples are dropped from both the abundance and metadata")
    mat <- mat[, empty_samples]
    filt_md <- sample_metadata |>
      filter(SampleID %in% colnames(mat))
  }

  dist <- vegdist(mat |> t(), method = "bray")

  # Calculate the average between subject Bray-Curtis
  dist_df <- dist |>
    as.matrix() |>
    as.data.frame() |>  
    rownames_to_column("id_a") |>
    pivot_longer(
      !id_a,
      names_to = "id_b",
      values_to = "distance"
    ) |>
    dplyr::left_join(
      filt_md |> select(SampleID, SubjectID),
      join_by(id_a == SampleID)
    ) |>
    rename(group_a = SubjectID) |>
    dplyr::left_join(
      filt_md |> select(SampleID, SubjectID),
      join_by(id_b == SampleID)
    ) |>
    rename(group_b = SubjectID)
  
  # Easiest way to group these is to make this complete
  # Inelegant but achieves the goal
  dist_df2 <- dist_df
  dist_df2$group_b <- dist_df$group_a
  dist_df2$group_a <- dist_df$group_b
  dist_df2$id_a <- dist_df$id_b
  dist_df2$id_b <- dist_df$id_a

  dist_df <- bind_rows(dist_df, dist_df2)

  # Turn back into a distance matrix
  subject_dist <- dist_df |>
    group_by(group_a, group_b) |>
    summarise(distance = mean(distance)) |>
    pivot_wider(names_from = group_a, values_from = distance) |>
    column_to_rownames("group_b") |>
    as.matrix() |>
    as.dist()
  
  # PERMANOVA with adonis2
  res <- adonis2(
    subject_dist ~ Label,
    subject_metadata[attr(subject_dist, "Labels"), ],
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