library("scMultiSim")

data(GRN_params_100)
GRN_params <- GRN_params_100

lig_params <- data.frame(
  target    = c(101, 102),
  regulator = c(103, 104),
  effect    = c(5.2, 5.9)
)

spatial_options <- function (...) {
  cci_opt <- list(
    params = lig_params,
    max.neighbors = 4,
    cell.type.interaction = "random"
  )
  list(
    rand.seed = 1,
    GRN = GRN_params,
    num.genes = 110,
    num.cells = 2000,
    num.cifs = 50,
    tree = Phyla1(),
    diff.cif.fraction = 0.8,
    do.velocity = FALSE,
    speed.up = T,
    intrinsic.noise = 0.1,
    dynamic.GRN = list(
      cell.per.step = 1,
      num.changing.edges = 5,
      weight.mean = 0,
      weight.sd = 4
    ),
    cci = c(cci_opt, list(...))
  )
}

results <- sim_true_counts(spatial_options(
  layout = "layers"
))

# cellid x y
#results[["cci_locs"]]
write.csv(results[["cci_locs"]], "in_sim/g110_c2k_0.1/cell_loc.csv", row.names=TRUE)
print("locs saved")

# count gene x cell
#results[["counts"]]
# save as cell x gene
write.csv(t(results[["counts"]]), "in_sim/g110_c2k_0.1/raw_count.csv", row.names=TRUE)
print("counts saved")

# cell specific grn
# results[["cell_specific_grn"]]
for (x in 1:2000){
  write.csv(results[["cell_specific_grn"]][[x]], 
            paste0("in_sim/g110_c2k_0.1/cell_specific_GRN/cell",x,".csv"), row.names=TRUE)
}

print("cell_specific_grn saved")





