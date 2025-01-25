library("scMultiSim")

# For 210 genes
GRN_params = read.csv("scMultiSim_DataPrepare/GRN_files/g200_TF6_1.csv")
lig_params <- data.frame(
  target    = c(202, 203),
  regulator = c(204, 205),
  effect    = c(5.2, 5.9)
)

# For 310 genes
# GRN_params = read.csv("scMultiSim_DataPrepare/GRN_files/g300_TF6_1.csv")
# lig_params <- data.frame(
#   target    = c(302, 303),
#   regulator = c(304, 305),
#   effect    = c(5.2, 5.9)
# )

# For 410 genes
# GRN_params = read.csv("scMultiSim_DataPrepare/GRN_files/g400_TF6_1.csv")
# lig_params <- data.frame(
#   target    = c(402, 403),
#   regulator = c(404, 405),
#   effect    = c(5.2, 5.9)
# )

# For 510 genes
# GRN_params_500 = read.csv("scMultiSim_DataPrepare/GRN_files/g500_TF6_1.csv")
# lig_params <- data.frame(
#   target    = c(502, 503),
#   regulator = c(504, 505),
#   effect    = c(5.2, 5.9)
# )

spatial_options <- function (...) {
  cci_opt <- list(
    params = lig_params,
    max.neighbors = 4,
    cell.type.interaction = "random"
  )
  list(
    rand.seed = 0,
    GRN = GRN_params,
    num.genes = 210,     # 310, 410, 510...
    num.cells = 2000,    # 6000
    num.cifs = 50,
    tree = Phyla1(),
    diff.cif.fraction = 0.8,
    do.velocity = FALSE,
    speed.up = T,
    intrinsic.noise = 0.1,   # 0.5, 0.8, 1.0
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
write.csv(results[["cci_locs"]], "in_sim/g210_c2k_0.1/cell_loc.csv", row.names=TRUE)
print("locs saved")

# count gene x cell
#results[["counts"]]
# save as cell x gene
write.csv(t(results[["counts"]]), "in_sim/g210_c2k_0.1/raw_count.csv", row.names=TRUE)
print("counts saved")

# cell specific grn
# results[["cell_specific_grn"]]
for (x in 1:2000){   
  write.csv(results[["cell_specific_grn"]][[x]], 
            paste0("in_sim/g210_c2k_0.1/cell_specific_GRN/cell",x,".csv"), row.names=TRUE)
}

print("cell_specific_grn saved")