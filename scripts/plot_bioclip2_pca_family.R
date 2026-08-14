#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "docs/clip/figures/bioclip2_pca_family_coords.csv"
out_prefix <- if (length(args) >= 2) args[[2]] else "docs/clip/figures/bioclip2_pca_family"

coords <- read.csv(input, stringsAsFactors = FALSE)
required <- c("pc1", "pc2", "family_label", "image_path")
missing <- setdiff(required, names(coords))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

coords$family <- sub(".*/by_family/([^/]+)/.*", "\\1", coords$image_path)
coords$family[coords$family == coords$image_path] <- paste0("family_", coords$family_label[coords$family == coords$image_path])

family_counts <- sort(table(coords$family), decreasing = TRUE)
top_n <- 10
top_families <- names(family_counts)[seq_len(min(top_n, length(family_counts)))]

coords$plot_family <- ifelse(coords$family %in% top_families, coords$family, "Other families")
legend_levels <- c(
  paste0(top_families, " (n=", as.integer(family_counts[top_families]), ")"),
  sprintf("Other families (n=%s)", format(sum(family_counts[setdiff(names(family_counts), top_families)]), big.mark = ","))
)
names(legend_levels) <- c(top_families, "Other families")
coords$plot_family_label <- unname(legend_levels[coords$plot_family])
coords$plot_family_label <- factor(coords$plot_family_label, levels = unname(legend_levels))

palette <- c(
  setNames(grDevices::hcl.colors(length(top_families), palette = "Dark 3"), unname(legend_levels[top_families])),
  setNames("#c9c9c9", unname(legend_levels["Other families"]))
)

subtitle <- sprintf(
  "%s field-collected seedling images, %s families, BioCLIP 2 embeddings projected from 768D to 2D PCA",
  format(nrow(coords), big.mark = ","),
  length(family_counts)
)

p <- ggplot(coords, aes(pc1, pc2, color = plot_family_label)) +
  geom_point(size = 0.65, alpha = 0.62, stroke = 0) +
  scale_color_manual(values = palette, name = NULL) +
  labs(
    title = "BioCLIP 2 Embedding Space for SeedLearn Seedlings",
    subtitle = subtitle,
    x = "Principal component 1",
    y = "Principal component 2",
    caption = "Points are individual images. The ten largest families are colored; remaining families are shown in grey."
  ) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, margin = margin(b = 5)),
    plot.subtitle = element_text(size = 10.5, color = "grey30", margin = margin(b = 12)),
    plot.caption = element_text(size = 9, color = "grey35", hjust = 0),
    legend.position = "bottom",
    legend.text = element_text(size = 8.2),
    legend.key.width = grid::unit(0.8, "lines"),
    legend.key.height = grid::unit(0.8, "lines"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey88", linewidth = 0.25),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 9, color = "grey35"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(14, 16, 12, 14)
  )

ggsave(paste0(out_prefix, ".png"), p, width = 11, height = 7.2, dpi = 320, bg = "white")
ggsave(paste0(out_prefix, ".pdf"), p, width = 11, height = 7.2, bg = "white")

summary_path <- paste0(out_prefix, "_summary.csv")
write.csv(
  data.frame(family = names(family_counts), n = as.integer(family_counts)),
  summary_path,
  row.names = FALSE
)

message("saved: ", paste0(out_prefix, ".png"))
message("saved: ", paste0(out_prefix, ".pdf"))
message("saved: ", summary_path)
