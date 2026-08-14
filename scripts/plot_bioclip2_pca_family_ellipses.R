#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
})

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "/private/tmp/seedlearn-nhn-snapshot/docs/clip/figures/bioclip2_pca_family_coords.csv"
out_prefix <- if (length(args) >= 2) args[[2]] else "/private/tmp/bioclip2_pca_family_ellipses_preview"

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

coords_top <- coords[coords$family %in% top_families, ]
coords_other <- coords[!coords$family %in% top_families, ]
coords_other$family <- "Other families"

coords_top$family <- factor(coords_top$family, levels = top_families)
coords_other$family <- factor(coords_other$family, levels = "Other families")

okabe_ito <- c(
  "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
  "#56B4E9", "#F0E442", "#332288", "#88CCEE", "#AA4499"
)
palette <- c(
  setNames(okabe_ito[seq_along(top_families)], top_families),
  "Other families" = "grey78"
)

centroids <- aggregate(cbind(pc1, pc2) ~ family, data = coords_top, FUN = median)
centroids$n <- as.integer(family_counts[as.character(centroids$family)])
centroids$label <- paste0(centroids$family, "\n", "n=", centroids$n)

label_families <- top_families[seq_len(min(6, length(top_families)))]
label_centroids <- centroids[as.character(centroids$family) %in% label_families, ]

legend_breaks <- c(top_families, "Other families")
legend_labels <- c(
  paste0(top_families, " (", format(as.integer(family_counts[top_families]), big.mark = ","), ")"),
  sprintf("Other families (%s)", format(nrow(coords_other), big.mark = ","))
)
names(legend_labels) <- legend_breaks

p <- ggplot() +
  geom_point(
    data = coords_other,
    aes(pc1, pc2, color = family),
    size = 0.34,
    alpha = 0.36,
    stroke = 0
  ) +
  geom_point(
    data = coords_top,
    aes(pc1, pc2, color = family),
    size = 0.5,
    alpha = 0.76,
    stroke = 0
  ) +
  stat_ellipse(
    data = coords_top,
    aes(pc1, pc2, color = family),
    type = "norm",
    level = 0.68,
    linewidth = 0.42,
    alpha = 0.8,
    show.legend = FALSE
  ) +
  ggrepel::geom_text_repel(
    data = label_centroids,
    aes(pc1, pc2, label = family, color = family),
    box.padding = 0.38,
    point.padding = 0.18,
    min.segment.length = 0,
    segment.color = "grey48",
    segment.size = 0.18,
    size = 2.75,
    fontface = "bold",
    show.legend = FALSE,
    seed = 42,
    max.overlaps = Inf
  ) +
  scale_color_manual(
    values = palette,
    breaks = legend_breaks,
    labels = legend_labels,
    name = "Family label"
  ) +
  labs(
    title = NULL,
    subtitle = NULL,
    x = "PC1",
    y = "PC2",
    caption = sprintf(
      "%s images, %s families. Ten largest families colored; remaining families in grey.",
      format(nrow(coords), big.mark = ","),
      length(family_counts)
    )
  ) +
  coord_equal() +
  theme_classic(base_size = 10.5) +
  theme(
    plot.caption = element_text(size = 7.7, color = "grey35", hjust = 0, margin = margin(t = 6)),
    legend.position = "right",
    legend.justification = c(1, 0),
    legend.title = element_text(face = "bold", size = 7.9),
    legend.text = element_text(size = 7.25),
    legend.key.size = grid::unit(0.34, "lines"),
    axis.title = element_text(size = 9.4),
    axis.text = element_text(size = 8.1, color = "grey35"),
    axis.line = element_line(color = "grey20", linewidth = 0.25),
    axis.ticks = element_line(color = "grey20", linewidth = 0.25),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(6, 8, 5, 6)
  ) +
  guides(color = guide_legend(override.aes = list(size = 1.8, alpha = 1), ncol = 1))

ggsave(paste0(out_prefix, ".png"), p, width = 6.8, height = 4.8, dpi = 450, bg = "white")
ggsave(paste0(out_prefix, ".pdf"), p, width = 6.8, height = 4.8, bg = "white")

summary_path <- paste0(out_prefix, "_summary.csv")
write.csv(
  data.frame(family = names(family_counts), n = as.integer(family_counts)),
  summary_path,
  row.names = FALSE
)

message("saved: ", paste0(out_prefix, ".png"))
message("saved: ", paste0(out_prefix, ".pdf"))
message("saved: ", summary_path)
