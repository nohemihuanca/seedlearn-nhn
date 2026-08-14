#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "docs/clip/figures/bioclip2_pca_family_coords.csv"
out_prefix <- if (length(args) >= 2) args[[2]] else "docs/clip/figures/seedlearn_family_distribution"

coords <- read.csv(input, stringsAsFactors = FALSE)
required <- c("image_path")
missing <- setdiff(required, names(coords))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

coords$family <- sub(".*/by_family/([^/]+)/.*", "\\1", coords$image_path)
coords$family[coords$family == coords$image_path] <- "unknown"
coords$individual_id <- sub(".*/([^/]+)/[^/]+$", "\\1", coords$image_path)

summary_df <- aggregate(
  image_path ~ family,
  data = coords,
  FUN = length
)
names(summary_df)[names(summary_df) == "image_path"] <- "images"

individual_counts <- aggregate(
  individual_id ~ family,
  data = unique(coords[, c("family", "individual_id")]),
  FUN = length
)
names(individual_counts)[names(individual_counts) == "individual_id"] <- "individuals"

summary_df <- merge(summary_df, individual_counts, by = "family")
summary_df <- summary_df[order(summary_df$individuals, decreasing = TRUE), ]

highlight_n <- 10
highlight <- summary_df$family[seq_len(min(highlight_n, nrow(summary_df)))]
summary_df$group <- ifelse(as.character(summary_df$family) %in% highlight, "largest families", "other families")
summary_df$family <- factor(summary_df$family, levels = rev(summary_df$family))

palette <- c("largest families" = "#0072B2", "other families" = "#c8c8c8")

p <- ggplot(summary_df, aes(individuals, family, fill = group)) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = individuals),
    hjust = -0.16,
    size = 2.25,
    color = "grey25"
  ) +
  scale_fill_manual(values = palette, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(
    title = "SeedLearn training set composition by family",
    subtitle = sprintf(
      "%s images from %s unique individuals across %s families",
      format(nrow(coords), big.mark = ","),
      format(sum(summary_df$individuals), big.mark = ","),
      nrow(summary_df)
    ),
    x = "Unique individuals",
    y = NULL,
    caption = "Counts are derived from BioCLIP 2 image paths; bars show unique individual IDs, not image counts."
  ) +
  theme_classic(base_size = 10.5) +
  theme(
    plot.title = element_text(face = "bold", size = 13.2, margin = margin(b = 4)),
    plot.subtitle = element_text(size = 9.2, color = "grey30", margin = margin(b = 8)),
    plot.caption = element_text(size = 8.1, color = "grey35", hjust = 0, margin = margin(t = 8)),
    axis.text.y = element_text(size = 6.8, color = "grey20", face = "italic"),
    axis.text.x = element_text(size = 8.2, color = "grey35"),
    axis.title.x = element_text(size = 9.5),
    axis.line = element_line(color = "grey20", linewidth = 0.25),
    axis.ticks = element_line(color = "grey20", linewidth = 0.25),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(8, 16, 8, 6)
  )

ggsave(paste0(out_prefix, ".png"), p, width = 7.2, height = 8.8, dpi = 450, bg = "white")
ggsave(paste0(out_prefix, ".pdf"), p, width = 7.2, height = 8.8, bg = "white")
write.csv(summary_df[, c("family", "individuals", "images")], paste0(out_prefix, "_summary.csv"), row.names = FALSE)

message("saved: ", paste0(out_prefix, ".png"))
message("saved: ", paste0(out_prefix, ".pdf"))
message("saved: ", paste0(out_prefix, "_summary.csv"))
