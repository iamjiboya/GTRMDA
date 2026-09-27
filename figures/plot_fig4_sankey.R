#!/usr/bin/env Rscript

# ggalluvial Figure 4: verifier-guided MDA re-ranking in two disease cases.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("^--file=", args)][1]
script_file <- sub("^--file=", "", script_arg)
root <- normalizePath(dirname(script_file))

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggalluvial)
  library(patchwork)
  library(dplyr)
})

requested_font <- Sys.getenv("GTRMDA_R_FONT", unset = "Times New Roman")
matched_font <- tryCatch(
  system2("fc-match", c("-f", "%{family}", requested_font), stdout = TRUE),
  error = function(e) ""
)
font_family <- if (grepl(requested_font, matched_font, fixed = TRUE)) {
  requested_font
} else {
  warning("Times New Roman is unavailable; using Liberation Serif")
  "Liberation Serif"
}
width_in <- 3.50
height_in <- 2.95

palette <- c(
  Promoted = "#91C9B7",
  Unchanged = "#C8D0D4",
  Demoted = "#C9A4C5"
)
palette_edge <- c(
  Promoted = "#4B8F7D",
  Unchanged = "#6F7B81",
  Demoted = "#875C7D"
)
pale_green <- "#F1F7F3"
text_color <- "#202020"

data <- read.csv(
  file.path(root, "fig4_case_ranking_data.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

short_label <- function(x) {
  x <- sub("^hsa-", "", x)
  sub("-(3p|5p)$", "", x)
}

prepare_panel <- function(panel_data) {
  panel_data <- panel_data |>
    filter(verified_rank <= 6) |>
    mutate(
      movement = case_when(
        raw_rank > verified_rank ~ "Promoted",
        raw_rank < verified_rank ~ "Demoted",
        TRUE ~ "Unchanged"
      ),
      movement = factor(movement, levels = c("Promoted", "Unchanged", "Demoted")),
      display = short_label(mirna),
      verified_display = sprintf("#%d   %.2f | %d", verified_rank,
                                 verifier_confidence, source_count),
      raw_key = display,
      verified_key = verified_display
    )

  raw_order <- panel_data |>
    arrange(desc(raw_rank)) |>
    pull(display)
  verified_order <- panel_data |>
    arrange(desc(verified_rank)) |>
    pull(verified_display)
  panel_data$raw_key <- factor(panel_data$raw_key, levels = raw_order)
  panel_data$verified_key <- factor(panel_data$verified_key, levels = verified_order)

  list(
    flows = panel_data,
    total = sum(panel_data$source_count),
    rank6_weight = panel_data$source_count[panel_data$verified_rank == 6]
  )
}

make_panel <- function(panel_data, disease) {
  prepared <- prepare_panel(panel_data)
  flows <- prepared$flows
  representative_raw <- flows$display[flows$representative == 1]
  representative_verified <- flows$verified_display[flows$representative == 1]

  ggplot(
    flows,
    aes(axis1 = raw_key, axis2 = verified_key, y = source_count)
  ) +
    annotate(
      "rect", xmin = 1.46, xmax = 2.48,
      ymin = prepared$rank6_weight, ymax = prepared$total,
      fill = pale_green, colour = NA
    ) +
    geom_alluvium(
      aes(fill = movement, colour = movement),
      width = 0.105, alpha = 0.70, knot.pos = 0.47,
      linewidth = 0.36, reverse = FALSE, decreasing = NA,
      show.legend = FALSE
    ) +
    geom_stratum(
      width = 0.105, fill = "white", colour = "#7D888E",
      linewidth = 0.45, reverse = FALSE, decreasing = NA,
      show.legend = FALSE
    ) +
    geom_point(
      stat = "stratum",
      aes(size = after_stat(count), fill = movement, colour = movement,
          alpha = after_stat(x) == 2),
      shape = 21, stroke = 0.55, reverse = FALSE, decreasing = NA,
      show.legend = FALSE
    ) +
    geom_text(
      stat = "stratum",
      aes(
        label = ifelse(after_stat(x) == 1, as.character(after_stat(stratum)), ""),
        fontface = ifelse(after_stat(stratum) %in% representative_raw,
                          "bold", "plain")
      ),
      nudge_x = -0.12, hjust = 1, family = font_family,
      size = 2.45, colour = text_color,
      reverse = FALSE, decreasing = NA
    ) +
    geom_text(
      stat = "stratum",
      aes(
        label = ifelse(after_stat(x) == 2, as.character(after_stat(stratum)), ""),
        fontface = ifelse(after_stat(stratum) %in% representative_verified,
                          "bold", "plain")
      ),
      nudge_x = 0.12, hjust = 0, family = font_family,
      size = 2.25, colour = text_color,
      reverse = FALSE, decreasing = NA
    ) +
    annotate(
      "text", x = 1, y = prepared$total + 0.72, label = "Raw rank",
      family = font_family, fontface = "bold", size = 2.75,
      colour = text_color
    ) +
    annotate(
      "text", x = 2, y = prepared$total + 0.72, label = "Verified",
      family = font_family, fontface = "bold", size = 2.75,
      colour = text_color
    ) +
    annotate(
      "text", x = 2.31, y = prepared$total + 0.20,
      label = "rank  conf. | src", family = font_family,
      size = 1.90, colour = "#6F7B81"
    ) +
    scale_x_discrete(limits = c("Raw", "Verified"), expand = c(0, 0)) +
    scale_fill_manual(values = palette, drop = FALSE, guide = "none") +
    scale_colour_manual(values = palette_edge, guide = "none", drop = FALSE) +
    scale_size_continuous(range = c(2.1, 4.0), guide = "none") +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0), guide = "none") +
    coord_cartesian(
      xlim = c(0.48, 2.56),
      ylim = c(0, prepared$total + 1.25),
      clip = "off",
      expand = FALSE
    ) +
    labs(title = disease, fill = NULL) +
    theme_void(base_family = font_family, base_size = 8.3) +
    theme(
      plot.title = element_text(
        family = font_family, face = "bold", size = 10,
        hjust = 0.5, margin = margin(b = 5.5)
      ),
      legend.position = "none",
      plot.margin = margin(t = 2, r = 5, b = 1, l = 5)
    )
}

p_breast <- make_panel(
  data[data$disease == "Breast Neoplasms", ],
  "Breast Neoplasms"
)
p_ad <- make_panel(
  data[data$disease == "Alzheimer's Disease", ],
  "Alzheimer's Disease"
)

legend_data <- data.frame(
  x = c(0.72, 2.18, 3.63),
  movement = factor(c("Promoted", "Unchanged", "Demoted"),
                    levels = c("Promoted", "Unchanged", "Demoted"))
)

legend_plot <- ggplot(legend_data, aes(x = x, y = 1, fill = movement,
                                       colour = movement)) +
  geom_point(shape = 22, size = 3.0, stroke = 0.5) +
  geom_text(aes(x = x + 0.18, label = movement), hjust = 0,
            family = font_family, size = 2.55, colour = text_color) +
  scale_fill_manual(values = palette, guide = "none") +
  scale_colour_manual(values = palette_edge, guide = "none") +
  coord_cartesian(xlim = c(0.42, 4.55), ylim = c(0.75, 1.25),
                  clip = "off", expand = FALSE) +
  theme_void(base_family = font_family) +
  theme(plot.margin = margin(0, 0, 0, 0))

p_breast <- p_breast + labs(tag = "a")
p_ad <- p_ad + labs(tag = "b")

figure <- legend_plot / (p_breast | p_ad) +
  plot_layout(heights = c(0.13, 1), widths = 1) &
  theme(
    plot.tag = element_text(
      family = font_family, face = "bold", size = 10,
      colour = text_color
    )
  )

svglite::svglite(
  file.path(root, "fig4.svg"), width = width_in,
  height = height_in, pointsize = 10
)
print(figure)
dev.off()

grDevices::cairo_pdf(
  file.path(root, "fig4.pdf"), width = width_in,
  height = height_in, family = font_family, pointsize = 10
)
print(figure)
dev.off()

grDevices::tiff(
  file.path(root, "fig4.tiff"), width = 4200, height = 3540,
  units = "px", res = 1200, compression = "lzw", type = "cairo",
  pointsize = 10
)
print(figure)
dev.off()

grDevices::png(
  file.path(root, "fig4_preview.png"), width = 2100, height = 1770,
  units = "px", res = 600, type = "cairo", pointsize = 10
)
print(figure)
dev.off()
