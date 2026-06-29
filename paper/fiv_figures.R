#!/usr/bin/env Rscript
# FIV figures from LIVE results (CSV emitted from results/fiv_tabfact_scaled.json).
# No jsonlite dependency. Run: Rscript fiv_figures.R  (from paper/).

suppressMessages({ library(ggplot2); library(scales) })
outdir <- "figs"; dir.create(outdir, showWarnings = FALSE)
rates <- read.csv("../results/fiv_rates.csv", stringsAsFactors = FALSE)
byfr  <- read.csv("../results/fiv_by_framing.csv", stringsAsFactors = FALSE)

ink <- "#1A1A1A"; base_c <- "#C0392B"; skep_c <- "#1F6FB2"; grid_c <- "#DADADA"
theme_paper <- function(bs = 11) theme_minimal(base_size = bs) +
  theme(text = element_text(color = ink),
        plot.title = element_text(face = "bold", size = bs + 1),
        plot.subtitle = element_text(color = "#555555", size = bs - 1),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = grid_c, linewidth = 0.3),
        legend.position = "top", legend.title = element_blank())

rates$condition <- factor(ifelse(rates$condition == "protocol", "+ deliberate protocol", "baseline"),
                          levels = c("baseline", "+ deliberate protocol"))

long <- rbind(
  data.frame(condition = rates$condition, metric = "Framing flip rate (FIV)",
             value = rates$flip_rate, lo = rates$flip_lo, hi = rates$flip_hi),
  data.frame(condition = rates$condition, metric = "Wrong-flip rate (vs gold)",
             value = rates$wrong_flip, lo = rates$wf_lo, hi = rates$wf_hi))
pA <- ggplot(long, aes(metric, value, fill = condition)) +
  geom_col(position = position_dodge(0.7), width = 0.62) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(0.7), width = 0.18) +
  geom_text(aes(label = percent(value, accuracy = 1)),
            position = position_dodge(0.7), vjust = -0.6, size = 3, color = ink) +
  scale_fill_manual(values = c("baseline" = base_c, "+ deliberate protocol" = skep_c)) +
  scale_y_continuous(labels = percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Framing-Induced Variance on TabFact (live Gemma 4 31B, n=500, K=8, R=3)",
       subtitle = "Same table + claim; only the question framing changes. 95% bootstrap CIs.",
       x = NULL, y = "rate") + theme_paper()
ggsave(file.path(outdir, "fig_fiv_rates.pdf"), pA, width = 7.2, height = 3.6, device = cairo_pdf)

byfr <- byfr[order(byfr$wrong_share), ]
byfr$framing <- factor(byfr$framing, levels = byfr$framing)
pB <- ggplot(byfr, aes(wrong_share, framing)) +
  geom_col(fill = base_c, width = 0.64) +
  geom_text(aes(label = percent(wrong_share, accuracy = 0.1)), hjust = -0.15, size = 3, color = ink) +
  scale_x_continuous(labels = percent, limits = c(0, max(byfr$wrong_share) * 1.18),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Which framings reward-hack the verdict hardest?",
       subtitle = "Per-framing share of WRONG verdicts (baseline). 'lead_refuted' is the most damaging.",
       x = "wrong-verdict share", y = NULL) + theme_paper()
ggsave(file.path(outdir, "fig_fiv_by_framing.pdf"), pB, width = 7.2, height = 3.6, device = cairo_pdf)

# ---- Specification curve: per-variant wrong-flip, baseline vs +protocol ----
spec <- read.csv("../results/fiv_spec_curve.csv", stringsAsFactors = FALSE)
spec <- spec[order(spec$base_wrong), ]
spec$variant <- factor(spec$variant, levels = spec$variant)
mean_base <- mean(spec$base_wrong); mean_prot <- mean(spec$prot_wrong)
pC <- ggplot(spec) +
  geom_segment(aes(x = prot_wrong, xend = base_wrong, y = variant, yend = variant),
               color = grid_c, linewidth = 1.1) +
  geom_point(aes(base_wrong, variant, color = "baseline"), size = 3.2) +
  geom_point(aes(prot_wrong, variant, color = "+ deliberate protocol"),
             size = 3.2, shape = 21, fill = "white", stroke = 1.2) +
  geom_vline(xintercept = mean_base, linetype = "dashed", color = base_c, linewidth = 0.4) +
  geom_vline(xintercept = mean_prot, linetype = "dashed", color = skep_c, linewidth = 0.4) +
  scale_color_manual(values = c("baseline" = base_c, "+ deliberate protocol" = skep_c)) +
  scale_x_continuous(labels = percent, limits = c(0, max(spec$base_wrong) * 1.15),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Specification curve: wrong-flip across 5 prompt variants",
       subtitle = "Wrong-flip is positive in every variant; the protocol shifts it down each time. Dashed = cross-variant mean.",
       x = "wrong-verdict share", y = NULL) + theme_paper()
ggsave(file.path(outdir, "fig_fiv_spec_curve.pdf"), pC, width = 7.2, height = 3.6, device = cairo_pdf)
cat("FIV figures written to", normalizePath(outdir), "\n")
