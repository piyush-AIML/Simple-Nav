# Baseline Report (provisional)

*Generated 2026-09-02T00:33:36* — dataset: `provisional`

> Built from 301 pre-extracted frames (source video missing). Held-out labels are executor-generated coarse area labels. Re-run with real mapping+evaluation walkthroughs before research conclusions.

> **Labels:** full-data KMeans (k=11, silhouette-selected) as pseudo ground truth; train places mapped to pseudo-clusters by majority vote; metrics measure map generalization to unseen frames, not physical accuracy

## Graph
- nodes: 12 · edges: 13 · k: 12 · exemplars: 36

## Metrics (held-out)

| metric | value |
|---|---|
| top1_accuracy_self_consistency | 0.8852 |
| top3_accuracy_self_consistency | 0.918 |
| false_jump_rate_self_consistency | 0.1698 |
| transition_detection_recall_self_consistency | 1.0 |
| transition_detection_precision_self_consistency | 0.8571 |
| pseudo_cluster_count | 11 |
| latency_mean_ms | 22.21 |
| latency_p95_ms | 25.63 |

## Split
- total 301 · train 240 · held-out 61