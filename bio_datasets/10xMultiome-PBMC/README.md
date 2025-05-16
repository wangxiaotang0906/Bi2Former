## Cell-Type-Level Regulatory Maps.

The given TF_data is the experimentally derived TF binding scores of CD4 cells. To further validate the biological relevance of our model, we compare the cumulative attention scores against experimentally derived TF binding scores, which reflect the actual activation strength of ATAC peaks in each cell type and serve as a proxy for the ground truth. The results show that the cumulative attention scores for CD4 cells exhibit strong agreement with CD4-specific TF binding signals, indicating that our model successfully identifies biologically meaningful regulatory relationships. 

<img src=".../figs/cd4_cd4.png">
Comparison between TF binding scores and cumulative attention scores within the same cell type (CD4). The cumulative attention scores produced by our model align closely with the TF binding intensity in CD4 cells, suggesting that the learned crossmodal interactions accurately capture cell-type-specific regulatory signals.

<img src=".../figs/cd4_cd14.png">
Comparison between TF binding scores from CD4 cells and cumulative attention scores in CD14 cells. CD14 cells exhibit distinct peak activation patterns with the CD4 ground truth, highlighting the distinct regulatory patterns across cell types.


