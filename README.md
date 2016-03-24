# Computer_Vision_Proj2_Local_Feature_Matching
Project 2 by Tiancheng Gong(tgong7)

Here are the settings and results:

|Name:	                          |Distance Ratio|Good Matches|Bad Matches|Total Matches|Accuracy|
|:--------------------------------|:-------------|:-----------|:----------|:------------|:-------|
|Notre Dame                       |0.6           |57          |14         |71           |80%     |
|Mount Rushmore                   |0.6           |110         |6          |116          |95%     |
|Episcopal Gaudi (without scaling)|0.85          |3           |66         |69           |4.3%    |
|Episcopal Gaudi (with scaling)   |0.85          |13          |44         |57           |29.5%   |

Extra Feature Implementation:<br/>
1. Adaptive threshold for Harris Score (mean value of the harris matrix)<br/>
2. Find the scale of the image during finding interesting points and apply the result to the feature width during getting features

See more details at: [Computer Vision Project 1 Website](http://all4win.github.io/projects/cv_proj1/index.html)
