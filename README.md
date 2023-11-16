# tma_to_wsi

c-MYC and BCL2 positivity are important prognostic factors for diffuse large B-cell lymphoma. However, manual quantification is subject to significant intra- and inter-observer variability. We developed an automated method for quantification in whole-slide images of tissue sections where manual quantification requires evaluating large areas of tissue with possibly heterogeneous staining. We train this method using annotations of tumor positivity in smaller tissue microarray cores where expression and staining are more homogeneous and then translate this model to whole-slide images. 

#### Methods
Methods: Our method applies a technique called attention-based multiple instance learning to regress the proportion of c-MYC-positive and BCL2-positive tumor cells from pathologist-scored tissue microarray cores. This technique does not require annotation of individual cell nuclei and is trained instead on core-level annotations of percent tumor positivity. We translate this model to scoring of whole-slide images by tessellating the slide into smaller core-sized tissue regions and calculating an aggregate score. Our method was trained on a public tissue microarray dataset from Stanford and applied to whole-slide images from a geographically diverse multi-center cohort produced by the Lymphoma Epidemiology of Outcomes study.

#### Results

In tissue microarrays, the automated method had Pearson correlations of 0.843 and 0.919 with pathologist scores for c-MYC and BCL2, respectively. When utilizing standard clinical thresholds, the sensitivity/specificity of our method was 0.743 / 0.963 for c-MYC and 0.938 / 0.951 for BCL2. For double-expressors, sensitivity and specificity were 0.720 and 0.974. When translated to the external WSI dataset scored by two pathologists, Pearson correlation was 0.753 & 0.883 for c-MYC and 0.749 & 0.765 for BCL2, and sensitivity/specificity was 0.857/0.991 & 0.706/0.930 for c-MYC, 0.856/0.719 & 0.855/0.690 for BCL2, and 0.890/1.00 & 0.598/0.952 for double-expressors. Survival analysis demonstrates that for progression-free survival, model-predicted TMA scores significantly stratify double-expressors and non double-expressors (p=0.0345), whereas pathologist scores do not (p=0.128). 

#### Conclusions

We conclude that proportion of positive stains can be regressed using attention-based multiple instance learning, that these models generalize well to whole slide images, and that our models can provide non-inferior stratification of progression-free survival outcomes.

## Datasets

For training, we utilized the publicly available DLBCL-Morph dataset from Stanford consisting of digitized images of 378 TMA cores of DLBCL stained for c-MYC and BCL2. This can be freely downloaded at https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119702520.

For testing, an external dataset consisting of 52 WSIs of DLBCL tissue sections stained for c-MYC and 56 WSI of DLBCL stained for BCL2. This external dataset came from the LEO study and can be accessed with permission at https://leocohort.org/contact-leo/.

## Preprocessing

For training, patch-wise features are extracted using the third residual block of a pretrained Resnet50 and individually spatially averaged to yield 1024-dimensional feature vectors for each 224x224 patch at 20x magnification. Please see the processing_postprocessing code to apply this step to TMAs.

For testing, coordinates of foreground patches were clustered using k-means clustering (see Figure 3) to yield an average cluster size of 45±7 patches with a range from 6 to 80. This cluster size coincides with the average number of non-overlapping patches obtained from TMAs in the previous step. K-means clustering based on coordinates was motivated by the fact that TMA-shaped regions (i.e., circles) extracted from WSIs would necessarily overlap. k-means clustering creates convex polygons which not only prevent this overlap but also approximate the shape of a circle. This resulted in several “mini-bags” for each WSI. These mini-bags were passed through respective pre-trained AB-MIL models to yield several predictions per WSI. Please see the processing_postprocessing code to apply this step to WSIs.

## Model training

CLAM (https://github.com/mahmoodlab/CLAM) was modified to perform regression tasks. Please use main.py to train a regression model on TMAs (i.e. Stanford dataset). Use task codes "stanford_bcl2", "standford_bcl2_by_patient", "stanford_myc", "standford_myc_by_patient" to train TMA models. 

## Inference time

Use eval.py to apply TMA-trained models to TMAs and WSIs. Use task codes "stanford_myc_by_patient_on_leo_wsi_as_tmas" and "stanford_bcl2_by_patient_on_leo_wsi_as_tmas".

## Postprocessing

See preprocessing_and_postprocessing to generate tables and figures.
