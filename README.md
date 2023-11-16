# tma_to_wsi
0) Datasets
For training, we utilized the publicly available DLBCL-Morph dataset from Stanford consisting of digitized images of 378 TMA cores of DLBCL stained for c-MYC and BCL2. This can be freely downloaded at https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119702520.
For testing, an external dataset consisting of 52 WSIs of DLBCL tissue sections stained for c-MYC and 56 WSI of DLBCL stained for BCL2. This external dataset came from the LEO study and can be accessed with permission at https://leocohort.org/contact-leo/.
1) Preprocessing
For training, patch-wise features are extracted using the third residual block of a pretrained Resnet50 and individually spatially averaged to yield 1024-dimensional feature vectors for each 224x224 patch at 20x magnification. Please see the processing_postprocessing code to apply this step to TMAs.
For testing, coordinates of foreground patches were clustered using k-means clustering (see Figure 3) to yield an average cluster size of 45±7 patches with a range from 6 to 80. This cluster size coincides with the average number of non-overlapping patches obtained from TMAs in the previous step. K-means clustering based on coordinates was motivated by the fact that TMA-shaped regions (i.e., circles) extracted from WSIs would necessarily overlap. k-means clustering creates convex polygons which not only prevent this overlap but also approximate the shape of a circle. This resulted in several “mini-bags” for each WSI. These mini-bags were passed through respective pre-trained AB-MIL models to yield several predictions per WSI. Please see the processing_postprocessing code to apply this step to WSIs.
2) Model training
CLAM (https://github.com/mahmoodlab/CLAM) was modified to perform regression tasks. Please use main.py to train a regression model on TMAs (i.e. Stanford dataset). Use task codes "stanford_bcl2", "standford_bcl2_by_patient", "stanford_myc", "standford_myc_by_patient" to train TMA models.  
3) Inference time
Use eval.py to apply TMA-trained models to TMAs and WSIs. Use task codes "stanford_myc_by_patient_on_leo_wsi_as_tmas" and "stanford_bcl2_by_patient_on_leo_wsi_as_tmas"
4) Postprocessing
See preprocessing_and_postprocessing to generate tables and figures.
