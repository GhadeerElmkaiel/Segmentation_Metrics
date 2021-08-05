# Metrics For Segmentation
Here there is an implimentation for segmentation metrics such as **IoU**, **Acc**, **FPR**, **Precision**, **Recall**, and **Confusion Matrix**

The code ```calc_all_metrics.py``` creates 5 different files:  
1- The first file contains the following metrics (Mean IoU, Median IoU, Mean Acc, Median Acc, FPR, Mean FPR, Median FPR, Recall, Mean Recall, Median Recall, Precision, Mean Precision, Median Precision) ```for example: SberMerged_b0_1024_160000_masks.csv```  
2- Confusion matrix as numbers. ```SberMerged_b0_1024_160000_masks_confusion_matrix.csv```    
3- Confusion matrix normalized for Recall. ```SberMerged_b0_1024_160000_masks_confusion_matrix_recall.csv```    
4- Confusion matrix normalized for Precision. ```SberMerged_b0_1024_160000_masks.csv```    
5- Confusion matrix for merged classes. ```SberMerged_b0_1024_160000_masks_confusion_matrix_merged.csv```    