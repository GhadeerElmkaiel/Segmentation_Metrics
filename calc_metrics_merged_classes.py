from utils.metrics import *
import numpy as np
import re
import glob
import PIL.Image as Image
import pandas as pd

split = "test/"
# root = "/home/ghadeer/Projects/Datasets/Sber2400/"
root = "/home/ghadeer/Projects/Datasets/SberMerged/"
root_to_masks = root + split
path_to_target_glass = root_to_masks+"Glass/"
path_to_target_all = root_to_masks+"All_Optical/"
path_to_target_G_M = root_to_masks+"Glass_and_Mirrors/"
path_to_target_Floor = root_to_masks+"Floor/"
path_to_target_All_Floor = root_to_masks+"All_Floor/"
path_to_target_mirror = root_to_masks+"Mirror/"
path_to_target_OOS = root_to_masks+"Other optical surface/"
path_to_target_FU = root_to_masks+"Floor under obstacle/"

# path_to_target_res = "/home/ghadeer/Projects/Trans2Seg/runs/visual/with_adaptive_palette-removed/"

res_root = "/home/ghadeer/Projects/SegFormer/results/SberMerged_b4_512_160000_images/"
# res_root = "runs/visual/training_on_both_50_merge_masjs/"
# res_root = "/home/ghadeer/Projects/Trans2Seg/runs/visual/training_on_both_50_merged_masks/Merged_mask/"

path_to_target_res = res_root+"Orig/"
path_to_target_flipped = res_root+ "Flipped/"
path_to_merged = res_root+ "Merged_mask/"
# path_to_target_and = res_root+ "And/"
# path_to_target_or = res_root+ "Or/"

path_to_testing = res_root
# path_to_testing = path_to_merged

images = []
for img in glob.glob(path_to_testing+"*"):
    images.append(re.split('[/]',img)[-1])

###########################
# images = only_glass

metrics = ["IoU", "Acc", "Recall", "Precision", "FPR"]
metrics_vals_placeholder = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}

IoU_res = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}
acc_res = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}
FPR_res = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}
precision_res = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}
recall_res = {"Glass": [], "Mirror":[], "OOS":[], "All_Optical":[], "Glass_Mirrors":[], "Floor":[], "FU":[], "All_Floor":[], "Back_Ground":[]}

confusion_matrix_merged = {"All_Optical": {}, "All_Floor":{}, "Back_Ground":{}}
for key in confusion_matrix_merged.keys():
    confusion_matrix_merged[key] = {"All_Optical": 0, "All_Floor":0, "Back_Ground":0}


#################################################################
src_palette = Image.open("/home/ghadeer/Projects/Datasets/Sber3500/all_palette.png")
src_palette = src_palette.convert("P", palette=Image.ADAPTIVE)
#################################################################

for name in images:
    mask_arrs = {}
    result_arrs = {}
    all_mask = Image.open(path_to_target_all+name)
    all_mask = all_mask.convert("L")
    all_arr = np.array(all_mask, dtype="?")

    glass_mask = Image.open(path_to_target_glass+name)
    glass_mask = glass_mask.convert("L")
    glass_arr = np.array(glass_mask, dtype="?")

    G_M_mask = Image.open(path_to_target_G_M+name)
    G_M_mask = G_M_mask.convert("L")
    G_M_arr = np.array(G_M_mask, dtype="?")

    Floor_mask = Image.open(path_to_target_Floor+name)
    Floor_mask = Floor_mask.convert("L")
    Floor_arr = np.array(Floor_mask, dtype="?")

    Mirror_mask = Image.open(path_to_target_mirror+name)
    Mirror_mask = Mirror_mask.convert("L")
    Mirror_arr = np.array(Mirror_mask, dtype="?")

    OOS_mask = Image.open(path_to_target_OOS+name)
    OOS_mask = OOS_mask.convert("L")
    OOS_arr = np.array(OOS_mask, dtype="?")

    FU_mask = Image.open(path_to_target_FU+name)
    FU_mask = FU_mask.convert("L")
    FU_arr = np.array(FU_mask, dtype="?")

    All_Floor_mask = Image.open(path_to_target_All_Floor+name)
    All_Floor_mask = All_Floor_mask.convert("L")
    All_Floor_arr = np.array(All_Floor_mask, dtype="?")

    mask_arrs["Glass"] = glass_arr
    mask_arrs["Mirror"] = Mirror_arr
    mask_arrs["OOS"] = OOS_arr
    mask_arrs["All_Optical"] = all_arr
    mask_arrs["Glass_Mirrors"] = G_M_arr
    mask_arrs["Floor"] = Floor_arr
    mask_arrs["FU"] = FU_arr
    mask_arrs["All_Floor"] = All_Floor_arr
    mask_arrs["Back_Ground"] = np.logical_not(np.logical_or(All_Floor_arr, all_arr))

    reslulted_mask = Image.open(path_to_testing+name)
    #################################################################
    reslulted_mask = reslulted_mask.quantize(palette=src_palette)
    #################################################################

    reslulted_mask = reslulted_mask.resize(all_mask.size, Image.BILINEAR)
    # reslulted_mask.show()
    reslulted_arr = np.array(reslulted_mask)
    img_size = reslulted_arr.size

    # vals = np.unique(reslulted_arr)
    # test = Image.open(path_to_testing+"1.png")
    # test.show()
    # vals = np.unique(test)
    
    reslulted_glass = reslulted_arr==1
    reslulted_mirror = reslulted_arr==0
    reslulted_oss = reslulted_arr==3
    reslulted_floor = reslulted_arr==4
    reslulted_fu = reslulted_arr==2


    reslulted_all_optical = np.logical_or( np.logical_or(reslulted_glass, reslulted_mirror),reslulted_oss)
    reslulted_glass_mirrors = np.logical_or(reslulted_glass, reslulted_mirror)
    reslulted_all_floor = np.logical_or(reslulted_floor, reslulted_fu)


    result_arrs["Glass"] = reslulted_glass
    result_arrs["Mirror"] = reslulted_mirror
    result_arrs["OOS"] = reslulted_oss
    result_arrs["All_Optical"] = reslulted_all_optical
    result_arrs["Glass_Mirrors"] = reslulted_glass_mirrors
    result_arrs["Floor"] = reslulted_floor
    result_arrs["FU"] = reslulted_fu
    result_arrs["All_Floor"] = reslulted_all_floor
    result_arrs["Back_Ground"] = np.logical_not(np.logical_or(reslulted_all_optical, reslulted_all_floor))
    # reslulted_all_optical = reslulted_arr==0
    # reslulted_floor = reslulted_arr==1
    ##############################################
    for key in result_arrs.keys():
        intersection = np.sum(np.logical_and(result_arrs[key], mask_arrs[key]))
        union = np.sum(np.logical_or(result_arrs[key], mask_arrs[key]))
        false_pos = np.sum(np.logical_and(result_arrs[key], np.logical_not(mask_arrs[key])))
        if union == 0:
            iou_res = 1
        else:
            iou_res = intersection/union

        acc_all = np.sum(np.logical_not(np.logical_xor(result_arrs[key], mask_arrs[key])))
        acc_val = acc_all/img_size

        real_mask_size = np.sum(mask_arrs[key])
        if real_mask_size > 0:
            recall_val = intersection/real_mask_size
        else:
            recall_val = 1

        pred_mask_size = np.sum(result_arrs[key])
        if pred_mask_size > 0:
            precision_val = intersection/pred_mask_size
        else:
            precision_val = 1

        not_mask_size = np.sum(np.logical_not(mask_arrs[key]))
        if not_mask_size > 0:
            FPR_val = false_pos/not_mask_size
        else:
            FPR_val = 0

        IoU_res[key].append(iou_res)
        acc_res[key].append(acc_val)
        precision_res[key].append(precision_val)
        recall_res[key].append(recall_val)
        FPR_res[key].append(FPR_val)

        # Clac Confusion Matrix
    for key in confusion_matrix_merged.keys():
        for key2 in confusion_matrix_merged.keys():
            intersection = np.sum(np.logical_and(result_arrs[key], mask_arrs[key2]))
            confusion_matrix_merged[key][key2] += intersection


final_res = {"IoU":{}, "Acc":{}, "Precision":{}, "Recall":{}, "FPR":{}}
for key in acc_res.keys():
    final_res["IoU"][key] = np.mean(IoU_res[key])
    final_res["Acc"][key] = np.mean(acc_res[key])
    final_res["Precision"][key] = np.mean(precision_res[key])
    final_res["Recall"][key] = np.mean(recall_res[key])
    final_res["FPR"][key] = np.mean(FPR_res[key])


dataframe = pd.DataFrame.from_dict(final_res["IoU"], orient='index', columns=["IoU"])
dataframe.insert(1, "Acc", final_res["Acc"].values())
dataframe.insert(1, "Precision", final_res["Precision"].values())
dataframe.insert(1, "Recall", final_res["Recall"].values())
dataframe.insert(1, "FPR", final_res["FPR"].values())

# Normalizing the confusion matrix
confusion_matrix_merged_sums = {}
for key2 in confusion_matrix_merged.keys():
    current_sum = 0
    for key in confusion_matrix_merged.keys():
        current_sum+= confusion_matrix_merged[key][key2]
    confusion_matrix_merged_sums[key2] = current_sum
for key2 in confusion_matrix_merged.keys():
    for key in confusion_matrix_merged.keys():
        confusion_matrix_merged[key][key2] = confusion_matrix_merged[key][key2]/confusion_matrix_merged_sums[key2]


confusion_matrix_merged_dataframe =  pd.DataFrame.from_dict(confusion_matrix_merged)

root_to_data = res_root.split('/')
csv_name = root_to_data[-2]
dataframe.to_csv(csv_name+".csv")
confusion_matrix_merged_dataframe.to_csv(csv_name+"_confusion_matrix_merged.csv")
print(dataframe)
print("Done")
