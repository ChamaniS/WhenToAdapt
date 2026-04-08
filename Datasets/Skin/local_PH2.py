import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os .environ ["CUDA_LAUNCH_BLOCKING"]="1"

import copy 
import time 
import glob 
import cv2 
import numpy as np 
import torch 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import torch .optim as optim 
from sklearn .metrics import jaccard_score 
from torch .utils .data import DataLoader ,Dataset 
import segmentation_models_pytorch as smp 
from torch .optim .lr_scheduler import CosineAnnealingWarmRestarts 
from models .UNET import UNET 
import torch .nn as nn 
DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
from config import Config 
config =Config ()
img_size =config .DATA_IMG_SIZE 
patch_size =config .MODEL_SWIN_PATCH_SIZE 

class SkinPairDataset (Dataset ):
    def __init__ (self ,img_dir ,mask_dir ,transform =None ,img_exts =None ,mask_exts =None ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .transform =transform 

        if img_exts is None :
            img_exts =(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        if mask_exts is None :
            mask_exts =(".png",".jpg",".bmp",".tif",".tiff")

        self .img_exts =tuple (e .lower ()for e in img_exts )
        self .mask_exts =tuple (e .lower ()for e in mask_exts )

        img_files =[]
        for ext in self .img_exts :
            img_files .extend (glob .glob (os .path .join (self .img_dir ,f"*{ext}")))
        img_files =sorted (img_files )

        pairs =[]
        missing_masks =0 

        for img_path in img_files :
            stem =os .path .splitext (os .path .basename (img_path ))[0 ]
            mask_path =None 

            for mext in self .mask_exts :
                candidate =os .path .join (self .mask_dir ,stem +mext )
                if os .path .exists (candidate ):
                    mask_path =candidate 
                    break 

            if mask_path is None :
                alt_patterns =[
                stem +"_mask",
                stem +"-mask",
                stem +"_lesion",
                stem .replace ("_lesion","")
                ]
                for pat in alt_patterns :
                    for mext in self .mask_exts :
                        candidate =os .path .join (self .mask_dir ,pat +mext )
                        if os .path .exists (candidate ):
                            mask_path =candidate 
                            break 
                    if mask_path :
                        break 

            if mask_path is None :
                missing_masks +=1 
                continue 

            pairs .append ((img_path ,mask_path ))

        if len (pairs )==0 :
            raise ValueError (
            f"No image-mask pairs found in {img_dir} / {mask_dir}. "
            f"Skipped {missing_masks} images due to missing masks."
            )

        self .pairs =pairs 

        if missing_masks >0 :
            print (f"Warning: {missing_masks} images in {img_dir} had no matching masks and were skipped.")

    def __len__ (self ):
        return len (self .pairs )

    def __getitem__ (self ,idx ):
        img_path ,mask_path =self .pairs [idx ]

        img =cv2 .imread (img_path ,cv2 .IMREAD_COLOR )
        if img is None :
            raise RuntimeError (f"Failed to read image: {img_path}")
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        mask =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
        if mask is None :
            raise RuntimeError (f"Failed to read mask: {mask_path}")

        if mask .ndim ==3 :
            mask =cv2 .cvtColor (mask ,cv2 .COLOR_BGR2GRAY )

        mask =np .asarray (mask )
        mask =(mask >0 ).astype (np .uint8 )

        if self .transform :
            augmented =self .transform (image =img ,mask =mask )
            img =augmented ["image"]
            mask =augmented ["mask"]
        else :
            img =img .astype (np .float32 )/255.0 
            img =np .transpose (img ,(2 ,0 ,1 ))
            mask =np .expand_dims (mask .astype (np .float32 ),0 )

        return img ,mask 

def get_loaders_train (dataset_class ,train_img ,train_mask ,batch_size ,transform ,num_workers ,img_exts =None ,mask_exts =None ):
    train_ds =dataset_class (train_img ,train_mask ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    train_loader =DataLoader (train_ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =True )
    return train_loader 

def get_loaders_val (dataset_class ,val_img ,val_mask ,batch_size ,transform ,num_workers ,img_exts =None ,mask_exts =None ):
    val_ds =dataset_class (val_img ,val_mask ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    val_loader =DataLoader (val_ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =False )
    return val_loader 

def get_loader_test (dataset_class ,test_img ,test_mask ,transform ,img_exts =None ,mask_exts =None ):
    test_ds =dataset_class (test_img ,test_mask ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    test_loader =DataLoader (test_ds ,batch_size =1 ,shuffle =False )
    return test_loader 


class ComboLoss (nn .Module ):
    def __init__ (self ,num_classes ):
        super ().__init__ ()
        self .dice =smp .losses .DiceLoss (smp .losses .MULTICLASS_MODE ,from_logits =True )
        self .ce =nn .CrossEntropyLoss ()

    def forward (self ,logits ,targets ):
        return 1.0 *self .dice (logits ,targets )+0.0 *self .ce (logits ,targets )


def safe_jaccard_per_class (targets ,preds ,num_classes ):
    return jaccard_score (
    targets .cpu ().flatten (),
    preds .cpu ().flatten (),
    average =None ,
    labels =list (range (num_classes )),
    zero_division =0 
    )

def accumulate_binary_counts (preds_label ,targets ):
    preds_fg =(preds_label ==1 )
    targets_fg =(targets ==1 )

    TP =((preds_fg ==1 )&(targets_fg ==1 )).sum ().item ()
    TN =((preds_fg ==0 )&(targets_fg ==0 )).sum ().item ()
    FP =((preds_fg ==1 )&(targets_fg ==0 )).sum ().item ()
    FN =((preds_fg ==0 )&(targets_fg ==1 )).sum ().item ()

    return TP ,TN ,FP ,FN 

def compute_metrics_from_counts (TP ,TN ,FP ,FN ,smooth =1e-6 ):
    dice_with_bg =(2 *TP +smooth )/(2 *TP +FP +FN +smooth )
    iou_with_bg =(TP +smooth )/(TP +FP +FN +smooth )
    acc =(TP +TN )/(TP +TN +FP +FN +smooth )
    precision =(TP +smooth )/(TP +FP +smooth )
    recall =(TP +smooth )/(TP +FN +smooth )
    specificity =(TN +smooth )/(TN +FP +smooth )

    if TP +FN ==0 :
        dice_no_bg =1.0 if (TP +FP ==0 )else 0.0 
        iou_no_bg =1.0 if (TP +FP ==0 )else 0.0 
    else :
        dice_no_bg =(2 *TP +smooth )/(2 *TP +FP +FN +smooth )
        iou_no_bg =(TP +smooth )/(TP +FP +FN +smooth )

    return {
    "dice_with_bg":dice_with_bg ,
    "dice_no_bg":dice_no_bg ,
    "iou_with_bg":iou_with_bg ,
    "iou_no_bg":iou_no_bg ,
    "accuracy":acc ,
    "precision":precision ,
    "recall":recall ,
    "specificity":specificity ,
    }

def format_metric_dict (prefix ,metrics ):
    return (
    f"[{prefix}] "
    f"Dice(w/ bg): {metrics['dice_with_bg']:.4f} | "
    f"Dice(no bg): {metrics['dice_no_bg']:.4f} | "
    f"IoU(w/ bg): {metrics['iou_with_bg']:.4f} | "
    f"IoU(no bg): {metrics['iou_no_bg']:.4f} | "
    f"Acc: {metrics['accuracy']:.4f} | "
    f"Prec: {metrics['precision']:.4f} | "
    f"Recall: {metrics['recall']:.4f} | "
    f"Spec: {metrics['specificity']:.4f}"
    )


def eval_performance (loader ,model ,loss_fn ,num_classes ):
    model .eval ()

    val_running_loss =0.0 
    valid_running_correct =0.0 
    valid_iou_score_class =[0.0 ]*num_classes 

    TP =TN =FP =FN =0 

    with torch .no_grad ():
        for x ,y in loader :
            if isinstance (x ,np .ndarray ):
                x =torch .from_numpy (x )
            if isinstance (y ,np .ndarray ):
                y =torch .from_numpy (y )

            x =x .to (DEVICE )
            y =y .long ().to (DEVICE )

            predictions =model (x )
            loss =loss_fn (predictions ,y )

            preds =torch .argmax (predictions ,dim =1 )

            equals =preds ==y 
            valid_running_correct +=torch .mean (equals .float ()).item ()
            val_running_loss +=loss .item ()

            iou_sklearn =safe_jaccard_per_class (y ,preds ,num_classes )
            for i in range (num_classes ):
                valid_iou_score_class [i ]+=iou_sklearn [i ]

            bTP ,bTN ,bFP ,bFN =accumulate_binary_counts (preds ,y )
            TP +=bTP 
            TN +=bTN 
            FP +=bFP 
            FN +=bFN 

    dataset_size =len (loader .dataset )
    epoch_loss =val_running_loss /max (len (loader ),1 )
    epoch_acc =100.0 *(valid_running_correct /max (len (loader ),1 ))

    epoch_iou_class =[v /max (len (loader ),1 )for v in valid_iou_score_class ]
    epoch_iou_withbackground =sum (epoch_iou_class )/num_classes 
    epoch_iou_nobackground =sum (epoch_iou_class [1 :])/(num_classes -1 )if num_classes >1 else 0.0 

    metrics =compute_metrics_from_counts (TP ,TN ,FP ,FN )

    return epoch_loss ,epoch_acc ,epoch_iou_withbackground ,epoch_iou_nobackground ,epoch_iou_class ,metrics 

def test (loader ,model ,loss_fn ,num_classes ):
    model .eval ()

    test_running_loss =0.0 
    test_running_correct =0.0 
    test_iou_score_class =[0.0 ]*num_classes 

    TP =TN =FP =FN =0 

    with torch .no_grad ():
        for x ,y in loader :
            if isinstance (x ,np .ndarray ):
                x =torch .from_numpy (x )
            if isinstance (y ,np .ndarray ):
                y =torch .from_numpy (y )

            x =x .to (DEVICE )
            y =y .long ().to (DEVICE )

            predictions =model (x )
            loss =loss_fn (predictions ,y )

            preds =torch .argmax (predictions ,dim =1 )

            equals =preds ==y 
            test_running_correct +=torch .mean (equals .float ()).item ()
            test_running_loss +=loss .item ()

            iou_sklearn =safe_jaccard_per_class (y ,preds ,num_classes )
            for i in range (num_classes ):
                test_iou_score_class [i ]+=iou_sklearn [i ]

            bTP ,bTN ,bFP ,bFN =accumulate_binary_counts (preds ,y )
            TP +=bTP 
            TN +=bTN 
            FP +=bFP 
            FN +=bFN 

    epoch_loss =test_running_loss /max (len (loader ),1 )
    epoch_acc =100.0 *(test_running_correct /max (len (loader ),1 ))
    epoch_iou_class =[v /max (len (loader ),1 )for v in test_iou_score_class ]
    epoch_iou_withbackground =sum (epoch_iou_class )/num_classes 
    epoch_iou_nobackground =sum (epoch_iou_class [1 :])/(num_classes -1 )if num_classes >1 else 0.0 

    metrics =compute_metrics_from_counts (TP ,TN ,FP ,FN )

    return epoch_loss ,epoch_acc ,epoch_iou_withbackground ,epoch_iou_nobackground ,epoch_iou_class ,metrics 

def train (train_loader ,model ,optimizer ,scheduler ,loss_fn ,num_classes ):
    model .train ()

    loop =tqdm (train_loader )
    total_loss ,total_correct =0.0 ,0.0 
    iou_classes =[0.0 ]*num_classes 

    TP =TN =FP =FN =0 

    for data ,targets in loop :
        if isinstance (data ,np .ndarray ):
            data =torch .from_numpy (data )
        if isinstance (targets ,np .ndarray ):
            targets =torch .from_numpy (targets )

        data =data .to (DEVICE )
        targets =targets .long ().to (DEVICE )

        preds =model (data )
        loss =loss_fn (preds ,targets )

        preds_label =torch .argmax (preds ,dim =1 )
        total_correct +=(preds_label ==targets ).float ().mean ().item ()
        total_loss +=loss .item ()

        ious =safe_jaccard_per_class (targets ,preds_label ,num_classes )
        for i in range (num_classes ):
            iou_classes [i ]+=ious [i ]

        bTP ,bTN ,bFP ,bFN =accumulate_binary_counts (preds_label ,targets )
        TP +=bTP 
        TN +=bTN 
        FP +=bFP 
        FN +=bFN 

        optimizer .zero_grad ()
        loss .backward ()
        optimizer .step ()
        loop .set_postfix (loss =loss .item ())

    scheduler .step ()

    N =max (len (train_loader ),1 )
    avg_loss =total_loss /N 
    avg_acc =100.0 *total_correct /N 
    avg_ious =[v /N for v in iou_classes ]

    metrics =compute_metrics_from_counts (TP ,TN ,FP ,FN )

    return avg_loss ,avg_acc ,sum (avg_ious )/num_classes ,sum (avg_ious [1 :])/(num_classes -1 )if num_classes >1 else 0.0 ,avg_ious ,metrics 

def main ():
    LEARNING_RATE =1e-4 
    BATCH_SIZE =1 
    NUM_EPOCHS =120 
    NUM_WORKERS =1 
    IMAGE_HEIGHT =224 
    IMAGE_WIDTH =224 

    dataset_class =SkinPairDataset 
    NUM_CLASSES =2 

    TRAIN_IMG_DIR =r"xxxx\Data\skinlesions\PH2\train\images"
    TRAIN_MASK_DIR =r"xxxx\Data\skinlesions\PH2\train\masks"
    VAL_IMG_DIR =r"xxxx\Data\skinlesions\PH2\val\images"
    VAL_MASK_DIR =r"xxxx\Data\skinlesions\PH2\val\masks"
    TEST_IMG_DIR =r"xxxx\Data\skinlesions\PH2\test\images"
    TEST_MASK_DIR =r"xxxx\Data\skinlesions\PH2\test\masks"

    img_exts =(".jpg",".jpeg",".bmp",".png")
    mask_exts =(".png",".bmp",".jpg",".jpeg")

    val_transform =A .Compose ([
    A .Resize (height =IMAGE_HEIGHT ,width =IMAGE_WIDTH ),
    A .Normalize (mean =[0.0 ,0.0 ,0.0 ],std =[1.0 ,1.0 ,1.0 ],max_pixel_value =255.0 ),
    ToTensorV2 (),
    ])

    train_transform =A .Compose ([
    A .Resize (height =IMAGE_HEIGHT ,width =IMAGE_WIDTH ),
    A .Normalize (mean =[0.0 ,0.0 ,0.0 ],std =[1.0 ,1.0 ,1.0 ],max_pixel_value =255.0 ),
    ToTensorV2 (),
    ])

    train_loader =get_loaders_train (
    dataset_class ,
    TRAIN_IMG_DIR ,
    TRAIN_MASK_DIR ,
    BATCH_SIZE ,
    train_transform ,
    NUM_WORKERS ,
    img_exts =img_exts ,
    mask_exts =mask_exts ,
    )

    val_loader =get_loaders_val (
    dataset_class ,
    VAL_IMG_DIR ,
    VAL_MASK_DIR ,
    BATCH_SIZE ,
    val_transform ,
    NUM_WORKERS ,
    img_exts =img_exts ,
    mask_exts =mask_exts ,
    )

    test_loader =get_loader_test (
    dataset_class ,
    TEST_IMG_DIR ,
    TEST_MASK_DIR ,
    val_transform ,
    img_exts =img_exts ,
    mask_exts =mask_exts ,
    )

    model =UNET (in_channels =3 ,out_channels =NUM_CLASSES ).to (DEVICE )

    for param in model .parameters ():
        param .requires_grad =True 

    total_params =sum (p .numel ()for p in model .parameters ())
    print (f"[INFO]: {total_params:,} total parameters.")
    total_trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    print (f"[INFO]: {total_trainable_params:,} trainable parameters.")

    loss_fn =ComboLoss (NUM_CLASSES )
    optimizer =optim .AdamW (model .parameters (),lr =LEARNING_RATE ,weight_decay =1e-4 )
    scheduler =CosineAnnealingWarmRestarts (optimizer ,T_0 =5 ,T_mult =1 ,eta_min =1e-6 )

    best_iou =0.0 

    def print_results (mode ,loss ,acc ,iou_bg ,iou_no_bg ,iou_classes ,metrics ):
        print (f"\n[{mode}] Loss: {loss:.4f} | Acc: {acc:.2f}% | IoU(w/ bg): {iou_bg:.4f} | IoU(no bg): {iou_no_bg:.4f}")
        print ("Per-Class IoU:"," | ".join ([f"C{i}: {iou_classes[i]:.4f}"for i in range (len (iou_classes ))]))
        print (
        f"[{mode}] "
        f"Dice(w/ bg): {metrics['dice_with_bg']:.4f} | "
        f"Dice(no bg): {metrics['dice_no_bg']:.4f} | "
        f"IoU(w/ bg): {metrics['iou_with_bg']:.4f} | "
        f"IoU(no bg): {metrics['iou_no_bg']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Specificity: {metrics['specificity']:.4f}"
        )

    for epoch in range (NUM_EPOCHS ):
        print (f"[INFO]: Epoch {epoch + 1} of {NUM_EPOCHS}")

        train_results =train (train_loader ,model ,optimizer ,scheduler ,loss_fn ,NUM_CLASSES )
        val_results =eval_performance (val_loader ,model ,loss_fn ,NUM_CLASSES )

        train_loss ,train_acc ,train_iou_with_bg ,train_iou_no_bg ,train_iou_classes ,train_metrics =train_results 
        val_loss ,val_acc ,val_iou_with_bg ,val_iou_no_bg ,val_iou_classes ,val_metrics =val_results 

        print_results ("TRAIN",train_loss ,train_acc ,train_iou_with_bg ,train_iou_no_bg ,train_iou_classes ,train_metrics )
        print_results ("VAL",val_loss ,val_acc ,val_iou_with_bg ,val_iou_no_bg ,val_iou_classes ,val_metrics )

        os .makedirs ("BestModels",exist_ok =True )
        if val_iou_no_bg >best_iou :
            best_iou =val_iou_no_bg 
            torch .save (model .state_dict (),"BestModels/best_model_centralized_Unet_ph2.pth")
            print ("Model saved!")

    print ("[INFO]: Testing the best model...")
    model .load_state_dict (torch .load ("BestModels/best_model_centralized_Unet_ph2.pth",map_location =DEVICE ))

    test_results =test (test_loader ,model ,loss_fn ,NUM_CLASSES )
    test_loss ,test_acc ,test_iou_with_bg ,test_iou_no_bg ,test_iou_classes ,test_metrics =test_results 

    print_results ("TEST",test_loss ,test_acc ,test_iou_with_bg ,test_iou_no_bg ,test_iou_classes ,test_metrics )

if __name__ =="__main__":
    main ()