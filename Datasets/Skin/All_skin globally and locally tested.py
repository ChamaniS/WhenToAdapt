import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os .environ ["CUDA_LAUNCH_BLOCKING"]="1"

import glob 
import cv2 
import numpy as np 
import copy 
import time 
import torch 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import torch .optim as optim 
from torch .utils .data import DataLoader ,ConcatDataset ,Dataset 
from torch .optim .lr_scheduler import CosineAnnealingWarmRestarts 
from models .UNET import UNET 
from models .DuckNet import DuckNet 
import torch .nn as nn 
import segmentation_models_pytorch as smp 

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"




def compute_metrics (pred ,target ,smooth =1e-6 ):
    pred =torch .sigmoid (pred )
    pred =(pred >0.5 ).float ()
    target =(target >0.5 ).float ()

    TP =(pred *target ).sum ().item ()
    TN =((1 -pred )*(1 -target )).sum ().item ()
    FP =(pred *(1 -target )).sum ().item ()
    FN =((1 -pred )*target ).sum ().item ()

    dice_with_bg =(2 *TP +smooth )/(2 *TP +FP +FN +smooth )
    iou_with_bg =(TP +smooth )/(TP +FP +FN +smooth )
    acc =(TP +TN )/(TP +TN +FP +FN +smooth )
    precision =(TP +smooth )/(TP +FP +smooth )
    recall =(TP +smooth )/(TP +FN +smooth )
    specificity =(TN +smooth )/(TN +FP +smooth )

    if target .sum ()==0 :
        dice_no_bg =1.0 if pred .sum ()==0 else 0.0 
        iou_no_bg =1.0 if pred .sum ()==0 else 0.0 
    else :
        intersection =(pred *target ).sum ().item ()
        dice_no_bg =(2 *intersection +smooth )/(pred .sum ().item ()+target .sum ().item ()+smooth )
        iou_no_bg =(intersection +smooth )/(pred .sum ().item ()+target .sum ().item ()-intersection +smooth )

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

def aggregate_metrics (metrics_list ):
    if not metrics_list :
        return {}
    agg ={}
    for key in metrics_list [0 ].keys ():
        agg [key ]=sum (m [key ]for m in metrics_list )/len (metrics_list )
    return agg 





class SkinPairDataset (Dataset ):
    def __init__ (self ,img_dir ,mask_dir ,transform =None ,img_exts =None ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .transform =transform 

        if img_exts is None :
            img_exts =(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        self .img_exts =tuple (e .lower ()for e in img_exts )

        img_files =[]
        for ext in self .img_exts :
            img_files .extend (glob .glob (os .path .join (self .img_dir ,f"*{ext}")))
        img_files =sorted (img_files )

        pairs =[]
        missing_masks =0 
        for img_path in img_files :
            stem =os .path .splitext (os .path .basename (img_path ))[0 ]

            candidates =glob .glob (os .path .join (self .mask_dir ,stem +".*"))
            mask_path =candidates [0 ]if candidates else None 

            if mask_path is None :
                alt_patterns =[stem +"_mask",stem +"-mask",stem +"_lesion",stem .replace ("_lesion","")]
                for pat in alt_patterns :
                    cand =glob .glob (os .path .join (self .mask_dir ,pat +".*"))
                    if cand :
                        mask_path =cand [0 ]
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

        if self .transform is not None :
            augmented =self .transform (image =img ,mask =mask )
            img =augmented ["image"]
            mask =augmented ["mask"]
        else :
            img =img .astype (np .float32 )/255.0 
            img =np .transpose (img ,(2 ,0 ,1 ))
            mask =np .expand_dims (mask .astype (np .float32 ),0 )
            img =torch .from_numpy (img )
            mask =torch .from_numpy (mask )

        return img ,mask 




def build_concat_dataset (dataset_class ,img_dirs ,mask_dirs ,transform =None ):
    datasets =[]
    for img_dir ,mask_dir in zip (img_dirs ,mask_dirs ):
        if os .path .exists (img_dir )and os .path .exists (mask_dir ):
            datasets .append (dataset_class (img_dir ,mask_dir ,transform =transform ))
        else :
            print (f"[WARN] Skipping missing dataset: {img_dir} or {mask_dir}")
    if len (datasets )==0 :
        raise RuntimeError ("No valid datasets found for the provided directories.")
    return ConcatDataset (datasets )

def build_single_dataset (dataset_class ,img_dir ,mask_dir ,transform =None ):
    if not os .path .exists (img_dir )or not os .path .exists (mask_dir ):
        raise FileNotFoundError (f"Missing dataset: {img_dir} or {mask_dir}")
    return dataset_class (img_dir ,mask_dir ,transform =transform )

def get_loaders (dataset_class ,train_img_dirs ,train_mask_dirs ,
val_img_dirs ,val_mask_dirs ,
test_img_dirs ,test_mask_dirs ,
batch_size ,train_transform ,val_transform ,
num_workers ):

    train_ds =build_concat_dataset (dataset_class ,train_img_dirs ,train_mask_dirs ,transform =train_transform )
    val_ds =build_concat_dataset (dataset_class ,val_img_dirs ,val_mask_dirs ,transform =val_transform )
    test_ds =build_concat_dataset (dataset_class ,test_img_dirs ,test_mask_dirs ,transform =val_transform )

    train_loader =DataLoader (train_ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =True )
    val_loader =DataLoader (val_ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =False )
    test_loader =DataLoader (test_ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =False )

    return train_loader ,val_loader ,test_loader 

def get_loader_from_dirs (img_dir ,mask_dir ,transform ,batch_size ,num_workers ,shuffle =False ):
    ds =build_single_dataset (SkinPairDataset ,img_dir ,mask_dir ,transform =transform )
    return DataLoader (ds ,batch_size =batch_size ,num_workers =num_workers ,shuffle =shuffle )




def get_loss_fn (net ,device ):
    return smp .losses .DiceLoss (mode ="binary",from_logits =True )




def train (train_loader ,model ,optimizer ,scheduler ,loss_fn ):
    model .train ()
    loop =tqdm (train_loader )
    total_loss ,total_correct =0.0 ,0.0 
    all_metrics =[]

    for data ,targets in loop :
        if isinstance (data ,np .ndarray ):
            data =torch .from_numpy (data )
        if isinstance (targets ,np .ndarray ):
            targets =torch .from_numpy (targets )

        data =data .to (DEVICE )
        targets =targets .to (DEVICE )

        preds =model (data )

        if preds .shape [1 ]==1 :
            loss =loss_fn (preds ,targets .unsqueeze (1 ).float ())
            preds_label =(torch .sigmoid (preds )>0.5 ).long ().squeeze (1 )
        else :
            loss =loss_fn (preds ,targets .long ())
            preds_label =torch .argmax (preds ,dim =1 )

        total_correct +=(preds_label ==targets ).float ().mean ().item ()
        total_loss +=loss .item ()

        if preds .shape [1 ]==1 :
            batch_metrics =compute_metrics (preds ,targets .unsqueeze (1 ))
            all_metrics .append (batch_metrics )

        optimizer .zero_grad ()
        loss .backward ()
        optimizer .step ()
        loop .set_postfix (loss =loss .item ())

    scheduler .step ()

    num_batches =max (len (train_loader ),1 )
    avg_loss =total_loss /num_batches 
    avg_acc =100.0 *total_correct /num_batches 
    avg_metrics =aggregate_metrics (all_metrics )if all_metrics else {}

    return avg_loss ,avg_acc ,avg_metrics 

@torch .no_grad ()
def eval_performance (loader ,model ,loss_fn ):
    model .eval ()
    val_running_loss =0.0 
    val_running_correct =0.0 
    all_metrics =[]

    for x ,y in loader :
        if isinstance (x ,np .ndarray ):
            x =torch .from_numpy (x )
        if isinstance (y ,np .ndarray ):
            y =torch .from_numpy (y )

        x =x .to (DEVICE )
        y =y .to (DEVICE )

        predictions =model (x )

        if predictions .shape [1 ]==1 :
            loss =loss_fn (predictions ,y .unsqueeze (1 ).float ())
            preds =(torch .sigmoid (predictions )>0.5 ).long ().squeeze (1 )
        else :
            loss =loss_fn (predictions ,y .long ())
            preds =torch .argmax (predictions ,dim =1 )

        val_running_correct +=(preds ==y ).float ().mean ().item ()
        val_running_loss +=loss .item ()

        if predictions .shape [1 ]==1 :
            batch_metrics =compute_metrics (predictions ,y .unsqueeze (1 ))
            all_metrics .append (batch_metrics )

    num_batches =max (len (loader ),1 )
    epoch_loss =val_running_loss /num_batches 
    epoch_acc =100.0 *val_running_correct /num_batches 
    avg_metrics =aggregate_metrics (all_metrics )if all_metrics else {}

    return epoch_loss ,epoch_acc ,avg_metrics 

@torch .no_grad ()
def test (loader ,model ,loss_fn ):
    model .eval ()
    test_running_loss =0.0 
    test_running_correct =0.0 
    all_metrics =[]

    for x ,y in loader :
        if isinstance (x ,np .ndarray ):
            x =torch .from_numpy (x )
        if isinstance (y ,np .ndarray ):
            y =torch .from_numpy (y )

        x =x .to (DEVICE )
        y =y .to (DEVICE )

        predictions =model (x )

        if predictions .shape [1 ]==1 :
            loss =loss_fn (predictions ,y .unsqueeze (1 ).float ())
            preds =(torch .sigmoid (predictions )>0.5 ).long ().squeeze (1 )
        else :
            loss =loss_fn (predictions ,y .long ())
            preds =torch .argmax (predictions ,dim =1 )

        test_running_correct +=(preds ==y ).float ().mean ().item ()
        test_running_loss +=loss .item ()

        if predictions .shape [1 ]==1 :
            batch_metrics =compute_metrics (predictions ,y .unsqueeze (1 ))
            all_metrics .append (batch_metrics )

    num_batches =max (len (loader ),1 )
    epoch_loss =test_running_loss /num_batches 
    epoch_acc =100.0 *test_running_correct /num_batches 
    avg_metrics =aggregate_metrics (all_metrics )if all_metrics else {}

    return epoch_loss ,epoch_acc ,avg_metrics 




def print_results (mode ,loss ,acc ,metrics ):
    print (f"\n[{mode}] Loss: {loss:.4f} | Acc: {acc:.2f}%")
    if metrics :
        ordered_keys =[
        "dice_with_bg",
        "dice_no_bg",
        "iou_with_bg",
        "iou_no_bg",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        ]
        for k in ordered_keys :
            if k in metrics :
                print (f"{k}: {metrics[k]:.4f}",end =" | ")
        print ()




def main ():
    LEARNING_RATE =1e-4 
    BATCH_SIZE =4 
    NUM_EPOCHS =1 
    NUM_WORKERS =1 
    IMAGE_HEIGHT =256 
    IMAGE_WIDTH =256 

    dataset_class =SkinPairDataset 

    train_img_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\train\images",
    r"xxxx\Data\skinlesions\PH2\train\images",
    r"xxxx\Data\skinlesions\ISIC2017\train\images",
    r"xxxx\Data\skinlesions\ISIC2018\train\images"
    ]
    train_mask_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\train\masks",
    r"xxxx\Data\skinlesions\PH2\train\masks",
    r"xxxx\Data\skinlesions\ISIC2017\train\masks",
    r"xxxx\Data\skinlesions\ISIC2018\train\masks"
    ]

    val_img_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\val\images",
    r"xxxx\Data\skinlesions\PH2\val\images",
    r"xxxx\Data\skinlesions\ISIC2017\val\images",
    r"xxxx\Data\skinlesions\ISIC2018\val\images"
    ]
    val_mask_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\val\masks",
    r"xxxx\Data\skinlesions\PH2\val\masks",
    r"xxxx\Data\skinlesions\ISIC2017\val\masks",
    r"xxxx\Data\skinlesions\ISIC2018\val\masks"
    ]

    test_img_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\test\images",
    r"xxxx\Data\skinlesions\PH2\test\images",
    r"xxxx\Data\skinlesions\ISIC2017\test\images",
    r"xxxx\Data\skinlesions\ISIC2018\test\images"
    ]
    test_mask_dirs =[
    r"xxxx\Data\skinlesions\HAM10K\test\masks",
    r"xxxx\Data\skinlesions\PH2\test\masks",
    r"xxxx\Data\skinlesions\ISIC2017\test\masks",
    r"xxxx\Data\skinlesions\ISIC2018\test\masks"
    ]

    val_transform =A .Compose ([
    A .Resize (IMAGE_HEIGHT ,IMAGE_WIDTH ),
    A .Normalize (mean =[0.485 ,0.456 ,0.406 ],
    std =[0.229 ,0.224 ,0.225 ]),
    ToTensorV2 (),
    ])
    train_transform =A .Compose ([
    A .Resize (IMAGE_HEIGHT ,IMAGE_WIDTH ),
    A .Normalize (mean =[0.485 ,0.456 ,0.406 ],
    std =[0.229 ,0.224 ,0.225 ]),
    ToTensorV2 (),
    ])

    train_loader ,val_loader ,test_loader_all =get_loaders (
    dataset_class ,
    train_img_dirs ,train_mask_dirs ,
    val_img_dirs ,val_mask_dirs ,
    test_img_dirs ,test_mask_dirs ,
    BATCH_SIZE ,train_transform ,val_transform ,NUM_WORKERS 
    )

    model =UNET (in_channels =3 ,out_channels =1 ).to (DEVICE )


    for param in model .parameters ():
        param .requires_grad =True 

    total_params =sum (p .numel ()for p in model .parameters ())
    print (f"[INFO]: {total_params:,} total parameters.")
    total_trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    print (f"[INFO]: {total_trainable_params:,} trainable parameters.")

    loss_fn =get_loss_fn (model ,DEVICE )
    optimizer =optim .AdamW (model .parameters (),lr =LEARNING_RATE ,weight_decay =1e-4 )
    scheduler =CosineAnnealingWarmRestarts (optimizer ,T_0 =5 ,T_mult =1 ,eta_min =1e-6 )

    best_iou =0.0 
    best_model_path ="BestModels/best_model_centralized.pth"
    os .makedirs ("BestModels",exist_ok =True )

    for epoch in range (NUM_EPOCHS ):
        print (f"[INFO]: Epoch {epoch + 1} of {NUM_EPOCHS}")

        train_loss ,train_acc ,train_metrics =train (train_loader ,model ,optimizer ,scheduler ,loss_fn )
        val_loss ,val_acc ,val_metrics =eval_performance (val_loader ,model ,loss_fn )

        print_results ("TRAIN",train_loss ,train_acc ,train_metrics )
        print_results ("VAL",val_loss ,val_acc ,val_metrics )

        if val_metrics and val_metrics .get ("iou_no_bg",0 )>best_iou :
            best_iou =val_metrics ["iou_no_bg"]
            torch .save (model .state_dict (),best_model_path )
            print ("Model saved!")

    print ("[INFO]: Testing the best model on ALL client test sets together...")
    model .load_state_dict (torch .load (best_model_path ,map_location =DEVICE ))
    global_test_loss ,global_test_acc ,global_test_metrics =test (test_loader_all ,model ,loss_fn )
    print_results ("GLOBAL TEST (ALL CLIENT TEST SETS)",global_test_loss ,global_test_acc ,global_test_metrics )

    print ("\n[INFO]: Testing the best model on EACH client test set separately...")
    per_client_results ={}
    for i ,cname in enumerate (["HAM10K","PH2","ISIC2017","ISIC2018"]):
        client_test_loader =get_loader_from_dirs (
        test_img_dirs [i ],
        test_mask_dirs [i ],
        val_transform ,
        batch_size =BATCH_SIZE ,
        num_workers =NUM_WORKERS ,
        shuffle =False 
        )
        c_loss ,c_acc ,c_metrics =test (client_test_loader ,model ,loss_fn )
        print_results (f"TEST - {cname}",c_loss ,c_acc ,c_metrics )
        per_client_results [cname ]={
        "loss":c_loss ,
        "acc":c_acc ,
        **c_metrics 
        }

if __name__ =="__main__":
    main ()