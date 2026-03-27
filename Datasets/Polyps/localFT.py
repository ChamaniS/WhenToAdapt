import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy ,time ,torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,RandomSampler 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
import shutil 
from glob import glob 
from PIL import Image 
import numpy as np 

from models .UNET import UNET 
from dataset import CVCDataset 




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
NUM_CLIENTS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
start_time =time .time ()


FINETUNE_EPOCHS =3 
FINETUNE_LR =1e-5 
FINETUNE_BATCH =8 

out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )




train_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
]
train_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
]
val_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images"
]
val_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks"
]
test_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images",
]
test_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks",
]
client_names =["Kvasir","ETIS","CVC-Colon","CVC-Clinic"]




def get_loader (img_dir ,mask_dir ,transform ,batch_size =8 ,shuffle =True ,oversample_factor =None ):
    ds =CVCDataset (img_dir ,mask_dir ,transform =transform )

    if oversample_factor is not None :
        sampler =RandomSampler (ds ,replacement =True ,num_samples =oversample_factor *len (ds ))
        return DataLoader (ds ,batch_size =batch_size ,sampler =sampler )
    else :
        return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )

def compute_metrics (pred ,target ,smooth =1e-6 ):
    """
    Robust metric computation that avoids numeric issues and ensures outputs
    are in [0,1]. Expects:
      - pred: raw logits tensor (Bx1xHxW) OR already logits; we apply sigmoid then threshold
      - target: ground-truth tensor (Bx1xHxW) or (BxHxW)
    """
    probs =torch .sigmoid (pred )
    pred_bin =(probs >0.5 ).to (torch .uint8 )
    tgt =(target >0.5 ).to (torch .uint8 )

    if tgt .dim ()==3 :
        tgt =tgt .unsqueeze (1 )
    if pred_bin .dim ()==3 :
        pred_bin =pred_bin .unsqueeze (1 )

    p =pred_bin .view (-1 ).to (torch .int64 )
    t =tgt .view (-1 ).to (torch .int64 )

    TP =int (((p ==1 )&(t ==1 )).sum ().item ())
    TN =int (((p ==0 )&(t ==0 )).sum ().item ())
    FP =int (((p ==1 )&(t ==0 )).sum ().item ())
    FN =int (((p ==0 )&(t ==1 )).sum ().item ())

    pred_sum =TP +FP 
    targ_sum =TP +FN 
    intersection =TP 

    denom_with_bg =(2 *TP +FP +FN )
    if denom_with_bg <=0 :
        dice_with_bg =1.0 
    else :
        dice_with_bg =(2 *TP +smooth )/(denom_with_bg +smooth )

    denom_iou_with_bg =(TP +FP +FN )
    if denom_iou_with_bg <=0 :
        iou_with_bg =1.0 
    else :
        iou_with_bg =(TP +smooth )/(denom_iou_with_bg +smooth )

    denom_no_bg =(pred_sum +targ_sum )
    if denom_no_bg <=0 :
        dice_no_bg =1.0 
    else :
        dice_no_bg =(2 *intersection +smooth )/(denom_no_bg +smooth )

    denom_iou_no_bg =(pred_sum +targ_sum -intersection )
    if denom_iou_no_bg <=0 :
        if pred_sum ==0 and targ_sum ==0 :
            iou_no_bg =1.0 
        else :
            iou_no_bg =0.0 
    else :
        iou_no_bg =(intersection +smooth )/(denom_iou_no_bg +smooth )

    total =TP +TN +FP +FN 
    if total <=0 :
        acc =1.0 
    else :
        acc =(TP +TN )/(total +smooth )

    if (TP +FP )<=0 :
        precision =0.0 if (TP +FP )==0 and TP ==0 else 1.0 
    else :
        precision =(TP +smooth )/(TP +FP +smooth )

    if (TP +FN )<=0 :
        recall =0.0 if (TP +FN )==0 and TP ==0 else 1.0 
    else :
        recall =(TP +smooth )/(TP +FN +smooth )

    if (TN +FP )<=0 :
        specificity =1.0 if (TN +FP )==0 and TN ==0 else 0.0 
    else :
        specificity =(TN +smooth )/(TN +FP +smooth )


    dice_with_bg =float (min (max (dice_with_bg ,0.0 ),1.0 ))
    dice_no_bg =float (min (max (dice_no_bg ,0.0 ),1.0 ))
    iou_with_bg =float (min (max (iou_with_bg ,0.0 ),1.0 ))
    iou_no_bg =float (min (max (iou_no_bg ,0.0 ),1.0 ))
    acc =float (min (max (acc ,0.0 ),1.0 ))
    precision =float (min (max (precision ,0.0 ),1.0 ))
    recall =float (min (max (recall ,0.0 ),1.0 ))
    specificity =float (min (max (specificity ,0.0 ),1.0 ))

    return dict (
    dice_with_bg =dice_with_bg ,dice_no_bg =dice_no_bg ,
    iou_with_bg =iou_with_bg ,iou_no_bg =iou_no_bg ,
    accuracy =acc ,precision =precision ,recall =recall ,specificity =specificity 
    )

def average_metrics (metrics_list ):
    avg ={}
    for k in metrics_list [0 ].keys ():
        avg [k ]=sum (m [k ]for m in metrics_list )/len (metrics_list )
    return avg 

def get_loss_fn (device ):
    return smp .losses .DiceLoss (mode ="binary",from_logits =True ).to (device )




def fedbn_average (models ,weights ):
    """Average only non-BN parameters across clients (FedBN)."""
    avg_state =copy .deepcopy (models [0 ].state_dict ())
    for key in avg_state .keys ():
        if "bn"in key .lower ()or "downsample.1"in key .lower ():
            continue 
        avg_state [key ]=sum (weights [i ]*models [i ].state_dict ()[key ]for i in range (len (models )))
    return avg_state 




def _mask_to_uint8 (mask_tensor ):
    m =mask_tensor .cpu ().numpy ()
    if m .ndim ==3 :
        m =np .squeeze (m ,axis =0 )
    m =(m >0.5 ).astype (np .uint8 )*255 
    return m 

def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )

def save_image (arr ,path ):
    Image .fromarray (arr ).save (path )

def save_test_predictions (model ,test_loader ,client_name ,out_base ,round_num =None ,mode ="before",device_arg =None ):
    """
    Save predicted masks (binary PNGs) for a client, but keep ONLY the latest results.
    Writes to: out_base/TestPreds/<client_name>/latest/

    - round_num and mode are accepted for API-compatibility but NOT used to create subfolders.
    - If a 'latest' folder exists it will be cleared before saving.
    """
    if device_arg is None :
        device =DEVICE 
    else :
        device =device_arg 

    model .eval ()
    latest_dir =os .path .join (out_base ,"TestPreds",client_name ,"latest")


    if os .path .exists (latest_dir ):
        shutil .rmtree (latest_dir )
    ensure_dir (latest_dir )


    dataset_filenames =None 
    try :
        ds =test_loader .dataset 
        if hasattr (ds ,"images")and isinstance (ds .images ,list ):
            dataset_filenames =ds .images 
        elif hasattr (ds ,"common_basenames")and isinstance (ds .common_basenames ,list ):
            dataset_filenames =ds .common_basenames 
        elif hasattr (ds ,"common")and isinstance (ds .common ,list ):
            dataset_filenames =ds .common 
    except Exception :
        dataset_filenames =None 

    model .to (device )
    saved =0 
    with torch .no_grad ():
        for batch_idx ,batch in enumerate (test_loader ):

            if isinstance (batch ,(list ,tuple ))and len (batch )==3 :
                data ,target ,fnames =batch 
            elif isinstance (batch ,(list ,tuple ))and len (batch )>=2 :
                data ,target =batch [0 ],batch [1 ]
                fnames =None 
            else :
                raise RuntimeError ("Unexpected batch format from test_loader")


            try :
                if target .dim ()==3 :
                    target =target .unsqueeze (1 )
            except Exception :
                pass 

            data =data .to (device )
            preds =model (data )
            probs =torch .sigmoid (preds )
            bin_mask =(probs >0.5 ).float ()

            bsz =data .size (0 )
            for b in range (bsz ):
                mask_t =bin_mask [b ].cpu ()
                mask_arr =_mask_to_uint8 (mask_t )





                if fnames is not None :
                    try :
                        orig_name =fnames [b ]
                        orig_name =os .path .basename (orig_name )
                        base ,_ =os .path .splitext (orig_name )
                        fname =f"{base}_pred.png"
                    except Exception :
                        fname =f"{client_name}_pred_{batch_idx}_{b}.png"
                elif dataset_filenames is not None :
                    global_idx =batch_idx *test_loader .batch_size +b 
                    if global_idx <len (dataset_filenames ):
                        orig_name =dataset_filenames [global_idx ]
                        base =os .path .splitext (orig_name )[0 ]
                        fname =f"{base}_pred.png"
                    else :
                        fname =f"{client_name}_pred_{batch_idx}_{b}.png"
                else :
                    fname =f"{client_name}_pred_{batch_idx}_{b}.png"

                save_image (mask_arr ,os .path .join (latest_dir ,fname ))
                saved +=1 

    print (f"[SavePreds] saved {saved} masks -> {latest_dir}")





def train_local (loader ,model ,loss_fn ,opt ):
    model .train ()
    total_loss ,metrics =0.0 ,[]
    for _ in range (LOCAL_EPOCHS ):
        for data ,target in tqdm (loader ,leave =False ):
            data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
            preds =model (data )
            loss =loss_fn (preds ,target )
            opt .zero_grad ();loss .backward ();opt .step ()
            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))
    avg_metrics =average_metrics (metrics )
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /len (loader .dataset ),avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss ,metrics =0.0 ,[]
    for data ,target in loader :
        data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
        preds =model (data )
        loss =loss_fn (preds ,target )
        total_loss +=loss .item ()
        metrics .append (compute_metrics (preds ,target ))
    avg_metrics =average_metrics (metrics )if metrics else {}
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /len (loader .dataset ),avg_metrics 




def fine_tune_local (img_dir ,mask_dir ,model ,epochs =FINETUNE_EPOCHS ,lr =FINETUNE_LR ,batch_size =FINETUNE_BATCH ):
    """
    Fine-tune a provided model on the local client's train data for a few epochs.
    Operates in-place on `model`. Returns (avg_loss, metrics).
    """
    tr_tf_local =A .Compose ([
    A .Resize (224 ,224 ),
    A .HorizontalFlip (p =0.2 ),
    A .VerticalFlip (p =0.1 ),
    A .RandomRotate90 (p =0.2 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])
    loader =get_loader (img_dir ,mask_dir ,tr_tf_local ,batch_size =batch_size ,shuffle =True )
    opt =optim .AdamW (model .parameters (),lr =lr )
    loss_fn =get_loss_fn (DEVICE )
    model .train ()
    total_loss =0.0 
    metrics =[]
    for e in range (epochs ):
        for data ,target in loader :
            data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
            preds =model (data )
            loss =loss_fn (preds ,target )
            opt .zero_grad ();loss .backward ();opt .step ()
            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))
    avg_metrics =average_metrics (metrics )if metrics else {}
    avg_loss =total_loss /max (1 ,len (loader .dataset ))
    print (f"[FineTune] finished {epochs} epochs, loss={avg_loss:.4f}")
    return avg_loss ,avg_metrics 




def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))


    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Per-client Dice")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_localFT.png"))
    plt .close ()


    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Per-client IoU")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_localFT.png"))
    plt .close ()




def main ():

    tr_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.5 ),
    A .RandomRotate90 (p =0.5 ),
    A .ColorJitter (brightness =0.2 ,contrast =0.2 ,saturation =0.2 ,p =0.5 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])


    val_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])

    global_model =UNET (in_channels =3 ,out_channels =1 ).to (DEVICE )
    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models ,weights =[],[]
        total_sz =0 


        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            opt =optim .AdamW (local_model .parameters (),lr =1e-4 )
            loss_fn =get_loss_fn (DEVICE )

            oversample_factor =1 
            train_loader =get_loader (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,
            oversample_factor =oversample_factor )
            val_loader =get_loader (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,shuffle =False )

            print (f"[Client {client_names[i]}]")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            local_models .append (local_model )
            sz =len (train_loader .dataset );weights .append (sz );total_sz +=sz 


        norm_weights =[w /total_sz for w in weights ]
        avg_state =fedbn_average (local_models ,norm_weights )
        global_model .load_state_dict (avg_state ,strict =False )


        for lm in local_models :
            lm .load_state_dict (avg_state ,strict =False )


        rm ={}
        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,shuffle =False )
            print (f"[Client {client_names[i]}] Test (before fine-tune)")
            _ ,test_metrics =evaluate (test_loader ,local_models [i ],get_loss_fn (DEVICE ),split ="Test")
            rm [f"client{i}_dice_no_bg"]=test_metrics ["dice_no_bg"]
            rm [f"client{i}_iou_no_bg"]=test_metrics ["iou_no_bg"]


            save_test_predictions (local_models [i ],test_loader ,client_names [i ],out_base =out_dir ,round_num =r +1 ,mode ="before",device_arg =DEVICE )




        print ("\n[Personalization] Running local fine-tuning on each client (lightweight)...")
        for i in range (NUM_CLIENTS ):
            print (f"[Client {client_names[i]}] Fine-tuning for {FINETUNE_EPOCHS} epochs (lr={FINETUNE_LR})")
            fine_tune_local (train_img_dirs [i ],train_mask_dirs [i ],local_models [i ],
            epochs =FINETUNE_EPOCHS ,lr =FINETUNE_LR ,batch_size =FINETUNE_BATCH )


        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,shuffle =False )
            print (f"[Client {client_names[i]}] Test (after fine-tune)")
            _ ,test_metrics_ft =evaluate (test_loader ,local_models [i ],get_loss_fn (DEVICE ),split ="Test_FineTune")
            rm [f"client{i}_dice_no_bg_ft"]=test_metrics_ft ["dice_no_bg"]
            rm [f"client{i}_iou_no_bg_ft"]=test_metrics_ft ["iou_no_bg"]


            save_test_predictions (local_models [i ],test_loader ,client_names [i ],out_base =out_dir ,round_num =r +1 ,mode ="after",device_arg =DEVICE )

        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

    end_time =time .time ()
    print (f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ =="__main__":
    main ()
