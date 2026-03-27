import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy ,time ,shutil 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,RandomSampler 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
from PIL import Image 
import numpy as np 

from models .UNET import UNET 
from dataset import CVCDataset 




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
NUM_CLIENTS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 


PFEDME_LAMBDA =10 
INNER_STEPS =10 
INNER_LR =5e-3 
OUTER_ETA =1e-4 
BETA =1.0 


MAX_SAVE_PER_CLIENT =100 
PRED_SAVE_DIR ="Outputs/pFedMe-test"

start_time =time .time ()
out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )




train_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images"
]
train_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks"
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
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images"
]
test_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks"
]
client_names =["Kvasir","ETIS","CVC-Colon","CVC-Clinic"]




def get_loader (img_dir ,mask_dir ,transform ,batch_size =4 ,shuffle =True ,oversample_factor =None ):
    ds =CVCDataset (img_dir ,mask_dir ,transform =transform )
    if oversample_factor is not None :
        sampler =RandomSampler (ds ,replacement =True ,num_samples =oversample_factor *len (ds ))
        return DataLoader (ds ,batch_size =batch_size ,sampler =sampler )
    else :
        return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )




def compute_metrics (pred ,target ,smooth =1e-6 ):
    probs =torch .sigmoid (pred )
    pred_bin =(probs >0.5 ).float ()

    if target .dim ()==3 :
        target =target .unsqueeze (1 )
    target_bin =(target >0.5 ).float ()

    if pred_bin .dim ()==4 and pred_bin .size (1 )==1 :
        pred_flat =pred_bin .view (pred_bin .size (0 ),-1 )
    else :
        pred_flat =pred_bin .view (pred_bin .size (0 ),-1 )

    if target_bin .dim ()==4 and target_bin .size (1 )==1 :
        targ_flat =target_bin .view (target_bin .size (0 ),-1 )
    else :
        targ_flat =target_bin .view (target_bin .size (0 ),-1 )

    TP =(pred_flat *targ_flat ).sum (dim =1 ).float ()
    FP =(pred_flat *(1.0 -targ_flat )).sum (dim =1 ).float ()
    FN =((1.0 -pred_flat )*targ_flat ).sum (dim =1 ).float ()
    TN =((1.0 -pred_flat )*(1.0 -targ_flat )).sum (dim =1 ).float ()

    pred_sum =(TP +FP )
    targ_sum =(TP +FN )
    inter =TP 

    denom_dice_fg =(pred_sum +targ_sum )
    dice_no_bg_per_image =torch .where (
    denom_dice_fg >0 ,
    (2.0 *inter +smooth )/(denom_dice_fg +smooth ),
    torch .ones_like (denom_dice_fg )
    )

    denom_iou_fg =(pred_sum +targ_sum -inter )
    iou_no_bg_per_image =torch .where (
    denom_iou_fg >0 ,
    (inter +smooth )/(denom_iou_fg +smooth ),
    torch .where ((pred_sum ==0 )&(targ_sum ==0 ),torch .ones_like (denom_iou_fg ),torch .zeros_like (denom_iou_fg ))
    )

    back_pred_sum =(TN +FN )
    back_targ_sum =(TN +FP )
    back_inter =TN 

    denom_dice_bg =(back_pred_sum +back_targ_sum )
    dice_bg_per_image =torch .where (
    denom_dice_bg >0 ,
    (2.0 *back_inter +smooth )/(denom_dice_bg +smooth ),
    torch .ones_like (denom_dice_bg )
    )

    denom_iou_bg =(back_pred_sum +back_targ_sum -back_inter )
    iou_bg_per_image =torch .where (
    denom_iou_bg >0 ,
    (back_inter +smooth )/(denom_iou_bg +smooth ),
    torch .where ((back_pred_sum ==0 )&(back_targ_sum ==0 ),torch .ones_like (denom_iou_bg ),torch .zeros_like (denom_iou_bg ))
    )

    total =TP +TN +FP +FN 
    acc_per_image =torch .where (total >0 ,(TP +TN )/(total +smooth ),torch .ones_like (total ))

    precision_per_image =torch .where ((TP +FP )>0 ,(TP +smooth )/(TP +FP +smooth ),torch .where (TP ==0 ,torch .zeros_like (TP ),torch .ones_like (TP )))
    recall_per_image =torch .where ((TP +FN )>0 ,(TP +smooth )/(TP +FN +smooth ),torch .where (TP ==0 ,torch .zeros_like (TP ),torch .ones_like (TP )))
    specificity_per_image =torch .where ((TN +FP )>0 ,(TN +smooth )/(TN +FP +smooth ),torch .where (TN ==0 ,torch .zeros_like (TN ),torch .ones_like (TN )))

    dice_no_bg =float (torch .clamp (dice_no_bg_per_image .mean (),0.0 ,1.0 ).item ())
    iou_no_bg =float (torch .clamp (iou_no_bg_per_image .mean (),0.0 ,1.0 ).item ())
    dice_with_bg =float (torch .clamp (0.5 *(dice_no_bg_per_image +dice_bg_per_image ).mean (),0.0 ,1.0 ).item ())
    iou_with_bg =float (torch .clamp (0.5 *(iou_no_bg_per_image +iou_bg_per_image ).mean (),0.0 ,1.0 ).item ())
    accuracy =float (torch .clamp (acc_per_image .mean (),0.0 ,1.0 ).item ())
    precision =float (torch .clamp (precision_per_image .mean (),0.0 ,1.0 ).item ())
    recall =float (torch .clamp (recall_per_image .mean (),0.0 ,1.0 ).item ())
    specificity =float (torch .clamp (specificity_per_image .mean (),0.0 ,1.0 ).item ())

    return dict (
    dice_with_bg =dice_with_bg ,
    dice_no_bg =dice_no_bg ,
    iou_with_bg =iou_with_bg ,
    iou_no_bg =iou_no_bg ,
    accuracy =accuracy ,
    precision =precision ,
    recall =recall ,
    specificity =specificity 
    )

def average_metrics (metrics_list ):
    if not metrics_list :
        return {}
    avg ={}
    for k in metrics_list [0 ].keys ():
        avg [k ]=sum (m [k ]for m in metrics_list )/len (metrics_list )
    return avg 

def get_loss_fn (device ):
    return smp .losses .DiceLoss (mode ="binary",from_logits =True ).to (device )


def average_models_weighted (models ,weights ):
    """Average all model parameters across clients (FedAvg-style)."""
    avg_sd =copy .deepcopy (models [0 ].state_dict ())
    for k in avg_sd .keys ():
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ].cpu ()for i in range (len (models )))
    return avg_sd 




def train_local_pfedme (loader ,model ,loss_fn ,inner_steps =INNER_STEPS ,inner_lr =INNER_LR ,
outer_eta =OUTER_ETA ,lam =PFEDME_LAMBDA ,device =DEVICE ):
    model .train ()
    total_loss =0.0 
    metrics =[]

    for epoch in range (LOCAL_EPOCHS ):
        for data ,target in tqdm (loader ,leave =False ):
            data ,target =data .to (device ),target .to (device ).unsqueeze (1 ).float ()


            theta =copy .deepcopy (model ).to (device )
            inner_opt =optim .SGD (theta .parameters (),lr =inner_lr ,momentum =0.9 )

            for _k in range (inner_steps ):
                inner_opt .zero_grad ()
                preds_theta =theta (data )
                loss_inner =loss_fn (preds_theta ,target )
                prox_reg =0.0 
                for (n_p ,p_w ),p_theta in zip (model .named_parameters (),theta .parameters ()):
                    prox_reg =prox_reg +((p_theta -p_w )**2 ).sum ()
                loss_inner =loss_inner +(lam /2.0 )*prox_reg 
                loss_inner .backward ()
                inner_opt .step ()


            with torch .no_grad ():
                for p_w ,p_theta in zip (model .parameters (),theta .parameters ()):
                    diff =(p_w .data -p_theta .data )
                    p_w .data =p_w .data -(outer_eta *lam )*diff 

            preds =model (data )
            loss =loss_fn (preds ,target )
            total_loss +=float (loss .item ())
            metrics .append (compute_metrics (preds .detach (),target ))

    avg_metrics =average_metrics (metrics )if metrics else {}
    print ("Train (pFedMe): "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss ,metrics =0.0 ,[]
    for data ,target in loader :
        data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
        preds =model (data )
        loss =loss_fn (preds ,target )
        total_loss +=float (loss .item ())
        metrics .append (compute_metrics (preds ,target ))
    avg_metrics =average_metrics (metrics )if metrics else {}
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 

def compute_personalized_model_from_w (w_model ,data_loader ,loss_fn ,inner_steps =INNER_STEPS ,inner_lr =INNER_LR ,lam =PFEDME_LAMBDA ):
    theta =copy .deepcopy (w_model ).to (DEVICE )
    theta .train ()
    inner_opt =optim .SGD (theta .parameters (),lr =inner_lr ,momentum =0.9 )

    steps_done =0 
    while steps_done <inner_steps :
        for data ,target in data_loader :
            data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
            inner_opt .zero_grad ()
            preds =theta (data )
            loss_inner =loss_fn (preds ,target )
            prox_reg =0.0 
            for p_w ,p_theta in zip (w_model .parameters (),theta .parameters ()):
                prox_reg =prox_reg +((p_theta -p_w )**2 ).sum ()
            loss_inner =loss_inner +(lam /2.0 )*prox_reg 
            loss_inner .backward ()
            inner_opt .step ()
            steps_done +=1 
            if steps_done >=inner_steps :
                break 

    return theta 




def save_test_predictions (model ,test_loader ,save_dir ,client_id ,max_save =MAX_SAVE_PER_CLIENT ,device =DEVICE ):
    """
    Saves up to `max_save` images for a client into save_dir.
    Overwrites the directory contents each call (caller may clear dir beforehand).
    Each example saves three pngs: img, gt, pred.
    """
    os .makedirs (save_dir ,exist_ok =True )
    saved =0 
    model .eval ()
    th =0.5 

    with torch .no_grad ():
        for idx ,(img ,mask )in enumerate (test_loader ):
            if saved >=max_save :
                break 
            img =img .to (device )
            mask =mask .to (device )

            preds =model (img )
            probs =torch .sigmoid (preds )
            pred_bin =(probs >th ).float ()



            img_np =img .detach ().cpu ().numpy ()
            mask_np =mask .detach ().cpu ().numpy ()
            pred_np =pred_bin .detach ().cpu ().numpy ()


            batch_size =img_np .shape [0 ]
            for b in range (batch_size ):
                if saved >=max_save :
                    break 


                im =img_np [b ]
                if im .ndim ==3 :
                    im =np .transpose (im ,(1 ,2 ,0 ))

                im =np .clip (im ,0.0 ,1.0 )
                im_uint8 =(im *255. ).astype (np .uint8 )


                m =mask_np [b ]
                if m .ndim ==3 and m .shape [0 ]==1 :
                    m =m [0 ]
                m_uint8 =(np .clip (m ,0.0 ,1.0 )*255. ).astype (np .uint8 )


                p =pred_np [b ]
                if p .ndim ==3 and p .shape [0 ]==1 :
                    p =p [0 ]
                p_uint8 =(np .clip (p ,0.0 ,1.0 )*255. ).astype (np .uint8 )


                img_fname =os .path .join (save_dir ,f"client{client_id}_img_{saved:04d}.png")
                gt_fname =os .path .join (save_dir ,f"client{client_id}_gt_{saved:04d}.png")
                pred_fname =os .path .join (save_dir ,f"client{client_id}_pred_{saved:04d}.png")

                Image .fromarray (im_uint8 ).save (img_fname )
                Image .fromarray (m_uint8 ).save (gt_fname )
                Image .fromarray (p_uint8 ).save (pred_fname )

                saved +=1 

    print (f"Saved {saved} prediction files for client {client_id} in {save_dir}")
    return saved 




def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))
    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round");plt .ylabel ("Dice");plt .title ("Per-client Dice (pFedMe)");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_pfedme.png"));plt .close ()

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round");plt .ylabel ("IoU");plt .title ("Per-client IoU (pFedMe)");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_pfedme.png"));plt .close ()




def main ():
    tr_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])
    val_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])


    global_model =smp .Unet (
    encoder_name ="resnet34",
    encoder_weights ="imagenet",
    in_channels =3 ,
    classes =1 ,
    ).to (DEVICE )
    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models ,weights =[],[]
        total_sz =0 


        if os .path .exists (PRED_SAVE_DIR ):

            shutil .rmtree (PRED_SAVE_DIR )
        os .makedirs (PRED_SAVE_DIR ,exist_ok =True )


        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            loss_fn =get_loss_fn (DEVICE )

            oversample_factor =1 
            train_loader =get_loader (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,
            oversample_factor =oversample_factor )
            val_loader =get_loader (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,shuffle =False )

            print (f"[Client {client_names[i]}] Local training (pFedMe λ={PFEDME_LAMBDA}, K={INNER_STEPS})")
            train_local_pfedme (train_loader ,local_model ,loss_fn ,
            inner_steps =INNER_STEPS ,inner_lr =INNER_LR ,
            outer_eta =OUTER_ETA ,lam =PFEDME_LAMBDA ,device =DEVICE )

            evaluate (val_loader ,local_model ,loss_fn ,split ="Val (w_i)")

            local_models .append (local_model )
            sz =len (train_loader .dataset );weights .append (sz );total_sz +=sz 


        norm_weights =[w /total_sz for w in weights ]
        avg_state =average_models_weighted (local_models ,norm_weights )

        new_global_state ={}
        old_state =global_model .state_dict ()
        for k in old_state .keys ():
            new_global_state [k ]=(1.0 -BETA )*old_state [k ].cpu ()+BETA *avg_state [k ]

        global_model .load_state_dict (new_global_state ,strict =False )

        for lm in local_models :
            lm .load_state_dict (new_global_state ,strict =False )


        rm ={}
        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,shuffle =False )
            print (f"[Client {client_names[i]}] Test (personalized θ̃)")


            data_for_inner =get_loader (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,batch_size =4 ,shuffle =True )
            theta_tilde =compute_personalized_model_from_w (local_models [i ],data_for_inner ,get_loss_fn (DEVICE ),
            inner_steps =INNER_STEPS ,inner_lr =INNER_LR ,lam =PFEDME_LAMBDA )


            _ ,test_metrics =evaluate (test_loader ,theta_tilde ,get_loss_fn (DEVICE ),split ="Test θ̃")
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )


            client_save_dir =os .path .join (PRED_SAVE_DIR ,f"client{i}")

            save_test_predictions (theta_tilde ,test_loader ,client_save_dir ,client_id =i ,max_save =MAX_SAVE_PER_CLIENT )


        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

    print (f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ =="__main__":
    main ()
