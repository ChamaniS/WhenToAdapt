import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy 
import time 
import math 
import shutil 
import numpy as np 
import torch 
import torch .optim as optim 
from torch .utils .data import DataLoader ,RandomSampler 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
from PIL import Image 

from models .UNET import UNET 
from dataset import CVCDataset 




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
NUM_CLIENTS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 

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




def get_loader (img_dir ,mask_dir ,transform ,batch_size =8 ,shuffle =True ,oversample_factor =None ):
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

    avg_sd =copy .deepcopy (models [0 ].state_dict ())
    for k in list (avg_sd .keys ()):
        t0 =avg_sd [k ]
        if not torch .is_tensor (t0 ):
            continue 
        if t0 .dtype .is_floating_point :
            acc =torch .zeros_like (t0 .cpu (),dtype =torch .float32 )
            for i in range (len (models )):
                v =models [i ].state_dict ()[k ].cpu ().to (torch .float32 )
                acc +=weights [i ]*v 
            avg_sd [k ]=acc 
        else :
            avg_sd [k ]=t0 
    return avg_sd 


def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )

def _mask_to_uint8 (mask_tensor ):

    if torch .is_tensor (mask_tensor ):
        mt =mask_tensor .squeeze ().cpu ().numpy ()
    else :
        mt =np .array (mask_tensor )

    mt_bin =(mt >0.5 ).astype (np .uint8 )*255 

    return mt_bin .astype (np .uint8 )

def save_image (np_arr ,path ):
    """
    Save a HxW uint8 numpy array as a PNG.
    """
    img =Image .fromarray (np_arr )
    img .save (path )




def save_test_predictions (model ,test_loader ,client_name ,out_base ,round_num =None ,device_arg =None ):

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
                continue 

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
            total_loss +=float (loss .item ())
            metrics .append (compute_metrics (preds .detach (),target ))
    avg_metrics =average_metrics (metrics )if metrics else {}
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val",print_diag =True ):
    model .eval ()
    total_loss ,metrics =0.0 ,[]

    min_logit =float ("inf")
    max_logit =-float ("inf")
    total_pos =0 
    total_pix =0 
    for data ,target in loader :
        data ,target =data .to (DEVICE ),target .to (DEVICE ).unsqueeze (1 ).float ()
        preds =model (data )
        loss =loss_fn (preds ,target )
        total_loss +=float (loss .item ())
        metrics .append (compute_metrics (preds ,target ))


        logits =preds 
        min_logit =min (min_logit ,float (logits .min ().item ()))
        max_logit =max (max_logit ,float (logits .max ().item ()))
        probs =torch .sigmoid (logits )
        total_pos +=int ((probs >0.5 ).sum ().item ())
        total_pix +=int (probs .numel ())

    avg_metrics =average_metrics (metrics )if metrics else {}
    if print_diag :
        pos_frac =total_pos /(total_pix +1e-12 )
        print (f"{split}: logit_min={min_logit:.4f} logit_max={max_logit:.4f} pred_pos_frac={pos_frac:.6f}")
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 


def positive_fraction (loader ,model ):
    model .eval ()
    total_pos =0 
    total_pix =0 
    with torch .no_grad ():
        for data ,_ in loader :
            data =data .to (DEVICE )
            logits =model (data )
            probs =torch .sigmoid (logits )
            total_pos +=int ((probs >0.5 ).sum ().item ())
            total_pix +=int (probs .numel ())
    return total_pos /(total_pix +1e-12 )




def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round");plt .ylabel ("Dice");plt .title ("Per-client Dice (FedADAM/FedAvg)");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_fedadam.png"));plt .close ()

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round");plt .ylabel ("IoU");plt .title ("Per-client IoU (FedADAM/FedAvg)");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_fedadam.png"));plt .close ()




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

    global_model =UNET (in_channels =3 ,out_channels =1 ).to (DEVICE )
    round_metrics =[]

    SAVE_PREDICTIONS =True 
    USE_FEDAVG =False 
    server_lr =0.05 
    beta1 =0.9 
    beta2 =0.99 
    tau =1e-2 
    eps =1e-8 
    server_step =0 

    max_param_update =0.05 
    max_global_update_norm =50.0 

    per_client_clip =None 

    server_m ={k :torch .zeros_like (v .cpu (),dtype =torch .float32 )for k ,v in global_model .state_dict ().items ()}
    server_v ={k :torch .zeros_like (v .cpu (),dtype =torch .float32 )for k ,v in global_model .state_dict ().items ()}

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r + 1}/{COMM_ROUNDS}]")
        local_models =[]
        weights =[]
        total_sz =0 


        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            opt =optim .AdamW (local_model .parameters (),lr =1e-4 )
            loss_fn =get_loss_fn (DEVICE )

            oversample_factor =1 
            train_loader =get_loader (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,
            oversample_factor =oversample_factor )
            val_loader =get_loader (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,shuffle =False )

            print (f"[Client {client_names[i]}] Local training (no FedProx)")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val",print_diag =False )

            local_models .append (local_model )
            sz =len (train_loader .dataset )
            weights .append (sz )
            total_sz +=sz 

        norm_weights =[w /total_sz for w in weights ]
        avg_state =average_models_weighted (local_models ,norm_weights )
        global_sd_cpu ={k :v .cpu ()for k ,v in global_model .state_dict ().items ()}

        if per_client_clip is not None :
            client_sds =[]
            for lm in local_models :
                sd ={}
                for k ,v in lm .state_dict ().items ():
                    if torch .is_tensor (v )and v .dtype .is_floating_point :
                        sd [k ]=v .cpu ().to (torch .float32 )
                client_sds .append (sd )
            client_deltas =[]
            for sd in client_sds :
                delta ={}
                total_norm_sq =0.0 
                for k ,v in sd .items ():
                    d =v -global_sd_cpu [k ].to (torch .float32 )
                    total_norm_sq +=float (d .norm ().item ()**2 )
                    delta [k ]=d 
                norm =math .sqrt (total_norm_sq )
                if norm >per_client_clip :
                    scale =per_client_clip /(norm +1e-12 )
                else :
                    scale =1.0 
                for k in delta :
                    delta [k ]=delta [k ]*scale 
                client_deltas .append (delta )
            avg_delta ={}
            for k in client_deltas [0 ].keys ():
                acc =torch .zeros_like (client_deltas [0 ][k ],dtype =torch .float32 )
                for i ,d in enumerate (client_deltas ):
                    acc +=norm_weights [i ]*d [k ]
                avg_delta [k ]=acc 
        else :
            avg_delta ={}
            for k ,v in avg_state .items ():
                if torch .is_tensor (v )and v .dtype .is_floating_point :
                    avg_delta [k ]=(v .to (torch .float32 )-global_sd_cpu [k ].to (torch .float32 )).clone ()

        for i ,lm in enumerate (local_models ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,shuffle =False )
            pf =positive_fraction (test_loader ,lm )
            print (f"[Before Agg] Client {client_names[i]} LOCAL pos_frac: {pf:.6f}")

        if USE_FEDAVG :
            for k ,delta in avg_delta .items ():
                global_sd_cpu [k ]=global_sd_cpu [k ].to (torch .float32 )+delta .to (torch .float32 )
            print ("Applied FedAvg aggregation (no server optimizer).")
        else :
            server_step +=1 
            updates ={}
            total_delta_sq =0.0 
            total_update_sq =0.0 
            for k ,delta in avg_delta .items ():
                g_t =delta .to (torch .float32 )
                if k not in server_m :
                    server_m [k ]=torch .zeros_like (g_t )
                    server_v [k ]=torch .zeros_like (g_t )
                server_m [k ]=beta1 *server_m [k ]+(1.0 -beta1 )*g_t 
                server_v [k ]=beta2 *server_v [k ]+(1.0 -beta2 )*(g_t *g_t )
                m_hat =server_m [k ]/(1.0 -beta1 **server_step )
                v_hat =server_v [k ]/(1.0 -beta2 **server_step )
                update =server_lr *m_hat /(torch .sqrt (v_hat +eps )+tau )
                update =torch .clamp (update ,-max_param_update ,max_param_update )
                updates [k ]=update 
                total_delta_sq +=float (g_t .norm ().item ()**2 )
                total_update_sq +=float (update .norm ().item ()**2 )

            delta_norm =math .sqrt (total_delta_sq )
            update_norm =math .sqrt (total_update_sq )
            scale =1.0 
            if update_norm >max_global_update_norm and update_norm >0 :
                scale =max_global_update_norm /(update_norm +1e-12 )
                print (f"Scaling global updates by {scale:.6f} (norm {update_norm:.3f} > {max_global_update_norm})")

            for k ,update in updates .items ():
                global_sd_cpu [k ]=global_sd_cpu [k ].to (torch .float32 )+(update *scale ).to (global_sd_cpu [k ].dtype )

            print (f"server_step={server_step} | avg_delta_norm={delta_norm:.6f} | avg_update_norm_before_scale={update_norm:.6f} | scale={scale:.6f}")


        global_model .load_state_dict (global_sd_cpu ,strict =False )


        rm ={}
        for i ,lm in enumerate (local_models ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,shuffle =False )
            print (f"[Client {client_names[i]}] Test (client local model)")
            _ ,test_metrics =evaluate (test_loader ,lm ,get_loss_fn (DEVICE ),split ="Test",print_diag =True )
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )


            if SAVE_PREDICTIONS :
                save_test_predictions (lm ,test_loader ,client_names [i ],out_dir ,round_num =r +1 ,device_arg =DEVICE )


        example_loader =get_loader (test_img_dirs [0 ],test_mask_dirs [0 ],val_tf ,shuffle =False )
        print (f"[After Agg] GLOBAL pos_frac on client {client_names[0]} test: {positive_fraction(example_loader, global_model):.6f}")


        for lm in local_models :
            lm .load_state_dict (global_sd_cpu ,strict =False )

        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

    print (f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ =="__main__":
    main ()
