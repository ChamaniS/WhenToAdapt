import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy 
import time 
import glob 
import random 
import cv2 
import numpy as np 
import torch 
import torch .optim as optim 
from torch .utils .data import DataLoader ,Dataset ,ConcatDataset 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
import sys 
from unet import UNET 
output_file =r"/lustre06/project/6008975/csj5/narvalenv/skin_pfedme.txt"
sys .stdout =open (output_file ,"w")

DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

client_names =["HAM10K","PH2","ISIC2017","ISIC2018"]
NUM_CLIENTS =len (client_names )

LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
start_time =time .time ()
out_dir ="Outputs_pFedMe"
os .makedirs (out_dir ,exist_ok =True )

PFEDME_LAMBDA =10.0 
INNER_STEPS =10 
INNER_LR =5e-3 
OUTER_ETA =1e-4 
BETA =1.0 

splits_root =r"/lustre06/project/6008975/csj5/skinlesions/"

client_ext_map ={
"HAM10K":((".jpg",),(".png",)),
"ISIC2017":((".jpg",),(".png",)),
"ISIC2018":((".jpg",),(".png",)),

"PH2":((".jpg",".png",".bmp",".jpeg"),(".png",".bmp",".jpg")),
}

def set_seed (seed =42 ):
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )

set_seed (42 )


class SkinPairDataset (Dataset ):
    def __init__ (self ,img_dir ,mask_dir ,transform =None ,img_exts =None ,mask_exts =None ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .transform =transform 

        if img_exts is None :
            img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        if mask_exts is None :
            mask_exts =(".png",".jpg",".bmp",".tif",".tiff")

        self .img_exts =tuple (e .lower ()for e in img_exts )
        self .mask_exts =tuple (e .lower ()for e in mask_exts )

        files =[]
        for ext in self .img_exts :
            files .extend (glob .glob (os .path .join (self .img_dir ,f"*{ext}")))
        files =sorted (files )

        pairs =[]
        missing_masks =0 

        for img_path in files :
            stem =os .path .splitext (os .path .basename (img_path ))[0 ]
            mask_path =None 

            for mext in self .mask_exts :
                candidate =os .path .join (self .mask_dir ,stem +mext )
                if os .path .exists (candidate ):
                    mask_path =candidate 
                    break 

            if mask_path is None :
                alt_candidates =(
                [os .path .join (self .mask_dir ,stem +"_mask"+mext )for mext in self .mask_exts ]+
                [os .path .join (self .mask_dir ,stem +"-mask"+mext )for mext in self .mask_exts ]+
                [os .path .join (self .mask_dir ,stem .replace ("_lesion","")+mext )for mext in self .mask_exts ]
                )
                for c in alt_candidates :
                    if os .path .exists (c ):
                        mask_path =c 
                        break 

            if mask_path is None :
                missing_masks +=1 
                continue 

            pairs .append ((img_path ,mask_path ))

        if len (pairs )==0 :
            raise ValueError (
            f"No image-mask pairs found in {img_dir} / {mask_dir}. Missing masks: {missing_masks}"
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
            raise RuntimeError (f"Failed to read image {img_path}")
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        mask =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
        if mask is None :
            raise RuntimeError (f"Failed to read mask {mask_path}")
        if mask .ndim ==3 :
            mask =cv2 .cvtColor (mask ,cv2 .COLOR_BGR2GRAY )

        mask =np .asarray (mask )
        mask =(mask >0 ).astype (np .uint8 )

        if self .transform is not None :
            augmented =self .transform (image =img ,mask =mask )
            img =augmented ["image"]
            mask =augmented ["mask"]
        else :
            img =img .transpose (2 ,0 ,1 ).astype (np .float32 )/255.0 
            mask =np .expand_dims (mask .astype (np .float32 ),0 )

        return img ,mask 

train_img_dirs =[]
train_mask_dirs =[]
val_img_dirs =[]
val_mask_dirs =[]
test_img_dirs =[]
test_mask_dirs =[]

required_subpaths =[
("train","images"),("train","masks"),
("val","images"),("val","masks"),
("test","images"),("test","masks"),
]

for cname in client_names :
    base =os .path .join (splits_root ,cname )
    missing =[]
    for split ,sub in required_subpaths :
        p =os .path .join (base ,split ,sub )
        if not os .path .isdir (p ):
            missing .append (p )
    if missing :
        raise FileNotFoundError (
        f"Missing required split folders for client '{cname}':\n"+"\n".join (missing )
        )

    train_img_dirs .append (os .path .join (base ,"train","images"))
    train_mask_dirs .append (os .path .join (base ,"train","masks"))
    val_img_dirs .append (os .path .join (base ,"val","images"))
    val_mask_dirs .append (os .path .join (base ,"val","masks"))
    test_img_dirs .append (os .path .join (base ,"test","images"))
    test_mask_dirs .append (os .path .join (base ,"test","masks"))

print ("Using these dataset splits:")
for i ,name in enumerate (client_names ):
    print (f"Client {i}: {name}")
    print (f"  train imgs: {train_img_dirs[i]}  masks: {train_mask_dirs[i]}")
    print (f"  val   imgs: {val_img_dirs[i]}  masks: {val_mask_dirs[i]}")
    print (f"  test  imgs: {test_img_dirs[i]}  masks: {test_mask_dirs[i]}")

def get_loader (img_dir ,mask_dir ,transform ,client_name =None ,batch_size =4 ,shuffle =True ):
    if client_name is not None and client_name in client_ext_map :
        img_exts ,mask_exts =client_ext_map [client_name ]
    else :
        img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        mask_exts =(".png",".jpg",".bmp",".tif",".tiff")

    ds =SkinPairDataset (
    img_dir ,
    mask_dir ,
    transform =transform ,
    img_exts =img_exts ,
    mask_exts =mask_exts 
    )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )

def get_global_test_loader (transform ,batch_size =4 ):
    datasets =[]
    for i ,cname in enumerate (client_names ):
        if cname in client_ext_map :
            img_exts ,mask_exts =client_ext_map [cname ]
        else :
            img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
            mask_exts =(".png",".jpg",".bmp",".tif",".tiff")

        ds =SkinPairDataset (
        test_img_dirs [i ],
        test_mask_dirs [i ],
        transform =transform ,
        img_exts =img_exts ,
        mask_exts =mask_exts ,
        )
        datasets .append (ds )

    global_test_ds =ConcatDataset (datasets )
    return DataLoader (global_test_ds ,batch_size =batch_size ,shuffle =False )

def compute_metrics (pred ,target ,smooth =1e-6 ):
    probs =torch .sigmoid (pred )
    pred_bin =(probs >0.5 ).float ()

    if target .dim ()==3 :
        target =target .unsqueeze (1 )
    target_bin =(target >0.5 ).float ()

    pred_flat =pred_bin .view (pred_bin .size (0 ),-1 )
    targ_flat =target_bin .view (target_bin .size (0 ),-1 )

    TP =(pred_flat *targ_flat ).sum (dim =1 ).float ()
    FP =(pred_flat *(1.0 -targ_flat )).sum (dim =1 ).float ()
    FN =((1.0 -pred_flat )*targ_flat ).sum (dim =1 ).float ()
    TN =((1.0 -pred_flat )*(1.0 -targ_flat )).sum (dim =1 ).float ()

    pred_sum =TP +FP 
    targ_sum =TP +FN 
    inter =TP 

    denom_dice_fg =pred_sum +targ_sum 
    dice_no_bg_per_image =torch .where (
    denom_dice_fg >0 ,
    (2.0 *inter +smooth )/(denom_dice_fg +smooth ),
    torch .ones_like (denom_dice_fg )
    )

    denom_iou_fg =pred_sum +targ_sum -inter 
    iou_no_bg_per_image =torch .where (
    denom_iou_fg >0 ,
    (inter +smooth )/(denom_iou_fg +smooth ),
    torch .where ((pred_sum ==0 )&(targ_sum ==0 ),torch .ones_like (denom_iou_fg ),torch .zeros_like (denom_iou_fg ))
    )

    back_pred_sum =TN +FN 
    back_targ_sum =TN +FP 
    back_inter =TN 

    denom_dice_bg =back_pred_sum +back_targ_sum 
    dice_bg_per_image =torch .where (
    denom_dice_bg >0 ,
    (2.0 *back_inter +smooth )/(denom_dice_bg +smooth ),
    torch .ones_like (denom_dice_bg )
    )

    denom_iou_bg =back_pred_sum +back_targ_sum -back_inter 
    iou_bg_per_image =torch .where (
    denom_iou_bg >0 ,
    (back_inter +smooth )/(denom_iou_bg +smooth ),
    torch .where ((back_pred_sum ==0 )&(back_targ_sum ==0 ),torch .ones_like (denom_iou_bg ),torch .zeros_like (denom_iou_bg ))
    )

    total =TP +TN +FP +FN 
    acc_per_image =torch .where (total >0 ,(TP +TN )/(total +smooth ),torch .ones_like (total ))

    precision_per_image =torch .where (
    (TP +FP )>0 ,
    (TP +smooth )/(TP +FP +smooth ),
    torch .where (TP ==0 ,torch .zeros_like (TP ),torch .ones_like (TP ))
    )
    recall_per_image =torch .where (
    (TP +FN )>0 ,
    (TP +smooth )/(TP +FN +smooth ),
    torch .where (TP ==0 ,torch .zeros_like (TP ),torch .ones_like (TP ))
    )
    specificity_per_image =torch .where (
    (TN +FP )>0 ,
    (TN +smooth )/(TN +FP +smooth ),
    torch .where (TN ==0 ,torch .zeros_like (TN ),torch .ones_like (TN ))
    )

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
    for k in avg_sd .keys ():
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ].detach ().cpu ()for i in range (len (models )))
    return avg_sd 

def train_local_pfedme (
loader ,
model ,
loss_fn ,
inner_steps =INNER_STEPS ,
inner_lr =INNER_LR ,
outer_eta =OUTER_ETA ,
lam =PFEDME_LAMBDA ,
device =DEVICE 
):

    model .train ()
    total_loss =0.0 
    metrics =[]

    for _epoch in range (LOCAL_EPOCHS ):
        for data ,target in tqdm (loader ,leave =False ):
            if isinstance (data ,np .ndarray ):
                data =torch .from_numpy (data )
            if isinstance (target ,np .ndarray ):
                target =torch .from_numpy (target )

            data =data .to (device )
            target =target .to (device )
            if target .dim ()==3 :
                target =target .unsqueeze (1 )
            target =target .float ()

            theta =copy .deepcopy (model ).to (device )
            theta .train ()
            inner_opt =optim .SGD (theta .parameters (),lr =inner_lr ,momentum =0.9 )

            for _ in range (inner_steps ):
                inner_opt .zero_grad ()
                preds_theta =theta (data )
                loss_inner =loss_fn (preds_theta ,target )

                prox_reg =0.0 
                for p_w ,p_theta in zip (model .parameters (),theta .parameters ()):
                    prox_reg =prox_reg +((p_theta -p_w ).pow (2 )).sum ()

                loss_inner =loss_inner +(lam /2.0 )*prox_reg 
                loss_inner .backward ()
                inner_opt .step ()

            with torch .no_grad ():
                for p_w ,p_theta in zip (model .parameters (),theta .parameters ()):
                    p_w .data =p_w .data -(outer_eta *lam )*(p_w .data -p_theta .data )

            preds =model (data )
            loss =loss_fn (preds ,target )
            total_loss +=float (loss .item ())
            metrics .append (compute_metrics (preds .detach (),target ))

    avg_metrics =average_metrics (metrics )
    if avg_metrics :
        print ("Train (pFedMe): "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss ,metrics =0.0 ,[]

    for data ,target in loader :
        if isinstance (data ,np .ndarray ):
            data =torch .from_numpy (data )
        if isinstance (target ,np .ndarray ):
            target =torch .from_numpy (target )

        data =data .to (DEVICE )
        target =target .to (DEVICE )
        if target .dim ()==3 :
            target =target .unsqueeze (1 )
        target =target .float ()

        preds =model (data )
        loss =loss_fn (preds ,target )

        total_loss +=float (loss .item ())
        metrics .append (compute_metrics (preds ,target ))

    avg_metrics =average_metrics (metrics )
    if avg_metrics :
        print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /max (1 ,len (loader .dataset )),avg_metrics 

def compute_personalized_model_from_w (
w_model ,
data_loader ,
loss_fn ,
inner_steps =INNER_STEPS ,
inner_lr =INNER_LR ,
lam =PFEDME_LAMBDA ,
device =DEVICE 
):
    theta =copy .deepcopy (w_model ).to (device )
    theta .train ()
    inner_opt =optim .SGD (theta .parameters (),lr =inner_lr ,momentum =0.9 )

    steps_done =0 
    while steps_done <inner_steps :
        for data ,target in data_loader :
            if isinstance (data ,np .ndarray ):
                data =torch .from_numpy (data )
            if isinstance (target ,np .ndarray ):
                target =torch .from_numpy (target )

            data =data .to (device )
            target =target .to (device )
            if target .dim ()==3 :
                target =target .unsqueeze (1 )
            target =target .float ()

            inner_opt .zero_grad ()
            preds =theta (data )
            loss_inner =loss_fn (preds ,target )

            prox_reg =0.0 
            for p_w ,p_theta in zip (w_model .parameters (),theta .parameters ()):
                prox_reg =prox_reg +((p_theta -p_w ).pow (2 )).sum ()

            loss_inner =loss_inner +(lam /2.0 )*prox_reg 
            loss_inner .backward ()
            inner_opt .step ()

            steps_done +=1 
            if steps_done >=inner_steps :
                break 

    return theta 

def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Per-client Dice (pFedMe)")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_pfedme.png"))
    plt .close ()

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Per-client IoU (pFedMe)")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_pfedme.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_dice_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test Dice")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Global Test Dice Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_dice_no_bg_pfedme.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_iou_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test IoU")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Global Test IoU Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_iou_no_bg_pfedme.png"))
    plt .close ()

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
    global_test_loader =get_global_test_loader (val_tf ,batch_size =4 )
    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r + 1}/{COMM_ROUNDS}]")

        local_models =[]
        weights =[]
        total_sz =0 

        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            loss_fn =get_loss_fn (DEVICE )

            train_loader =get_loader (
            train_img_dirs [i ],
            train_mask_dirs [i ],
            tr_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =True 
            )

            val_loader =get_loader (
            val_img_dirs [i ],
            val_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =False 
            )

            print (f"[Client {client_names[i]}] Local training (pFedMe)")
            train_local_pfedme (
            train_loader ,
            local_model ,
            loss_fn ,
            inner_steps =INNER_STEPS ,
            inner_lr =INNER_LR ,
            outer_eta =OUTER_ETA ,
            lam =PFEDME_LAMBDA ,
            device =DEVICE 
            )

            evaluate (val_loader ,local_model ,loss_fn ,split ="Val (w_i)")

            local_models .append (local_model )
            sz =len (train_loader .dataset )
            weights .append (sz )
            total_sz +=sz 

        if total_sz ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your split folders and filenames.")
        norm_weights =[w /total_sz for w in weights ]
        avg_state =average_models_weighted (local_models ,norm_weights )
        new_global_state ={}
        old_state =global_model .state_dict ()
        for k in old_state .keys ():
            new_global_state [k ]=(1.0 -BETA )*old_state [k ].detach ().cpu ()+BETA *avg_state [k ]
        global_model .load_state_dict (new_global_state ,strict =True )
        rm ={}
        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (
            test_img_dirs [i ],
            test_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =False 
            )

            print (f"[Client {client_names[i]}] Test (personalized theta)")
            inner_train_loader =get_loader (
            train_img_dirs [i ],
            train_mask_dirs [i ],
            tr_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =True 
            )

            theta_tilde =compute_personalized_model_from_w (
            local_models [i ],
            inner_train_loader ,
            get_loss_fn (DEVICE ),
            inner_steps =INNER_STEPS ,
            inner_lr =INNER_LR ,
            lam =PFEDME_LAMBDA ,
            device =DEVICE 
            )

            _ ,test_metrics =evaluate (test_loader ,theta_tilde ,get_loss_fn (DEVICE ),split ="Test (theta)")
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )
        global_test_loss ,global_test_metrics =evaluate (
        global_test_loader ,
        global_model ,
        get_loss_fn (DEVICE ),
        split ="Global Test"
        )

        rm ["global_test_loss"]=global_test_loss 
        rm ["global_dice_no_bg"]=global_test_metrics .get ("dice_no_bg",0 )
        rm ["global_iou_no_bg"]=global_test_metrics .get ("iou_no_bg",0 )
        rm ["global_accuracy"]=global_test_metrics .get ("accuracy",0 )
        rm ["global_precision"]=global_test_metrics .get ("precision",0 )
        rm ["global_recall"]=global_test_metrics .get ("recall",0 )
        rm ["global_specificity"]=global_test_metrics .get ("specificity",0 )

        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

        print (
        f"[GLOBAL TEST AFTER ROUND {r + 1}] "
        f"Dice(no bg): {rm['global_dice_no_bg']:.4f} | "
        f"IoU(no bg): {rm['global_iou_no_bg']:.4f} | "
        f"Acc: {rm['global_accuracy']:.4f} | "
        f"Prec: {rm['global_precision']:.4f} | "
        f"Recall: {rm['global_recall']:.4f} | "
        f"Spec: {rm['global_specificity']:.4f}"
        )

    end_time =time .time ()
    print (f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ =="__main__":
    main ()