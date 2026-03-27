import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from torchvision import io as tv_io 
try :
    tv_io .set_image_backend ('PIL')
except Exception :
    pass 

import copy 
import time 
import random 
import shutil 
from typing import List ,Tuple 
from PIL import Image 
import numpy as np 

import matplotlib 
matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import torch 
import torch .nn as nn 
from torch .optim import AdamW 
from torch .utils .data import DataLoader ,Dataset ,ConcatDataset ,WeightedRandomSampler 
from torchvision import transforms 
from torchvision .datasets .folder import default_loader 
import timm 
import torchvision 
from tqdm import tqdm 
import pandas as pd 
from sklearn .metrics import (
precision_score ,recall_score ,f1_score ,balanced_accuracy_score ,
cohen_kappa_score ,confusion_matrix ,accuracy_score 
)

from skimage import exposure 


CLIENT_ROOTS =[
r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Shenzhen",
r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Montgomery",
r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\TBX11K",
r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES =["Shenzhen","Montgomery","TBX11K","Pakistan"]
OUTPUT_DIR =r"./tubo_his_avg"
out_dir_root =r"./tubo_his_avg"
ARCH ="densenet169"
PRETRAINED =True 
IMG_SIZE =224 
BATCH_SIZE =4 
WORKERS =4 
LOCAL_EPOCHS =6 
COMM_ROUNDS =10 
LR =1e-4 
WEIGHT_DECAY =1e-5 
USE_AMP =False 
PIN_MEMORY =True 
DROPOUT_P =0.5 
SEED =42 
CLASS_NAMES =["normal","positive"]

USE_HIST_MATCH =True 
REFERENCE_CLIENT_IDX =0 
N_REF_SAMPLES =128 
REF_RESIZE =(IMG_SIZE ,IMG_SIZE )
SAVE_HARMONIZED_SAMPLES =True 


COMPARE_FILES ={
"Shenzhen":"CHNCXR_0001_0.png",
"Montgomery":"MCUCXR_0001_0.png",
"TBX11K":"h0001.png",
"Pakistan":"TB.1.jpg"
}

os .makedirs (OUTPUT_DIR ,exist_ok =True )


def set_seed (seed =SEED ):
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed_all (seed )

set_seed ()
DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")

class PathListDataset (Dataset ):
    def __init__ (self ,samples :List [Tuple [str ,int ]],transform =None ,loader =default_loader ):
        self .samples =list (samples )
        self .transform =transform 
        self .loader =loader 

    def __len__ (self ):
        return len (self .samples )

    def __getitem__ (self ,idx ):
        path ,label =self .samples [idx ]
        img =self .loader (path ).convert ("RGB")
        if self .transform :
            img =self .transform (img )
        return img ,label 

def gather_samples_from_client_split (client_root :str ,split :str ,class_names :List [str ]):
    split_dir =os .path .join (client_root ,split )
    if not os .path .isdir (split_dir ):
        print (f"Warning: Missing '{split}' in {client_root} -> returning empty list")
        return []
    samples =[]
    canon_map ={c .lower ():i for i ,c in enumerate (class_names )}
    for cls_folder in sorted (os .listdir (split_dir )):
        cls_path =os .path .join (split_dir ,cls_folder )
        if not os .path .isdir (cls_path ):
            continue 
        key =cls_folder .lower ()
        if key not in canon_map :
            print (f"Warning: unknown class folder '{cls_folder}' in {split_dir}; skipping")
            continue 
        label =canon_map [key ]
        for fn in sorted (os .listdir (cls_path )):
            if fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                samples .append ((os .path .join (cls_path ,fn ),label ))
    return samples 

def make_multi_client_dataloaders (client_roots ,batch_size =BATCH_SIZE ,image_size =IMG_SIZE ,workers =WORKERS ,pin_memory =PIN_MEMORY ):
    normalize =transforms .Normalize (mean =[0.485 ,0.456 ,0.406 ],std =[0.229 ,0.224 ,0.225 ])
    train_tf =transforms .Compose ([
    transforms .RandomResizedCrop (image_size ,scale =(0.8 ,1.0 )),
    transforms .RandomHorizontalFlip (),
    transforms .RandomRotation (5 ),
    transforms .ColorJitter (brightness =0.05 ,contrast =0.05 ),
    transforms .ToTensor (),
    normalize 
    ])
    val_tf =transforms .Compose ([
    transforms .Resize ((image_size ,image_size )),
    transforms .ToTensor (),
    normalize 
    ])

    train_samples_all ,val_samples_all ,test_samples_all =[],[],[]
    per_client_dataloaders =[]
    per_client_test_dsets ={}

    for client_root in client_roots :
        tr =gather_samples_from_client_split (client_root ,"train",CLASS_NAMES )
        va =gather_samples_from_client_split (client_root ,"val",CLASS_NAMES )
        te =gather_samples_from_client_split (client_root ,"test",CLASS_NAMES )
        print (f"[DATA] client {client_root} -> train:{len(tr)} val:{len(va)} test:{len(te)}")
        train_samples_all .extend (tr );val_samples_all .extend (va );test_samples_all .extend (te )

        train_ds =PathListDataset (tr ,transform =train_tf )
        val_ds =PathListDataset (va ,transform =val_tf )
        test_ds =PathListDataset (te ,transform =val_tf )

        per_client_dataloaders .append ({
        "train":DataLoader (train_ds ,batch_size =batch_size ,shuffle =True ,num_workers =workers ,pin_memory =pin_memory ),
        "val":DataLoader (val_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory ),
        "test":DataLoader (test_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory ),
        "train_ds":train_ds 
        })
        per_client_test_dsets [client_root ]=test_ds 

    combined_train_ds =PathListDataset (train_samples_all ,transform =train_tf )
    combined_val_ds =PathListDataset (val_samples_all ,transform =val_tf )
    combined_test_ds =PathListDataset (test_samples_all ,transform =val_tf )
    dataloaders_combined ={
    "train":DataLoader (combined_train_ds ,batch_size =batch_size ,shuffle =True ,num_workers =workers ,pin_memory =pin_memory ),
    "val":DataLoader (combined_val_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory ),
    "test":DataLoader (combined_test_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory )
    }
    sizes ={"train":len (combined_train_ds ),"val":len (combined_val_ds ),"test":len (combined_test_ds )}
    return dataloaders_combined ,sizes ,CLASS_NAMES ,combined_train_ds ,per_client_dataloaders ,per_client_test_dsets 

def compute_class_weights_from_dataset (dataset ):
    targets =[s [1 ]for s in dataset .samples ]
    if len (targets )==0 :
        return torch .ones (len (CLASS_NAMES ),dtype =torch .float32 )
    counts =np .bincount (targets ,minlength =len (CLASS_NAMES )).astype (np .float32 )
    total =counts .sum ()
    weights =total /(counts +1e-8 )
    weights =weights /np .mean (weights )
    return torch .tensor (weights ,dtype =torch .float32 )

def make_weighted_sampler_from_dataset (dataset ):
    targets =[s [1 ]for s in dataset .samples ]
    counts =np .bincount (targets ,minlength =len (CLASS_NAMES )).astype (np .float32 )
    inv_freq =1.0 /np .maximum (counts ,1.0 )
    weights =[float (inv_freq [t ])for t in targets ]
    return WeightedRandomSampler (weights ,num_samples =len (weights ),replacement =True )

def create_model (num_classes ,arch =ARCH ,pretrained =PRETRAINED ):
    if arch .startswith ("densenet")and hasattr (torchvision .models ,arch ):
        model =getattr (torchvision .models ,arch )(pretrained =pretrained )
        if hasattr (model ,"classifier"):
            in_ch =model .classifier .in_features 
            model .classifier =nn .Sequential (nn .Dropout (p =DROPOUT_P ),nn .Linear (in_ch ,num_classes ))
        elif hasattr (model ,"fc"):
            in_ch =model .fc .in_features 
            model .fc =nn .Sequential (nn .Dropout (p =DROPOUT_P ),nn .Linear (in_ch ,num_classes ))
        return model 
    else :
        model =timm .create_model (arch ,pretrained =pretrained ,num_classes =num_classes )
        return model 

def count_parameters (model ):
    return sum (p .numel ()for p in model .parameters ()if p .requires_grad )

def average_models_weighted (models :List [torch .nn .Module ],weights :List [float ]):
    if len (models )==0 :
        raise ValueError ("No models to average")
    if len (models )!=len (weights ):
        raise ValueError ("models and weights must have same length")
    sum_w =float (sum (weights ))
    norm_weights =[w /sum_w for w in weights ]
    base_sd =models [0 ].state_dict ()
    avg_sd ={}
    with torch .no_grad ():
        for k ,v0 in base_sd .items ():
            acc =torch .zeros_like (v0 ,dtype =torch .float32 ,device ="cpu")
            for m ,w in zip (models ,norm_weights ):
                vm =m .state_dict ()[k ].cpu ().to (dtype =torch .float32 )
                acc +=float (w )*vm 
            try :
                acc =acc .to (dtype =v0 .dtype )
            except Exception :
                acc =acc 
            avg_sd [k ]=acc 
    return avg_sd 


def train_local (model ,dataloader ,criterion ,optimizer ,device ,epochs =LOCAL_EPOCHS ,use_amp =False ):
    model .to (device )
    scaler =torch .cuda .amp .GradScaler ()if (use_amp and device .type =="cuda")else None 
    logs =[]
    for ep in range (epochs ):
        model .train ()
        running_loss =0.0 
        correct =0 
        total =0 
        pbar =tqdm (dataloader ,desc =f"LocalTrain ep{ep+1}/{epochs}",leave =False )
        for x ,y in pbar :
            x ,y =x .to (device ),y .to (device )
            optimizer .zero_grad ()
            with torch .cuda .amp .autocast (enabled =(scaler is not None )):
                out =model (x )
                loss =criterion (out ,y )
            if scaler :
                scaler .scale (loss ).backward ();scaler .step (optimizer );scaler .update ()
            else :
                loss .backward ();optimizer .step ()
            running_loss +=float (loss .item ())*x .size (0 )
            _ ,preds =out .max (1 )
            correct +=(preds ==y ).sum ().item ()
            total +=x .size (0 )
            pbar .set_postfix (loss =running_loss /total if total >0 else 0.0 ,acc =correct /total if total >0 else 0.0 )
        epoch_loss =running_loss /max (1 ,total )
        epoch_acc =correct /max (1 ,total )
        logs .append ((epoch_loss ,epoch_acc ))
    return logs 

@torch .no_grad ()
def evaluate_model (model ,dataloader ,device ,criterion =None ,return_per_class =False ,class_names =None ):
    model .eval ()
    all_y =[];all_pred =[]
    total_loss =0.0 ;n =0 
    for x ,y in tqdm (dataloader ,desc ="Eval",leave =False ):
        x ,y =x .to (device ),y .to (device )
        out =model (x )
        _ ,preds =out .max (1 )
        all_y .extend (y .cpu ().numpy ().tolist ())
        all_pred .extend (preds .cpu ().numpy ().tolist ())
        if criterion is not None :
            loss =criterion (out ,y )
            total_loss +=float (loss .item ())*x .size (0 )
        n +=x .size (0 )
    if n ==0 and len (all_y )==0 :
        return {}
    acc =accuracy_score (all_y ,all_pred )
    prec_macro =precision_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    rec_macro =recall_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    f1_macro =f1_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    bal =balanced_accuracy_score (all_y ,all_pred )
    kappa =cohen_kappa_score (all_y ,all_pred )
    metrics ={
    "accuracy":float (acc ),
    "precision_macro":float (prec_macro ),
    "recall_macro":float (rec_macro ),
    "f1_macro":float (f1_macro ),
    "balanced_acc":float (bal ),
    "cohen_kappa":float (kappa )
    }
    if criterion is not None :
        metrics ["loss"]=float (total_loss /max (1 ,n ))if n >0 else float ('nan')

    if return_per_class :
        if class_names is None :
            raise ValueError ("class_names must be provided when return_per_class=True")
        num_classes =len (class_names )
        cm =confusion_matrix (all_y ,all_pred ,labels =list (range (num_classes )))
        per_class_prec =precision_score (all_y ,all_pred ,labels =list (range (num_classes )),average =None ,zero_division =0 )
        per_class_rec =recall_score (all_y ,all_pred ,labels =list (range (num_classes )),average =None ,zero_division =0 )
        per_class_f1 =f1_score (all_y ,all_pred ,labels =list (range (num_classes )),average =None ,zero_division =0 )

        tp =np .diag (cm ).astype (float )
        fp =cm .sum (axis =0 )-tp 
        fn =cm .sum (axis =1 )-tp 
        tn =cm .sum ()-(tp +fp +fn )
        per_class_spec =np .divide (tn ,(tn +fp ),out =np .zeros_like (tn ),where =(tn +fp )!=0 )
        support =cm .sum (axis =1 ).astype (float )
        per_class_acc =np .divide (tp ,support ,out =np .zeros_like (tp ),where =support !=0 )

        metrics .update ({
        "confusion_matrix":cm ,
        "per_class_precision":[float (x )for x in per_class_prec .tolist ()],
        "per_class_recall":[float (x )for x in per_class_rec .tolist ()],
        "per_class_f1":[float (x )for x in per_class_f1 .tolist ()],
        "per_class_specificity":[float (x )for x in per_class_spec .tolist ()],
        "per_class_correct":[int (x )for x in tp .tolist ()],
        "per_class_accuracy":[float (x )for x in per_class_acc .tolist ()],
        "per_class_support":[int (x )for x in support .tolist ()]
        })
    return metrics 

def _list_all_train_images_from_client (client_root ,max_samples =None ):
    train_dir =os .path .join (client_root ,"train")
    img_paths =[]
    if not os .path .isdir (train_dir ):
        return img_paths 
    for cls in sorted (os .listdir (train_dir )):
        cls_dir =os .path .join (train_dir ,cls )
        if not os .path .isdir (cls_dir ):
            continue 
        for fn in sorted (os .listdir (cls_dir )):
            if fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                img_paths .append (os .path .join (cls_dir ,fn ))
    if max_samples is not None :
        img_paths =img_paths [:max_samples ]
    return img_paths 

def compute_reference_image_from_client (ref_client_root ,n_samples =N_REF_SAMPLES ,resize_to =REF_RESIZE ):
    files =_list_all_train_images_from_client (ref_client_root ,max_samples =n_samples )
    if len (files )==0 :
        raise ValueError (f"No train images found in reference client: {ref_client_root}")
    acc =None 
    count =0 
    for p in files :
        im =Image .open (p ).convert ("RGB")
        if resize_to is not None :
            im =im .resize (resize_to ,resample =Image .BILINEAR )
        arr =np .array (im ).astype (np .float32 )
        if acc is None :
            acc =arr 
        else :
            acc +=arr 
        count +=1 
    avg =(acc /count ).astype (np .uint8 )
    return avg 

def _client_dst_has_files (dst_root ):
    if not os .path .isdir (dst_root ):
        return False 
    for split in ("train","val","test"):
        splitdir =os .path .join (dst_root ,split )
        if not os .path .isdir (splitdir ):
            continue 
        for root ,_ ,files in os .walk (splitdir ):
            for fn in files :
                if fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    return True 
    return False 

def histogram_match_client (src_client_root ,dst_client_root ,reference_image ):
    for split in ("train","val","test"):
        src_split =os .path .join (src_client_root ,split )
        if not os .path .isdir (src_split ):
            continue 
        for cls in sorted (os .listdir (src_split )):
            src_cls_dir =os .path .join (src_split ,cls )
            if not os .path .isdir (src_cls_dir ):
                continue 
            dst_cls_dir =os .path .join (dst_client_root ,split ,cls )
            os .makedirs (dst_cls_dir ,exist_ok =True )
            for fn in sorted (os .listdir (src_cls_dir )):
                if not fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    continue 
                src_p =os .path .join (src_cls_dir ,fn )
                dst_p =os .path .join (dst_cls_dir ,fn )
                if os .path .exists (dst_p ):
                    continue 
                try :
                    img =np .array (Image .open (src_p ).convert ("RGB"))
                    try :
                        matched =exposure .match_histograms (img ,reference_image ,channel_axis =-1 )
                    except TypeError :
                        matched =exposure .match_histograms (img ,reference_image ,multichannel =True )
                    matched =np .clip (matched ,0 ,255 ).astype (np .uint8 )
                    Image .fromarray (matched ).save (dst_p )
                except Exception :

                    shutil .copy2 (src_p ,dst_p )

def create_histogram_matched_client_roots (client_roots ,client_names ,out_base ,reference_client_idx =0 ,n_ref_samples =N_REF_SAMPLES ,resize_ref =REF_RESIZE ):
    base =os .path .join (out_base ,"HistMatch")
    os .makedirs (base ,exist_ok =True )
    ref_save =os .path .join (base ,"reference_image.png")

    if os .path .exists (ref_save ):
        try :
            ref_img =np .array (Image .open (ref_save ).convert ("RGB"))
            print (f"[HistMatch] Reusing existing reference image at {ref_save}")
        except Exception :
            print (f"[HistMatch] Failed to load existing reference image {ref_save}, recomputing...")
            ref_root =client_roots [reference_client_idx ]
            ref_img =compute_reference_image_from_client (ref_root ,n_samples =n_ref_samples ,resize_to =resize_ref )
            Image .fromarray (ref_img ).save (ref_save )
            print (f"[HistMatch] Reference image saved to {ref_save}")
    else :
        ref_root =client_roots [reference_client_idx ]
        print (f"[HistMatch] Computing reference image from client {client_names[reference_client_idx]} ({ref_root}) ...")
        ref_img =compute_reference_image_from_client (ref_root ,n_samples =n_ref_samples ,resize_to =resize_ref )
        Image .fromarray (ref_img ).save (ref_save )
        print (f"[HistMatch] Reference image saved to {ref_save}")

    hm_roots =[]
    for i ,cr in enumerate (client_roots ):
        cname =client_names [i ]if i <len (client_names )else f"client_{i}"
        dst =os .path .join (base ,cname )
        if _client_dst_has_files (dst ):
            print (f"[HistMatch] Found existing harmonized images for client '{cname}' -> skipping re-harmonization for this client.")
            hm_roots .append (dst )
            continue 
        os .makedirs (dst ,exist_ok =True )
        print (f"[HistMatch] Harmonizing client {cname} ...")
        histogram_match_client (cr ,dst ,ref_img )
        hm_roots .append (dst )
    print ("[HistMatch] Done creating histogram-matched client roots.")
    return hm_roots ,ref_save 

def save_hist_match_examples (hm_client_root ,orig_client_root ,dest_dir ,n_samples =7 ):
    os .makedirs (dest_dir ,exist_ok =True )
    out_path =os .path .join (dest_dir ,"hist_match_examples.png")
    if os .path .exists (out_path ):
        print (f"[HistMatch] QA image already exists at {out_path}, skipping creation.")
        return 
    orig_train =os .path .join (orig_client_root ,"train")
    if not os .path .isdir (orig_train ):
        return 
    candidates =[]
    for cls in sorted (os .listdir (orig_train )):
        clsdir =os .path .join (orig_train ,cls )
        if not os .path .isdir (clsdir ):
            continue 
        for fn in sorted (os .listdir (clsdir )):
            if fn .lower ().endswith ((".png",".jpg",".jpeg")):
                candidates .append ((cls ,fn ,clsdir ))
    candidates =candidates [:n_samples ]
    if not candidates :
        return 
    top_imgs =[];mid_imgs =[];names =[]
    for cls ,fn ,clsdir in candidates :
        orig_p =os .path .join (clsdir ,fn )
        hm_p =os .path .join (hm_client_root ,"train",cls ,fn )
        if not os .path .exists (hm_p ):
            continue 
        orig =np .array (Image .open (orig_p ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE )))
        matched =np .array (Image .open (hm_p ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE )))
        top_imgs .append (orig );mid_imgs .append (matched );names .append (fn )
    n =len (top_imgs )
    if n ==0 :
        return 
    fig ,axs =plt .subplots (3 ,n ,figsize =(3 *n ,6 ))
    if n ==1 :
        axs =np .array ([[axs [0 ]],[axs [1 ]],[axs [2 ]]])
    for i in range (n ):
        axs [0 ,i ].imshow (top_imgs [i ]);axs [0 ,i ].axis ('off');axs [0 ,i ].set_title (names [i ],fontsize =8 )
        axs [1 ,i ].imshow (mid_imgs [i ]);axs [1 ,i ].axis ('off')
        ax =axs [2 ,i ]
        for ch in range (3 ):
            ax .hist (top_imgs [i ][:,:,ch ].ravel (),bins =128 ,alpha =0.4 )
            ax .hist (mid_imgs [i ][:,:,ch ].ravel (),bins =128 ,alpha =0.4 ,histtype ='step')
        ax .set_xticks ([]);ax .set_yticks ([])
    plt .tight_layout ()
    plt .savefig (out_path ,dpi =150 );plt .close ()
    print (f"[HistMatch] Saved visual QA to {out_path}")


def _find_file_recursive (root_dir :str ,target_name :str ):
    if not os .path .isdir (root_dir ):
        return None 
    target_lower =target_name .lower ()
    for root ,_ ,files in os .walk (root_dir ):
        for fn in files :
            if fn .lower ()==target_lower :
                return os .path .join (root ,fn )
    target_base =os .path .splitext (target_lower )[0 ]
    for root ,_ ,files in os .walk (root_dir ):
        for fn in files :
            if os .path .splitext (fn .lower ())[0 ]==target_base :
                return os .path .join (root ,fn )
    return None 

def make_comparison_grid_for_client (orig_client_root :str ,hm_client_root :str ,client_name :str ,desired_fname :str ,out_dir_root :str ,img_size =IMG_SIZE ,amp_factor =4.0 ):
    out_base =os .path .join (out_dir_root ,"ComparisonGrids")
    os .makedirs (out_base ,exist_ok =True )
    out_png =os .path .join (out_base ,f"comparison_{client_name}.png")

    orig_path =_find_file_recursive (orig_client_root ,desired_fname )
    if orig_path is None :
        print (f"[GRID] Original file '{desired_fname}' not found under {orig_client_root} for client {client_name}. Skipping grid.")
        return 

    hm_path =_find_file_recursive (hm_client_root ,desired_fname )
    if hm_path is None :
        print (f"[GRID] Harmonized file '{desired_fname}' not found under {hm_client_root} for client {client_name}. Skipping grid.")
        return 

    try :
        orig =Image .open (orig_path ).convert ("RGB").resize ((img_size ,img_size ))
        hm =Image .open (hm_path ).convert ("RGB").resize ((img_size ,img_size ))
        orig_np =np .array (orig ).astype (np .float32 )
        hm_np =np .array (hm ).astype (np .float32 )
        diff =np .abs (orig_np -hm_np )
        amplified =np .clip (diff *amp_factor ,0 ,255 ).astype (np .uint8 )

        if amplified .max ()<8 and diff .max ()>0 :
            amplified =(diff /(diff .max ()+1e-8 )*255.0 ).astype (np .uint8 )

        fig ,axs =plt .subplots (3 ,1 ,figsize =(6 ,9 ))
        axs [0 ].imshow (orig_np .astype (np .uint8 ));axs [0 ].axis ('off');axs [0 ].set_title ("Original (raw)",fontsize =10 )
        axs [1 ].imshow (hm_np .astype (np .uint8 ));axs [1 ].axis ('off');axs [1 ].set_title ("Harmonized (hist-match)",fontsize =10 )
        axs [2 ].imshow (amplified );axs [2 ].axis ('off');axs [2 ].set_title ("Amplified difference (|orig - harmonized|)",fontsize =10 )
        plt .suptitle (f"{client_name}: {desired_fname}",fontsize =12 )
        plt .tight_layout (rect =[0 ,0.03 ,1 ,0.95 ])
        plt .savefig (out_png ,dpi =150 ,bbox_inches ='tight');plt .close (fig )
        print (f"[GRID] Saved comparison grid for {client_name} -> {out_png}")
    except Exception as e :
        print (f"[GRID] Failed creating grid for {client_name}: {e}")


def _fmt_pct (x ):
    if x is None or (isinstance (x ,float )and (np .isnan (x )or np .isinf (x ))):
        return "   n/a "
    return f"{100.0 * x:6.2f}%"

def print_metrics_summary (metrics :dict ,class_names :list ,title :str ="Test"):
    print ()
    print (f"{title} Per-class metrics:")
    per_prec =metrics .get ("per_class_precision",[])
    per_rec =metrics .get ("per_class_recall",[])
    per_f1 =metrics .get ("per_class_f1",[])
    per_spec =metrics .get ("per_class_specificity",[])
    per_acc =metrics .get ("per_class_accuracy",[])
    per_support =metrics .get ("per_class_support",[])
    per_correct =metrics .get ("per_class_correct",[])


    print ("  {:12s} | {:>6s} | {:>6s} | {:>6s} | {:>6s} | {:>6s} | {:>6s}".format (
    "","acc","prec","rec","f1","spec","supp"
    ))

    for i ,cname in enumerate (class_names ):
        acc =per_acc [i ]if i <len (per_acc )else np .nan 
        prec =per_prec [i ]if i <len (per_prec )else np .nan 
        rec =per_rec [i ]if i <len (per_rec )else np .nan 
        f1 =per_f1 [i ]if i <len (per_f1 )else np .nan 
        spec =per_spec [i ]if i <len (per_spec )else np .nan 
        supp =per_support [i ]if i <len (per_support )else 0 
        print ("  {:12s} | {:>6s} | {:>6s} | {:>6s} | {:>6s} | {:>6s} | {:>6d}".format (
        cname [:12 ],
        _fmt_pct (acc ),
        _fmt_pct (prec ),
        _fmt_pct (rec ),
        _fmt_pct (f1 ),
        _fmt_pct (spec ),
        int (supp )
        ))

    print ()
    print (f"{title} Macro / mean metrics:")
    acc =metrics .get ("accuracy",np .nan )
    bal =metrics .get ("balanced_acc",np .nan )
    prec_macro =metrics .get ("precision_macro",np .nan )
    rec_macro =metrics .get ("recall_macro",np .nan )
    f1_macro =metrics .get ("f1_macro",np .nan )
    kappa =metrics .get ("cohen_kappa",np .nan )
    mean_spec =None 
    if "per_class_specificity"in metrics :
        try :
            mean_spec =float (np .mean (metrics .get ("per_class_specificity",[np .nan ])))
        except Exception :
            mean_spec =None 

    print (f"  Mean class accuracy (balanced acc): {100.0*bal:6.2f}%")
    print (f"  Macro precision : {100.0*prec_macro:6.2f}%")
    print (f"  Macro recall    : {100.0*rec_macro:6.2f}%")
    print (f"  Macro F1        : {100.0*f1_macro:6.2f}%")
    if mean_spec is not None and not np .isnan (mean_spec ):
        print (f"  Mean specificity: {100.0*mean_spec:6.2f}%")
    else :
        print (f"  Mean specificity:    n/a")
    print (f"  Cohen's kappa    : {kappa:6.4f}")
    print ()


def main ():
    print ("DEVICE:",DEVICE )

    if USE_HIST_MATCH :
        try :
            hm_roots ,ref_img_path =create_histogram_matched_client_roots (CLIENT_ROOTS ,CLIENT_NAMES ,OUTPUT_DIR ,reference_client_idx =REFERENCE_CLIENT_IDX )
            client_roots_used =hm_roots 
            print (f"[Main] Using histogram-matched client roots under {os.path.join(OUTPUT_DIR,'HistMatch')}")
        except Exception as e :
            print ("[Main] Histogram matching failed, falling back to original client roots. Error:",e )
            client_roots_used =CLIENT_ROOTS 
            ref_img_path =None 
    else :
        client_roots_used =CLIENT_ROOTS 
        ref_img_path =None 

    if USE_HIST_MATCH and SAVE_HARMONIZED_SAMPLES :
        qa_dir =os .path .join (OUTPUT_DIR ,"HistMatch_QA")
        os .makedirs (qa_dir ,exist_ok =True )
        for i ,cr in enumerate (CLIENT_ROOTS ):
            cname =CLIENT_NAMES [i ]if i <len (CLIENT_NAMES )else f"client_{i}"
            hm_root =client_roots_used [i ]
            dest =os .path .join (qa_dir ,cname )
            try :
                save_hist_match_examples (hm_root ,cr ,dest ,n_samples =7 )
            except Exception as e :
                print (f"[HistMatch QA] failed for {cname}: {e}")

    try :
        for i ,cname in enumerate (CLIENT_NAMES ):
            desired =COMPARE_FILES .get (cname )
            if desired is None :
                print (f"[GRID] No compare filename specified for {cname}, skipping.")
                continue 
            orig_root =CLIENT_ROOTS [i ]
            hm_root =client_roots_used [i ]if i <len (client_roots_used )else None 
            if hm_root is None :
                print (f"[GRID] No harmonized root for client {cname}, skipping grid.")
                continue 
            make_comparison_grid_for_client (orig_root ,hm_root ,cname ,desired ,OUTPUT_DIR ,img_size =IMG_SIZE ,amp_factor =4.0 )
    except Exception as e :
        print ("[GRID] Error while creating comparison grids:",e )

    combined_loaders ,combined_sizes ,class_names ,combined_train_ds ,per_client_dataloaders ,per_client_test_dsets =make_multi_client_dataloaders (
    client_roots_used ,batch_size =BATCH_SIZE ,image_size =IMG_SIZE ,workers =WORKERS ,pin_memory =PIN_MEMORY and (DEVICE .type =="cuda")
    )
    num_classes =len (class_names )
    print ("class names:",class_names )

    client_train_sizes =[len (per_client_dataloaders [i ]['train'].dataset )for i in range (len (per_client_dataloaders ))]
    total_train =sum (client_train_sizes )if sum (client_train_sizes )>0 else 1 
    print ("client train sizes:",client_train_sizes )


    global_model =create_model (num_classes =num_classes ,arch =ARCH ,pretrained =PRETRAINED ).to (DEVICE )
    print (f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")

    round_results =[]
    global_test_acc_fname =os .path .join (OUTPUT_DIR ,"global_test_accuracy_rounds.png")
    global_test_loss_fname =os .path .join (OUTPUT_DIR ,"global_test_loss_rounds.png")
    per_client_acc_history ={i :[]for i in range (len (per_client_dataloaders ))}
    per_client_loss_history ={i :[]for i in range (len (per_client_dataloaders ))}

    for r in range (COMM_ROUNDS ):
        print ("\n"+"="*60 )
        print (f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print ("="*60 )
        local_models =[]
        weights =[]
        round_summary ={"round":r +1 }

        for i ,client in enumerate (per_client_dataloaders ):
            print (f"\n[CLIENT {i}] {CLIENT_NAMES[i]}: local training")
            local_model =copy .deepcopy (global_model )
            train_ds =client ['train'].dataset 
            client_cw =compute_class_weights_from_dataset (train_ds ).to (DEVICE )
            criterion =nn .CrossEntropyLoss (weight =client_cw )
            optimizer =AdamW (local_model .parameters (),lr =LR ,weight_decay =WEIGHT_DECAY )


            logs =train_local (local_model ,client ['train'],criterion ,optimizer ,DEVICE ,epochs =LOCAL_EPOCHS ,use_amp =USE_AMP )
            last_train_loss ,last_train_acc =logs [-1 ]
            print (f"[CLIENT {i}] last local epoch loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
            round_summary [f"client{i}_train_loss"]=float (last_train_loss )
            round_summary [f"client{i}_train_acc"]=float (last_train_acc )

            print (f"[CLIENT {i}] local validation")
            local_val_metrics =evaluate_model (local_model ,client ['val'],DEVICE ,criterion =criterion )
            print (f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary [f"client{i}_localval_loss"]=float (local_val_metrics .get ("loss",np .nan ))
            round_summary [f"client{i}_localval_acc"]=float (local_val_metrics .get ("accuracy",np .nan ))

            local_models .append (local_model .cpu ())
            w =float (client_train_sizes [i ])/float (total_train )
            weights .append (w )
            print (f"[CLIENT {i}] aggregation weight: {w:.4f}")


        print ("\nAggregating local models (FedAvg weighted)")
        avg_state =average_models_weighted (local_models ,weights )
        avg_state_on_device ={k :v .to (DEVICE )for k ,v in avg_state .items ()}
        global_model .load_state_dict (avg_state_on_device )
        global_model .to (DEVICE )

        print ("\nGlobal validation on combined val sets...")
        combined_val_dsets =[per_client_dataloaders [i ]['val'].dataset for i in range (len (per_client_dataloaders ))]
        combined_val =ConcatDataset (combined_val_dsets )
        combined_val_loader =DataLoader (combined_val ,batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =PIN_MEMORY and (DEVICE .type =="cuda"))

        combined_train_targets =[]
        for i in range (len (per_client_dataloaders )):
            combined_train_targets .extend ([s [1 ]for s in per_client_dataloaders [i ]['train'].dataset .samples ])
        counts =np .bincount (combined_train_targets ,minlength =num_classes ).astype (np .float32 )
        counts [counts ==0 ]=1.0 
        weights_arr =1.0 /counts 
        weights_arr =weights_arr *(len (weights_arr )/weights_arr .sum ())
        combined_class_weights =torch .tensor (weights_arr ,dtype =torch .float32 ).to (DEVICE )
        combined_criterion =nn .CrossEntropyLoss (weight =combined_class_weights )

        global_val_metrics =evaluate_model (global_model ,combined_val_loader ,DEVICE ,criterion =combined_criterion )
        print ("Global combined val metrics:",global_val_metrics )
        round_summary ["global_val_loss"]=float (global_val_metrics .get ("loss",np .nan ))
        round_summary ["global_val_acc"]=float (global_val_metrics .get ("accuracy",np .nan ))

        print ("\nGlobal TEST on combined test (all clients)")
        combined_test_dsets =[per_client_dataloaders [i ]['test'].dataset for i in range (len (per_client_dataloaders ))]
        combined_test =ConcatDataset (combined_test_dsets )
        combined_test_loader =DataLoader (combined_test ,batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =PIN_MEMORY and (DEVICE .type =="cuda"))

        global_test_metrics =evaluate_model (global_model ,combined_test_loader ,DEVICE ,criterion =combined_criterion ,return_per_class =True ,class_names =class_names )

        print ("\nGlobal combined TEST metrics summary:")
        print_metrics_summary (global_test_metrics ,class_names ,title ="GLOBAL COMBINED")

        round_summary ["global_test_loss"]=float (global_test_metrics .get ("loss",np .nan ))
        round_summary ["global_test_acc"]=float (global_test_metrics .get ("accuracy",np .nan ))

        try :
            cm =global_test_metrics .get ("confusion_matrix",None )
            per_prec =global_test_metrics .get ("per_class_precision",[])
            per_rec =global_test_metrics .get ("per_class_recall",[])
            per_f1 =global_test_metrics .get ("per_class_f1",[])
            per_spec =global_test_metrics .get ("per_class_specificity",[])
            per_acc =global_test_metrics .get ("per_class_accuracy",[])
            per_support =global_test_metrics .get ("per_class_support",[])
            per_correct =global_test_metrics .get ("per_class_correct",[])
            combined_rows =[]
            if cm is not None :
                for ci ,cname in enumerate (class_names ):
                    combined_rows .append ({
                    "class":cname ,
                    "support":int (per_support [ci ])if ci <len (per_support )else 0 ,
                    "correct":int (per_correct [ci ])if ci <len (per_correct )else 0 ,
                    "acc":float (per_acc [ci ])if ci <len (per_acc )else np .nan ,
                    "prec":float (per_prec [ci ])if ci <len (per_prec )else np .nan ,
                    "rec":float (per_rec [ci ])if ci <len (per_rec )else np .nan ,
                    "f1":float (per_f1 [ci ])if ci <len (per_f1 )else np .nan ,
                    "spec":float (per_spec [ci ])if ci <len (per_spec )else np .nan 
                    })
            macro_row ={
            "class":"macro",
            "support":int (cm .sum ())if cm is not None else sum ([r ["support"]for r in combined_rows ])if combined_rows else 0 ,
            "correct":int (np .sum ([r ["correct"]for r in combined_rows ]))if combined_rows else 0 ,
            "acc":float (global_test_metrics .get ("balanced_acc",np .nan )),
            "prec":float (global_test_metrics .get ("precision_macro",np .nan )),
            "rec":float (global_test_metrics .get ("recall_macro",np .nan )),
            "f1":float (global_test_metrics .get ("f1_macro",np .nan )),
            "spec":float (np .mean (per_spec ))if len (per_spec )>0 else float (np .nan )
            }
            combined_rows .append (macro_row )
            df_combined =pd .DataFrame (combined_rows )
            combined_csv =os .path .join (OUTPUT_DIR ,"combined_test_metrics.csv")
            df_combined .to_csv (combined_csv ,index =False )
            if cm is not None :
                pd .DataFrame (cm ,index =class_names ,columns =class_names ).to_csv (os .path .join (OUTPUT_DIR ,"combined_confusion_matrix.csv"))
            print (f"Saved combined per-class test metrics CSV to: {combined_csv}")
        except Exception as e :
            print ("Warning saving/printing combined per-class metrics:",e )

        per_client_test_metrics =[]
        for i ,client in enumerate (per_client_dataloaders ):
            print (f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            client_train_ds =client ['train'].dataset 
            client_cw =compute_class_weights_from_dataset (client_train_ds ).to (DEVICE )
            client_criterion =nn .CrossEntropyLoss (weight =client_cw )
            cl_metrics =evaluate_model (global_model ,client ['test'],DEVICE ,criterion =client_criterion ,return_per_class =True ,class_names =class_names )

            print (f"\n[CLIENT {i}] Global model evaluated on client {CLIENT_NAMES[i]} test set")
            print_metrics_summary (cl_metrics ,class_names ,title =f"[CLIENT {i}]")

            try :
                cmc =cl_metrics .get ("confusion_matrix",None )
                rows =[]
                if cmc is not None :
                    per_support =cl_metrics .get ("per_class_support",[])
                    per_correct =cl_metrics .get ("per_class_correct",[])
                    per_accs =cl_metrics .get ("per_class_accuracy",[])
                    per_precs =cl_metrics .get ("per_class_precision",[])
                    per_recs =cl_metrics .get ("per_class_recall",[])
                    per_f1s =cl_metrics .get ("per_class_f1",[])
                    per_specs =cl_metrics .get ("per_class_specificity",[])
                    for i_c ,cname in enumerate (class_names ):
                        rows .append ({
                        "class":cname ,
                        "support":int (per_support [i_c ])if i_c <len (per_support )else int (cmc [i_c ].sum ())if cmc is not None else 0 ,
                        "correct":int (per_correct [i_c ])if i_c <len (per_correct )else int (np .diag (cmc )[i_c ])if cmc is not None else 0 ,
                        "acc":float (per_accs [i_c ])if i_c <len (per_accs )else np .nan ,
                        "prec":float (per_precs [i_c ])if i_c <len (per_precs )else np .nan ,
                        "rec":float (per_recs [i_c ])if i_c <len (per_recs )else np .nan ,
                        "f1":float (per_f1s [i_c ])if i_c <len (per_f1s )else np .nan ,
                        "spec":float (per_specs [i_c ])if i_c <len (per_specs )else np .nan 
                        })
                macro ={
                "class":"macro",
                "support":int (cmc .sum ())if cmc is not None else 0 ,
                "correct":int (np .sum ([r ["correct"]for r in rows ]))if rows else 0 ,
                "acc":cl_metrics .get ("balanced_acc",np .nan ),
                "prec":cl_metrics .get ("precision_macro",np .nan ),
                "rec":cl_metrics .get ("recall_macro",np .nan ),
                "f1":cl_metrics .get ("f1_macro",np .nan ),
                "spec":float (np .mean (cl_metrics .get ("per_class_specificity",[np .nan ])))if "per_class_specificity"in cl_metrics else np .nan 
                }
                if rows :
                    rows .append (macro )
                dfc =pd .DataFrame (rows )
                safe_name =CLIENT_NAMES [i ].replace (" ","_")
                client_csv =os .path .join (OUTPUT_DIR ,f"{safe_name}_test_metrics_round{r+1}.csv")
                dfc .to_csv (client_csv ,index =False )
                if cmc is not None :
                    pd .DataFrame (cmc ,index =class_names ,columns =class_names ).to_csv (os .path .join (OUTPUT_DIR ,f"{safe_name}_confusion_matrix_round{r+1}.csv"))
                print (f"Saved per-client metrics CSV to {client_csv}")
            except Exception as e :
                print ("Warning saving per-client metrics CSV:",e )

            per_client_test_metrics .append (cl_metrics )
            round_summary [f"client{i}_test_loss"]=float (cl_metrics .get ("loss",np .nan ))
            round_summary [f"client{i}_test_acc"]=float (cl_metrics .get ("accuracy",np .nan ))
            per_client_acc_history [i ].append (float (cl_metrics .get ("accuracy",np .nan )))
            per_client_loss_history [i ].append (float (cl_metrics .get ("loss",np .nan )))

        ckpt ={
        "round":r +1 ,
        "model_state":global_model .state_dict (),
        "global_val_metrics":global_val_metrics ,
        "global_test_metrics":global_test_metrics ,
        "per_client_test_metrics":per_client_test_metrics ,
        "client_names":CLIENT_NAMES ,
        "class_names":class_names 
        }
        ckpt_path =os .path .join (OUTPUT_DIR ,f"global_round_{r+1}.pth")
        torch .save (ckpt ,ckpt_path )
        print ("Saved checkpoint:",ckpt_path )

        round_results .append (round_summary )
        df =pd .DataFrame (round_results )
        csv_path =os .path .join (OUTPUT_DIR ,"fl_round_results.csv")
        df .to_csv (csv_path ,index =False )
        print ("Saved per-round summary CSV to",csv_path )

        rounds =list (range (1 ,len (round_results )+1 ))
        gtest_acc =[rr .get ("global_test_acc",0.0 )for rr in round_results ]
        gtest_loss =[rr .get ("global_test_loss",0.0 )for rr in round_results ]
        plt .figure (figsize =(6 ,4 ))
        plt .plot (rounds ,gtest_acc )
        plt .xlabel ("Global Round");plt .ylabel ("Test Accuracy");plt .title ("Global Test Accuracy")
        plt .savefig (global_test_acc_fname );plt .close ()
        plt .figure (figsize =(6 ,4 ))
        plt .plot (rounds ,gtest_loss )
        plt .xlabel ("Global Round");plt .ylabel ("Test Loss");plt .title ("Global Test Loss")
        plt .savefig (global_test_loss_fname );plt .close ()

        per_client_acc_fname =os .path .join (OUTPUT_DIR ,"per_client_test_accuracy_rounds.png")
        plt .figure (figsize =(8 ,5 ))
        for i ,name in enumerate (CLIENT_NAMES ):
            plt .plot (range (1 ,len (per_client_acc_history [i ])+1 ),per_client_acc_history [i ],label =name )
        plt .xlabel ("Global Round");plt .ylabel ("Test Accuracy");plt .title ("Per-client Test Accuracy");plt .legend ()
        plt .savefig (per_client_acc_fname );plt .close ()

        per_client_loss_fname =os .path .join (OUTPUT_DIR ,"per_client_test_loss_rounds.png")
        plt .figure (figsize =(8 ,5 ))
        for i ,name in enumerate (CLIENT_NAMES ):
            plt .plot (range (1 ,len (per_client_loss_history [i ])+1 ),per_client_loss_history [i ],label =name )
        plt .xlabel ("Global Round");plt .ylabel ("Test Loss");plt .title ("Per-client Test Loss");plt .legend ()
        plt .savefig (per_client_loss_fname );plt .close ()

    final_model_path =os .path .join (OUTPUT_DIR ,"global_final.pth")
    torch .save ({"model_state":global_model .state_dict (),"class_names":class_names },final_model_path )
    print ("Federated training finished. Final global model saved to:",final_model_path )

if __name__ =="__main__":
    main ()
