
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
import numpy as np 
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
import matplotlib .pyplot as plt 
from typing import List ,Tuple 

CLIENT_ROOTS =[
r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen",
r"xxxxx\Projects\Data\Tuberculosis_Data\Montgomery",
r"xxxxx\Projects\Data\Tuberculosis_Data\TBX11K",
r"xxxxx\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES =["Shenzhen","Montgomery","TBX11K","Pakistan"]
OUTPUT_DIR =r"./fl_outputs_ditto"
ARCH ="densenet169"
PRETRAINED =True 
IMG_SIZE =224 
BATCH_SIZE =4 
WORKERS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
LR =1e-4 
LR_PERSONAL =1e-4 
WEIGHT_DECAY =1e-5 
USE_AMP =False 
PIN_MEMORY =True 
DROPOUT_P =0.5 
SEED =42 
CLASS_NAMES =["normal","positive"]

PERSONALIZATION_MU =1.0 

def set_seed (seed =SEED ):
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed_all (seed )
set_seed ()

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
os .makedirs (OUTPUT_DIR ,exist_ok =True )

class PathListDataset (Dataset ):
    def __init__ (self ,samples :List [Tuple [str ,int ]],transform =None ,loader =default_loader ):
        self .samples =list (samples )
        self .transform =transform 
        self .loader =loader 
    def __len__ (self ):
        return len (self .samples )
    def __getitem__ (self ,idx ):
        path ,label =self .samples [idx ]
        img =self .loader (path )
        if self .transform :
            img =self .transform (img )
        return img ,label 

def gather_samples_from_client_split (client_root :str ,split :str ,class_names :List [str ]):
    split_dir =os .path .join (client_root ,split )
    if not os .path .isdir (split_dir ):
        raise ValueError (f"Missing {split} in {client_root}")
    samples =[]
    canon_map ={c .lower ():i for i ,c in enumerate (class_names )}
    for cls_folder in os .listdir (split_dir ):
        cls_path =os .path .join (split_dir ,cls_folder )
        if not os .path .isdir (cls_path ):continue 
        key =cls_folder .lower ()
        if key not in canon_map :
            print (f"Warning: unknown class folder '{cls_folder}' in {split_dir}; skipping")
            continue 
        label =canon_map [key ]
        for fn in os .listdir (cls_path ):
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
        "train":DataLoader (train_ds ,batch_size =batch_size ,shuffle =True ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda")),
        "val":DataLoader (val_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda")),
        "test":DataLoader (test_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda")),
        "train_ds":train_ds 
        })
        per_client_test_dsets [client_root ]=test_ds 

    combined_train_ds =PathListDataset (train_samples_all ,transform =train_tf )
    combined_val_ds =PathListDataset (val_samples_all ,transform =val_tf )
    combined_test_ds =PathListDataset (test_samples_all ,transform =val_tf )

    dataloaders_combined ={
    "train":DataLoader (combined_train_ds ,batch_size =batch_size ,shuffle =True ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda")),
    "val":DataLoader (combined_val_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda")),
    "test":DataLoader (combined_test_ds ,batch_size =batch_size ,shuffle =False ,num_workers =workers ,pin_memory =pin_memory and (DEVICE .type =="cuda"))
    }
    sizes ={"train":len (combined_train_ds ),"val":len (combined_val_ds ),"test":len (combined_test_ds )}
    return dataloaders_combined ,sizes ,CLASS_NAMES ,combined_train_ds ,per_client_dataloaders ,per_client_test_dsets 

def compute_class_weights_from_dataset (dataset ):
    targets =[s [1 ]for s in dataset .samples ]
    counts =np .bincount (targets ,minlength =len (CLASS_NAMES )).astype (np .float32 )
    total =counts .sum ()
    weights =total /(counts +1e-8 )
    weights =weights /np .mean (weights )
    return torch .tensor (weights ,dtype =torch .float32 )

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
    if sum_w ==0.0 :
        raise ValueError ("Sum of weights is zero")
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

def l2_distance_params (model_a ,model_b ):
    total =0.0 
    for pa ,pb in zip (model_a .parameters (),model_b .parameters ()):
        total +=torch .sum ((pa -pb .detach ())**2 )
    return total 

def train_local_ditto_classification (train_loader ,local_global_model ,personal_model ,
criterion ,opt_global ,opt_personal ,mu ,device ,epochs =LOCAL_EPOCHS ,use_amp =False ):
    local_global_model .to (device )
    personal_model .to (device )
    scaler =torch .cuda .amp .GradScaler ()if (use_amp and device .type =="cuda")else None 
    for ep in range (epochs ):
        local_global_model .train ()
        personal_model .train ()
        pbar =tqdm (train_loader ,desc =f"LocalDitto ep{ep+1}/{epochs}",leave =False )
        for x ,y in pbar :
            x ,y =x .to (device ),y .to (device )

            opt_global .zero_grad ()
            with torch .cuda .amp .autocast (enabled =(scaler is not None )):
                out_g =local_global_model (x )
                loss_g =criterion (out_g ,y )
            if scaler :
                scaler .scale (loss_g ).backward ();scaler .step (opt_global )
            else :
                loss_g .backward ();opt_global .step ()

            opt_personal .zero_grad ()
            with torch .cuda .amp .autocast (enabled =(scaler is not None )):
                out_p =personal_model (x )
                loss_p =criterion (out_p ,y )
                prox =0.5 *mu *l2_distance_params (personal_model ,local_global_model )
                total_personal_loss =loss_p +prox 
            if scaler :
                scaler .scale (total_personal_loss ).backward ();scaler .step (opt_personal );scaler .update ()
            else :
                total_personal_loss .backward ();opt_personal .step ()
    return 

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
    if n ==0 :
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
        metrics ["loss"]=float (total_loss /max (1 ,n ))
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

def print_client_summary (metrics :dict ,client_idx :int ,client_name :str ,class_names :List [str ],label =""):
    if metrics is None or metrics =={}:
        print (f"[CLIENT {client_idx}] {label} Summary metrics: (no samples)")
        return 
    acc =metrics .get ("accuracy",float ("nan"))
    prec =metrics .get ("precision_macro",float ("nan"))
    rec =metrics .get ("recall_macro",float ("nan"))
    f1 =metrics .get ("f1_macro",float ("nan"))
    kappa =metrics .get ("cohen_kappa",float ("nan"))
    per_specs =metrics .get ("per_class_specificity",[])
    mean_spec =float (np .mean (per_specs ))if len (per_specs )>0 else float ("nan")
    print (f"[CLIENT {client_idx}] {label} Summary metrics:")
    print (f"  Accuracy       : {acc:.4f}")
    print (f"  Precision (mac): {prec:.4f}")
    print (f"  Recall (mac)   : {rec:.4f}")
    print (f"  F1 (mac)       : {f1:.4f}")
    if not np .isnan (mean_spec ):
        print (f"  Mean Specificity: {mean_spec:.4f}")
    else :
        print (f"  Mean Specificity: n/a")
    print (f"  Cohen's kappa  : {kappa:.4f}")
    print ("")
    if "per_class_precision"in metrics :
        print (f"  Per-class metrics (order = {class_names}):")
        header =["Class","Support","Correct","Acc","Prec","Rec","F1","Spec"]
        print ("    "+"{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format (*header ))
        cm =metrics .get ("confusion_matrix",None )
        tp_counts =np .diag (cm ).astype (int )if cm is not None else [0 ]*len (class_names )
        supports =cm .sum (axis =1 ).astype (int )if cm is not None else [0 ]*len (class_names )
        precisions =metrics .get ("per_class_precision",[])
        recalls =metrics .get ("per_class_recall",[])
        f1s =metrics .get ("per_class_f1",[])
        specs =metrics .get ("per_class_specificity",[])
        accs =metrics .get ("per_class_accuracy",[])
        for ci ,cname in enumerate (class_names ):
            s =supports [ci ]if ci <len (supports )else 0 
            ccount =tp_counts [ci ]if ci <len (tp_counts )else 0 
            acc_val =accs [ci ]if ci <len (accs )else np .nan 
            pval =precisions [ci ]if ci <len (precisions )else np .nan 
            rval =recalls [ci ]if ci <len (recalls )else np .nan 
            fval =f1s [ci ]if ci <len (f1s )else np .nan 
            sval =specs [ci ]if ci <len (specs )else np .nan 
            print ("    {:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format (
            cname ,int (s ),int (ccount ),
            float (acc_val )if not np .isnan (acc_val )else 0.0 ,
            float (pval )if not np .isnan (pval )else 0.0 ,
            float (rval )if not np .isnan (rval )else 0.0 ,
            float (fval )if not np .isnan (fval )else 0.0 ,
            float (sval )if not np .isnan (sval )else 0.0 
            ))
    else :
        print ("  (per-class metrics not available)")
    print ("\n")




def plot_global_vs_personal_accuracy (rounds_history_global ,rounds_history_personal ,out_dir ,client_names ):
    rounds =list (range (1 ,len (rounds_history_global )+1 ))
    plt .figure (figsize =(8 ,5 ))
    for cid in range (len (client_names )):
        global_vals =[rm .get (f"client{cid}_global_acc",np .nan )for rm in rounds_history_global ]
        personal_vals =[rm .get (f"client{cid}_personal_acc",np .nan )for rm in rounds_history_personal ]
        plt .plot (rounds ,personal_vals ,label =f"{client_names[cid]} (personal)",linewidth =2 )
        plt .plot (rounds ,global_vals ,label =f"{client_names[cid]} (global)",linestyle ="--",linewidth =1.5 )
    plt .xlabel ("Global Round");plt .ylabel ("Test Accuracy");plt .title ("Per-client: GLOBAL vs PERSONAL Accuracy")
    plt .legend (ncol =2 ,fontsize ="small")
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"per_client_global_vs_personal_accuracy.png"))
    plt .close ()

def plot_global_vs_personal_loss (rounds_history_global ,rounds_history_personal ,out_dir ,client_names ):
    rounds =list (range (1 ,len (rounds_history_global )+1 ))
    plt .figure (figsize =(8 ,5 ))
    for cid in range (len (client_names )):
        global_vals =[rm .get (f"client{cid}_global_loss",np .nan )for rm in rounds_history_global ]
        personal_vals =[rm .get (f"client{cid}_personal_loss",np .nan )for rm in rounds_history_personal ]
        plt .plot (rounds ,personal_vals ,label =f"{client_names[cid]} (personal)",linewidth =2 )
        plt .plot (rounds ,global_vals ,label =f"{client_names[cid]} (global)",linestyle ="--",linewidth =1.5 )
    plt .xlabel ("Global Round");plt .ylabel ("Test Loss");plt .title ("Per-client: GLOBAL vs PERSONAL Loss")
    plt .legend (ncol =2 ,fontsize ="small")
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"per_client_global_vs_personal_loss.png"))
    plt .close ()




def main ():
    print ("DEVICE:",DEVICE )
    combined_loaders ,combined_sizes ,class_names ,combined_train_ds ,per_client_dataloaders ,per_client_test_dsets =make_multi_client_dataloaders (
    CLIENT_ROOTS ,batch_size =BATCH_SIZE ,image_size =IMG_SIZE ,workers =WORKERS ,pin_memory =PIN_MEMORY and (DEVICE .type =="cuda")
    )
    num_classes =len (class_names )
    print ("class names:",class_names )

    client_train_sizes =[len (per_client_dataloaders [i ]['train'].dataset )for i in range (len (per_client_dataloaders ))]
    total_train =sum (client_train_sizes )if sum (client_train_sizes )>0 else 1 
    print ("client train sizes:",client_train_sizes )


    global_model =create_model (num_classes =num_classes ,arch =ARCH ,pretrained =PRETRAINED ).to (DEVICE )
    personal_models =[copy .deepcopy (global_model ).to (DEVICE )for _ in range (len (per_client_dataloaders ))]

    print (f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")


    rounds_history_global =[]
    rounds_history_personal =[]

    round_results =[]

    for r in range (COMM_ROUNDS ):
        print ("\n"+"="*60 )
        print (f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print ("="*60 )
        local_models =[]
        weights =[]
        round_summary ={"round":r +1 }


        for i ,client in enumerate (per_client_dataloaders ):
            print (f"\n[CLIENT {i}] {CLIENT_NAMES[i]}: local Ditto training")
            local_global =copy .deepcopy (global_model ).to (DEVICE )
            personal =personal_models [i ]
            personal .to (DEVICE )

            train_ds =client ['train'].dataset 
            client_cw =compute_class_weights_from_dataset (train_ds ).to (DEVICE )
            criterion =nn .CrossEntropyLoss (weight =client_cw )

            opt_global =AdamW (local_global .parameters (),lr =LR ,weight_decay =WEIGHT_DECAY )
            opt_personal =AdamW (personal .parameters (),lr =LR_PERSONAL ,weight_decay =WEIGHT_DECAY )

            train_local_ditto_classification (client ['train'],local_global ,personal ,
            criterion ,opt_global ,opt_personal ,
            PERSONALIZATION_MU ,DEVICE ,epochs =LOCAL_EPOCHS ,use_amp =USE_AMP )

            print (f"[CLIENT {i}] local validation (local_global)")
            local_val_metrics =evaluate_model (local_global ,client ['val'],DEVICE ,criterion =criterion ,return_per_class =True ,class_names =class_names )
            print (f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary [f"client{i}_localval_loss"]=float (local_val_metrics .get ("loss",np .nan ))
            round_summary [f"client{i}_localval_acc"]=float (local_val_metrics .get ("accuracy",np .nan ))

            local_models .append (local_global .cpu ())
            w =float (client_train_sizes [i ])/float (total_train )
            weights .append (w )
            print (f"[CLIENT {i}] aggregation weight: {w:.4f}")


            personal_models [i ]=personal .cpu ()if DEVICE .type !="cuda"else personal 


        print ("\nAggregating local global models (FedAvg weighted)")
        avg_state =average_models_weighted (local_models ,weights )
        avg_state_on_device ={k :v .to (DEVICE )for k ,v in avg_state .items ()}
        global_model .load_state_dict (avg_state_on_device )
        global_model .to (DEVICE )


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


        combined_test_dsets =[per_client_dataloaders [i ]['test'].dataset for i in range (len (per_client_dataloaders ))]
        combined_test =ConcatDataset (combined_test_dsets )
        combined_test_loader =DataLoader (combined_test ,batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =PIN_MEMORY and (DEVICE .type =="cuda"))
        global_test_metrics =evaluate_model (global_model ,combined_test_loader ,DEVICE ,criterion =combined_criterion ,return_per_class =True ,class_names =class_names )
        print ("Global combined TEST metrics summary:",{k :global_test_metrics .get (k )for k in ["accuracy","loss","f1_macro","precision_macro","recall_macro","balanced_acc","cohen_kappa"]})
        round_summary ["global_test_loss"]=float (global_test_metrics .get ("loss",np .nan ))
        round_summary ["global_test_acc"]=float (global_test_metrics .get ("accuracy",np .nan ))

        per_client_test_metrics =[]
        round_global_metrics ={}
        round_personal_metrics ={}

        for i ,client in enumerate (per_client_dataloaders ):
            print (f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set - GLOBAL model")
            client_train_ds =client ['train'].dataset 
            client_cw =compute_class_weights_from_dataset (client_train_ds ).to (DEVICE )
            client_criterion =nn .CrossEntropyLoss (weight =client_cw )


            cl_metrics_global =evaluate_model (global_model ,client ['test'],DEVICE ,criterion =client_criterion ,return_per_class =True ,class_names =class_names )
            print_client_summary (cl_metrics_global ,i ,CLIENT_NAMES [i ],class_names ,label ="GLOBAL")


            print (f"[CLIENT {i}] Test PERSONAL model")
            personal =personal_models [i ]
            personal .to (DEVICE )
            cl_metrics_personal =evaluate_model (personal ,client ['test'],DEVICE ,criterion =client_criterion ,return_per_class =True ,class_names =class_names )
            print_client_summary (cl_metrics_personal ,i ,CLIENT_NAMES [i ],class_names ,label ="PERSONAL")

            per_client_test_metrics .append ({
            "client_idx":i ,
            "global_metrics":cl_metrics_global ,
            "personal_metrics":cl_metrics_personal 
            })

            round_global_metrics [f"client{i}_global_acc"]=float (cl_metrics_global .get ("accuracy",np .nan ))
            round_global_metrics [f"client{i}_global_loss"]=float (cl_metrics_global .get ("loss",np .nan ))
            round_personal_metrics [f"client{i}_personal_acc"]=float (cl_metrics_personal .get ("accuracy",np .nan ))
            round_personal_metrics [f"client{i}_personal_loss"]=float (cl_metrics_personal .get ("loss",np .nan ))


            round_summary [f"client{i}_test_loss_global"]=float (cl_metrics_global .get ("loss",np .nan ))
            round_summary [f"client{i}_test_acc_global"]=float (cl_metrics_global .get ("accuracy",np .nan ))
            round_summary [f"client{i}_test_loss_personal"]=float (cl_metrics_personal .get ("loss",np .nan ))
            round_summary [f"client{i}_test_acc_personal"]=float (cl_metrics_personal .get ("accuracy",np .nan ))

        rounds_history_global .append (round_global_metrics )
        rounds_history_personal .append (round_personal_metrics )


        plot_global_vs_personal_accuracy (rounds_history_global ,rounds_history_personal ,OUTPUT_DIR ,CLIENT_NAMES )
        plot_global_vs_personal_loss (rounds_history_global ,rounds_history_personal ,OUTPUT_DIR ,CLIENT_NAMES )


        ckpt ={
        "round":r +1 ,
        "model_state":global_model .state_dict (),
        "global_val_metrics":global_val_metrics ,
        "global_test_metrics":global_test_metrics ,
        "per_client_test_metrics":per_client_test_metrics ,
        "client_names":CLIENT_NAMES ,
        "class_names":class_names ,
        "personalization_mu":PERSONALIZATION_MU 
        }
        ckpt_path =os .path .join (OUTPUT_DIR ,f"global_round_{r+1}_ditto.pth")
        torch .save (ckpt ,ckpt_path )
        print ("Saved checkpoint:",ckpt_path )

        round_results .append (round_summary )
        df =pd .DataFrame (round_results )
        csv_path =os .path .join (OUTPUT_DIR ,"fl_round_results_ditto.csv")
        df .to_csv (csv_path ,index =False )
        print ("Saved per-round summary CSV to",csv_path )

    final_model_path =os .path .join (OUTPUT_DIR ,"global_final_ditto.pth")
    torch .save ({"model_state":global_model .state_dict (),"class_names":class_names },final_model_path )
    print ("Final global model saved to:",final_model_path )
    for i ,pm in enumerate (personal_models ):
        pm_cpu =pm .cpu ()if DEVICE .type =="cuda"else pm 
        pm_path =os .path .join (OUTPUT_DIR ,f"personal_model_client{i}_final_ditto.pth")
        torch .save ({"model_state":(pm_cpu .state_dict ()),"class_names":class_names },pm_path )
        print (f"Saved personal model for client {i} to {pm_path}")
    print ("Federated Ditto training finished.")

if __name__ =="__main__":
    main ()
