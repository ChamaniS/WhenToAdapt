import os 
import time 
import copy 
import random 
import json 
import csv 
import numpy as np 
from collections import defaultdict 

import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,ConcatDataset 
from torchvision import datasets ,transforms 
from torchvision .models import efficientnet_b0 
from tqdm import tqdm 
from sklearn .metrics import (
confusion_matrix ,
classification_report ,
accuracy_score ,
precision_score ,
recall_score ,
f1_score ,
cohen_kappa_score 
)
import matplotlib .pyplot as plt 




SEED =42 
DATA_ROOT =r"xxxx\Data\Brain\FINAL"
OUTPUT_DIR ="brain_tumor_federated"
MODEL_NAME ="efficientnet_b0_brain_tumor_fedavg.pth"

BATCH_SIZE =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
LR =1e-4 
NUM_WORKERS =0 
IMG_SIZE =224 

CLIENT_NAMES =["Sartajbhuvaji","rm1000","thomasdubail","figshare"]

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
PIN_MEMORY =DEVICE .type =="cuda"
scaler =torch .cuda .amp .GradScaler (enabled =(DEVICE .type =="cuda"))




def set_seed (seed =42 ):
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    torch .backends .cudnn .deterministic =False 
    torch .backends .cudnn .benchmark =True 

set_seed (SEED )
os .makedirs (OUTPUT_DIR ,exist_ok =True )




mean =[0.485 ,0.456 ,0.406 ]
std =[0.229 ,0.224 ,0.225 ]

train_tfms =transforms .Compose ([
transforms .Resize ((IMG_SIZE ,IMG_SIZE )),
transforms .RandomHorizontalFlip (p =0.5 ),
transforms .RandomRotation (10 ),
transforms .RandomAffine (degrees =0 ,translate =(0.05 ,0.05 ),scale =(0.95 ,1.05 )),
transforms .ColorJitter (brightness =0.1 ,contrast =0.1 ,saturation =0.1 ),
transforms .ToTensor (),
transforms .Normalize (mean =mean ,std =std ),
])

eval_tfms =transforms .Compose ([
transforms .Resize ((IMG_SIZE ,IMG_SIZE )),
transforms .ToTensor (),
transforms .Normalize (mean =mean ,std =std ),
])




def set_dataset_paths (root ,client_names ):
    """
    Expected structure:
    DATA_ROOT/
        Sartajbhuvaji/train, val, test
        rm1000/train, val, test
        thomasdubail/train, val, test
        figshare/train, val, test
    Each split folder must contain class subfolders.
    """
    paths ={}
    for client in client_names :
        paths [client ]={
        "train":os .path .join (root ,client ,"train"),
        "val":os .path .join (root ,client ,"val"),
        "test":os .path .join (root ,client ,"test"),
        }
    return paths 

def check_class_alignment (datasets_list ):
    base_classes =datasets_list [0 ].classes 
    base_class_to_idx =datasets_list [0 ].class_to_idx 

    for i ,ds in enumerate (datasets_list [1 :],start =2 ):
        if ds .classes !=base_classes :
            raise ValueError (
            f"Class mismatch detected in dataset {i}.\n"
            f"Expected classes: {base_classes}\n"
            f"Found classes   : {ds.classes}\n"
            "All clients must have the same class folder names."
            )
        if ds .class_to_idx !=base_class_to_idx :
            raise ValueError (
            f"Class-to-index mismatch detected in dataset {i}.\n"
            "All clients must use identical class folder naming."
            )

    return base_classes ,base_class_to_idx 

def build_client_datasets (paths_dict ,split ,transform ):
    ds_list =[]
    for client in CLIENT_NAMES :
        split_dir =paths_dict [client ][split ]
        if not os .path .isdir (split_dir ):
            raise FileNotFoundError (f"Missing directory: {split_dir}")
        ds =datasets .ImageFolder (split_dir ,transform =transform )
        ds_list .append (ds )

    classes ,class_to_idx =check_class_alignment (ds_list )
    return ds_list ,classes ,class_to_idx 

def build_combined_dataset (ds_list ):
    if len (ds_list )==1 :
        return ds_list [0 ]
    return ConcatDataset (ds_list )

def count_samples (ds ):
    if isinstance (ds ,ConcatDataset ):
        return sum (len (d )for d in ds .datasets )
    return len (ds )

def build_model (num_classes ):
    model =efficientnet_b0 (pretrained =True )
    in_features =model .classifier [1 ].in_features 
    model .classifier [1 ]=nn .Linear (in_features ,num_classes )
    return model 

def compute_specificity_from_cm (cm ):
    """
    Multiclass specificity for each class using one-vs-rest:
    specificity_i = TN_i / (TN_i + FP_i)
    """
    num_classes =cm .shape [0 ]
    per_class_specificity =[]

    total =cm .sum ()
    for i in range (num_classes ):
        tp =cm [i ,i ]
        fp =cm [:,i ].sum ()-tp 
        fn =cm [i ,:].sum ()-tp 
        tn =total -tp -fp -fn 

        denom =tn +fp 
        specificity =tn /denom if denom >0 else 0.0 
        per_class_specificity .append (specificity )

    macro_specificity =float (np .mean (per_class_specificity ))
    return per_class_specificity ,macro_specificity 

def average_state_dicts_weighted (models ,weights ):
    avg_sd =copy .deepcopy (models [0 ].state_dict ())
    for k in avg_sd .keys ():
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ]for i in range (len (models )))
    return avg_sd 

def run_epoch (model ,loader ,criterion ,optimizer =None ,train =True ):
    if train :
        model .train ()
    else :
        model .eval ()

    running_loss =0.0 
    all_preds =[]
    all_targets =[]

    loop =tqdm (loader ,desc ="Train"if train else "Eval",leave =False )

    for images ,labels in loop :
        images =images .to (DEVICE ,non_blocking =True )
        labels =labels .to (DEVICE ,non_blocking =True )

        if train :
            optimizer .zero_grad (set_to_none =True )
            with torch .cuda .amp .autocast (enabled =(DEVICE .type =="cuda")):
                outputs =model (images )
                loss =criterion (outputs ,labels )

            scaler .scale (loss ).backward ()
            scaler .step (optimizer )
            scaler .update ()
        else :
            with torch .no_grad ():
                outputs =model (images )
                loss =criterion (outputs ,labels )

        running_loss +=loss .item ()*images .size (0 )
        preds =torch .argmax (outputs ,dim =1 )

        all_preds .extend (preds .detach ().cpu ().numpy ())
        all_targets .extend (labels .detach ().cpu ().numpy ())

        loop .set_postfix (loss =float (loss .item ()))

    epoch_loss =running_loss /len (loader .dataset )
    epoch_acc =accuracy_score (all_targets ,all_preds )
    return epoch_loss ,epoch_acc ,all_targets ,all_preds 

def evaluate_loader (model ,loader ,criterion ,class_names ,title_prefix ="test",save_dir =OUTPUT_DIR ,save_cm =True ):
    loss ,acc ,targets ,preds =run_epoch (model ,loader ,criterion ,optimizer =None ,train =False )

    cm =confusion_matrix (targets ,preds )
    precision_macro =precision_score (targets ,preds ,average ="macro",zero_division =0 )
    recall_macro =recall_score (targets ,preds ,average ="macro",zero_division =0 )
    f1_macro =f1_score (targets ,preds ,average ="macro",zero_division =0 )
    kappa =cohen_kappa_score (targets ,preds )
    per_class_specificity ,macro_specificity =compute_specificity_from_cm (cm )

    print (f"\n=== {title_prefix.upper()} ===")
    print (f"Loss        : {loss:.4f}")
    print (f"Accuracy    : {acc:.4f}")
    print (f"Precision   : {precision_macro:.4f}")
    print (f"Recall      : {recall_macro:.4f}")
    print (f"F1-score    : {f1_macro:.4f}")
    print (f"Kappa       : {kappa:.4f}")
    print (f"Specificity : {macro_specificity:.4f}")

    print ("\nPer-class Specificity:")
    for idx ,cls_name in enumerate (class_names ):
        print (f"{cls_name:15s}: {per_class_specificity[idx]:.4f}")

    print ("\nClassification Report:")
    print (classification_report (targets ,preds ,target_names =class_names ,digits =4 ,zero_division =0 ))

    print ("Confusion Matrix:")
    print (cm )

    if save_cm :
        plt .figure (figsize =(7 ,6 ))
        plt .imshow (cm ,interpolation ="nearest")
        plt .title (f"Confusion Matrix - {title_prefix}")
        plt .colorbar ()
        tick_marks =np .arange (len (class_names ))
        plt .xticks (tick_marks ,class_names ,rotation =45 ,ha ="right")
        plt .yticks (tick_marks ,class_names )

        for i in range (cm .shape [0 ]):
            for j in range (cm .shape [1 ]):
                plt .text (j ,i ,str (cm [i ,j ]),ha ="center",va ="center")

        plt .ylabel ("True Label")
        plt .xlabel ("Predicted Label")
        plt .tight_layout ()
        plt .savefig (os .path .join (save_dir ,f"confusion_matrix_{title_prefix}.png"),dpi =300 )
        plt .close ()

    return {
    "split":title_prefix ,
    "loss":loss ,
    "accuracy":acc ,
    "precision_macro":precision_macro ,
    "recall_macro":recall_macro ,
    "f1_macro":f1_macro ,
    "kappa":kappa ,
    "specificity_macro":macro_specificity ,
    "per_class_specificity":per_class_specificity ,
    "cm":cm .tolist (),
    }

def save_metrics_csv (results ,path ):
    with open (path ,"w",newline ="")as f :
        writer =csv .writer (f )
        writer .writerow ([
        "split",
        "loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "kappa",
        "specificity_macro"
        ])
        for r in results :
            writer .writerow ([
            r ["split"],
            f'{r["loss"]:.6f}',
            f'{r["accuracy"]:.6f}',
            f'{r["precision_macro"]:.6f}',
            f'{r["recall_macro"]:.6f}',
            f'{r["f1_macro"]:.6f}',
            f'{r["kappa"]:.6f}',
            f'{r["specificity_macro"]:.6f}'
            ])

def plot_round_curves (history ,out_dir ):
    rounds =np .arange (1 ,len (history ["round"])+1 )

    plt .figure (figsize =(10 ,5 ))
    plt .plot (rounds ,history ["train_loss"],label ="Train Loss")
    plt .plot (rounds ,history ["val_loss"],label ="Val Loss")
    plt .xlabel ("Communication Round")
    plt .ylabel ("Loss")
    plt .title ("Federated Training Loss")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"fed_train_val_loss.png"),dpi =300 )
    plt .close ()

    plt .figure (figsize =(10 ,5 ))
    plt .plot (rounds ,history ["train_acc"],label ="Train Acc")
    plt .plot (rounds ,history ["val_acc"],label ="Val Acc")
    plt .xlabel ("Communication Round")
    plt .ylabel ("Accuracy")
    plt .title ("Federated Training Accuracy")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"fed_train_val_acc.png"),dpi =300 )
    plt .close ()

    plt .figure (figsize =(10 ,5 ))
    plt .plot (rounds ,history ["global_test_acc"],label ="Global Test Acc")
    plt .xlabel ("Communication Round")
    plt .ylabel ("Accuracy")
    plt .title ("Global Test Accuracy Across Rounds")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"fed_global_test_acc.png"),dpi =300 )
    plt .close ()




def main ():
    paths =set_dataset_paths (DATA_ROOT ,CLIENT_NAMES )


    train_datasets ,class_names ,class_to_idx =build_client_datasets (paths ,"train",train_tfms )
    val_datasets ,_ ,_ =build_client_datasets (paths ,"val",eval_tfms )
    test_datasets ,_ ,_ =build_client_datasets (paths ,"test",eval_tfms )


    train_ds_all =build_combined_dataset (train_datasets )
    val_ds_all =build_combined_dataset (val_datasets )
    test_ds_all =build_combined_dataset (test_datasets )

    num_classes =len (class_names )

    print ("Classes:",class_names )
    print ("Train samples (all clients):",count_samples (train_ds_all ))
    print ("Val samples   (all clients):",count_samples (val_ds_all ))
    print ("Test samples  (all clients):",count_samples (test_ds_all ))

    for client ,ds_tr ,ds_va ,ds_te in zip (CLIENT_NAMES ,train_datasets ,val_datasets ,test_datasets ):
        print (f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")


    train_loaders ={}
    val_loaders ={}
    test_loaders ={}

    for client ,ds_tr ,ds_va ,ds_te in zip (CLIENT_NAMES ,train_datasets ,val_datasets ,test_datasets ):
        train_loaders [client ]=DataLoader (
        ds_tr ,
        batch_size =BATCH_SIZE ,
        shuffle =True ,
        num_workers =NUM_WORKERS ,
        pin_memory =PIN_MEMORY 
        )
        val_loaders [client ]=DataLoader (
        ds_va ,
        batch_size =BATCH_SIZE ,
        shuffle =False ,
        num_workers =NUM_WORKERS ,
        pin_memory =PIN_MEMORY 
        )
        test_loaders [client ]=DataLoader (
        ds_te ,
        batch_size =BATCH_SIZE ,
        shuffle =False ,
        num_workers =NUM_WORKERS ,
        pin_memory =PIN_MEMORY 
        )

    global_val_loader =DataLoader (
    val_ds_all ,
    batch_size =BATCH_SIZE ,
    shuffle =False ,
    num_workers =NUM_WORKERS ,
    pin_memory =PIN_MEMORY 
    )

    global_test_loader =DataLoader (
    test_ds_all ,
    batch_size =BATCH_SIZE ,
    shuffle =False ,
    num_workers =NUM_WORKERS ,
    pin_memory =PIN_MEMORY 
    )


    global_model =build_model (num_classes ).to (DEVICE )
    criterion =nn .CrossEntropyLoss ()

    best_val_loss =float ("inf")
    best_model_wts =copy .deepcopy (global_model .state_dict ())

    history =defaultdict (list )
    round_metrics =[]
    start_time =time .time ()

    for r in range (COMM_ROUNDS ):
        print (f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        local_models =[]
        local_weights =[]

        round_train_losses =[]
        round_train_accs =[]


        for client_name in CLIENT_NAMES :
            print (f"\n[Client {client_name}]")

            local_model =copy .deepcopy (global_model ).to (DEVICE )
            optimizer =optim .AdamW (local_model .parameters (),lr =LR ,weight_decay =1e-4 )

            train_loader =train_loaders [client_name ]
            val_loader =val_loaders [client_name ]

            client_epoch_losses =[]
            client_epoch_accs =[]

            for ep in range (LOCAL_EPOCHS ):
                tr_loss ,tr_acc ,_ ,_ =run_epoch (
                local_model ,
                train_loader ,
                criterion ,
                optimizer =optimizer ,
                train =True 
                )
                client_epoch_losses .append (tr_loss )
                client_epoch_accs .append (tr_acc )
                print (f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss ,val_acc ,_ ,_ =run_epoch (
            local_model ,
            val_loader ,
            criterion ,
            optimizer =None ,
            train =False 
            )
            print (f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models .append (local_model )
            local_weights .append (len (train_loader .dataset ))

            round_train_losses .append (float (np .mean (client_epoch_losses )))
            round_train_accs .append (float (np .mean (client_epoch_accs )))

        total_train_size =sum (local_weights )
        if total_train_size ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your dataset splits.")


        norm_weights =[w /total_train_size for w in local_weights ]
        global_model .load_state_dict (average_state_dicts_weighted (local_models ,norm_weights ))


        global_val_loss ,global_val_acc ,_ ,_ =run_epoch (
        global_model ,
        global_val_loader ,
        criterion ,
        optimizer =None ,
        train =False 
        )


        if global_val_loss <best_val_loss :
            best_val_loss =global_val_loss 
            best_model_wts =copy .deepcopy (global_model .state_dict ())
            torch .save (best_model_wts ,os .path .join (OUTPUT_DIR ,MODEL_NAME ))
            print ("\nSaved best global model.")

        rm ={
        "round":r +1 ,
        "train_loss":float (np .mean (round_train_losses )),
        "train_acc":float (np .mean (round_train_accs )),
        "val_loss":global_val_loss ,
        "val_acc":global_val_acc ,
        }




        print ("\n"+"="*30 )
        print (f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print ("="*30 )

        global_test_result =evaluate_loader (
        global_model ,
        global_test_loader ,
        criterion ,
        class_names ,
        title_prefix =f"global_round_{r + 1}",
        save_dir =OUTPUT_DIR ,
        save_cm =True 
        )

        rm ["global_test_loss"]=global_test_result ["loss"]
        rm ["global_test_acc"]=global_test_result ["accuracy"]
        rm ["global_test_precision"]=global_test_result ["precision_macro"]
        rm ["global_test_recall"]=global_test_result ["recall_macro"]
        rm ["global_test_f1"]=global_test_result ["f1_macro"]
        rm ["global_test_kappa"]=global_test_result ["kappa"]
        rm ["global_test_specificity"]=global_test_result ["specificity_macro"]

        print ("\n"+"="*30 )
        print (f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print ("="*30 )

        for client_name in CLIENT_NAMES :
            client_result =evaluate_loader (
            global_model ,
            test_loaders [client_name ],
            criterion ,
            class_names ,
            title_prefix =f"{client_name}_round_{r + 1}",
            save_dir =OUTPUT_DIR ,
            save_cm =True 
            )

            rm [f"{client_name}_test_loss"]=client_result ["loss"]
            rm [f"{client_name}_test_acc"]=client_result ["accuracy"]
            rm [f"{client_name}_test_precision"]=client_result ["precision_macro"]
            rm [f"{client_name}_test_recall"]=client_result ["recall_macro"]
            rm [f"{client_name}_test_f1"]=client_result ["f1_macro"]
            rm [f"{client_name}_test_kappa"]=client_result ["kappa"]
            rm [f"{client_name}_test_specificity"]=client_result ["specificity_macro"]

        round_metrics .append (rm )

        history ["round"].append (r +1 )
        history ["train_loss"].append (rm ["train_loss"])
        history ["train_acc"].append (rm ["train_acc"])
        history ["val_loss"].append (rm ["val_loss"])
        history ["val_acc"].append (rm ["val_acc"])
        history ["global_test_acc"].append (rm ["global_test_acc"])

        print (
        f"\n[ROUND {r + 1}] "
        f"Train Loss: {rm['train_loss']:.4f} | Train Acc: {rm['train_acc']:.4f} | "
        f"Val Loss: {rm['val_loss']:.4f} | Val Acc: {rm['val_acc']:.4f} | "
        f"Global Test Acc: {rm['global_test_acc']:.4f}"
        )

        plot_round_curves (history ,OUTPUT_DIR )

    elapsed =time .time ()-start_time 
    print (f"\nFederated training finished in {elapsed / 60:.2f} minutes")


    global_model .load_state_dict (best_model_wts )




    final_results =[]

    print ("\n==============================")
    print ("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print ("==============================")
    result_all =evaluate_loader (
    global_model ,
    global_test_loader ,
    criterion ,
    class_names ,
    title_prefix ="all_clients_final",
    save_dir =OUTPUT_DIR ,
    save_cm =True 
    )
    final_results .append (result_all )

    print ("\n==============================")
    print ("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print ("==============================")
    for client_name in CLIENT_NAMES :
        result_client =evaluate_loader (
        global_model ,
        test_loaders [client_name ],
        criterion ,
        class_names ,
        title_prefix =f"{client_name}_final",
        save_dir =OUTPUT_DIR ,
        save_cm =True 
        )
        final_results .append (result_client )


    with open (os .path .join (OUTPUT_DIR ,"federated_round_metrics.json"),"w")as f :
        json .dump (round_metrics ,f ,indent =2 )

    with open (os .path .join (OUTPUT_DIR ,"federated_round_metrics.csv"),"w",newline ="")as f :
        writer =csv .DictWriter (f ,fieldnames =list (round_metrics [0 ].keys ()))
        writer .writeheader ()
        writer .writerows (round_metrics )


    save_metrics_csv (final_results ,os .path .join (OUTPUT_DIR ,"final_test_metrics.csv"))
    with open (os .path .join (OUTPUT_DIR ,"final_test_metrics.json"),"w")as f :
        json .dump (final_results ,f ,indent =2 )

    print ("\nSaved outputs to:")
    print (os .path .join (OUTPUT_DIR ,MODEL_NAME ))
    print (os .path .join (OUTPUT_DIR ,"federated_round_metrics.csv"))
    print (os .path .join (OUTPUT_DIR ,"federated_round_metrics.json"))
    print (os .path .join (OUTPUT_DIR ,"final_test_metrics.csv"))
    print (os .path .join (OUTPUT_DIR ,"final_test_metrics.json"))

if __name__ =="__main__":
    main ()