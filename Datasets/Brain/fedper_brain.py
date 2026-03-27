import os 
import time 
import copy 
import random 
import json 
import csv 
import numpy as np 
from collections import defaultdict 
import pandas as pd 
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
DATA_ROOT =r"C:\Users\csj5\Projects\Data\Brain\FINAL"
OUTPUT_DIR ="brain_tumor_fedper"
MODEL_NAME ="efficientnet_b0_brain_tumor_fedper.pth"

BATCH_SIZE =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
LR =1e-4 
WEIGHT_DECAY =1e-4 
NUM_WORKERS =0 
IMG_SIZE =224 

CLIENT_NAMES =["Sartajbhuvaji","rm1000","thomasdubail","figshare"]

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
PIN_MEMORY =DEVICE .type =="cuda"
USE_AMP =DEVICE .type =="cuda"

os .makedirs (OUTPUT_DIR ,exist_ok =True )




def set_seed (seed =42 ):
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    torch .backends .cudnn .deterministic =False 
    torch .backends .cudnn .benchmark =True 

set_seed (SEED )




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

def compute_specificity_from_cm (cm ):
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
    plt .plot (rounds ,history ["mean_train_loss"],label ="Mean Train Loss")
    plt .plot (rounds ,history ["mean_val_loss"],label ="Mean Val Loss")
    plt .xlabel ("Communication Round")
    plt .ylabel ("Loss")
    plt .title ("FedPer Training Loss")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"fedper_train_val_loss.png"),dpi =300 )
    plt .close ()

    plt .figure (figsize =(10 ,5 ))
    plt .plot (rounds ,history ["mean_train_acc"],label ="Mean Train Acc")
    plt .plot (rounds ,history ["mean_val_acc"],label ="Mean Val Acc")
    plt .xlabel ("Communication Round")
    plt .ylabel ("Accuracy")
    plt .title ("FedPer Training Accuracy")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"fedper_train_val_acc.png"),dpi =300 )
    plt .close ()

def is_head_key (key :str )->bool :
    k =key .lower ()
    return (
    k .startswith ("classifier.")
    or k .startswith ("fc.")
    or k .startswith ("head.")
    )

def average_shared_state_dicts (local_models ,weights ):
    """
    FedPer aggregation:
    average only shared parameters, keep head/classifier local.
    """
    avg_sd =copy .deepcopy (local_models [0 ].state_dict ())
    keys =list (avg_sd .keys ())

    for k in keys :
        if is_head_key (k ):
            continue 

        avg_tensor =None 
        for i ,model in enumerate (local_models ):
            tensor =model .state_dict ()[k ].detach ().cpu ()
            w =float (weights [i ])
            if avg_tensor is None :
                avg_tensor =w *tensor 
            else :
                avg_tensor +=w *tensor 
        avg_sd [k ]=avg_tensor 

    return avg_sd 

def load_shared_params (model ,shared_state ):
    """
    Copy only shared parameters from shared_state into model.
    Head parameters remain untouched.
    """
    model_sd =model .state_dict ()
    for k ,v in shared_state .items ():
        if k in model_sd and not is_head_key (k ):
            model_sd [k ]=v .clone ().to (model_sd [k ].device )
    model .load_state_dict (model_sd )

def compute_class_weights_from_dataset (dataset ,num_classes ):
    targets =[s [1 ]for s in dataset .samples ]
    counts =np .bincount (targets ,minlength =num_classes ).astype (np .float32 )
    total =counts .sum ()
    weights =total /(counts +1e-8 )
    weights =weights /np .mean (weights )
    return torch .tensor (weights ,dtype =torch .float32 )

def build_model (num_classes ):
    try :
        from torchvision .models import EfficientNet_B0_Weights 
        model =efficientnet_b0 (weights =EfficientNet_B0_Weights .DEFAULT )
    except Exception :
        model =efficientnet_b0 (pretrained =True )

    in_features =model .classifier [1 ].in_features 
    model .classifier [1 ]=nn .Linear (in_features ,num_classes )
    return model 

def run_epoch (model ,loader ,criterion ,optimizer =None ,train =True ):
    model .train ()if train else model .eval ()

    running_loss =0.0 
    all_preds =[]
    all_targets =[]

    loop =tqdm (loader ,desc ="Train"if train else "Eval",leave =False )

    for images ,labels in loop :
        images =images .to (DEVICE ,non_blocking =True )
        labels =labels .to (DEVICE ,non_blocking =True )

        if train :
            optimizer .zero_grad (set_to_none =True )
            with torch .cuda .amp .autocast (enabled =USE_AMP ):
                outputs =model (images )
                loss =criterion (outputs ,labels )
            if USE_AMP :
                scaler .scale (loss ).backward ()
                scaler .step (optimizer )
                scaler .update ()
            else :
                loss .backward ()
                optimizer .step ()
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

    cm =confusion_matrix (targets ,preds ,labels =list (range (len (class_names ))))
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




def main ():
    global scaler 
    scaler =torch .cuda .amp .GradScaler (enabled =USE_AMP )

    paths =set_dataset_paths (DATA_ROOT ,CLIENT_NAMES )


    train_datasets ,class_names ,class_to_idx =build_client_datasets (paths ,"train",train_tfms )
    val_datasets ,_ ,_ =build_client_datasets (paths ,"val",eval_tfms )
    test_datasets ,_ ,_ =build_client_datasets (paths ,"test",eval_tfms )

    num_classes =len (class_names )


    train_ds_all =build_combined_dataset (train_datasets )
    val_ds_all =build_combined_dataset (val_datasets )
    test_ds_all =build_combined_dataset (test_datasets )

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


    global_model =build_model (num_classes ).to (DEVICE )
    client_models ={
    client :copy .deepcopy (global_model ).to (DEVICE )
    for client in CLIENT_NAMES 
    }

    print (f"\nGlobal model created with {sum(p.numel() for p in global_model.parameters()):,} parameters")


    client_class_weights ={
    client :compute_class_weights_from_dataset (train_datasets [i ],num_classes ).to (DEVICE )
    for i ,client in enumerate (CLIENT_NAMES )
    }


    shared_state =global_model .state_dict ()


    round_results =[]
    history =defaultdict (list )
    per_client_test_acc_history ={client :[]for client in CLIENT_NAMES }
    per_client_test_loss_history ={client :[]for client in CLIENT_NAMES }
    per_client_val_acc_history ={client :[]for client in CLIENT_NAMES }
    per_client_val_loss_history ={client :[]for client in CLIENT_NAMES }

    best_mean_val_loss =float ("inf")
    best_shared_state =copy .deepcopy (shared_state )

    start_time =time .time ()

    for r in range (COMM_ROUNDS ):
        print ("\n"+"="*70 )
        print (f"COMMUNICATION ROUND {r + 1}/{COMM_ROUNDS} - FEDPER")
        print ("="*70 )

        local_models =[]
        local_weights =[]
        round_summary ={"round":r +1 }

        train_losses =[]
        train_accs =[]
        val_losses =[]
        val_accs =[]


        for i ,client_name in enumerate (CLIENT_NAMES ):
            print (f"\n[Client {client_name}]")


            local_model =copy .deepcopy (client_models [client_name ]).to (DEVICE )


            load_shared_params (local_model ,shared_state )

            criterion =nn .CrossEntropyLoss (weight =client_class_weights [client_name ])
            optimizer =optim .AdamW (local_model .parameters (),lr =LR ,weight_decay =WEIGHT_DECAY )

            client_epoch_losses =[]
            client_epoch_accs =[]

            for ep in range (LOCAL_EPOCHS ):
                tr_loss ,tr_acc ,_ ,_ =run_epoch (
                local_model ,
                train_loaders [client_name ],
                criterion ,
                optimizer =optimizer ,
                train =True 
                )
                client_epoch_losses .append (tr_loss )
                client_epoch_accs .append (tr_acc )
                print (f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss ,val_acc ,_ ,_ =run_epoch (
            local_model ,
            val_loaders [client_name ],
            criterion ,
            optimizer =None ,
            train =False 
            )
            print (f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


            client_models [client_name ]=copy .deepcopy (local_model ).to (DEVICE )


            w =len (train_loaders [client_name ].dataset )
            local_weights .append (w )
            local_models .append (copy .deepcopy (local_model ).cpu ())

            train_losses .append (float (np .mean (client_epoch_losses )))
            train_accs .append (float (np .mean (client_epoch_accs )))
            val_losses .append (float (val_loss ))
            val_accs .append (float (val_acc ))

            round_summary [f"{client_name}_train_loss"]=float (np .mean (client_epoch_losses ))
            round_summary [f"{client_name}_train_acc"]=float (np .mean (client_epoch_accs ))
            round_summary [f"{client_name}_val_loss"]=float (val_loss )
            round_summary [f"{client_name}_val_acc"]=float (val_acc )

        total_train_size =sum (local_weights )
        if total_train_size ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your dataset splits.")

        norm_weights =[w /total_train_size for w in local_weights ]


        print ("\nAggregating shared layers only (FedPer)...")
        avg_shared =average_shared_state_dicts (local_models ,norm_weights )


        shared_state =copy .deepcopy (global_model .state_dict ())
        for k ,v in avg_shared .items ():
            shared_state [k ]=v .clone ()
        global_model .load_state_dict (shared_state )


        for client_name in CLIENT_NAMES :
            load_shared_params (client_models [client_name ],shared_state )


        mean_val_loss =float (np .mean (val_losses ))
        mean_val_acc =float (np .mean (val_accs ))

        if mean_val_loss <best_mean_val_loss :
            best_mean_val_loss =mean_val_loss 
            best_shared_state =copy .deepcopy (shared_state )
            torch .save (best_shared_state ,os .path .join (OUTPUT_DIR ,MODEL_NAME ))
            print ("\nSaved best shared model state.")

        round_summary ["mean_train_loss"]=float (np .mean (train_losses ))
        round_summary ["mean_train_acc"]=float (np .mean (train_accs ))
        round_summary ["mean_val_loss"]=mean_val_loss 
        round_summary ["mean_val_acc"]=mean_val_acc 


        print ("\n"+"="*30 )
        print (f"PERSONALIZED CLIENT TESTS AFTER ROUND {r + 1}")
        print ("="*30 )

        for client_name in CLIENT_NAMES :
            client_eval_criterion =nn .CrossEntropyLoss (weight =client_class_weights [client_name ])
            result_client =evaluate_loader (
            client_models [client_name ],
            test_loaders [client_name ],
            client_eval_criterion ,
            class_names ,
            title_prefix =f"{client_name}_round_{r + 1}",
            save_dir =OUTPUT_DIR ,
            save_cm =True 
            )

            round_summary [f"{client_name}_test_loss"]=result_client ["loss"]
            round_summary [f"{client_name}_test_acc"]=result_client ["accuracy"]
            round_summary [f"{client_name}_test_precision"]=result_client ["precision_macro"]
            round_summary [f"{client_name}_test_recall"]=result_client ["recall_macro"]
            round_summary [f"{client_name}_test_f1"]=result_client ["f1_macro"]
            round_summary [f"{client_name}_test_kappa"]=result_client ["kappa"]
            round_summary [f"{client_name}_test_specificity"]=result_client ["specificity_macro"]

            per_client_test_acc_history [client_name ].append (result_client ["accuracy"])
            per_client_test_loss_history [client_name ].append (result_client ["loss"])

        round_results .append (round_summary )

        history ["round"].append (r +1 )
        history ["mean_train_loss"].append (round_summary ["mean_train_loss"])
        history ["mean_train_acc"].append (round_summary ["mean_train_acc"])
        history ["mean_val_loss"].append (round_summary ["mean_val_loss"])
        history ["mean_val_acc"].append (round_summary ["mean_val_acc"])

        print (
        f"\n[ROUND {r + 1}] "
        f"Mean Train Loss: {round_summary['mean_train_loss']:.4f} | "
        f"Mean Train Acc: {round_summary['mean_train_acc']:.4f} | "
        f"Mean Val Loss: {round_summary['mean_val_loss']:.4f} | "
        f"Mean Val Acc: {round_summary['mean_val_acc']:.4f}"
        )


        ckpt ={
        "round":r +1 ,
        "shared_state":shared_state ,
        "client_states":{client :client_models [client ].state_dict ()for client in CLIENT_NAMES },
        "client_names":CLIENT_NAMES ,
        "class_names":class_names ,
        }
        ckpt_path =os .path .join (OUTPUT_DIR ,f"fedper_round_{r + 1}.pth")
        torch .save (ckpt ,ckpt_path )
        print ("Saved checkpoint:",ckpt_path )


        df_rounds =pd .DataFrame (round_results )
        csv_path =os .path .join (OUTPUT_DIR ,"fedper_round_results.csv")
        df_rounds .to_csv (csv_path ,index =False )
        print ("Saved per-round summary CSV to",csv_path )

        plot_round_curves (history ,OUTPUT_DIR )


        plt .figure (figsize =(8 ,5 ))
        for client_name in CLIENT_NAMES :
            plt .plot (
            range (1 ,len (per_client_test_acc_history [client_name ])+1 ),
            per_client_test_acc_history [client_name ],
            label =client_name 
            )
        plt .xlabel ("Communication Round")
        plt .ylabel ("Test Accuracy")
        plt .title ("Per-client Test Accuracy")
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (os .path .join (OUTPUT_DIR ,"per_client_test_accuracy_rounds.png"),dpi =300 )
        plt .close ()

        plt .figure (figsize =(8 ,5 ))
        for client_name in CLIENT_NAMES :
            plt .plot (
            range (1 ,len (per_client_test_loss_history [client_name ])+1 ),
            per_client_test_loss_history [client_name ],
            label =client_name 
            )
        plt .xlabel ("Communication Round")
        plt .ylabel ("Test Loss")
        plt .title ("Per-client Test Loss")
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (os .path .join (OUTPUT_DIR ,"per_client_test_loss_rounds.png"),dpi =300 )
        plt .close ()

    elapsed =time .time ()-start_time 
    print (f"\nFederated training finished in {elapsed / 60:.2f} minutes")


    global_model .load_state_dict (best_shared_state )




    final_results =[]

    print ("\n==============================")
    print ("FINAL TEST RESULTS: EACH CLIENT PERSONALIZED MODEL")
    print ("==============================")

    for i ,client_name in enumerate (CLIENT_NAMES ):
        client_eval_criterion =nn .CrossEntropyLoss (weight =client_class_weights [client_name ])
        result_client =evaluate_loader (
        client_models [client_name ],
        test_loaders [client_name ],
        client_eval_criterion ,
        class_names ,
        title_prefix =f"{client_name}_final",
        save_dir =OUTPUT_DIR ,
        save_cm =True 
        )
        final_results .append (result_client )


    torch .save (
    {
    "shared_state":global_model .state_dict (),
    "client_states":{client :client_models [client ].state_dict ()for client in CLIENT_NAMES },
    "class_names":class_names ,
    "client_names":CLIENT_NAMES ,
    },
    os .path .join (OUTPUT_DIR ,"fedper_final_checkpoint.pth")
    )


    with open (os .path .join (OUTPUT_DIR ,"fedper_round_metrics.json"),"w")as f :
        json .dump (round_results ,f ,indent =2 )

    with open (os .path .join (OUTPUT_DIR ,"fedper_round_metrics.csv"),"w",newline ="")as f :
        writer =csv .DictWriter (f ,fieldnames =list (round_results [0 ].keys ()))
        writer .writeheader ()
        writer .writerows (round_results )

    save_metrics_csv (final_results ,os .path .join (OUTPUT_DIR ,"fedper_final_test_metrics.csv"))
    with open (os .path .join (OUTPUT_DIR ,"fedper_final_test_metrics.json"),"w")as f :
        json .dump (final_results ,f ,indent =2 )

    print ("\nSaved outputs to:")
    print (os .path .join (OUTPUT_DIR ,MODEL_NAME ))
    print (os .path .join (OUTPUT_DIR ,"fedper_final_checkpoint.pth"))
    print (os .path .join (OUTPUT_DIR ,"fedper_round_metrics.csv"))
    print (os .path .join (OUTPUT_DIR ,"fedper_round_metrics.json"))
    print (os .path .join (OUTPUT_DIR ,"fedper_final_test_metrics.csv"))
    print (os .path .join (OUTPUT_DIR ,"fedper_final_test_metrics.json"))

if __name__ =="__main__":
    main ()