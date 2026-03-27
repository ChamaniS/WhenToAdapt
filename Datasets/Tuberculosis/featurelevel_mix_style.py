import os
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torchvision import io as tv_io 
try :
    tv_io .set_image_backend ('PIL')
except Exception :
    pass 

import copy ,random ,time 
from typing import List 
import numpy as np 
import torch 
import torch .nn as nn 
from torch .optim import AdamW 
from torch .utils .data import DataLoader ,Dataset ,ConcatDataset 
from torchvision import transforms 
from torchvision .datasets .folder import default_loader 
import torchvision 
from tqdm import tqdm 
import matplotlib .pyplot as plt 
from PIL import Image 
from sklearn .metrics import (
precision_score ,recall_score ,f1_score ,balanced_accuracy_score ,
cohen_kappa_score ,confusion_matrix ,accuracy_score 
)


UPLOADED_SAMPLE_PATH ="/mnt/data/bcea4669-5b03-46c7-9fe6-f80536ff4b98.png"




CLIENT_ROOTS =[
r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen",
r"xxxxx\Projects\Data\Tuberculosis_Data\Montgomery",
r"xxxxx\Projects\Data\Tuberculosis_Data\TBX11K",
r"xxxxx\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES =["Shenzhen","Montgomery","TBX11K","Pakistan"]
OUTPUT_DIR ="./fl_outputs_mixstyle_singlehist_perclass"
ARCH ="densenet169"
PRETRAINED =True 
USE_MIXSTYLE =True 
MIX_P =0.5 
MIX_ALPHA =0.1 
IMG_SIZE =224 
BATCH_SIZE =4 
WORKERS =4 
LOCAL_EPOCHS =6 
COMM_ROUNDS =10 
LR =1e-4 
WEIGHT_DECAY =1e-5 
DROPOUT_P =0.5 
SEED =42 
CLASS_NAMES =["normal","positive"]
DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


N_SAMPLES =7 


GLOBAL_TEST_ACC_FN =os .path .join (OUTPUT_DIR ,"global_test_accuracy_rounds.png")
GLOBAL_TEST_LOSS_FN =os .path .join (OUTPUT_DIR ,"global_test_loss_rounds.png")
PER_CLIENT_ACC_FN =os .path .join (OUTPUT_DIR ,"per_client_test_accuracy_rounds.png")
PER_CLIENT_LOSS_FN =os .path .join (OUTPUT_DIR ,"per_client_test_loss_rounds.png")

os .makedirs (OUTPUT_DIR ,exist_ok =True )

def set_seed (seed =SEED ):
    random .seed (seed );np .random .seed (seed );torch .manual_seed (seed )
    if torch .cuda .is_available ():torch .cuda .manual_seed_all (seed )
set_seed ()




class MixStyle (nn .Module ):
    def __init__ (self ,p =0.5 ,alpha =0.1 ,eps =1e-6 ):
        super ().__init__ ()
        self .p =p ;self .alpha =alpha ;self .eps =eps 
    def forward (self ,x :torch .Tensor )->torch .Tensor :
        if (not self .training )or (torch .rand (1 ).item ()>self .p )or x .size (0 )<=1 :
            return x 
        B ,C ,H ,W =x .size ()
        x_view =x .view (B ,C ,-1 )
        mu =x_view .mean (dim =2 ).view (B ,C ,1 ,1 )
        var =x_view .var (dim =2 ,unbiased =False ).view (B ,C ,1 ,1 )
        sigma =(var +self .eps ).sqrt ()
        x_norm =(x -mu )/sigma 
        lm =np .random .beta (self .alpha ,self .alpha ,size =B ).astype (np .float32 )
        lm =torch .from_numpy (lm ).to (x .device ).view (B ,1 ,1 ,1 )
        perm =torch .randperm (B ).to (x .device )
        mu2 =mu [perm ];sigma2 =sigma [perm ]
        mu_mix =mu *lm +mu2 *(1.0 -lm )
        sigma_mix =sigma *lm +sigma2 *(1.0 -lm )
        out =x_norm *sigma_mix +mu_mix 
        return out 




class PathListDataset (Dataset ):
    def __init__ (self ,samples :List [tuple ],transform =None ,loader =default_loader ):
        self .samples =list (samples )
        self .transform =transform 
        self .loader =loader 
    def __len__ (self ):return len (self .samples )
    def __getitem__ (self ,idx ):
        path ,label =self .samples [idx ]
        img =self .loader (path ).convert ("RGB")
        if self .transform :img =self .transform (img )
        return img ,label 

def gather_samples_from_client_split (client_root :str ,split :str ,class_names :List [str ]):
    split_dir =os .path .join (client_root ,split )
    if not os .path .isdir (split_dir ):
        return []
    samples =[]
    canon_map ={c .lower ():i for i ,c in enumerate (class_names )}
    for cls_folder in os .listdir (split_dir ):
        cls_path =os .path .join (split_dir ,cls_folder )
        if not os .path .isdir (cls_path ):continue 
        key =cls_folder .lower ()
        if key not in canon_map :continue 
        label =canon_map [key ]
        for fn in os .listdir (cls_path ):
            if fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                samples .append ((os .path .join (cls_path ,fn ),label ))
    return samples 

def make_multi_client_dataloaders (client_roots ):
    normalize =transforms .Normalize (mean =[0.485 ,0.456 ,0.406 ],std =[0.229 ,0.224 ,0.225 ])
    train_tf =transforms .Compose ([
    transforms .RandomResizedCrop (IMG_SIZE ,scale =(0.8 ,1.0 )),
    transforms .RandomHorizontalFlip (),
    transforms .RandomRotation (5 ),
    transforms .ColorJitter (brightness =0.05 ,contrast =0.05 ),
    transforms .ToTensor (),
    normalize 
    ])
    val_tf =transforms .Compose ([
    transforms .Resize ((IMG_SIZE ,IMG_SIZE )),
    transforms .ToTensor (),
    normalize 
    ])

    per_client =[]
    skipped_indices =[]
    for idx ,root in enumerate (client_roots ):
        tr =gather_samples_from_client_split (root ,"train",CLASS_NAMES )
        va =gather_samples_from_client_split (root ,"val",CLASS_NAMES )
        te =gather_samples_from_client_split (root ,"test",CLASS_NAMES )
        print (f"[DATA] client {root} -> train:{len(tr)} val:{len(va)} test:{len(te)}")
        if len (tr )==0 :
            print (f"WARNING: Skipping client '{CLIENT_NAMES[idx]}' (no train samples). Check path: {root}")
            skipped_indices .append (idx )
            continue 
        per_client .append ({
        "root":root ,
        "name":CLIENT_NAMES [idx ],
        "train":DataLoader (PathListDataset (tr ,transform =train_tf ),batch_size =BATCH_SIZE ,shuffle =True ,num_workers =WORKERS ,pin_memory =True ),
        "val":DataLoader (PathListDataset (va ,transform =val_tf ),batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =True ),
        "test":DataLoader (PathListDataset (te ,transform =val_tf ),batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =True ),
        "train_samples":tr ,"val_samples":va ,"test_samples":te ,
        "train_tf":train_tf ,"val_tf":val_tf 
        })
    if len (per_client )==0 :
        raise RuntimeError ("No clients with train samples found. Fix CLIENT_ROOTS and ensure each has a 'train' folder with images.")
    if skipped_indices :
        print (f"NOTE: skipped clients indices {skipped_indices} because they had zero train samples.")
    return per_client 

def compute_class_weights_from_samples (samples ):
    targets =[s [1 ]for s in samples ]
    if len (targets )==0 :
        return torch .ones (len (CLASS_NAMES ),dtype =torch .float32 )
    counts =np .bincount (targets ,minlength =len (CLASS_NAMES )).astype (np .float32 )
    total =counts .sum ()if counts .sum ()>0 else 1.0 
    weights =total /(counts +1e-8 )
    weights =weights /np .mean (weights )
    return torch .tensor (weights ,dtype =torch .float32 )




def create_model (num_classes ):
    base =getattr (torchvision .models ,ARCH )(pretrained =PRETRAINED )
    mixlayer =MixStyle (p =MIX_P ,alpha =MIX_ALPHA )if USE_MIXSTYLE else nn .Identity ()
    if hasattr (base ,"features")and isinstance (base .features ,nn .Sequential ):
        base .features .add_module ("mixstyle",mixlayer )
    elif hasattr (base ,"layer1")and isinstance (base .layer1 ,nn .Sequential ):
        base .layer1 .add_module ("mixstyle",mixlayer )
    else :
        base .mixstyle =mixlayer 
    if hasattr (base ,"classifier"):
        in_ch =base .classifier .in_features 
        base .classifier =nn .Sequential (nn .Dropout (p =DROPOUT_P ),nn .Linear (in_ch ,num_classes ))
    elif hasattr (base ,"fc"):
        in_ch =base .fc .in_features 
        base .fc =nn .Sequential (nn .Dropout (p =DROPOUT_P ),nn .Linear (in_ch ,num_classes ))
    else :
        raise RuntimeError ("Unknown model head; edit create_model accordingly.")
    return base 

def count_parameters (model ):
    return sum (p .numel ()for p in model .parameters ()if p .requires_grad )

def average_models_weighted (models :List [torch .nn .Module ],weights :List [float ]):
    if len (models )==0 :raise ValueError ("No models to average")
    if len (models )!=len (weights ):raise ValueError ("models and weights must match length")
    sum_w =float (sum (weights ))
    if sum_w ==0.0 :raise ValueError ("Sum of weights is zero")
    norm_w =[w /sum_w for w in weights ]
    base_sd =models [0 ].state_dict ()
    avg_sd ={}
    with torch .no_grad ():
        for k ,v0 in base_sd .items ():
            acc =torch .zeros_like (v0 ,dtype =torch .float32 ,device ="cpu")
            for m ,w in zip (models ,norm_w ):
                vm =m .state_dict ()[k ].cpu ().to (dtype =torch .float32 )
                acc +=float (w )*vm 
            try :acc =acc .to (dtype =v0 .dtype )
            except Exception :pass 
            avg_sd [k ]=acc 
    return avg_sd 




def train_local (model ,dataloader ,criterion ,optimizer ,device ,epochs =LOCAL_EPOCHS ):
    model .to (device );model .train ()
    for ep in range (epochs ):
        running_loss =0.0 ;correct =0 ;total =0 
        pbar =tqdm (dataloader ,desc =f"LocalTrain ep{ep+1}/{epochs}",leave =False )
        for x ,y in pbar :
            x ,y =x .to (device ),y .to (device )
            optimizer .zero_grad ()
            out =model (x )
            loss =criterion (out ,y )
            loss .backward ();optimizer .step ()
            running_loss +=float (loss .item ())*x .size (0 )
            _ ,preds =out .max (1 )
            correct +=(preds ==y ).sum ().item ()
            total +=x .size (0 )
            pbar .set_postfix (loss =running_loss /total if total >0 else 0.0 ,acc =correct /total if total >0 else 0.0 )
    return 

@torch .no_grad ()
def evaluate_model (model ,dataloader ,device ,criterion =None ,return_per_class =False ,class_names =None ):
    model .eval ()
    all_y ,all_pred =[],[]
    total_loss =0.0 ;n =0 
    for x ,y in tqdm (dataloader ,desc ="Eval",leave =False ):
        x ,y =x .to (device ),y .to (device )
        out =model (x )
        _ ,preds =out .max (1 )
        all_y .extend (y .cpu ().numpy ().tolist ());all_pred .extend (preds .cpu ().numpy ().tolist ())
        if criterion is not None :total_loss +=float (criterion (out ,y ).item ())*x .size (0 )
        n +=x .size (0 )
    if n ==0 :return {}
    acc =accuracy_score (all_y ,all_pred )
    prec_macro =precision_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    rec_macro =recall_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    f1_macro =f1_score (all_y ,all_pred ,average ="macro",zero_division =0 )
    bal =balanced_accuracy_score (all_y ,all_pred )
    kappa =cohen_kappa_score (all_y ,all_pred )
    metrics ={"accuracy":float (acc ),"precision_macro":float (prec_macro ),
    "recall_macro":float (rec_macro ),"f1_macro":float (f1_macro ),
    "balanced_acc":float (bal ),"cohen_kappa":float (kappa )}
    if criterion is not None :metrics ["loss"]=float (total_loss /max (1 ,n ))
    if return_per_class :
        if class_names is None :raise ValueError ("class_names required")
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




def unnormalize_tensor_to_uint8 (tensor ,mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )):
    t =tensor .cpu ().numpy ()
    if t .ndim ==3 :
        t =t .transpose (1 ,2 ,0 )
    mean =np .array (mean ).reshape (1 ,1 ,3 );std =np .array (std ).reshape (1 ,1 ,3 )
    img =(t *std +mean )
    img =np .clip (img *255.0 ,0 ,255 ).astype (np .uint8 )
    return img 

def compute_grayscale_histogram_uint8 (arr_uint8 ,bins =256 ):
    """
    arr_uint8: HxWx3 uint8
    returns hist (bins,) normalized to [0,1]
    """
    import numpy as _np 
    gray =(0.2989 *arr_uint8 [:,:,0 ]+0.5870 *arr_uint8 [:,:,1 ]+0.1140 *arr_uint8 [:,:,2 ]).astype (_np .uint8 )
    hist ,edges =_np .histogram (gray .ravel (),bins =bins ,range =(0 ,255 ),density =False )

    hist =hist .astype (float )/(hist .sum ()+1e-8 )
    return hist ,edges 

def save_comparison_grid_singlehist (client_idx ,client_entry ,round_num ,out_base =OUTPUT_DIR ,n_samples =N_SAMPLES ):
    """
    Save a single composite image containing:
    - top row: original images (n columns)
    - middle row: harmonized images (n columns)
    - bottom row: combined grayscale histogram overlay (orig vs harm) (n columns)
    """
    cname =client_entry .get ("name",f"client{client_idx}")
    dest_dir =os .path .join (out_base ,"HarmonizedSamples",cname ,f"round_{round_num}")
    os .makedirs (dest_dir ,exist_ok =True )
    samples =client_entry .get ("train_samples",[])
    val_tf =client_entry .get ("val_tf",None )
    if not samples :
        print (f"[HARM] No train samples for {cname}, skipping harmonized saves.")
        return 
    rng =np .random .RandomState (seed =SEED +round_num +client_idx )
    idxs =rng .choice (range (len (samples )),size =min (n_samples ,len (samples )),replace =False )

    orig_imgs =[]
    harm_imgs =[]
    histograms =[]
    for s_i in idxs :
        path ,_ =samples [s_i ]
        try :
            pil =Image .open (path ).convert ("RGB")
        except Exception as e :
            print (f"[HARM] Could not open {path}: {e}")
            continue 
        orig_arr =np .array (pil ).astype (np .uint8 )

        if val_tf is None :
            print ("[HARM] No val transform provided; skipping.")
            continue 
        t =val_tf (pil )
        harm_arr =unnormalize_tensor_to_uint8 (t )

        h_h ,h_w =harm_arr .shape [0 ],harm_arr .shape [1 ]
        orig_resized =np .array (pil .resize ((h_w ,h_h ),resample =Image .BILINEAR )).astype (np .uint8 )

        hist_o ,edges =compute_grayscale_histogram_uint8 (orig_resized )
        hist_h ,_ =compute_grayscale_histogram_uint8 (harm_arr )
        orig_imgs .append (orig_resized )
        harm_imgs .append (harm_arr )
        histograms .append ((hist_o ,hist_h ,edges ))
    if len (orig_imgs )==0 :
        print (f"[HARM] No valid samples for {cname} to save.")
        return 

    ncol =len (orig_imgs )

    fig_h =3 *(IMG_SIZE /72 )
    fig_w =ncol *(IMG_SIZE /72 )
    fig ,axs =plt .subplots (3 ,ncol ,figsize =(fig_w ,fig_h ))
    if ncol ==1 :
        axs =np .array ([[axs [0 ]],[axs [1 ]],[axs [2 ]]])
    for j in range (ncol ):
        axs [0 ,j ].imshow (orig_imgs [j ]);axs [0 ,j ].axis ('off');axs [0 ,j ].set_title (os .path .basename (samples [idxs [j ]][0 ]),fontsize =8 )
        axs [1 ,j ].imshow (harm_imgs [j ]);axs [1 ,j ].axis ('off')

        hist_o ,hist_h ,edges =histograms [j ]
        centers =(edges [:-1 ]+edges [1 :])/2.0 
        axs [2 ,j ].plot (centers ,hist_o ,label ='orig',linewidth =1 )
        axs [2 ,j ].plot (centers ,hist_h ,label ='harm',linewidth =1 )
        axs [2 ,j ].set_xlim (0 ,255 )
        axs [2 ,j ].set_ylim (0 ,max (hist_o .max (),hist_h .max ())*1.05 +1e-8 )
        axs [2 ,j ].set_xticks ([]);axs [2 ,j ].set_yticks ([])
    plt .suptitle (f"Harmonized samples: {cname} (round {round_num})",fontsize =12 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.95 ])
    grid_path =os .path .join (dest_dir ,f"comparison_grid_{cname}_round{round_num}.png")
    fig .savefig (grid_path ,dpi =150 );plt .close (fig )
    print (f"[HARM] Saved comparison grid for {cname} round {round_num} -> {grid_path}")




def save_round_series_plots (round_results ,per_client_acc_history ,per_client_loss_history ,out_dir =OUTPUT_DIR ):
    rounds =list (range (1 ,len (round_results )+1 ))

    gtest_acc =[rr .get ("global_test_acc",0.0 )for rr in round_results ]
    gtest_loss =[rr .get ("global_test_loss",0.0 )for rr in round_results ]

    plt .figure (figsize =(6 ,4 ))
    plt .plot (rounds ,gtest_acc ,marker ='o')
    plt .xlabel ("Global Round");plt .ylabel ("Test Accuracy");plt .title ("Global Test Accuracy")
    plt .grid (True )
    plt .savefig (GLOBAL_TEST_ACC_FN );plt .close ()

    plt .figure (figsize =(6 ,4 ))
    plt .plot (rounds ,gtest_loss ,marker ='o')
    plt .xlabel ("Global Round");plt .ylabel ("Test Loss");plt .title ("Global Test Loss")
    plt .grid (True )
    plt .savefig (GLOBAL_TEST_LOSS_FN );plt .close ()

    plt .figure (figsize =(8 ,5 ))
    for i ,name in enumerate (CLIENT_NAMES ):
        vals =per_client_acc_history .get (i ,[])
        plt .plot (range (1 ,len (vals )+1 ),vals ,marker ='o',label =name )
    plt .xlabel ("Global Round");plt .ylabel ("Test Accuracy");plt .title ("Per-client Test Accuracy");plt .legend ();plt .grid (True )
    plt .savefig (PER_CLIENT_ACC_FN );plt .close ()

    plt .figure (figsize =(8 ,5 ))
    for i ,name in enumerate (CLIENT_NAMES ):
        vals =per_client_loss_history .get (i ,[])
        plt .plot (range (1 ,len (vals )+1 ),vals ,marker ='o',label =name )
    plt .xlabel ("Global Round");plt .ylabel ("Test Loss");plt .title ("Per-client Test Loss");plt .legend ();plt .grid (True )
    plt .savefig (PER_CLIENT_LOSS_FN );plt .close ()




def pretty_print_per_class_metrics (client_idx ,client_name ,metrics ):
    """
    Example desired format reproduced here.
    """
    acc =metrics .get ("accuracy",np .nan )
    prec =metrics .get ("precision_macro",np .nan )
    rec =metrics .get ("recall_macro",np .nan )
    f1 =metrics .get ("f1_macro",np .nan )
    kappa =metrics .get ("cohen_kappa",np .nan )

    mean_spec =None 
    if "per_class_specificity"in metrics :
        specs =metrics .get ("per_class_specificity",[])
        if len (specs )>0 :
            mean_spec =float (np .mean (specs ))

    print (f"[CLIENT {client_idx}] {client_name}")
    print (f"  Accuracy       : {acc:.4f}")
    print (f"  Precision (mac): {prec:.4f}")
    print (f"  Recall (mac)   : {rec:.4f}")
    print (f"  F1 (mac)       : {f1:.4f}")
    if mean_spec is not None :
        print (f"  Mean Specificity: {mean_spec:.4f}")
    else :
        print (f"  Mean Specificity: n/a")
    print (f"  Cohen's kappa  : {kappa:.4f}\n")

    if "per_class_precision"in metrics :
        print (f"  Per-class metrics (order = {CLASS_NAMES}):")
        header =["Class","Support","Correct","Acc","Prec","Rec","F1","Spec"]
        print ("    "+"{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format (*header ))
        cm =metrics .get ("confusion_matrix",None )
        tp_counts =np .diag (cm ).astype (int )if cm is not None else [0 ]*len (CLASS_NAMES )
        supports =cm .sum (axis =1 ).astype (int )if cm is not None else [0 ]*len (CLASS_NAMES )
        precisions =metrics .get ("per_class_precision",[])
        recalls =metrics .get ("per_class_recall",[])
        f1s =metrics .get ("per_class_f1",[])
        specs =metrics .get ("per_class_specificity",[])
        accs =metrics .get ("per_class_accuracy",[])
        for ci ,cname in enumerate (CLASS_NAMES ):
            s =int (supports [ci ])if ci <len (supports )else 0 
            ccount =int (tp_counts [ci ])if ci <len (tp_counts )else 0 
            acc_val =float (accs [ci ])if ci <len (accs )else np .nan 
            pval =float (precisions [ci ])if ci <len (precisions )else np .nan 
            rval =float (recalls [ci ])if ci <len (recalls )else np .nan 
            fval =float (f1s [ci ])if ci <len (f1s )else np .nan 
            sval =float (specs [ci ])if ci <len (specs )else np .nan 
            print ("    {:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format (
            cname ,s ,ccount ,acc_val ,pval ,rval ,fval ,sval 
            ))
    else :
        print ("  (no per-class metrics available)\n")




def main ():
    print ("DEVICE:",DEVICE )
    clients =make_multi_client_dataloaders (CLIENT_ROOTS )

    active_client_names =[c ["name"]for c in clients ]
    print ("Active clients:",active_client_names )
    client_train_sizes =[len (c ["train_samples"])for c in clients ]
    total_train =sum (client_train_sizes )if sum (client_train_sizes )>0 else 1 
    print ("client train sizes:",client_train_sizes )

    global_model =create_model (len (CLASS_NAMES )).to (DEVICE )
    print (f"Global model {ARCH} created with {count_parameters(global_model):,} params. MixStyle={USE_MIXSTYLE}")

    round_results =[]
    per_client_acc_history ={i :[]for i in range (len (clients ))}
    per_client_loss_history ={i :[]for i in range (len (clients ))}

    for r in range (COMM_ROUNDS ):
        print ("\n"+"="*60 )
        print (f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print ("="*60 )
        local_models =[];weights =[];round_summary ={"round":r +1 }

        for i ,client in enumerate (clients ):
            print (f"\n[CLIENT {i}] {client.get('name')}: local training")
            local_model =copy .deepcopy (global_model )
            train_ds =client ["train_samples"]
            client_cw =compute_class_weights_from_samples (train_ds ).to (DEVICE )
            criterion =nn .CrossEntropyLoss (weight =client_cw )
            optimizer =AdamW (local_model .parameters (),lr =LR ,weight_decay =WEIGHT_DECAY )
            train_local (local_model ,client ["train"],criterion ,optimizer ,DEVICE ,epochs =LOCAL_EPOCHS )
            local_models .append (local_model .cpu ())
            w =float (len (train_ds ))/float (total_train )
            weights .append (w )
            print (f"[CLIENT {i}] aggregation weight: {w:.4f}")


        print ("\nAggregating local models (FedAvg weighted)")
        avg_state =average_models_weighted (local_models ,weights )
        avg_state_on_device ={k :v .to (DEVICE )for k ,v in avg_state .items ()}
        global_model .load_state_dict (avg_state_on_device )
        global_model .to (DEVICE )


        combined_val_dsets =[c ["val"].dataset for c in clients ]
        combined_val =ConcatDataset (combined_val_dsets )
        combined_val_loader =DataLoader (combined_val ,batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =True )

        combined_train_targets =[]
        for c in clients :
            combined_train_targets .extend ([s [1 ]for s in c ["train_samples"]])
        counts =np .bincount (combined_train_targets ,minlength =len (CLASS_NAMES )).astype (np .float32 )
        counts [counts ==0 ]=1.0 
        weights_arr =1.0 /counts 
        weights_arr =weights_arr *(len (weights_arr )/weights_arr .sum ())
        combined_class_weights =torch .tensor (weights_arr ,dtype =torch .float32 ).to (DEVICE )
        combined_criterion =nn .CrossEntropyLoss (weight =combined_class_weights )
        global_val_metrics =evaluate_model (global_model ,combined_val_loader ,DEVICE ,criterion =combined_criterion )
        print ("Global combined val metrics:",global_val_metrics )
        round_summary ["global_val_loss"]=float (global_val_metrics .get ("loss",np .nan ))
        round_summary ["global_val_acc"]=float (global_val_metrics .get ("accuracy",np .nan ))


        combined_test_dsets =[c ["test"].dataset for c in clients ]
        combined_test =ConcatDataset (combined_test_dsets )
        combined_test_loader =DataLoader (combined_test ,batch_size =BATCH_SIZE ,shuffle =False ,num_workers =WORKERS ,pin_memory =True )
        global_test_metrics =evaluate_model (global_model ,combined_test_loader ,DEVICE ,criterion =combined_criterion ,return_per_class =False )
        print ("Global combined TEST metrics summary:",{k :global_test_metrics .get (k )for k in ["accuracy","loss","f1_macro","precision_macro","recall_macro","balanced_acc","cohen_kappa"]})
        round_summary ["global_test_loss"]=float (global_test_metrics .get ("loss",np .nan ))
        round_summary ["global_test_acc"]=float (global_test_metrics .get ("accuracy",np .nan ))


        per_client_test_metrics =[]
        for i ,client in enumerate (clients ):
            print (f"\n--- Global TEST on client {i} ({client.get('name')}) test set ---")
            client_train_ds =client ["train_samples"]
            client_cw =compute_class_weights_from_samples (client_train_ds ).to (DEVICE )
            client_criterion =nn .CrossEntropyLoss (weight =client_cw )

            cl_metrics =evaluate_model (global_model ,client ["test"],DEVICE ,criterion =client_criterion ,return_per_class =True ,class_names =CLASS_NAMES )


            pretty_print_per_class_metrics (i ,client .get ("name"),cl_metrics )


            per_client_acc_history [i ].append (float (cl_metrics .get ("accuracy",np .nan )))
            per_client_loss_history [i ].append (float (cl_metrics .get ("loss",np .nan )))

            per_client_test_metrics .append (cl_metrics )


            try :
                save_comparison_grid_singlehist (i ,client ,round_num =r +1 ,out_base =OUTPUT_DIR ,n_samples =N_SAMPLES )
            except Exception as e :
                print (f"[HARM] Error saving comparison grid for client {client.get('name')}: {e}")


        ckpt ={
        "round":r +1 ,
        "model_state":global_model .state_dict (),
        "global_val_metrics":global_val_metrics ,
        "global_test_metrics":global_test_metrics ,
        "per_client_test_metrics":per_client_test_metrics ,
        "client_names":[c .get ("name")for c in clients ],
        "class_names":CLASS_NAMES 
        }
        ckpt_path =os .path .join (OUTPUT_DIR ,f"global_round_{r+1}.pth")
        torch .save (ckpt ,ckpt_path )
        print ("Saved checkpoint:",ckpt_path )


        round_results .append (round_summary )
        save_round_series_plots (round_results ,per_client_acc_history ,per_client_loss_history ,out_dir =OUTPUT_DIR )


    final_model_path =os .path .join (OUTPUT_DIR ,"global_final.pth")
    torch .save ({"model_state":global_model .state_dict (),"class_names":CLASS_NAMES },final_model_path )
    print ("Federated training finished. Final global model saved to:",final_model_path )

if __name__ =="__main__":
    main ()
