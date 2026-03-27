import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy 
import time 
import glob 
import cv2 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,Dataset ,ConcatDataset 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
from models .UNET import UNET 


DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

client_names =["HAM10K","PH2","ISIC2017","ISIC2018"]
NUM_CLIENTS =len (client_names )

LOCAL_EPOCHS =8 
COMM_ROUNDS =10 
start_time =time .time ()
out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )
USE_SGD =True 
SGD_LR =1e-3 
ADAMW_LR =1e-4 
GRAD_CLIP_NORM =1.0 
DEBUG_SCAFFOLD =False 

FOREGROUND_BCE_WEIGHT =5.0 
DICE_WEIGHT =1.0 
BCE_WEIGHT =1.0 

splits_root =r"C:\Users\csj5\Projects\Data\skinlesions"

client_ext_map ={
"HAM10K":((".jpg",),(".png",)),
"ISIC2017":((".jpg",),(".png",)),
"ISIC2018":((".jpg",),(".png",)),
"PH2":((".jpg",".png",".bmp",".jpeg"),(".png",".bmp",".jpg")),
}

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
    print (f"  val   imgs: {val_img_dirs[i]}    masks: {val_mask_dirs[i]}")
    print (f"  test  imgs: {test_img_dirs[i]}   masks: {test_mask_dirs[i]}")


def get_loader (img_dir ,mask_dir ,transform ,client_name =None ,batch_size =4 ,shuffle =True ):
    if client_name is not None and client_name in client_ext_map :
        img_exts ,mask_exts =client_ext_map [client_name ]
    else :
        img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        mask_exts =(".png",".jpg",".bmp",".tif",".tiff")

    ds =SkinPairDataset (img_dir ,mask_dir ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    return DataLoader (
    ds ,
    batch_size =batch_size ,
    shuffle =shuffle ,
    num_workers =0 ,
    pin_memory =torch .cuda .is_available ()
    )

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
    return DataLoader (
    global_test_ds ,
    batch_size =batch_size ,
    shuffle =False ,
    num_workers =0 ,
    pin_memory =torch .cuda .is_available ()
    )

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

    return dict (
    dice_with_bg =dice_with_bg ,
    dice_no_bg =dice_no_bg ,
    iou_with_bg =iou_with_bg ,
    iou_no_bg =iou_no_bg ,
    accuracy =acc ,
    precision =precision ,
    recall =recall ,
    specificity =specificity ,
    )

def average_metrics (metrics_list ):
    if len (metrics_list )==0 :
        return {}
    avg ={}
    for k in metrics_list [0 ].keys ():
        avg [k ]=sum (m [k ]for m in metrics_list )/len (metrics_list )
    return avg 

class WeightedDiceBCELoss (nn .Module ):
    def __init__ (self ,pos_weight =5.0 ,dice_weight =1.0 ,bce_weight =1.0 ):
        super ().__init__ ()
        self .dice =smp .losses .DiceLoss (mode ="binary",from_logits =True )
        self .bce =nn .BCEWithLogitsLoss (pos_weight =torch .tensor ([pos_weight ],dtype =torch .float32 ))
        self .dice_weight =dice_weight 
        self .bce_weight =bce_weight 

    def forward (self ,logits ,targets ):
        bce =self .bce (logits ,targets )
        dice =self .dice (logits ,targets )
        return self .bce_weight *bce +self .dice_weight *dice 

def get_loss_fn (device ):
    loss_fn =WeightedDiceBCELoss (
    pos_weight =FOREGROUND_BCE_WEIGHT ,
    dice_weight =DICE_WEIGHT ,
    bce_weight =BCE_WEIGHT 
    ).to (device )
    return loss_fn 

def build_model ():
    try :
        return UNET (in_channels =3 ,num_classes =1 )
    except TypeError :
        return UNET (in_channels =3 ,out_channels =1 )

def average_model_state_dict (models ,weights ):
    base_sd =copy .deepcopy (models [0 ].state_dict ())

    for k in base_sd .keys ():
        tensor0 =models [0 ].state_dict ()[k ]
        if torch .is_floating_point (tensor0 ):
            base_sd [k ]=sum (
            weights [i ]*models [i ].state_dict ()[k ].detach ().cpu ()
            for i in range (len (models ))
            )
        else :
            base_sd [k ]=models [0 ].state_dict ()[k ].detach ().cpu ().clone ()

    return base_sd 

def print_batch_sanity (data ,target ,preds =None ,prefix =""):
    with torch .no_grad ():
        target_fg =float ((target >0.5 ).float ().mean ().item ())
        msg =f"{prefix} target foreground ratio: {target_fg:.6f}"
        if preds is not None :
            pred_prob =torch .sigmoid (preds )
            msg +=f" | pred prob mean: {float(pred_prob.mean().item()):.6f}"
            msg +=f" | pred prob max: {float(pred_prob.max().item()):.6f}"
        print (msg )


def train_local_scaffold (loader ,model ,loss_fn ,opt ,global_c ,client_c ,round_idx =0 ,client_idx =0 ):
    model .train ()
    total_loss =0.0 
    metrics =[]

    w_init ={name :param .data .clone ().detach ()for name ,param in model .named_parameters ()}

    local_steps =0 
    num_batches =0 
    eta =opt .param_groups [0 ].get ("lr",None )
    if eta is None or eta <=0 :
        raise ValueError ("Optimizer learning rate must be > 0 for SCAFFOLD.")

    for epoch in range (LOCAL_EPOCHS ):
        for batch_idx ,(data ,target )in enumerate (tqdm (loader ,leave =False )):
            num_batches +=1 

            if isinstance (data ,np .ndarray ):
                data =torch .from_numpy (data )
            if isinstance (target ,np .ndarray ):
                target =torch .from_numpy (target )

            data =data .to (DEVICE )
            target =target .to (DEVICE ).unsqueeze (1 ).float ()

            preds =model (data )
            loss =loss_fn (preds ,target )

            if round_idx ==0 and client_idx ==0 and epoch ==0 and batch_idx ==0 :
                print_batch_sanity (data ,target ,preds ,prefix ="[Sanity]")

            opt .zero_grad ()
            loss .backward ()

            with torch .no_grad ():
                for name ,param in model .named_parameters ():
                    if param .grad is None :
                        continue 
                    if name in global_c and name in client_c :
                        correction =global_c [name ].to (param .grad .device )-client_c [name ].to (param .grad .device )
                        param .grad .add_ (correction )

            torch .nn .utils .clip_grad_norm_ (model .parameters (),GRAD_CLIP_NORM )
            opt .step ()

            total_loss +=float (loss .item ())
            metrics .append (compute_metrics (preds .detach (),target ))
            local_steps +=1 

    client_c_new ={}
    if local_steps >0 :
        for name ,param in model .named_parameters ():
            delta_w =w_init [name ].to (DEVICE )-param .data .clone ().detach ()
            client_c_new [name ]=(
            client_c [name ].to (DEVICE )
            -global_c [name ].to (DEVICE )
            +(delta_w /(eta *local_steps ))
            ).detach ().cpu ().clone ()
    else :
        for name ,_ in model .named_parameters ():
            client_c_new [name ]=client_c [name ].detach ().cpu ().clone ()

    avg_metrics =average_metrics (metrics )
    avg_loss_per_batch =total_loss /max (1 ,num_batches )

    if avg_metrics :
        print ("Train (SCAFFOLD): "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    else :
        print ("Train (SCAFFOLD): no metrics")

    return local_steps ,avg_loss_per_batch ,avg_metrics ,client_c_new 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss ,metrics =0.0 ,[]
    num_batches =0 

    for data ,target in loader :
        num_batches +=1 

        if isinstance (data ,np .ndarray ):
            data =torch .from_numpy (data )
        if isinstance (target ,np .ndarray ):
            target =torch .from_numpy (target )

        data =data .to (DEVICE )
        target =target .to (DEVICE ).unsqueeze (1 ).float ()

        preds =model (data )
        loss =loss_fn (preds ,target )

        total_loss +=float (loss .item ())
        metrics .append (compute_metrics (preds ,target ))

    avg_metrics =average_metrics (metrics )
    if avg_metrics :
        print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    else :
        print (f"{split}: no metrics")

    return total_loss /max (1 ,num_batches ),avg_metrics 

def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Per-client Dice (SCAFFOLD)")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_scaffold.png"))
    plt .close ()

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Per-client IoU (SCAFFOLD)")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_scaffold.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_dice_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test Dice")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Global Test Dice Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_dice_no_bg_scaffold.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_iou_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test IoU")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Global Test IoU Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_iou_no_bg_scaffold.png"))
    plt .close ()


def main ():
    tr_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.5 ),
    A .RandomRotate90 (p =0.5 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])

    val_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])

    global_model =build_model ().to (DEVICE )
    global_test_loader =get_global_test_loader (val_tf ,batch_size =4 )

    global_c ={
    name :torch .zeros_like (param .data ).cpu ()
    for name ,param in global_model .named_parameters ()
    }

    client_cs =[
    {
    name :torch .zeros_like (param .data ).cpu ()
    for name ,param in global_model .named_parameters ()
    }
    for _ in range (NUM_CLIENTS )
    ]

    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models_info =[]
        total_sz =0 

        global_c_for_clients ={k :v .to (DEVICE ).clone ().detach ()for k ,v in global_c .items ()}

        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )

            if USE_SGD :
                opt =optim .SGD (local_model .parameters (),lr =SGD_LR ,momentum =0.0 )
            else :
                opt =optim .AdamW (local_model .parameters (),lr =ADAMW_LR )

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

            print (f"[Client {client_names[i]}] Local training (SCAFFOLD)")
            local_steps ,train_loss ,train_metrics ,client_c_new_cpu =train_local_scaffold (
            train_loader ,
            local_model ,
            loss_fn ,
            opt ,
            global_c =global_c_for_clients ,
            client_c ={k :v .to (DEVICE )for k ,v in client_cs [i ].items ()},
            round_idx =r ,
            client_idx =i 
            )

            client_c_old_cpu =client_cs [i ]
            client_cs [i ]={
            name :client_c_new_cpu [name ].cpu ().clone ().detach ()
            for name in client_c_new_cpu .keys ()
            }

            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            client_dataset_size =len (train_loader .dataset )
            local_models_info .append ((local_model ,client_c_old_cpu ,client_cs [i ],client_dataset_size ))
            total_sz +=client_dataset_size 

        if total_sz ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your split folders and masks.")

        model_list_for_avg =[info [0 ]for info in local_models_info ]
        weights =[info [3 ]for info in local_models_info ]
        norm_weights =[w /total_sz for w in weights ]

        avg_state =average_model_state_dict (model_list_for_avg ,norm_weights )
        global_model .load_state_dict (avg_state ,strict =False )

        c_delta_accum ={
        name :torch .zeros_like (param .data ).cpu ()
        for name ,param in global_model .named_parameters ()
        }

        for idx ,(_ ,client_c_old_cpu ,client_c_new_cpu ,client_sz )in enumerate (local_models_info ):
            w =norm_weights [idx ]
            for name in c_delta_accum .keys ():
                c_delta_accum [name ]+=w *(client_c_new_cpu [name ].cpu ()-client_c_old_cpu [name ].cpu ())

        for name in global_c .keys ():
            global_c [name ]=(global_c [name ]+c_delta_accum [name ]).clone ().detach ()

        global_test_loss ,global_test_metrics =evaluate (
        global_test_loader ,
        global_model ,
        get_loss_fn (DEVICE ),
        split ="Global Test"
        )

        rm ={
        "global_test_loss":global_test_loss ,
        "global_dice_no_bg":global_test_metrics .get ("dice_no_bg",0 ),
        "global_iou_no_bg":global_test_metrics .get ("iou_no_bg",0 ),
        "global_accuracy":global_test_metrics .get ("accuracy",0 ),
        "global_precision":global_test_metrics .get ("precision",0 ),
        "global_recall":global_test_metrics .get ("recall",0 ),
        "global_specificity":global_test_metrics .get ("specificity",0 ),
        }

        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (
            test_img_dirs [i ],
            test_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =False 
            )
            print (f"[Client {client_names[i]}] Test")
            _ ,test_metrics =evaluate (test_loader ,global_model ,get_loss_fn (DEVICE ),split ="Test")
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )

        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

        print (
        f"[GLOBAL TEST AFTER ROUND {r+1}] "
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