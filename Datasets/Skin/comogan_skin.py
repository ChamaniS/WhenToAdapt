import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib 
matplotlib .use ("Agg")

import copy 
import time 
import glob 
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

from PIL import Image ,ImageOps ,ImageEnhance 
import sys 
from unet import UNET 
output_file =r"/xxxxxxxx/skin_comogan.txt"
sys .stdout =open (output_file ,"w")




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

client_names =["HAM10K","PH2","ISIC2017","ISIC2018"]
NUM_CLIENTS =len (client_names )

LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
IMG_SIZE =224 
BATCH_SIZE =4 
SEED =42 

start_time =time .time ()

out_dir ="Outputs_comogan"
os .makedirs (out_dir ,exist_ok =True )

splits_root =r"/xxxxskinlesions/"

client_ext_map ={
"HAM10K":((".jpg",),(".png",)),
"ISIC2017":((".jpg",),(".png",)),
"ISIC2018":((".jpg",),(".png",)),
"PH2":((".jpg",".png",".bmp",".jpeg"),(".png",".bmp",".jpg")),
}

selected_grid_targets ={
"HAM10K":["ISIC_0032968.jpg"],
"PH2":["IMD405.BMP"],
"ISIC2017":["ISIC_0014191.JPG"],
"ISIC2018":["ISIC_0014715.jpg"],
}

def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )
    return path 

def image_list_in_dir (d ):
    exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    if not os .path .isdir (d ):
        return []
    out =[]
    for root ,_ ,files in os .walk (d ):
        for fn in files :
            if fn .lower ().endswith (exts ):
                out .append (os .path .join (root ,fn ))
    return sorted (out )

def find_file_case_insensitive (dirpath ,desired_name ):
    if not os .path .isdir (dirpath ):
        return None 
    desired_lower =desired_name .lower ()
    files =[f for f in os .listdir (dirpath )if f .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
    for f in files :
        if f .lower ()==desired_lower :
            return f 
    desired_base =os .path .splitext (desired_lower )[0 ]
    for f in files :
        if os .path .splitext (f .lower ())[0 ]==desired_base :
            return f 
    return None 

def harmonize_image_stub (p ):
    img =Image .open (p ).convert ("RGB")
    img =ImageOps .autocontrast (img ,cutoff =0 )

    seed =sum (bytearray (os .path .basename (p ).encode ("utf-8")))%1000 
    ctr =0.9 +(seed %11 )*0.02 
    bri =0.95 +((seed //7 )%9 )*0.02 

    img =ImageEnhance .Contrast (img ).enhance (ctr )
    img =ImageEnhance .Brightness (img ).enhance (bri )

    arr =np .array (img ).astype (np .float32 )
    r_off =((seed %11 )-5 )*1.2 
    g_off =(((seed //7 )%9 )-4 )*0.8 
    b_off =(((seed //35 )%7 )-3 )*1.0 

    arr [...,0 ]=np .clip (arr [...,0 ]+r_off ,0 ,255 )
    arr [...,1 ]=np .clip (arr [...,1 ]+g_off ,0 ,255 )
    arr [...,2 ]=np .clip (arr [...,2 ]+b_off ,0 ,255 )

    return Image .fromarray (arr .astype (np .uint8 ))

def harmonize_split_images (client_img_dir ,client_name ,split_name ,out_base ):
    dst_dir =ensure_dir (os .path .join (out_base ,"harmonized",client_name ,split_name ,"images"))

    if not os .path .isdir (client_img_dir ):
        print (f"[HARM] missing source directory: {client_img_dir}")
        return dst_dir 

    imgs =image_list_in_dir (client_img_dir )
    print (f"[HARM] {client_name}/{split_name}: found {len(imgs)} images")

    for idx ,p in enumerate (imgs ):
        try :
            out_img =harmonize_image_stub (p )
            base =os .path .basename (p )
            dst =os .path .join (dst_dir ,base )
            out_img .save (dst )
            if idx <3 :
                print (f"[HARM] wrote {dst}")
        except Exception as e :
            print (f"[HARM] error processing {p}: {e}")

    return dst_dir 

def make_comparison_grid_single_filenames (orig_dir ,hm_dir ,client_name ,out_base ,desired_filenames ):
    base =ensure_dir (os .path .join (out_base ,"ComparisonGrid"))

    for desired in desired_filenames :
        found =find_file_case_insensitive (orig_dir ,desired )
        if found is None :
            print (f"[VIS] '{desired}' not found in {orig_dir} for {client_name}; skipping.")
            continue 

        orig_path =os .path .join (orig_dir ,found )

        hm_found =find_file_case_insensitive (hm_dir ,found )if hm_dir and os .path .isdir (hm_dir )else None 
        hm_path =os .path .join (hm_dir ,hm_found )if hm_found else None 

        try :
            orig =Image .open (orig_path ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE ))
            if hm_path and os .path .exists (hm_path ):
                hm =Image .open (hm_path ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE ))
            else :
                print (f"[VIS] harmonized file for '{found}' not found in {hm_dir}; using original as fallback.")
                hm =orig .copy ()

            orig_np =np .array (orig ).astype (np .uint8 )
            hm_np =np .array (hm ).astype (np .uint8 )

            diff =np .abs (hm_np .astype (np .float32 )-orig_np .astype (np .float32 ))
            amplified =np .clip (diff *3.0 ,0 ,255 ).astype (np .uint8 )
            if amplified .max ()<8 and diff .max ()>0 :
                amplified =(diff /(diff .max ()+1e-8 )*255.0 ).astype (np .uint8 )

            fig ,axs =plt .subplots (3 ,1 ,figsize =(4 ,9 ))
            axs [0 ].imshow (orig_np )
            axs [0 ].axis ("off")
            axs [0 ].set_title (f"Raw: {found}",fontsize =9 )

            axs [1 ].imshow (hm_np )
            axs [1 ].axis ("off")
            axs [1 ].set_title ("CoMoGAN-style harmonized",fontsize =9 )

            axs [2 ].imshow (amplified )
            axs [2 ].axis ("off")
            axs [2 ].set_title ("Amplified diff",fontsize =9 )

            plt .tight_layout ()
            name ,_ =os .path .splitext (found )
            out_png =os .path .join (base ,f"comparison_{client_name}_{name}.png")
            plt .savefig (out_png ,dpi =150 )
            plt .close (fig )
            print (f"[VIS] saved: {out_png}")

        except Exception as e :
            print (f"[VIS] failed for {client_name}, file {found}: {e}")

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


harm_train_img_dirs =[]
harm_val_img_dirs =[]
harm_test_img_dirs =[]

for i ,cname in enumerate (client_names ):
    print (f"\n[HARMONIZING CLIENT] {cname}")
    harm_train_img_dirs .append (harmonize_split_images (train_img_dirs [i ],cname ,"train",out_dir ))
    harm_val_img_dirs .append (harmonize_split_images (val_img_dirs [i ],cname ,"val",out_dir ))
    harm_test_img_dirs .append (harmonize_split_images (test_img_dirs [i ],cname ,"test",out_dir ))

    targets =selected_grid_targets .get (cname ,[])
    if targets :
        make_comparison_grid_single_filenames (
        val_img_dirs [i ],
        harm_val_img_dirs [i ],
        cname ,
        out_dir ,
        targets 
        )

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
        harm_test_img_dirs [i ],
        test_mask_dirs [i ],
        transform =transform ,
        img_exts =img_exts ,
        mask_exts =mask_exts ,
        )
        datasets .append (ds )

    global_test_ds =ConcatDataset (datasets )
    return DataLoader (global_test_ds ,batch_size =batch_size ,shuffle =False )

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

def get_loss_fn (device ):
    return smp .losses .DiceLoss (mode ="binary",from_logits =True ).to (device )

def average_models_weighted (models ,weights ):
    avg_sd =copy .deepcopy (models [0 ].state_dict ())
    for k in avg_sd .keys ():
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ]for i in range (len (models )))
    return avg_sd 

def train_local (loader ,model ,loss_fn ,opt ):
    model .train ()
    total_loss ,metrics =0.0 ,[]

    for _ in range (LOCAL_EPOCHS ):
        for data ,target in tqdm (loader ,leave =False ):
            if isinstance (data ,np .ndarray ):
                data =torch .from_numpy (data )
            if isinstance (target ,np .ndarray ):
                target =torch .from_numpy (target )

            data =data .to (DEVICE )
            target =target .to (DEVICE ).unsqueeze (1 ).float ()

            preds =model (data )
            loss =loss_fn (preds ,target )

            opt .zero_grad ()
            loss .backward ()
            opt .step ()

            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))

    avg_metrics =average_metrics (metrics )
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return (total_loss /len (loader .dataset ))if len (loader .dataset )>0 else 0.0 ,avg_metrics 

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
        target =target .to (DEVICE ).unsqueeze (1 ).float ()

        preds =model (data )
        loss =loss_fn (preds ,target )

        total_loss +=loss .item ()
        metrics .append (compute_metrics (preds ,target ))

    avg_metrics =average_metrics (metrics )
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return (total_loss /len (loader .dataset ))if len (loader .dataset )>0 else 0.0 ,avg_metrics 

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
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_comogan.png"))
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
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_comogan.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_dice_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test Dice")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Global Test Dice Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_dice_no_bg_comogan.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_iou_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test IoU")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Global Test IoU Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_iou_no_bg_comogan.png"))
    plt .close ()

def main ():
    tr_tf =A .Compose ([
    A .Resize (IMG_SIZE ,IMG_SIZE ),
    A .Normalize (mean =[0 ]*3 ,std =[1 ]*3 ),
    ToTensorV2 ()
    ])
    val_tf =tr_tf 

    global_model =UNET (in_channels =3 ,out_channels =1 ).to (DEVICE )
    global_test_loader =get_global_test_loader (val_tf ,batch_size =BATCH_SIZE )

    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models ,weights =[],[]
        total_sz =0 

        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            opt =optim .AdamW (local_model .parameters (),lr =1e-4 )
            loss_fn =get_loss_fn (DEVICE )

            train_loader =get_loader (
            harm_train_img_dirs [i ],
            train_mask_dirs [i ],
            tr_tf ,
            client_name =client_names [i ],
            batch_size =BATCH_SIZE ,
            shuffle =True 
            )
            val_loader =get_loader (
            harm_val_img_dirs [i ],
            val_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =BATCH_SIZE ,
            shuffle =False 
            )

            print (f"[Client {client_names[i]}]")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            local_models .append (local_model )
            sz =len (train_loader .dataset )if hasattr (train_loader .dataset ,"__len__")else 0 
            weights .append (sz )
            total_sz +=sz 

        if total_sz ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your split folders and files.")

        norm_weights =[w /total_sz for w in weights ]
        global_model .load_state_dict (average_models_weighted (local_models ,norm_weights ))

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
            harm_test_img_dirs [i ],
            test_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =BATCH_SIZE ,
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