import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy 
import time 
import glob 
import shutil 
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
from PIL import Image 

from models .UNET import UNET 




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

client_names =["HAM10K","PH2","ISIC2017","ISIC2018"]
NUM_CLIENTS =len (client_names )

LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
start_time =time .time ()

out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )
splits_root =r"C:\Users\csj5\Projects\Data\skinlesions"

USE_FDA =True 
FDA_L =0.05 
reference_client_idx =0 

selected_grid_targets ={
"HAM10K":["ISIC 0032968.jpg"],
"PH2":["IMD 405.BMP"],
"ISIC2017":["ISIC 0014191.JPG"],
"ISIC2018":["ISIC 0014715.jpg"],
}

client_ext_map ={
"HAM10K":((".jpg",),(".png",)),
"ISIC2017":((".jpg",),(".png",)),
"ISIC2018":((".jpg",),(".png",)),
"PH2":((".jpg",".png",".bmp",".jpeg"),(".png",".bmp",".jpg",".jpeg")),
}


def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )

def list_files_with_exts (folder ,exts ):
    files =[]
    for ext in exts :
        files .extend (glob .glob (os .path .join (folder ,f"*{ext}")))
        files .extend (glob .glob (os .path .join (folder ,f"*{ext.upper()}")))
    return sorted (list (set (files )))

def get_exts_for_client (client_name ):
    if client_name in client_ext_map :
        return client_ext_map [client_name ]
    img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    mask_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    return img_exts ,mask_exts 

def normalize_stem (name ):
    stem =os .path .splitext (os .path .basename (name ))[0 ]
    return "".join (ch .lower ()for ch in stem if ch .isalnum ())

def build_image_index (folder ):
    idx ={}
    if not os .path .isdir (folder ):
        return idx 
    for fn in os .listdir (folder ):
        if fn .lower ().endswith ((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
            idx [normalize_stem (fn )]=os .path .join (folder ,fn )
    return idx 

def find_matching_image_across_dirs (target_name ,dirs ):
    key =normalize_stem (target_name )
    for d in dirs :
        idx =build_image_index (d )
        if key in idx :
            return idx [key ]
    return None 

def find_mask_for_image (img_path ,mask_dir ,mask_exts ):
    stem =os .path .splitext (os .path .basename (img_path ))[0 ]

    direct_candidates =[os .path .join (mask_dir ,stem +ext )for ext in mask_exts ]
    for c in direct_candidates :
        if os .path .exists (c ):
            return c 

    alt_candidates =[]
    for ext in mask_exts :
        alt_candidates .append (os .path .join (mask_dir ,stem +"_mask"+ext ))
        alt_candidates .append (os .path .join (mask_dir ,stem +"-mask"+ext ))
        alt_candidates .append (os .path .join (mask_dir ,stem .replace ("_lesion","")+ext ))
        alt_candidates .append (os .path .join (mask_dir ,stem .replace ("-lesion","")+ext ))

    for c in alt_candidates :
        if os .path .exists (c ):
            return c 

    return None 

def load_rgb_uint8 (path ):
    img =Image .open (path ).convert ("RGB")
    return np .array (img ).astype (np .uint8 )

def save_rgb_uint8 (arr ,path ):
    Image .fromarray (arr .astype (np .uint8 )).save (path )




class SkinPairDataset (Dataset ):
    def __init__ (self ,img_dir ,mask_dir ,transform =None ,img_exts =None ,mask_exts =None ):
        self .img_dir =img_dir 
        self .mask_dir =mask_dir 
        self .transform =transform 

        if img_exts is None :
            img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        if mask_exts is None :
            mask_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

        self .img_exts =tuple (e .lower ()for e in img_exts )
        self .mask_exts =tuple (e .lower ()for e in mask_exts )

        image_files =list_files_with_exts (self .img_dir ,self .img_exts )

        pairs =[]
        missing_masks =0 

        for img_path in image_files :
            mask_path =find_mask_for_image (img_path ,self .mask_dir ,self .mask_exts )
            if mask_path is None :
                missing_masks +=1 
                continue 
            pairs .append ((img_path ,mask_path ))

        if len (pairs )==0 :
            raise ValueError (
            f"No image-mask pairs found in:\n  {img_dir}\n  {mask_dir}\n"
            f"Missing masks: {missing_masks}"
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

        mask =(np .asarray (mask )>0 ).astype (np .uint8 )

        if self .transform is not None :
            augmented =self .transform (image =img ,mask =mask )
            img =augmented ["image"]
            mask =augmented ["mask"]
        else :
            img =img .transpose (2 ,0 ,1 ).astype (np .float32 )/255.0 
            mask =np .expand_dims (mask .astype (np .float32 ),0 )

        return img ,mask 




def _get_first_reference_image (ref_dir ,resize_to =None ):
    files =[]
    for ext in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"):
        files .extend (glob .glob (os .path .join (ref_dir ,f"*{ext}")))
        files .extend (glob .glob (os .path .join (ref_dir ,f"*{ext.upper()}")))
    files =sorted (list (set (files )))
    if not files :
        raise ValueError (f"No images found in reference dir: {ref_dir}")

    first =files [0 ]
    ref =Image .open (first ).convert ("RGB")
    if resize_to is not None :
        ref =ref .resize (resize_to ,resample =Image .BILINEAR )
    return np .array (ref ).astype (np .uint8 )

def fda_swap_amplitude (src_img ,ref_img ,L =0.05 ):
    src =src_img .astype (np .float32 )
    ref =ref_img .astype (np .float32 )

    h ,w ,_ =src .shape 

    if ref .shape [0 ]!=h or ref .shape [1 ]!=w :
        ref =np .array (Image .fromarray (ref .astype (np .uint8 )).resize ((w ,h ),resample =Image .BILINEAR )).astype (np .float32 )

    b_h =max (1 ,int (np .floor (h *L )))
    b_w =max (1 ,int (np .floor (w *L )))

    c_h =h //2 
    c_w =w //2 

    out =np .zeros_like (src ,dtype =np .uint8 )

    for ch in range (3 ):
        src_f =np .fft .fft2 (src [:,:,ch ])
        src_fshift =np .fft .fftshift (src_f )
        src_amp =np .abs (src_fshift )
        src_pha =np .angle (src_fshift )

        ref_f =np .fft .fft2 (ref [:,:,ch ])
        ref_fshift =np .fft .fftshift (ref_f )
        ref_amp =np .abs (ref_fshift )

        h1 =max (0 ,c_h -b_h )
        h2 =min (h ,c_h +b_h )
        w1 =max (0 ,c_w -b_w )
        w2 =min (w ,c_w +b_w )

        src_amp [h1 :h2 ,w1 :w2 ]=ref_amp [h1 :h2 ,w1 :w2 ]

        combined =src_amp *np .exp (1j *src_pha )
        combined_ishift =np .fft .ifftshift (combined )
        rec =np .fft .ifft2 (combined_ishift )
        rec =np .real (rec )
        rec =np .clip (rec ,0 ,255 ).astype (np .uint8 )
        out [:,:,ch ]=rec 

    return out 

def create_fda_dataset_for_client (src_img_dir ,src_mask_dir ,dst_img_dir ,dst_mask_dir ,
ref_img ,img_exts ,mask_exts ,L =0.05 ):
    ensure_dir (dst_img_dir )
    ensure_dir (dst_mask_dir )

    dataset =SkinPairDataset (
    src_img_dir ,src_mask_dir ,
    transform =None ,
    img_exts =img_exts ,
    mask_exts =mask_exts 
    )

    for img_path ,mask_path in dataset .pairs :
        img =load_rgb_uint8 (img_path )
        harmonized =fda_swap_amplitude (img ,ref_img ,L =L )

        dst_img_path =os .path .join (dst_img_dir ,os .path .basename (img_path ))
        dst_mask_path =os .path .join (dst_mask_dir ,os .path .basename (mask_path ))

        save_rgb_uint8 (harmonized ,dst_img_path )
        shutil .copy2 (mask_path ,dst_mask_path )

def create_fda_datasets (train_img_dirs ,train_mask_dirs ,
val_img_dirs ,val_mask_dirs ,
test_img_dirs ,test_mask_dirs ,
client_names ,out_base ,reference_client_idx =0 ,L =0.05 ):
    base =os .path .join (out_base ,"FDA_Harmonized")
    ensure_dir (base )

    ref_dir =train_img_dirs [reference_client_idx ]
    print (f"[FDA] Using reference client '{client_names[reference_client_idx]}' from: {ref_dir}")
    ref_img =_get_first_reference_image (ref_dir )

    train_img_out ,train_mask_out =[],[]
    val_img_out ,val_mask_out =[],[]
    test_img_out ,test_mask_out =[],[]

    for i ,cname in enumerate (client_names ):
        img_exts ,mask_exts =get_exts_for_client (cname )
        client_base =os .path .join (base ,cname )

        t_img =os .path .join (client_base ,"train","images")
        t_msk =os .path .join (client_base ,"train","masks")
        v_img =os .path .join (client_base ,"val","images")
        v_msk =os .path .join (client_base ,"val","masks")
        te_img =os .path .join (client_base ,"test","images")
        te_msk =os .path .join (client_base ,"test","masks")

        print (f"[FDA] Harmonizing client: {cname}")
        create_fda_dataset_for_client (train_img_dirs [i ],train_mask_dirs [i ],t_img ,t_msk ,ref_img ,img_exts ,mask_exts ,L =L )
        create_fda_dataset_for_client (val_img_dirs [i ],val_mask_dirs [i ],v_img ,v_msk ,ref_img ,img_exts ,mask_exts ,L =L )
        create_fda_dataset_for_client (test_img_dirs [i ],test_mask_dirs [i ],te_img ,te_msk ,ref_img ,img_exts ,mask_exts ,L =L )

        train_img_out .append (t_img )
        train_mask_out .append (t_msk )
        val_img_out .append (v_img )
        val_mask_out .append (v_msk )
        test_img_out .append (te_img )
        test_mask_out .append (te_msk )

    print ("[FDA] Dataset creation finished.")
    return train_img_out ,train_mask_out ,val_img_out ,val_mask_out ,test_img_out ,test_mask_out 




def save_comparison_grid (original_img_path ,harmonized_img_path ,save_path ,title =None ,diff_amp =4.0 ):
    orig =load_rgb_uint8 (original_img_path )
    harm =load_rgb_uint8 (harmonized_img_path )

    if orig .shape !=harm .shape :
        orig =np .array (Image .fromarray (orig ).resize ((harm .shape [1 ],harm .shape [0 ]),resample =Image .BILINEAR )).astype (np .uint8 )

    diff_color =np .abs (orig .astype (np .int16 )-harm .astype (np .int16 )).astype (np .uint8 )
    diff_amp_img =np .clip (diff_color .astype (np .float32 )*diff_amp ,0 ,255 ).astype (np .uint8 )

    fig ,axs =plt .subplots (3 ,1 ,figsize =(8 ,10 ))
    if title is None :
        title =os .path .basename (original_img_path )

    axs [0 ].imshow (orig )
    axs [0 ].axis ("off")
    axs [0 ].set_title ("Original",fontsize =12 )

    axs [1 ].imshow (harm )
    axs [1 ].axis ("off")
    axs [1 ].set_title ("FDA Harmonized",fontsize =12 )

    axs [2 ].imshow (diff_amp_img )
    axs [2 ].axis ("off")
    axs [2 ].set_title ("Amplified Absolute Difference",fontsize =12 )

    fig .suptitle (title ,fontsize =14 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.96 ])
    ensure_dir (os .path .dirname (save_path ))
    plt .savefig (save_path ,dpi =160 )
    plt .close (fig )

def save_selected_comparison_grids (original_train_img_dirs ,original_val_img_dirs ,original_test_img_dirs ,
harmonized_train_img_dirs ,harmonized_val_img_dirs ,harmonized_test_img_dirs ,
client_names ,out_base ,selected_targets ):
    grids_base =os .path .join (out_base ,"ComparisonGrids")
    ensure_dir (grids_base )

    for i ,cname in enumerate (client_names ):
        targets =selected_targets .get (cname ,[])
        if not targets :
            continue 

        original_dirs =[original_train_img_dirs [i ],original_val_img_dirs [i ],original_test_img_dirs [i ]]
        harmonized_dirs =[harmonized_train_img_dirs [i ],harmonized_val_img_dirs [i ],harmonized_test_img_dirs [i ]]

        client_out =os .path .join (grids_base ,cname )
        ensure_dir (client_out )

        for target in targets :
            orig_path =find_matching_image_across_dirs (target ,original_dirs )
            harm_path =find_matching_image_across_dirs (target ,harmonized_dirs )

            if orig_path is None :
                print (f"[Grid] Could not find original image for '{cname} - {target}'")
                continue 
            if harm_path is None :
                print (f"[Grid] Could not find FDA image for '{cname} - {target}'")
                continue 

            safe_name ="".join (ch if ch .isalnum ()else "_"for ch in os .path .splitext (target )[0 ]).strip ("_")
            save_path =os .path .join (client_out ,f"comparison_grid_{safe_name}.png")
            save_comparison_grid (
            orig_path ,
            harm_path ,
            save_path ,
            title =f"{cname} - {target}",
            diff_amp =4.0 
            )
            print (f"[Grid] Saved: {save_path}")




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

    if target .sum ().item ()==0 :
        dice_no_bg =1.0 if pred .sum ().item ()==0 else 0.0 
        iou_no_bg =1.0 if pred .sum ().item ()==0 else 0.0 
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
        if torch .is_floating_point (avg_sd [k ]):
            avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ]for i in range (len (models )))
        else :
            avg_sd [k ]=models [0 ].state_dict ()[k ].clone ()
    return avg_sd 




def get_loader (img_dir ,mask_dir ,transform ,client_name =None ,batch_size =4 ,shuffle =True ):
    if client_name is not None and client_name in client_ext_map :
        img_exts ,mask_exts =client_ext_map [client_name ]
    else :
        img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        mask_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

    ds =SkinPairDataset (img_dir ,mask_dir ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle ,num_workers =0 ,pin_memory =torch .cuda .is_available ())

def get_global_test_loader (test_img_dirs ,test_mask_dirs ,transform ,batch_size =4 ):
    datasets =[]
    for i ,cname in enumerate (client_names ):
        img_exts ,mask_exts =get_exts_for_client (cname )
        ds =SkinPairDataset (
        test_img_dirs [i ],
        test_mask_dirs [i ],
        transform =transform ,
        img_exts =img_exts ,
        mask_exts =mask_exts 
        )
        datasets .append (ds )

    global_test_ds =ConcatDataset (datasets )
    return DataLoader (global_test_ds ,batch_size =batch_size ,shuffle =False ,num_workers =0 ,pin_memory =torch .cuda .is_available ())




def train_local (loader ,model ,loss_fn ,opt ):
    model .train ()
    total_loss =0.0 
    metrics =[]
    n_steps =0 

    for _ in range (LOCAL_EPOCHS ):
        for batch in tqdm (loader ,leave =False ):
            if isinstance (batch ,(list ,tuple ))and len (batch )>=2 :
                data ,target =batch [0 ],batch [1 ]
            else :
                raise RuntimeError ("Unexpected batch format in train_local")

            if target .dim ()==3 :
                target =target .unsqueeze (1 ).float ()
            elif target .dim ()==4 :
                target =target .float ()
            else :
                raise RuntimeError (f"Unexpected target shape: {target.shape}")

            data =data .to (DEVICE )
            target =target .to (DEVICE )

            preds =model (data )
            loss =loss_fn (preds ,target )

            opt .zero_grad ()
            loss .backward ()
            opt .step ()

            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))
            n_steps +=1 

    avg_metrics =average_metrics (metrics )
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /max (1 ,n_steps ),avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss =0.0 
    metrics =[]
    n_steps =0 

    for batch in loader :
        if isinstance (batch ,(list ,tuple ))and len (batch )>=2 :
            data ,target =batch [0 ],batch [1 ]
        else :
            raise RuntimeError ("Unexpected batch format in evaluate")

        if target .dim ()==3 :
            target =target .unsqueeze (1 ).float ()
        elif target .dim ()==4 :
            target =target .float ()
        else :
            raise RuntimeError (f"Unexpected target shape: {target.shape}")

        data =data .to (DEVICE )
        target =target .to (DEVICE )

        preds =model (data )
        loss =loss_fn (preds ,target )

        total_loss +=loss .item ()
        metrics .append (compute_metrics (preds ,target ))
        n_steps +=1 

    avg_metrics =average_metrics (metrics )if metrics else {}
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    return total_loss /max (1 ,n_steps ),avg_metrics 




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
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_fda.png"))
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
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_fda.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_dice_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test Dice")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Global Test Dice Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_dice_no_bg_fda.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_iou_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global test IoU")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Global Test IoU Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_iou_no_bg_fda.png"))
    plt .close ()




def main ():
    required_subpaths =[
    ("train","images"),("train","masks"),
    ("val","images"),("val","masks"),
    ("test","images"),("test","masks"),
    ]

    train_img_dirs =[]
    train_mask_dirs =[]
    val_img_dirs =[]
    val_mask_dirs =[]
    test_img_dirs =[]
    test_mask_dirs =[]

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

    tr_tf =A .Compose ([
    A .RandomRotate90 (p =0.5 ),
    A .HorizontalFlip (p =0.5 ),
    A .VerticalFlip (p =0.2 ),
    A .ShiftScaleRotate (shift_limit =0.0625 ,scale_limit =0.1 ,rotate_limit =15 ,p =0.5 ),
    A .RandomBrightnessContrast (brightness_limit =0.2 ,contrast_limit =0.2 ,p =0.5 ),
    A .HueSaturationValue (hue_shift_limit =10 ,sat_shift_limit =15 ,val_shift_limit =10 ,p =0.3 ),
    A .RandomGamma (gamma_limit =(80 ,120 ),p =0.3 ),
    A .Resize (224 ,224 ),
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 ()
    ])

    val_tf =A .Compose ([
    A .Resize (224 ,224 ),
    A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),
    ToTensorV2 ()
    ])




    if USE_FDA :
        (train_img_dirs_used ,train_mask_dirs_used ,
        val_img_dirs_used ,val_mask_dirs_used ,
        test_img_dirs_used ,test_mask_dirs_used )=create_fda_datasets (
        train_img_dirs ,train_mask_dirs ,
        val_img_dirs ,val_mask_dirs ,
        test_img_dirs ,test_mask_dirs ,
        client_names ,out_dir ,
        reference_client_idx =reference_client_idx ,
        L =FDA_L 
        )
    else :
        train_img_dirs_used =train_img_dirs 
        train_mask_dirs_used =train_mask_dirs 
        val_img_dirs_used =val_img_dirs 
        val_mask_dirs_used =val_mask_dirs 
        test_img_dirs_used =test_img_dirs 
        test_mask_dirs_used =test_mask_dirs 




    save_selected_comparison_grids (
    original_train_img_dirs =train_img_dirs ,
    original_val_img_dirs =val_img_dirs ,
    original_test_img_dirs =test_img_dirs ,
    harmonized_train_img_dirs =train_img_dirs_used ,
    harmonized_val_img_dirs =val_img_dirs_used ,
    harmonized_test_img_dirs =test_img_dirs_used ,
    client_names =client_names ,
    out_base =out_dir ,
    selected_targets =selected_grid_targets 
    )




    global_test_loader =get_global_test_loader (
    test_img_dirs_used ,
    test_mask_dirs_used ,
    val_tf ,
    batch_size =4 
    )

    global_model =UNET (in_channels =3 ,num_classes =1 ).to (DEVICE )
    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")

        local_models =[]
        weights =[]
        total_sz =0 

        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            opt =optim .AdamW (local_model .parameters (),lr =1e-4 )
            loss_fn =get_loss_fn (DEVICE )

            train_loader =get_loader (
            train_img_dirs_used [i ],
            train_mask_dirs_used [i ],
            tr_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =True 
            )
            val_loader =get_loader (
            val_img_dirs_used [i ],
            val_mask_dirs_used [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =False 
            )

            print (f"\n[Client {client_names[i]}]")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            local_models .append (local_model )
            sz =len (train_loader .dataset )
            weights .append (sz )
            total_sz +=sz 

        if total_sz ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your split folders.")

        norm_weights =[w /total_sz for w in weights ]
        global_model .load_state_dict (average_models_weighted (local_models ,norm_weights ))

        rm ={
        "global_test_loss":0.0 ,
        }

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

        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (
            test_img_dirs_used [i ],
            test_mask_dirs_used [i ],
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