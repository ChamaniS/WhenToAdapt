import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy 
import time 
import glob 
import shutil 
import random 
import stat 
from PIL import Image 

import cv2 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,Dataset ,ConcatDataset ,RandomSampler 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 
from models .UNET import UNET 


DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"

client_names =["HAM10K","PH2","ISIC2017","ISIC2018"]
NUM_CLIENTS =len (client_names )
reference_idx =0 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
CUT_EPOCHS =60 
CUT_BATCH_SIZE =4 
CUT_LR_G =2e-4 
CUT_LR_D =2e-4 
CUT_LAMBDA_NCE =1.0 
CUT_NCE_LAYERS =[1 ,2 ,3 ,4 ]
CUT_NUM_PATCHES =256 

IMG_SIZE =224 
start_time =time .time ()
out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )
splits_root =r"C:\Users\csj5\Projects\Data\skinlesions"

client_ext_map ={
"HAM10K":((".jpg",),(".png",)),
"ISIC2017":((".jpg",),(".png",)),
"ISIC2018":((".jpg",),(".png",)),
"PH2":((".jpg",".png",".bmp",".jpeg"),(".png",".bmp",".jpg",".jpeg")),
}

cut_root =os .path .join (out_dir ,"CUT")
harm_root =os .path .join (out_dir ,"CUT_Harmonized")
os .makedirs (cut_root ,exist_ok =True )
os .makedirs (harm_root ,exist_ok =True )


def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )

def copy_tree_force (src ,dst ):
    if not os .path .exists (src ):
        raise FileNotFoundError (f"Source not found: {src}")
    if os .path .exists (dst ):
        shutil .rmtree (dst ,onerror =_on_rm_error )
    shutil .copytree (src ,dst )

def _on_rm_error (func ,path ,exc_info ):
    os .chmod (path ,stat .S_IWRITE )
    func (path )

def list_image_files (folder ):
    exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    files =[]
    for p in glob .glob (os .path .join (folder ,"*")):
        if os .path .isfile (p )and p .lower ().endswith (exts ):
            files .append (p )
    return sorted (files )

def find_mask_by_stem (mask_dir ,stem ,mask_exts ):
    mask_exts =tuple (e .lower ()for e in mask_exts )
    candidates =[]

    for ext in mask_exts :
        candidates .append (os .path .join (mask_dir ,stem +ext ))
        candidates .append (os .path .join (mask_dir ,stem +"_mask"+ext ))
        candidates .append (os .path .join (mask_dir ,stem +"-mask"+ext ))


    candidates .append (os .path .join (mask_dir ,stem .replace ("_lesion","")+".png"))
    candidates .append (os .path .join (mask_dir ,stem .replace ("_lesion","")+".jpg"))

    for c in candidates :
        if os .path .exists (c ):
            return c 
    return None 

def save_pil (arr ,path ):
    Image .fromarray (arr ).save (path )

def _mask_to_uint8 (mask_tensor ):
    m =mask_tensor .detach ().cpu ().numpy ()
    if m .ndim ==3 :
        m =np .squeeze (m ,axis =0 )
    m =(m >0.5 ).astype (np .uint8 )*255 
    return m 

def _image_tensor_to_uint8 (img_tensor ):
    arr =img_tensor .detach ().cpu ().numpy ()
    if arr .ndim ==3 :
        arr =arr .transpose (1 ,2 ,0 )
    arr =np .clip (arr *255.0 ,0 ,255 ).astype (np .uint8 )
    return arr 





class SkinPairDataset (Dataset ):
    """
    Pairs image and mask by filename stem.
    Returns:
        image: Tensor [C,H,W] if transform is used
        mask:  Tensor [H,W] or [1,H,W] depending on transform
    """
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

        img_files =[]
        for ext in self .img_exts :
            img_files .extend (glob .glob (os .path .join (self .img_dir ,f"*{ext}")))
        img_files =sorted (img_files )

        pairs =[]
        missing =0 
        for img_path in img_files :
            stem =os .path .splitext (os .path .basename (img_path ))[0 ]
            mask_path =find_mask_by_stem (self .mask_dir ,stem ,self .mask_exts )
            if mask_path is None :
                missing +=1 
                continue 
            pairs .append ((img_path ,mask_path ))

        if len (pairs )==0 :
            raise ValueError (f"No image-mask pairs found in {img_dir} / {mask_dir}")

        self .pairs =pairs 
        if missing >0 :
            print (f"Warning: {missing} images in {img_dir} had no matching masks and were skipped.")

    def __len__ (self ):
        return len (self .pairs )

    def __getitem__ (self ,idx ):
        img_path ,mask_path =self .pairs [idx ]

        img =cv2 .imread (img_path ,cv2 .IMREAD_COLOR )
        if img is None :
            raise RuntimeError (f"Failed to read image: {img_path}")
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        mask =cv2 .imread (mask_path ,cv2 .IMREAD_UNCHANGED )
        if mask is None :
            raise RuntimeError (f"Failed to read mask: {mask_path}")
        if mask .ndim ==3 :
            mask =cv2 .cvtColor (mask ,cv2 .COLOR_BGR2GRAY )
        mask =(mask >0 ).astype (np .uint8 )

        if self .transform is not None :
            out =self .transform (image =img ,mask =mask )
            img =out ["image"]
            mask =out ["mask"]
        else :
            img =img .transpose (2 ,0 ,1 ).astype (np .float32 )/255.0 
            mask =np .expand_dims (mask .astype (np .float32 ),0 )

        return img ,mask 


class ImageFolderSimple (Dataset ):
    """
    Image-only dataset for CUT training.
    Returns a tensor in [0,1].
    """
    def __init__ (self ,folder ,size =(224 ,224 ),augment =False ):
        self .files =list_image_files (folder )
        if len (self .files )==0 :
            raise ValueError (f"No images found in {folder}")
        self .size =size 
        self .augment =augment 

        self .base_tf =A .Compose ([
        A .Resize (size [0 ],size [1 ]),
        ToTensorV2 ()
        ])

        self .aug_tf =A .Compose ([
        A .HorizontalFlip (p =0.5 ),
        A .RandomRotate90 (p =0.5 ),
        A .ShiftScaleRotate (shift_limit =0.05 ,scale_limit =0.05 ,rotate_limit =10 ,p =0.5 ),
        ])

    def __len__ (self ):
        return len (self .files )

    def __getitem__ (self ,idx ):
        p =self .files [idx %len (self .files )]
        img =cv2 .imread (p ,cv2 .IMREAD_COLOR )
        if img is None :
            raise RuntimeError (f"Failed to read image: {p}")
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        if self .augment :
            aug =self .aug_tf (image =img )
            img =aug ["image"]

        out =self .base_tf (image =img )
        img =out ["image"].float ()/255.0 
        return img 





def weights_init_normal (m ):
    classname =m .__class__ .__name__ 
    if classname .find ("Conv")!=-1 :
        if hasattr (m ,"weight")and m .weight is not None :
            nn .init .normal_ (m .weight .data ,0.0 ,0.02 )
    elif classname .find ("BatchNorm2d")!=-1 or classname .find ("InstanceNorm2d")!=-1 :
        if hasattr (m ,"weight")and m .weight is not None :
            nn .init .normal_ (m .weight .data ,1.0 ,0.02 )
        if hasattr (m ,"bias")and m .bias is not None :
            nn .init .constant_ (m .bias .data ,0.0 )

def conv_block (in_ch ,out_ch ,down =True ):
    if down :
        return nn .Sequential (
        nn .Conv2d (in_ch ,out_ch ,kernel_size =4 ,stride =2 ,padding =1 ,bias =False ),
        nn .InstanceNorm2d (out_ch ),
        nn .ReLU (inplace =True )
        )
    else :
        return nn .Sequential (
        nn .ConvTranspose2d (in_ch ,out_ch ,kernel_size =4 ,stride =2 ,padding =1 ,bias =False ),
        nn .InstanceNorm2d (out_ch ),
        nn .ReLU (inplace =True )
        )

class CUTGenerator (nn .Module ):
    """
    Simple encoder-decoder generator used for CUT-style training.
    Output is tanh -> [-1, 1].
    """
    def __init__ (self ,in_ch =3 ,ngf =64 ):
        super ().__init__ ()
        self .enc1 =nn .Sequential (
        nn .Conv2d (in_ch ,ngf ,kernel_size =7 ,stride =1 ,padding =3 ,bias =False ),
        nn .InstanceNorm2d (ngf ),
        nn .ReLU (inplace =True )
        )
        self .enc2 =conv_block (ngf ,ngf *2 ,down =True )
        self .enc3 =conv_block (ngf *2 ,ngf *4 ,down =True )
        self .enc4 =conv_block (ngf *4 ,ngf *8 ,down =True )
        self .enc5 =conv_block (ngf *8 ,ngf *8 ,down =True )

        self .dec5 =conv_block (ngf *8 ,ngf *8 ,down =False )
        self .dec4 =conv_block (ngf *16 ,ngf *4 ,down =False )
        self .dec3 =conv_block (ngf *8 ,ngf *2 ,down =False )
        self .dec2 =conv_block (ngf *4 ,ngf ,down =False )
        self .final =nn .Sequential (
        nn .Conv2d (ngf *2 ,in_ch ,kernel_size =7 ,stride =1 ,padding =3 ),
        nn .Tanh ()
        )

    def encode (self ,x ):
        f1 =self .enc1 (x )
        f2 =self .enc2 (f1 )
        f3 =self .enc3 (f2 )
        f4 =self .enc4 (f3 )
        f5 =self .enc5 (f4 )
        return [f1 ,f2 ,f3 ,f4 ,f5 ]

    def decode (self ,feats ):
        f1 ,f2 ,f3 ,f4 ,f5 =feats 
        d5 =self .dec5 (f5 )
        d5 =torch .cat ([d5 ,f4 ],dim =1 )

        d4 =self .dec4 (d5 )
        d4 =torch .cat ([d4 ,f3 ],dim =1 )

        d3 =self .dec3 (d4 )
        d3 =torch .cat ([d3 ,f2 ],dim =1 )

        d2 =self .dec2 (d3 )
        d2 =torch .cat ([d2 ,f1 ],dim =1 )

        out =self .final (d2 )
        return out 

    def forward (self ,x ):
        feats =self .encode (x )
        out =self .decode (feats )
        return out ,feats 


class PatchDiscriminator (nn .Module ):
    def __init__ (self ,in_ch =3 ,ndf =64 ):
        super ().__init__ ()
        self .model =nn .Sequential (
        nn .Conv2d (in_ch ,ndf ,4 ,2 ,1 ),
        nn .LeakyReLU (0.2 ,inplace =True ),

        nn .Conv2d (ndf ,ndf *2 ,4 ,2 ,1 ),
        nn .InstanceNorm2d (ndf *2 ),
        nn .LeakyReLU (0.2 ,inplace =True ),

        nn .Conv2d (ndf *2 ,ndf *4 ,4 ,2 ,1 ),
        nn .InstanceNorm2d (ndf *4 ),
        nn .LeakyReLU (0.2 ,inplace =True ),

        nn .Conv2d (ndf *4 ,1 ,4 ,1 ,1 )
        )

    def forward (self ,x ):
        return self .model (x )


class PatchSampleMLP (nn .Module ):
    """
    Small 1x1 MLP head for PatchNCE.
    """
    def __init__ (self ,in_channels ,hidden_dim =256 ):
        super ().__init__ ()
        self .net =nn .Sequential (
        nn .Conv2d (in_channels ,hidden_dim ,kernel_size =1 ,bias =True ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (hidden_dim ,hidden_dim ,kernel_size =1 ,bias =True )
        )

    def forward (self ,feat ):
        return self .net (feat )


class PatchNCELoss (nn .Module ):
    def __init__ (self ,temperature =0.07 ):
        super ().__init__ ()
        self .temperature =temperature 
        self .ce =nn .CrossEntropyLoss ()

    def forward (self ,q ,k ):
        q =nn .functional .normalize (q ,dim =1 )
        k =nn .functional .normalize (k ,dim =1 )
        logits =torch .matmul (q ,k .T )/self .temperature 
        labels =torch .arange (logits .size (0 ),device =logits .device )
        return self .ce (logits ,labels )


def sample_patches (feat_map ,n_patches ):
    """
    feat_map: [B,C,H,W]
    returns flattened patch matrix [B*n_patches, C]
    """
    B ,C ,H ,W =feat_map .shape 
    total =H *W 
    num =min (n_patches ,total )
    idxs =torch .randperm (total ,device =feat_map .device )[:num ]
    flat =feat_map .view (B ,C ,-1 )[:,:,idxs ]
    flat =flat .permute (0 ,2 ,1 ).reshape (-1 ,C )
    return flat 





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

    dice_no_bg =max (0.0 ,min (1.0 ,dice_no_bg ))
    iou_no_bg =max (0.0 ,min (1.0 ,iou_no_bg ))

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
    total_loss =0.0 
    metrics =[]

    for _ in range (LOCAL_EPOCHS ):
        for data ,target in tqdm (loader ,leave =False ):
            if target .dim ()==3 :
                target =target .unsqueeze (1 )
            data =data .to (DEVICE )
            target =target .to (DEVICE ).float ()

            preds =model (data )
            loss =loss_fn (preds ,target )

            opt .zero_grad ()
            loss .backward ()
            opt .step ()

            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))

    avg_metrics =average_metrics (metrics )
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    avg_loss =total_loss /max (1 ,len (loader ))
    return avg_loss ,avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss =0.0 
    metrics =[]
    n_batches =0 

    for data ,target in loader :
        if target .dim ()==3 :
            target =target .unsqueeze (1 )
        data =data .to (DEVICE )
        target =target .to (DEVICE ).float ()

        preds =model (data )
        loss =loss_fn (preds ,target )

        total_loss +=loss .item ()
        metrics .append (compute_metrics (preds ,target ))
        n_batches +=1 

    avg_metrics =average_metrics (metrics )
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in avg_metrics .items ()]))
    avg_loss =total_loss /max (1 ,n_batches )
    return avg_loss ,avg_metrics 





def get_loader (img_dir ,mask_dir ,transform ,client_name =None ,batch_size =4 ,shuffle =True ):
    if client_name is not None and client_name in client_ext_map :
        img_exts ,mask_exts =client_ext_map [client_name ]
    else :
        img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        mask_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

    ds =SkinPairDataset (img_dir ,mask_dir ,transform =transform ,img_exts =img_exts ,mask_exts =mask_exts )
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle ,num_workers =0 )

def get_global_test_loader (test_img_dirs ,test_mask_dirs ,transform ,batch_size =4 ):
    datasets =[]
    for i ,cname in enumerate (client_names ):
        if cname in client_ext_map :
            img_exts ,mask_exts =client_ext_map [cname ]
        else :
            img_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
            mask_exts =(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

        ds =SkinPairDataset (
        test_img_dirs [i ],
        test_mask_dirs [i ],
        transform =transform ,
        img_exts =img_exts ,
        mask_exts =mask_exts ,
        )
        datasets .append (ds )

    global_ds =ConcatDataset (datasets )
    return DataLoader (global_ds ,batch_size =batch_size ,shuffle =False ,num_workers =0 )





def train_cut (domainA_dir ,domainB_dir ,save_dir ,epochs =CUT_EPOCHS ,device =DEVICE ,
lambda_nce =CUT_LAMBDA_NCE ,nce_layers =CUT_NCE_LAYERS ,n_patches =CUT_NUM_PATCHES ,
lr_g =CUT_LR_G ,lr_d =CUT_LR_D ,batch_size =CUT_BATCH_SIZE ):
    """
    Train CUT-style generator mapping A -> B.
    This is a practical CUT approximation:
    GAN loss + PatchNCE loss.
    """
    os .makedirs (save_dir ,exist_ok =True )

    dsA =ImageFolderSimple (domainA_dir ,size =(IMG_SIZE ,IMG_SIZE ),augment =True )
    dsB =ImageFolderSimple (domainB_dir ,size =(IMG_SIZE ,IMG_SIZE ),augment =True )

    max_samples =max (len (dsA ),len (dsB ))
    samplerA =RandomSampler (dsA ,replacement =True ,num_samples =max_samples )
    samplerB =RandomSampler (dsB ,replacement =True ,num_samples =max_samples )

    loaderA =DataLoader (dsA ,batch_size =batch_size ,sampler =samplerA ,drop_last =True ,num_workers =0 )
    loaderB =DataLoader (dsB ,batch_size =batch_size ,sampler =samplerB ,drop_last =True ,num_workers =0 )

    G =CUTGenerator ().to (device )
    D =PatchDiscriminator ().to (device )
    G .apply (weights_init_normal )
    D .apply (weights_init_normal )


    ngf =64 
    layer_channels ={
    1 :ngf ,
    2 :ngf *2 ,
    3 :ngf *4 ,
    4 :ngf *8 ,
    5 :ngf *8 
    }

    nce_mlps ={}
    for l in nce_layers :
        nce_mlps [str (l )]=PatchSampleMLP (layer_channels [l ],hidden_dim =256 ).to (device )

    nce_loss_fn =PatchNCELoss ().to (device )
    gan_loss =nn .MSELoss ().to (device )

    mlp_params =[]
    for m in nce_mlps .values ():
        mlp_params .extend (list (m .parameters ()))

    opt_G =optim .Adam (list (G .parameters ())+mlp_params ,lr =lr_g ,betas =(0.5 ,0.999 ))
    opt_D =optim .Adam (D .parameters (),lr =lr_d ,betas =(0.5 ,0.999 ))

    real_label =0.9 
    fake_label =0.0 

    print (f"[CUT] Training {domainA_dir} -> {domainB_dir} for {epochs} epochs")

    iterB =iter (loaderB )
    for epoch in range (epochs ):
        loop =tqdm (loaderA ,desc =f"CUT Epoch {epoch+1}/{epochs}")
        for real_A in loop :
            try :
                real_B =next (iterB )
            except StopIteration :
                iterB =iter (loaderB )
                real_B =next (iterB )

            real_A =real_A .to (device )
            real_B =real_B .to (device )

            inp_A =real_A *2.0 -1.0 
            inp_B =real_B *2.0 -1.0 


            fake_B ,feats_A =G (inp_A )
            feats_fake =G .encode (fake_B )




            opt_D .zero_grad ()

            pred_real =D (inp_B )
            valid =torch .full_like (pred_real ,real_label ,device =device )
            loss_D_real =gan_loss (pred_real ,valid )

            pred_fake =D (fake_B .detach ())
            fake =torch .full_like (pred_fake ,fake_label ,device =device )
            loss_D_fake =gan_loss (pred_fake ,fake )

            loss_D =0.5 *(loss_D_real +loss_D_fake )
            loss_D .backward ()
            opt_D .step ()




            opt_G .zero_grad ()

            pred_fake_for_g =D (fake_B )
            valid_for_g =torch .full_like (pred_fake_for_g ,real_label ,device =device )
            loss_G_GAN =gan_loss (pred_fake_for_g ,valid_for_g )

            loss_NCE =0.0 
            for l in nce_layers :
                idx =l -1 
                feat_q =feats_A [idx ]
                feat_k =feats_fake [idx ]

                proj =nce_mlps [str (l )]
                q_proj =proj (feat_q )
                k_proj =proj (feat_k )

                q_flat =sample_patches (q_proj ,n_patches )
                k_flat =sample_patches (k_proj ,n_patches )


                min_len =min (q_flat .size (0 ),k_flat .size (0 ))
                q_flat =q_flat [:min_len ]
                k_flat =k_flat [:min_len ]

                loss_NCE =loss_NCE +nce_loss_fn (q_flat ,k_flat )

            loss_NCE =loss_NCE *lambda_nce 
            loss_G =loss_G_GAN +loss_NCE 
            loss_G .backward ()
            opt_G .step ()

            loop .set_postfix ({
            "loss_G":float (loss_G .item ()),
            "loss_D":float (loss_D .item ()),
            "loss_NCE":float (loss_NCE .item ())if isinstance (loss_NCE ,torch .Tensor )else float (loss_NCE )
            })

        ckpt ={
        "G":G .state_dict (),
        "D":D .state_dict (),
        "mlps":{k :v .state_dict ()for k ,v in nce_mlps .items ()},
        "epoch":epoch +1 
        }
        torch .save (ckpt ,os .path .join (save_dir ,f"cut_epoch_{epoch+1}.pth"))

    final_path =os .path .join (save_dir ,"cut_final.pth")
    torch .save (
    {
    "G":G .state_dict (),
    "mlps":{k :v .state_dict ()for k ,v in nce_mlps .items ()},
    "epoch":epochs 
    },
    final_path 
    )
    print (f"[CUT] Finished and saved to {final_path}")
    return G .cpu ()

def load_cut_generator (ckpt_path ,device =DEVICE ):
    G =CUTGenerator ().to (device )
    ck =torch .load (ckpt_path ,map_location =device )
    if "G"in ck :
        G .load_state_dict (ck ["G"])
    else :
        G .load_state_dict (ck )
    G .eval ()
    return G 

@torch .no_grad ()
def harmonize_folder_with_generator (generator ,src_img_dir ,dst_img_dir ,
mask_src_dir =None ,mask_dst_dir =None ,
device =DEVICE ,size =(IMG_SIZE ,IMG_SIZE )):
    """
    Applies generator to all images in src_img_dir and saves harmonized images to dst_img_dir.
    Masks are copied unchanged by stem to mask_dst_dir.
    """
    ensure_dir (dst_img_dir )
    if mask_src_dir is not None and mask_dst_dir is not None :
        ensure_dir (mask_dst_dir )

    tf =A .Compose ([A .Resize (size [0 ],size [1 ]),ToTensorV2 ()])
    generator =generator .to (device )
    generator .eval ()

    image_files =list_image_files (src_img_dir )
    if len (image_files )==0 :
        raise ValueError (f"No images found in {src_img_dir}")

    for p in image_files :
        stem =os .path .splitext (os .path .basename (p ))[0 ]

        img =cv2 .imread (p ,cv2 .IMREAD_COLOR )
        if img is None :
            raise RuntimeError (f"Failed to read image: {p}")
        img =cv2 .cvtColor (img ,cv2 .COLOR_BGR2RGB )

        out =tf (image =img )
        x =out ["image"].float ().unsqueeze (0 ).to (device )/255.0 
        x =x *2.0 -1.0 

        fake ,_ =generator (x )
        fake =((fake .squeeze (0 ).clamp (-1 ,1 )+1.0 )/2.0 ).cpu ()

        fake_np =(fake .numpy ().transpose (1 ,2 ,0 )*255.0 ).astype (np .uint8 )
        save_pil (fake_np ,os .path .join (dst_img_dir ,os .path .basename (p )))

        if mask_src_dir is not None and mask_dst_dir is not None :

            mask_path =find_mask_by_stem (mask_src_dir ,stem ,(".png",".jpg",".jpeg",".bmp",".tif",".tiff"))
            if mask_path is None :

                continue 
            shutil .copy (mask_path ,os .path .join (mask_dst_dir ,os .path .basename (mask_path )))





def plot_metrics (round_metrics ,out_dir ):
    rounds =list (range (1 ,len (round_metrics )+1 ))

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_dice_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =client_names [cid ])
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Per-client Dice")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_cut_fedavg.png"))
    plt .close ()

    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =client_names [cid ])
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Per-client IoU")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_cut_fedavg.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_dice_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global Dice")
    plt .xlabel ("Global Round")
    plt .ylabel ("Dice")
    plt .title ("Global Dice Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_dice_no_bg_cut_fedavg.png"))
    plt .close ()

    plt .figure ()
    vals =[rm .get ("global_iou_no_bg",0 )for rm in round_metrics ]
    plt .plot (rounds ,vals ,label ="Global IoU")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Global IoU Across All Clients")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"global_iou_no_bg_cut_fedavg.png"))
    plt .close ()





def save_comparison_grid (original_dir ,hm_dir ,client_name ,out_base ,n_samples =7 ):
    base_dest =os .path .join (out_base ,"ComparisonGrid",client_name )
    ensure_dir (base_dest )
    diff_dest =os .path .join (base_dest ,"diffs")
    ensure_dir (diff_dest )

    orig_files =list_image_files (original_dir )[:n_samples ]
    if len (orig_files )==0 :
        return 

    top_imgs ,mid_imgs ,diff_imgs ,titles =[],[],[],[]

    for orig_path in orig_files :
        fname =os .path .basename (orig_path )
        hm_path =os .path .join (hm_dir ,fname )
        if not os .path .exists (hm_path ):
            continue 

        orig =np .array (Image .open (orig_path ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE )))
        hm =np .array (Image .open (hm_path ).convert ("RGB").resize ((IMG_SIZE ,IMG_SIZE )))
        diff =np .clip (np .abs (orig .astype (np .int32 )-hm .astype (np .int32 ))*4 ,0 ,255 ).astype (np .uint8 )

        top_imgs .append (orig )
        mid_imgs .append (hm )
        diff_imgs .append (diff )
        titles .append (fname )

        save_pil (orig ,os .path .join (base_dest ,f"orig_{fname}"))
        save_pil (hm ,os .path .join (base_dest ,f"hm_{fname}"))
        save_pil (diff ,os .path .join (diff_dest ,f"diff_{fname}"))

    n =len (top_imgs )
    if n ==0 :
        return 

    fig ,axs =plt .subplots (3 ,n ,figsize =(3 *n ,6 ))
    if n ==1 :
        axs =np .array ([[axs [0 ]],[axs [1 ]],[axs [2 ]]])

    for i in range (n ):
        axs [0 ,i ].imshow (top_imgs [i ])
        axs [0 ,i ].axis ("off")
        axs [0 ,i ].set_title (titles [i ][:12 ])

        axs [1 ,i ].imshow (mid_imgs [i ])
        axs [1 ,i ].axis ("off")

        axs [2 ,i ].imshow (diff_imgs [i ])
        axs [2 ,i ].axis ("off")

    fig .suptitle (f"Harmonized (CUT) vs Original: {client_name}",fontsize =16 ,y =0.98 )
    fig .text (0.01 ,0.82 ,"Original",fontsize =12 ,va ="center",rotation ="vertical")
    fig .text (0.01 ,0.50 ,"Harmonized",fontsize =12 ,va ="center",rotation ="vertical")
    fig .text (0.01 ,0.18 ,"Amplified Difference",fontsize =12 ,va ="center",rotation ="vertical")
    plt .tight_layout ()
    plt .savefig (os .path .join (base_dest ,f"comparison_{client_name}.png"))
    plt .close ()
    print (f"Saved comparison grid for {client_name} at {base_dest}")





def main ():

    tr_tf =A .Compose ([
    A .Resize (IMG_SIZE ,IMG_SIZE ),
    A .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ]),
    ToTensorV2 ()
    ])
    val_tf =A .Compose ([
    A .Resize (IMG_SIZE ,IMG_SIZE ),
    A .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ]),
    ToTensorV2 ()
    ])




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




    cut_models ={}
    for i in range (NUM_CLIENTS ):
        if i ==reference_idx :
            print (f"[CUT] Skipping reference client: {client_names[i]}")
            continue 

        save_dir =os .path .join (cut_root ,f"{client_names[i]}_to_{client_names[reference_idx]}")
        final_ckpt =os .path .join (save_dir ,"cut_final.pth")

        if os .path .exists (final_ckpt ):
            print (f"[CUT] Loading existing generator for {client_names[i]} -> {client_names[reference_idx]}")
            cut_models [i ]=load_cut_generator (final_ckpt ,device =DEVICE ).cpu ()
        else :
            G =train_cut (
            domainA_dir =train_img_dirs [i ],
            domainB_dir =train_img_dirs [reference_idx ],
            save_dir =save_dir ,
            epochs =CUT_EPOCHS ,
            device =DEVICE ,
            lambda_nce =CUT_LAMBDA_NCE ,
            nce_layers =CUT_NCE_LAYERS ,
            n_patches =CUT_NUM_PATCHES ,
            lr_g =CUT_LR_G ,
            lr_d =CUT_LR_D ,
            batch_size =CUT_BATCH_SIZE 
            )
            cut_models [i ]=G .cpu ()




    hm_train_dirs =[]
    hm_train_mask_dirs =[]
    hm_val_dirs =[]
    hm_val_mask_dirs =[]
    hm_test_dirs =[]
    hm_test_mask_dirs =[]

    for i in range (NUM_CLIENTS ):
        cname =client_names [i ]
        dst_base =os .path .join (harm_root ,cname )

        dst_train =os .path .join (dst_base ,"train","images")
        dst_train_mask =os .path .join (dst_base ,"train","masks")
        dst_val =os .path .join (dst_base ,"val","images")
        dst_val_mask =os .path .join (dst_base ,"val","masks")
        dst_test =os .path .join (dst_base ,"test","images")
        dst_test_mask =os .path .join (dst_base ,"test","masks")

        if i ==reference_idx :
            print (f"[HARM] Copying reference client unchanged: {cname}")
            copy_tree_force (train_img_dirs [i ],dst_train )
            copy_tree_force (train_mask_dirs [i ],dst_train_mask )
            copy_tree_force (val_img_dirs [i ],dst_val )
            copy_tree_force (val_mask_dirs [i ],dst_val_mask )
            copy_tree_force (test_img_dirs [i ],dst_test )
            copy_tree_force (test_mask_dirs [i ],dst_test_mask )
        else :
            print (f"[HARM] Harmonizing {cname} -> {client_names[reference_idx]}")
            G =cut_models [i ]
            harmonize_folder_with_generator (
            G ,train_img_dirs [i ],dst_train ,
            mask_src_dir =train_mask_dirs [i ],mask_dst_dir =dst_train_mask ,
            device =DEVICE ,size =(IMG_SIZE ,IMG_SIZE )
            )
            harmonize_folder_with_generator (
            G ,val_img_dirs [i ],dst_val ,
            mask_src_dir =val_mask_dirs [i ],mask_dst_dir =dst_val_mask ,
            device =DEVICE ,size =(IMG_SIZE ,IMG_SIZE )
            )
            harmonize_folder_with_generator (
            G ,test_img_dirs [i ],dst_test ,
            mask_src_dir =test_mask_dirs [i ],mask_dst_dir =dst_test_mask ,
            device =DEVICE ,size =(IMG_SIZE ,IMG_SIZE )
            )

        hm_train_dirs .append (dst_train )
        hm_train_mask_dirs .append (dst_train_mask )
        hm_val_dirs .append (dst_val )
        hm_val_mask_dirs .append (dst_val_mask )
        hm_test_dirs .append (dst_test )
        hm_test_mask_dirs .append (dst_test_mask )

    print ("[HARM] Harmonization complete.")
    print ("[HARM] Output:",harm_root )




    visuals_base =os .path .join (out_dir ,"HarmonizedSamples_CUT")
    for i in range (NUM_CLIENTS ):
        cname =client_names [i ]
        save_comparison_grid (val_img_dirs [i ],hm_val_dirs [i ],cname ,visuals_base ,n_samples =7 )




    global_model =UNET (in_channels =3 ,num_classes =1 ).to (DEVICE )



    global_test_loader =get_global_test_loader (hm_test_dirs ,hm_test_mask_dirs ,val_tf ,batch_size =4 )
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
            hm_train_dirs [i ],
            hm_train_mask_dirs [i ],
            tr_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =True 
            )
            val_loader =get_loader (
            hm_val_dirs [i ],
            hm_val_mask_dirs [i ],
            val_tf ,
            client_name =client_names [i ],
            batch_size =4 ,
            shuffle =False 
            )

            print (f"[Client {client_names[i]}] Local training")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            local_models .append (local_model )
            sz =len (train_loader .dataset )
            weights .append (sz )
            total_sz +=sz 

        if total_sz ==0 :
            raise RuntimeError ("Total training size across clients is 0. Check your folders.")

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
            hm_test_dirs [i ],
            hm_test_mask_dirs [i ],
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
    print (f"\nTotal runtime: {(end_time - start_time):.2f} seconds")


if __name__ =="__main__":
    main ()