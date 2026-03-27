import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os ,copy ,time ,torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
from tqdm import tqdm 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 

from models .UNET import UNET 
from dataset import CVCDataset 




os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
NUM_CLIENTS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
start_time =time .time ()

out_dir ="Outputs"
os .makedirs (out_dir ,exist_ok =True )




train_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
]
train_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",

]
val_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images"
]
val_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks"
]
test_img_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\images",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images"
]
test_mask_dirs =[
r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks"
]
client_names =["Kvasir","ETIS","CVC-Colon","CVC-Clinic"]




import numpy as np 
from PIL import Image 
import shutil 

def _unnormalize_image (tensor ,mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )):
    """
    tensor: torch tensor CxHxW normalized by mean/std
    returns uint8 HxWx3 numpy array
    """
    arr =tensor .cpu ().numpy ()
    if arr .ndim ==3 :
        c ,h ,w =arr .shape 
        arr =arr .transpose (1 ,2 ,0 )
    else :

        arr =arr 
    arr =arr *np .array (std ).reshape (1 ,1 ,3 )+np .array (mean ).reshape (1 ,1 ,3 )
    arr =np .clip (arr *255.0 ,0 ,255 ).astype (np .uint8 )
    return arr 

def _mask_to_uint8 (mask_tensor ):
    """
    mask_tensor: torch tensor (1,H,W) or (H,W)
    returns uint8 HxW
    """
    m =mask_tensor .cpu ().numpy ()
    if m .ndim ==3 :
        m =np .squeeze (m ,axis =0 )
    m =(m >0.5 ).astype (np .uint8 )*255 
    return m 

def save_image (arr ,path ):
    Image .fromarray (arr ).save (path )

def ensure_dir (path ):
    os .makedirs (path ,exist_ok =True )




def get_loader (img_dir ,mask_dir ,transform ,batch_size =32 ,shuffle =True ,return_filename =True ):
    """
    Builds a DataLoader. If your CVCDataset supports return_filename parameter,
    we request filenames to enable saving predictions with the original filenames.
    """

    try :
        ds =CVCDataset (img_dir ,mask_dir ,transform =transform ,return_filename =return_filename )
    except TypeError :
        ds =CVCDataset (img_dir ,mask_dir ,transform =transform )
        return_filename =False 
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )

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
    return dict (dice_with_bg =dice_with_bg ,dice_no_bg =dice_no_bg ,
    iou_with_bg =iou_with_bg ,iou_no_bg =iou_no_bg ,
    accuracy =acc ,precision =precision ,recall =recall ,specificity =specificity )

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
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ]for i in range (len (models )))
    return avg_sd 




def save_transformed_samples (img_dir ,mask_dir ,transform ,client_name ,out_base ,n_samples =8 ,prefix ="harmonized"):
    ds =CVCDataset (img_dir ,mask_dir ,transform =transform )
    dest =os .path .join (out_base ,f"{client_name}",prefix )
    ensure_dir (dest )
    num =min (n_samples ,len (ds ))
    for i in range (num ):
        try :
            item =ds [i ]

            if isinstance (item ,tuple )and len (item )>=2 :
                img_t ,mask_t =item [0 ],item [1 ]
            else :
                raise ValueError ("Unexpected dataset __getitem__ return")
        except Exception as e :
            raise 


        if isinstance (img_t ,np .ndarray ):
            img_arr =img_t .astype (np .uint8 )
            save_image (img_arr ,os .path .join (dest ,f"{client_name}_img_{i}.png"))
        else :
            img_arr =_unnormalize_image (img_t )
            save_image (img_arr ,os .path .join (dest ,f"{client_name}_img_{i}.png"))


        if isinstance (mask_t ,np .ndarray ):
            m_arr =(mask_t >0.5 ).astype (np .uint8 )*255 
            save_image (m_arr ,os .path .join (dest ,f"{client_name}_mask_{i}.png"))
        else :
            m_arr =_mask_to_uint8 (mask_t )
            save_image (m_arr ,os .path .join (dest ,f"{client_name}_mask_{i}.png"))

def save_test_predictions (global_model ,test_loader ,client_name ,out_base =None ,round_num =None ,max_to_save =16 ,device_arg =None ):
    """
    Save predicted masks (as .png) for test images for a client.
    - If the DataLoader/dataset returns filenames (third element in batch), those filenames will be used (keeps original order).
    - If filenames are not available, fallback names are used.
    """

    if out_base is None :
        out_base =out_dir 

    if device_arg is None :
        device =DEVICE if "DEVICE"in globals ()else torch .device ("cpu")
    else :
        device =device_arg 

    global_model .eval ()
    latest_dir =os .path .join (out_base ,"TestPreds",client_name ,"latest")


    if os .path .exists (latest_dir ):
        shutil .rmtree (latest_dir )
    ensure_dir (latest_dir )

    saved =0 
    with torch .no_grad ():
        for idx ,batch in enumerate (test_loader ):

            if isinstance (batch ,(list ,tuple ))and len (batch )==3 :
                data ,target ,fnames =batch 
            elif isinstance (batch ,(list ,tuple ))and len (batch )==2 :
                data ,target =batch 
                fnames =None 
            else :

                raise RuntimeError ("Unexpected batch format from test_loader")


            if target .dim ()==3 :

                target =target .unsqueeze (1 )


            data =data .to (device )
            preds =global_model (data )
            probs =torch .sigmoid (preds )
            bin_mask =(probs >0.5 ).float ()


            b_sz =data .size (0 )
            for b in range (b_sz ):
                mask_t =bin_mask [b ].cpu ()
                mask_arr =_mask_to_uint8 (mask_t )


                if fnames is not None :
                    try :

                        orig_name =fnames [b ]
                        base ,ext =os .path .splitext (orig_name )
                        fname =f"{base}_pred.png"
                    except Exception :
                        fname =f"{client_name}_pred_mask_{idx}_{b}.png"
                else :
                    fname =f"{client_name}_pred_mask_{idx}_{b}.png"

                save_image (mask_arr ,os .path .join (latest_dir ,fname ))

                saved +=1 
                if saved >=max_to_save :
                    break 
            if saved >=max_to_save :
                break 

    print (f"Saved {saved} prediction masks for {client_name} in {latest_dir}")

def make_comparison_grid_and_histograms_updated (img_dir ,mask_dir ,val_transform ,visual_transform ,
client_name ,out_base ,n_samples =7 ,diff_amp =4.0 ):
    base_dest =os .path .join (out_base ,"HarmonizedSamples",client_name )
    diffs_dest =os .path .join (base_dest ,"diffs")
    hist_dest =os .path .join (base_dest ,"histograms")
    ensure_dir (base_dest );ensure_dir (diffs_dest );ensure_dir (hist_dest )

    fnames =sorted ([f for f in os .listdir (img_dir )if f .lower ().endswith (('.png','.jpg','.jpeg'))])[:n_samples ]
    if len (fnames )==0 :
        print (f"No images found in {img_dir} for {client_name}")
        return 

    top_imgs ,mid_imgs ,diff_color_imgs ,short_names =[],[],[],[]

    for fname in fnames :
        img_path =os .path .join (img_dir ,fname )
        orig_pil =Image .open (img_path ).convert ("RGB")
        orig_np =np .array (orig_pil )


        vis_out =visual_transform (image =orig_np )
        vis_img =vis_out ['image']
        top =vis_img .astype (np .uint8 )if isinstance (vis_img ,np .ndarray )else np .array (vis_img ).astype (np .uint8 )


        vt_out =val_transform (image =orig_np )
        img_t =vt_out ['image']
        if isinstance (img_t ,torch .Tensor ):
            mid =_unnormalize_image (img_t )
        elif isinstance (img_t ,np .ndarray ):
            hnp =img_t 
            if hnp .dtype !=np .uint8 :
                if hnp .max ()<=1.5 :
                    hnp =(np .clip (hnp ,0 ,1 )*255 ).astype (np .uint8 )
                else :
                    hnp =np .clip (hnp ,0 ,255 ).astype (np .uint8 )
            mid =hnp .astype (np .uint8 )
        else :
            mid =top .copy ()


        h_h ,h_w =mid .shape [0 ],mid .shape [1 ]
        orig_resized =np .array (orig_pil .resize ((h_w ,h_h ),resample =Image .BILINEAR )).astype (np .uint8 )
        if top .shape [0 ]!=h_h or top .shape [1 ]!=h_w :
            top =np .array (Image .fromarray (top ).resize ((h_w ,h_h ),resample =Image .BILINEAR )).astype (np .uint8 )


        diff_color =np .abs (orig_resized .astype (int )-mid .astype (int )).astype (np .uint8 )


        amp =np .clip ((diff_color .astype (float )*diff_amp ),0 ,255 ).astype (np .uint8 )


        diff_fname =os .path .join (diffs_dest ,f"diff_color_{fname}")
        amp_fname =os .path .join (diffs_dest ,f"diff_color_amp_{fname}")
        save_image (diff_color ,diff_fname )
        save_image (amp ,amp_fname )


        fig ,ax =plt .subplots (1 ,3 ,figsize =(12 ,3 ))
        colors =['r','g','b']
        for ch in range (3 ):
            ax [ch ].hist (orig_resized [:,:,ch ].ravel (),bins =256 ,alpha =0.6 ,label ='orig',color =colors [ch ])
            ax [ch ].hist (mid [:,:,ch ].ravel (),bins =256 ,alpha =0.6 ,label ='harm',color =colors [ch ],histtype ='step')
            ax [ch ].legend (fontsize =6 )
            ax [ch ].set_title (['R','G','B'][ch ])
        plt .tight_layout ()
        hist_path =os .path .join (hist_dest ,f"hist_{fname}.png")
        plt .savefig (hist_path )
        plt .close (fig )


        top_imgs .append (orig_resized )
        mid_imgs .append (mid )
        diff_color_imgs .append (amp )
        short_names .append (fname if len (fname )<=20 else fname [:17 ]+"...")


    n =len (top_imgs )
    fig_w =max (3 *n ,8 )
    fig_h =6 
    fig ,axs =plt .subplots (3 ,n ,figsize =(fig_w ,fig_h ))
    if n ==1 :
        axs =np .array ([[axs [0 ]],[axs [1 ]],[axs [2 ]]])

    fig .suptitle (f"Harmonized (Input-level MixStyle) vs. Original: {client_name}",fontsize =16 ,y =0.98 )
    fig .text (0.01 ,0.82 ,"Original\nimages",fontsize =12 ,va ='center',rotation ='vertical')
    fig .text (0.01 ,0.50 ,"Harmonized\nimages",fontsize =12 ,va ='center',rotation ='vertical')
    fig .text (0.01 ,0.18 ,"Amplified\nDifference",fontsize =12 ,va ='center',rotation ='vertical')

    for i in range (n ):
        axs [0 ,i ].imshow (top_imgs [i ]);axs [0 ,i ].axis ('off');axs [0 ,i ].set_title (short_names [i ],fontsize =8 )
        axs [1 ,i ].imshow (mid_imgs [i ]);axs [1 ,i ].axis ('off')
        axs [2 ,i ].imshow (diff_color_imgs [i ]);axs [2 ,i ].axis ('off')

    plt .tight_layout (rect =[0.03 ,0.03 ,0.98 ,0.94 ])
    plt .subplots_adjust (wspace =0.01 ,hspace =0.02 )
    grid_path =os .path .join (base_dest ,f"comparison_grid_{client_name}.png")
    plt .savefig (grid_path ,dpi =150 )
    plt .close (fig )

    print (f"Saved comparison grid: {grid_path}")
    print (f"Saved color diffs in: {diffs_dest} (raw + amplified).")
    print (f"Saved RGB histograms in: {hist_dest} (histograms only).")




def input_mixstyle (images ,p =0.6 ,alpha =0.2 ,eps =1e-6 ):
    """
    Apply MixStyle-like mixing on input images (B x C x H x W).
    This is a lightweight, non-invasive way to test MixStyle without modifying UNET internals.
    - images: torch.Tensor (B,C,H,W), float (expected normalized or in [0,1])
    - p: probability to apply mixing for this batch
    - alpha: Beta distribution parameter for mixing coefficient sampling
    Returns images of same shape and dtype as input (torch.Tensor).
    """
    if not images .is_floating_point ():
        images =images .float ()
    if (not images .is_cuda )and DEVICE .startswith ("cuda"):
        images =images .to (DEVICE )

    B ,C ,H ,W =images .shape 
    if B <=1 :
        return images 

    if np .random .rand ()>p :
        return images 


    x =images .view (B ,C ,-1 )
    mu =x .mean (dim =2 ).view (B ,C ,1 ,1 )
    var =x .var (dim =2 ,unbiased =False ).view (B ,C ,1 ,1 )
    sigma =torch .sqrt (var +eps )


    x_norm =(images -mu )/sigma 


    lam =np .random .beta (alpha ,alpha ,size =B ).astype (np .float32 )
    lam =torch .from_numpy (lam ).to (images .device ).view (B ,1 ,1 ,1 )


    perm =torch .randperm (B ).to (images .device )
    mu2 =mu [perm ]
    sigma2 =sigma [perm ]

    mu_mix =mu *lam +mu2 *(1.0 -lam )
    sigma_mix =sigma *lam +sigma2 *(1.0 -lam )

    out =x_norm *sigma_mix +mu_mix 
    return out 




def train_local (loader ,model ,loss_fn ,opt ):
    model .train ()
    total_loss ,metrics =0.0 ,[]
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
                raise RuntimeError (f"Unexpected target dims: {target.shape}")



            try :
                data =input_mixstyle (data ,p =0.6 ,alpha =0.2 )
            except Exception :

                pass 


            data ,target =data .to (DEVICE ),target .to (DEVICE )
            preds =model (data )
            loss =loss_fn (preds ,target )
            opt .zero_grad ();loss .backward ();opt .step ()
            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))
    avg_metrics =average_metrics (metrics )
    if len (loader .dataset )>0 :
        avg_loss =total_loss /len (loader .dataset )
    else :
        avg_loss =total_loss 
    print ("Train: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return avg_loss ,avg_metrics 

@torch .no_grad ()
def evaluate (loader ,model ,loss_fn ,split ="Val"):
    model .eval ()
    total_loss ,metrics =0.0 ,[]
    n_items =0 
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
            raise RuntimeError (f"Unexpected target dims: {target.shape}")

        data ,target =data .to (DEVICE ),target .to (DEVICE )
        preds =model (data )
        loss =loss_fn (preds ,target )
        total_loss +=loss .item ()
        metrics .append (compute_metrics (preds ,target ))
        n_items +=1 
    avg_metrics =average_metrics (metrics )if metrics else {}
    avg_loss =(total_loss /n_items )if n_items >0 else 0.0 
    print (f"{split}: "+" | ".join ([f"{k}: {v:.4f}"for k ,v in (avg_metrics or {}).items ()]))
    return avg_loss ,avg_metrics 




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
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_harm4.png"))
    plt .close ()


    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round")
    plt .ylabel ("IoU")
    plt .title ("Per-client IoU ")
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_harm4.png"))
    plt .close ()




def main ():



    '''

    tr_tf = A.Compose([
        # geometric / scale / crop
        A.RandomResizedCrop(256, 256, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.35, rotate_limit=25, p=0.6),

        # realistic deformation
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.15),

        # photometric
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),

        # blur / noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=3)
        ], p=0.2),
        A.GaussNoise(var_limit=(5.0, 50.0), p=0.2),

        # final resize/normalize
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    '''
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


    visual_val_tf =A .Compose ([
    A .Resize (224 ,224 ),
    ])

    print ("Saving harmonized samples (val-style) and augmented samples (train-style) for each client...")
    visuals_base =os .path .join (out_dir ,"HarmonizedSamples")
    for i in range (NUM_CLIENTS ):
        cname =client_names [i ]

        save_transformed_samples (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,cname ,visuals_base ,n_samples =7 ,prefix ="harmonized")

        save_transformed_samples (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,cname ,visuals_base ,n_samples =7 ,prefix ="augmented")


        make_comparison_grid_and_histograms_updated (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,visual_val_tf ,cname ,visuals_base ,n_samples =7 )
    print ("Saved harmonized/augmented samples and comparison grids to Outputs/HarmonizedSamples/")

    global_model =UNET (in_channels =3 ,out_channels =1 ).to (DEVICE )

    round_metrics =[]

    for r in range (COMM_ROUNDS ):
        print (f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models ,weights =[],[]
        total_sz =0 

        for i in range (NUM_CLIENTS ):
            local_model =copy .deepcopy (global_model ).to (DEVICE )
            opt =optim .AdamW (local_model .parameters (),lr =1e-4 )
            loss_fn =get_loss_fn (DEVICE )

            train_loader =get_loader (train_img_dirs [i ],train_mask_dirs [i ],tr_tf ,batch_size =32 ,shuffle =True ,return_filename =True )
            val_loader =get_loader (val_img_dirs [i ],val_mask_dirs [i ],val_tf ,batch_size =32 ,shuffle =False ,return_filename =True )

            print (f"[Client {client_names[i]}]")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")

            local_models .append (local_model )
            sz =len (train_loader .dataset );weights .append (sz );total_sz +=sz 


        norm_weights =[w /total_sz for w in weights ]
        global_model .load_state_dict (average_models_weighted (local_models ,norm_weights ))


        rm ={}
        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (test_img_dirs [i ],test_mask_dirs [i ],val_tf ,batch_size =32 ,shuffle =False ,return_filename =True )
            print (f"[Client {client_names[i]}] Test")
            _ ,test_metrics =evaluate (test_loader ,global_model ,get_loss_fn (DEVICE ),split ="Test")
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )

            save_test_predictions (global_model ,test_loader ,client_names [i ],out_base =out_dir ,round_num =(r +1 ),max_to_save =int (len (test_loader .dataset )),device_arg =DEVICE )

        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )

    end_time =time .time ()
    print (f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ =="__main__":
    main ()
