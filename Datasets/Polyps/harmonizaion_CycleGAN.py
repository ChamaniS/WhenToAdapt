
import os 
os .environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os ,copy ,time ,random ,math ,shutil ,stat 
from glob import glob 
from PIL import Image 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader ,Dataset 
import torchvision .transforms as T 
from tqdm import tqdm 
import albumentations as A 
from albumentations .pytorch import ToTensorV2 
import segmentation_models_pytorch as smp 
import matplotlib .pyplot as plt 


from models .UNET import UNET 
from dataset import CVCDataset 




def _on_rm_error (func ,path ,exc_info ):
    """Error handler for rmtree: change file to writable and retry."""
    os .chmod (path ,stat .S_IWRITE )
    func (path )

def copy_tree_force (src ,dst ):
    """
    Copy directory tree from src -> dst.
    If dst exists, remove it entirely first (safe fallback for Python <3.8).
    """
    if not os .path .exists (src ):
        raise FileNotFoundError (f"Source not found: {src}")
    if os .path .exists (dst ):
        shutil .rmtree (dst ,onerror =_on_rm_error )
    shutil .copytree (src ,dst )




DEVICE ="cuda"if torch .cuda .is_available ()else "cpu"
NUM_CLIENTS =4 
LOCAL_EPOCHS =12 
COMM_ROUNDS =10 
CYCLEGAN_EPOCHS =30 
BATCH_CYCLEGAN =4 
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
reference_idx =0 




class ImageFolderSimple (Dataset ):
    def __init__ (self ,folder ,size =(224 ,224 ),augment =False ):
        self .files =sorted ([p for p in glob (os .path .join (folder ,"*"))if p .lower ().endswith (('.png','.jpg','.jpeg'))])
        self .size =size 
        self .augment =augment 
        self .base_trans =T .Compose ([T .Resize (self .size ),T .CenterCrop (self .size ),T .ToTensor ()])

        self .aug_trans =T .Compose ([
        T .RandomHorizontalFlip (),
        T .RandomRotation (10 )
        ])
    def __len__ (self ):return max (1 ,len (self .files ))
    def __getitem__ (self ,idx ):
        p =self .files [idx %len (self .files )]
        img =Image .open (p ).convert ('RGB')
        if self .augment :
            img =self .aug_trans (img )
        t =self .base_trans (img )
        return t 





def conv_block (in_ch ,out_ch ,k =3 ,stride =1 ,padding =1 ,norm =True ,relu =True ):
    layers =[nn .Conv2d (in_ch ,out_ch ,kernel_size =k ,stride =stride ,padding =padding ,bias =not norm )]
    if norm :
        layers .append (nn .InstanceNorm2d (out_ch ,affine =False ))
    if relu :
        layers .append (nn .ReLU (True ))
    return nn .Sequential (*layers )

class ResnetBlock (nn .Module ):
    def __init__ (self ,ch ):
        super ().__init__ ()
        self .block =nn .Sequential (
        nn .ReflectionPad2d (1 ),
        nn .Conv2d (ch ,ch ,3 ,1 ,0 ,bias =False ),
        nn .InstanceNorm2d (ch ),
        nn .ReLU (True ),
        nn .ReflectionPad2d (1 ),
        nn .Conv2d (ch ,ch ,3 ,1 ,0 ,bias =False ),
        nn .InstanceNorm2d (ch ),
        )
    def forward (self ,x ):
        return x +self .block (x )

class ResnetGenerator (nn .Module ):
    def __init__ (self ,in_ch =3 ,out_ch =3 ,ngf =64 ,nblocks =6 ):
        super ().__init__ ()
        model =[nn .ReflectionPad2d (3 ),
        nn .Conv2d (in_ch ,ngf ,7 ,1 ,0 ,bias =False ),
        nn .InstanceNorm2d (ngf ),
        nn .ReLU (True )]

        n_down =2 
        mult =1 
        for i in range (n_down ):
            mult_prev =mult 
            mult *=2 
            model +=[nn .Conv2d (ngf *mult_prev ,ngf *mult ,3 ,2 ,1 ,bias =False ),
            nn .InstanceNorm2d (ngf *mult ),
            nn .ReLU (True )]

        for i in range (nblocks ):
            model +=[ResnetBlock (ngf *mult )]

        for i in range (n_down ):
            mult_prev =mult 
            mult //=2 
            model +=[nn .ConvTranspose2d (ngf *mult_prev ,ngf *mult ,3 ,2 ,1 ,output_padding =1 ,bias =False ),
            nn .InstanceNorm2d (ngf *mult ),
            nn .ReLU (True )]
        model +=[nn .ReflectionPad2d (3 ),nn .Conv2d (ngf ,out_ch ,7 ,1 ,0 ),nn .Tanh ()]
        self .model =nn .Sequential (*model )
    def forward (self ,x ):return self .model (x )

class NLayerDiscriminator (nn .Module ):
    def __init__ (self ,in_ch =3 ,ndf =64 ,n_layers =3 ):
        super ().__init__ ()
        kw =4 ;padw =1 
        sequence =[nn .Conv2d (in_ch ,ndf ,kw ,2 ,padw ),nn .LeakyReLU (0.2 ,True )]
        nf_mult =1 
        for n in range (1 ,n_layers ):
            nf_mult_prev =nf_mult 
            nf_mult =min (2 **n ,8 )
            sequence +=[nn .Conv2d (ndf *nf_mult_prev ,ndf *nf_mult ,kw ,2 ,padw ,bias =False ),
            nn .InstanceNorm2d (ndf *nf_mult ),
            nn .LeakyReLU (0.2 ,True )]
        nf_mult_prev =nf_mult 
        nf_mult =min (2 **n_layers ,8 )
        sequence +=[nn .Conv2d (ndf *nf_mult_prev ,ndf *nf_mult ,kw ,1 ,padw ,bias =False ),
        nn .InstanceNorm2d (ndf *nf_mult ),
        nn .LeakyReLU (0.2 ,True )]
        sequence +=[nn .Conv2d (ndf *nf_mult ,1 ,kw ,1 ,padw )]
        self .model =nn .Sequential (*sequence )
    def forward (self ,x ):return self .model (x )




def weights_init_normal (m ):
    classname =m .__class__ .__name__ 
    if classname .find ('Conv')!=-1 :
        nn .init .normal_ (m .weight .data ,0.0 ,0.02 )
    elif classname .find ('BatchNorm2d')!=-1 or classname .find ('InstanceNorm2d')!=-1 :
        if hasattr (m ,'weight')and m .weight is not None :
            nn .init .normal_ (m .weight .data ,1.0 ,0.02 )
        if hasattr (m ,'bias')and m .bias is not None :
            nn .init .constant_ (m .bias .data ,0.0 )

class ImagePool :

    def __init__ (self ,pool_size =50 ):
        self .pool_size =pool_size 
        self .images =[]
    def query (self ,images ):
        if self .pool_size ==0 :
            return images 
        return_images =[]
        for image in images :
            image =torch .unsqueeze (image .data ,0 )
            if len (self .images )<self .pool_size :
                self .images .append (image )
                return_images .append (image )
            else :
                if random .random ()>0.5 :
                    idx =random .randint (0 ,self .pool_size -1 )
                    tmp =self .images [idx ].clone ()
                    self .images [idx ]=image 
                    return_images .append (tmp )
                else :
                    return_images .append (image )
        return torch .cat (return_images ,0 )




def train_cyclegan (domainA_dir ,domainB_dir ,save_dir ,epochs =CYCLEGAN_EPOCHS ,device =DEVICE ):
    os .makedirs (save_dir ,exist_ok =True )

    dsA =ImageFolderSimple (domainA_dir ,size =(224 ,224 ),augment =True )
    dsB =ImageFolderSimple (domainB_dir ,size =(224 ,224 ),augment =True )
    loaderA =DataLoader (dsA ,batch_size =BATCH_CYCLEGAN ,shuffle =True ,drop_last =True ,num_workers =2 )
    loaderB =DataLoader (dsB ,batch_size =BATCH_CYCLEGAN ,shuffle =True ,drop_last =True ,num_workers =2 )


    G_A2B =ResnetGenerator ().to (device )
    G_B2A =ResnetGenerator ().to (device )
    D_A =NLayerDiscriminator ().to (device )
    D_B =NLayerDiscriminator ().to (device )
    for net in [G_A2B ,G_B2A ,D_A ,D_B ]:
        net .apply (weights_init_normal )


    criterion_GAN =nn .MSELoss ().to (device )
    criterion_cycle =nn .L1Loss ().to (device )
    criterion_identity =nn .L1Loss ().to (device )

    optimizer_G =optim .Adam (list (G_A2B .parameters ())+list (G_B2A .parameters ()),lr =1e-4 ,betas =(0.5 ,0.999 ))
    optimizer_D_A =optim .Adam (D_A .parameters (),lr =1e-4 ,betas =(0.5 ,0.999 ))
    optimizer_D_B =optim .Adam (D_B .parameters (),lr =1e-4 ,betas =(0.5 ,0.999 ))

    fake_A_pool =ImagePool (50 )
    fake_B_pool =ImagePool (50 )


    real_label =1.0 
    fake_label =0.0 

    print (f"[CycleGAN] Train {domainA_dir} <-> {domainB_dir} for {epochs} epochs")
    iterB =iter (loaderB )
    for epoch in range (epochs ):
        loop =tqdm (loaderA ,desc =f"Epoch {epoch+1}/{epochs}")
        for real_A in loop :
            try :
                real_B =next (iterB )
            except StopIteration :
                iterB =iter (loaderB )
                real_B =next (iterB )
            real_A =real_A .to (device )
            real_B =real_B .to (device )
            bs =real_A .size (0 )


            optimizer_G .zero_grad ()

            same_B =G_A2B (real_B )
            loss_id_B =criterion_identity (same_B ,real_B )*10.0 
            same_A =G_B2A (real_A )
            loss_id_A =criterion_identity (same_A ,real_A )*10.0 


            fake_B =G_A2B (real_A )
            pred_fake_B =D_B (fake_B )
            valid =torch .full_like (pred_fake_B ,real_label ,device =device )
            loss_GAN_A2B =criterion_GAN (pred_fake_B ,valid )

            fake_A =G_B2A (real_B )
            pred_fake_A =D_A (fake_A )
            validA =torch .full_like (pred_fake_A ,real_label ,device =device )
            loss_GAN_B2A =criterion_GAN (pred_fake_A ,validA )


            rec_A =G_B2A (fake_B )
            loss_cycle_A =criterion_cycle (rec_A ,real_A )*5.0 
            rec_B =G_A2B (fake_A )
            loss_cycle_B =criterion_cycle (rec_B ,real_B )*5.0 

            loss_G =loss_GAN_A2B +loss_GAN_B2A +loss_cycle_A +loss_cycle_B +loss_id_A +loss_id_B 
            loss_G .backward ()
            optimizer_G .step ()


            optimizer_D_A .zero_grad ()
            pred_real =D_A (real_A )
            valid_label =torch .full_like (pred_real ,real_label ,device =device )
            loss_D_real =criterion_GAN (pred_real ,valid_label )

            fake_A_detached =fake_A_pool .query (fake_A .detach ())
            pred_fake =D_A (fake_A_detached )
            fake_label_tensor =torch .full_like (pred_fake ,fake_label ,device =device )
            loss_D_fake =criterion_GAN (pred_fake ,fake_label_tensor )

            loss_D_A =(loss_D_real +loss_D_fake )*0.5 
            loss_D_A .backward ()
            optimizer_D_A .step ()


            optimizer_D_B .zero_grad ()
            pred_real_B =D_B (real_B )
            valid_label_B =torch .full_like (pred_real_B ,real_label ,device =device )
            loss_D_real_B =criterion_GAN (pred_real_B ,valid_label_B )

            fake_B_detached =fake_B_pool .query (fake_B .detach ())
            pred_fake_B =D_B (fake_B_detached )
            fake_label_tensorB =torch .full_like (pred_fake_B ,fake_label ,device =device )
            loss_D_fake_B =criterion_GAN (pred_fake_B ,fake_label_tensorB )

            loss_D_B =(loss_D_real_B +loss_D_fake_B )*0.5 
            loss_D_B .backward ()
            optimizer_D_B .step ()

            loop .set_postfix ({
            "loss_G":loss_G .item (),
            "loss_D_A":loss_D_A .item (),
            "loss_D_B":loss_D_B .item ()
            })


        torch .save ({
        'G_A2B':G_A2B .state_dict (),
        'G_B2A':G_B2A .state_dict (),
        'D_A':D_A .state_dict (),
        'D_B':D_B .state_dict (),
        'opt_G':optimizer_G .state_dict ()
        },os .path .join (save_dir ,f"cyclegan_epoch_{epoch+1}.pth"))


    torch .save ({'G_A2B':G_A2B .state_dict (),'G_B2A':G_B2A .state_dict ()},os .path .join (save_dir ,f"cyclegan_final.pth"))
    print (f"[CycleGAN] finished and saved to {save_dir}")
    return G_A2B .cpu (),G_B2A .cpu ()




def harmonize_folder_with_generator (generator ,src_dir ,dst_dir ,mask_src_dir =None ,mask_dst_dir =None ,device =DEVICE ,size =(224 ,224 )):
    os .makedirs (dst_dir ,exist_ok =True )
    if mask_src_dir and mask_dst_dir :
        os .makedirs (mask_dst_dir ,exist_ok =True )
    tf =T .Compose ([T .Resize (size ),T .CenterCrop (size ),T .ToTensor ()])
    generator =generator .to (device )
    generator .eval ()
    with torch .no_grad ():
        for p in sorted ([f for f in glob (os .path .join (src_dir ,"*"))if f .lower ().endswith (('.png','.jpg','.jpeg'))]):
            img =Image .open (p ).convert ('RGB')
            inp =tf (img ).unsqueeze (0 ).to (device )*2.0 -1.0 
            out =generator (inp )
            out =(out .squeeze (0 ).detach ().cpu ().clamp (-1 ,1 )+1.0 )/2.0 
            out_img =T .ToPILImage ()(out )
            basename =os .path .basename (p )
            out_img .save (os .path .join (dst_dir ,basename ))

            if mask_src_dir and mask_dst_dir :
                src_mask_p =os .path .join (mask_src_dir ,basename )
                if os .path .exists (src_mask_p ):
                    shutil .copy (src_mask_p ,os .path .join (mask_dst_dir ,basename ))
                else :

                    base ,_ =os .path .splitext (basename )
                    for ext in ('.png','.jpg','.jpeg','.bmp'):
                        alt =os .path .join (mask_src_dir ,base +ext )
                        if os .path .exists (alt ):
                            shutil .copy (alt ,os .path .join (mask_dst_dir ,base +ext ))
                            break 




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
val_tf =A .Compose ([A .Resize (224 ,224 ),A .Normalize (mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )),ToTensorV2 ()])
visual_val_tf =A .Compose ([A .Resize (224 ,224 )])




def _unnormalize_image (tensor ,mean =(0.485 ,0.456 ,0.406 ),std =(0.229 ,0.224 ,0.225 )):
    arr =tensor .cpu ().numpy ()
    if arr .ndim ==3 :
        c ,h ,w =arr .shape 
        arr =arr .transpose (1 ,2 ,0 )
    arr =arr *np .array (std ).reshape (1 ,1 ,3 )+np .array (mean ).reshape (1 ,1 ,3 )
    return np .clip (arr *255.0 ,0 ,255 ).astype (np .uint8 )
def _mask_to_uint8 (mask_tensor ):
    m =mask_tensor .cpu ().numpy ()
    if m .ndim ==3 :
        m =np .squeeze (m ,axis =0 )
    m =(m >0.5 ).astype (np .uint8 )*255 
    return m 
def ensure_dir (path ):os .makedirs (path ,exist_ok =True )
def save_image (arr ,path ):Image .fromarray (arr ).save (path )


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
    if out_base is None :out_base =out_dir 
    device =DEVICE if device_arg is None else device_arg 
    global_model .eval ()
    latest_dir =os .path .join (out_base ,"TestPreds",client_name ,"latest")
    if os .path .exists (latest_dir ):shutil .rmtree (latest_dir )
    ensure_dir (latest_dir )
    saved =0 
    with torch .no_grad ():
        for idx ,batch in enumerate (test_loader ):
            if isinstance (batch ,(list ,tuple ))and len (batch )==3 :
                data ,target ,fnames =batch 
            elif isinstance (batch ,(list ,tuple ))and len (batch )>=2 :
                data ,target =batch [0 ],batch [1 ]
                fnames =None 
            else :
                raise RuntimeError ("Unexpected batch format from test_loader")
            if target .dim ()==3 :
                target =target .unsqueeze (1 )
            data =data .to (device )
            preds =global_model (data )
            probs =torch .sigmoid (preds )
            bin_mask =(probs >0.5 ).float ()
            bsz =data .size (0 )
            for b in range (bsz ):
                mask_t =bin_mask [b ].cpu ()
                mask_arr =_mask_to_uint8 (mask_t )
                if fnames is not None :
                    try :
                        orig_name =fnames [b ]
                        base ,_ =os .path .splitext (orig_name )
                        fname =f"{base}_pred.png"
                    except Exception :
                        fname =f"{client_name}_pred_mask_{idx}_{b}.png"
                else :
                    fname =f"{client_name}_pred_mask_{idx}_{b}.png"
                save_image (mask_arr ,os .path .join (latest_dir ,fname ))
                saved +=1 
                if saved >=max_to_save :break 
            if saved >=max_to_save :break 
    print (f"Saved {saved} prediction masks for {client_name} in {latest_dir}")


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
    if not metrics_list :return {}
    avg ={}
    for k in metrics_list [0 ].keys ():
        avg [k ]=sum (m [k ]for m in metrics_list )/len (metrics_list )
    return avg 
def get_loss_fn (device ):return smp .losses .DiceLoss (mode ="binary",from_logits =True ).to (device )
def average_models_weighted (models ,weights ):
    avg_sd =copy .deepcopy (models [0 ].state_dict ())
    for k in avg_sd .keys ():
        avg_sd [k ]=sum (weights [i ]*models [i ].state_dict ()[k ]for i in range (len (models )))
    return avg_sd 

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
            data ,target =data .to (DEVICE ),target .to (DEVICE )
            preds =model (data )
            loss =loss_fn (preds ,target )
            opt .zero_grad ();loss .backward ();opt .step ()
            total_loss +=loss .item ()
            metrics .append (compute_metrics (preds .detach (),target ))
    avg_metrics =average_metrics (metrics )
    avg_loss =total_loss /max (1 ,len (loader .dataset ))
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
    plt .xlabel ("Global Round");plt .ylabel ("Dice");plt .title ("Per-client Dice");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"dice_no_bg_cycle.png"));plt .close ()
    plt .figure ()
    for cid in range (NUM_CLIENTS ):
        vals =[rm .get (f"client{cid}_iou_no_bg",0 )for rm in round_metrics ]
        plt .plot (rounds ,vals ,label =f"{client_names[cid]}")
    plt .xlabel ("Global Round");plt .ylabel ("IoU");plt .title ("Per-client IoU");plt .legend ();plt .tight_layout ()
    plt .savefig (os .path .join (out_dir ,"iou_no_bg_cycle.png"));plt .close ()




def main ():

    cyclegan_models ={}
    for i in range (NUM_CLIENTS ):
        if i ==reference_idx :
            print (f"[HARM] Skipping CycleGAN for reference client {client_names[i]}")
            continue 
        a_dir =train_img_dirs [i ]
        b_dir =train_img_dirs [reference_idx ]
        save_dir =os .path .join (out_dir ,"CycleGAN",f"{client_names[i]}_to_{client_names[reference_idx]}")
        os .makedirs (save_dir ,exist_ok =True )

        final_ckpt =os .path .join (save_dir ,"cyclegan_final.pth")
        if os .path .exists (final_ckpt ):

            G_A2B =ResnetGenerator ()
            ck =torch .load (final_ckpt ,map_location ='cpu')
            G_A2B .load_state_dict (ck ['G_A2B'])
            G_A2B .to (DEVICE )
            cyclegan_models [i ]=G_A2B .cpu ()
            print (f"[HARM] Loaded existing CycleGAN generator for {client_names[i]} -> {client_names[reference_idx]}")
        else :
            G_A2B ,G_B2A =train_cyclegan (a_dir ,b_dir ,save_dir ,epochs =CYCLEGAN_EPOCHS ,device =DEVICE )
            cyclegan_models [i ]=G_A2B .cpu ()


    hist_base =os .path .join (out_dir ,"CycleGAN_Harmonized")
    hm_train_dirs =[]
    hm_train_mask_dirs =[]
    hm_val_dirs =[]
    hm_val_mask_dirs =[]
    hm_test_dirs =[]
    hm_test_mask_dirs =[]
    for i in range (NUM_CLIENTS ):
        cname =client_names [i ]
        if i ==reference_idx :

            dst_train =os .path .join (hist_base ,cname ,"train_images")
            dst_val =os .path .join (hist_base ,cname ,"val_images")
            dst_test =os .path .join (hist_base ,cname ,"test_images")
            dst_train_mask =os .path .join (hist_base ,cname ,"train_masks")
            dst_val_mask =os .path .join (hist_base ,cname ,"val_masks")
            dst_test_mask =os .path .join (hist_base ,cname ,"test_masks")

            copy_tree_force (train_img_dirs [i ],dst_train )
            copy_tree_force (val_img_dirs [i ],dst_val )
            copy_tree_force (test_img_dirs [i ],dst_test )
            copy_tree_force (train_mask_dirs [i ],dst_train_mask )
            copy_tree_force (val_mask_dirs [i ],dst_val_mask )
            copy_tree_force (test_mask_dirs [i ],dst_test_mask )
        else :
            G =cyclegan_models [i ]
            dst_train =os .path .join (hist_base ,cname ,"train_images");dst_val =os .path .join (hist_base ,cname ,"val_images");dst_test =os .path .join (hist_base ,cname ,"test_images")
            dst_train_mask =os .path .join (hist_base ,cname ,"train_masks");dst_val_mask =os .path .join (hist_base ,cname ,"val_masks");dst_test_mask =os .path .join (hist_base ,cname ,"test_masks")
            print (f"[HARM] Harmonizing {cname} -> {client_names[reference_idx]} (train/val/test)")
            ensure_dir (dst_train );ensure_dir (dst_val );ensure_dir (dst_test )
            ensure_dir (dst_train_mask );ensure_dir (dst_val_mask );ensure_dir (dst_test_mask )

            harmonize_folder_with_generator (G ,train_img_dirs [i ],dst_train ,mask_src_dir =train_mask_dirs [i ],mask_dst_dir =dst_train_mask ,device =DEVICE ,size =(224 ,224 ))
            harmonize_folder_with_generator (G ,val_img_dirs [i ],dst_val ,mask_src_dir =val_mask_dirs [i ],mask_dst_dir =dst_val_mask ,device =DEVICE ,size =(224 ,224 ))
            harmonize_folder_with_generator (G ,test_img_dirs [i ],dst_test ,mask_src_dir =test_mask_dirs [i ],mask_dst_dir =dst_test_mask ,device =DEVICE ,size =(224 ,224 ))
        hm_train_dirs .append (dst_train );hm_train_mask_dirs .append (dst_train_mask )
        hm_val_dirs .append (dst_val );hm_val_mask_dirs .append (dst_val_mask )
        hm_test_dirs .append (dst_test );hm_test_mask_dirs .append (dst_test_mask )

    print ("[HARM] Harmonization complete. Harmonized datasets written under:",hist_base )


    visuals_base =os .path .join (out_dir ,"HarmonizedSamples_CycleGAN")
    for i in range (NUM_CLIENTS ):
        cname =client_names [i ]
        save_transformed_samples (hm_val_dirs [i ],hm_val_mask_dirs [i ],val_tf ,cname ,visuals_base ,n_samples =7 ,prefix ="harmonized")
        save_transformed_samples (hm_train_dirs [i ],hm_train_mask_dirs [i ],tr_tf ,cname ,visuals_base ,n_samples =7 ,prefix ="augmented")

        make_comparison_grid_and_histograms_updated_original_vs_hm (val_img_dirs [i ],hm_val_dirs [i ],cname ,visuals_base )


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
            train_loader =get_loader (hm_train_dirs [i ],hm_train_mask_dirs [i ],tr_tf ,batch_size =32 ,shuffle =True ,return_filename =True )
            val_loader =get_loader (hm_val_dirs [i ],hm_val_mask_dirs [i ],val_tf ,batch_size =32 ,shuffle =False ,return_filename =True )
            print (f"[Client {client_names[i]}] Local training")
            train_local (train_loader ,local_model ,loss_fn ,opt )
            evaluate (val_loader ,local_model ,loss_fn ,split ="Val")
            local_models .append (local_model )
            sz =len (train_loader .dataset );weights .append (sz );total_sz +=sz 
        norm_weights =[w /total_sz for w in weights ]
        global_model .load_state_dict (average_models_weighted (local_models ,norm_weights ))

        rm ={}
        for i in range (NUM_CLIENTS ):
            test_loader =get_loader (hm_test_dirs [i ],hm_test_mask_dirs [i ],val_tf ,batch_size =32 ,shuffle =False ,return_filename =True )
            print (f"[Client {client_names[i]}] Global Test")
            _ ,test_metrics =evaluate (test_loader ,global_model ,get_loss_fn (DEVICE ),split ="Test")
            rm [f"client{i}_dice_no_bg"]=test_metrics .get ("dice_no_bg",0 )
            rm [f"client{i}_iou_no_bg"]=test_metrics .get ("iou_no_bg",0 )
            save_test_predictions (global_model ,test_loader ,client_names [i ],out_base =out_dir ,round_num =(r +1 ),max_to_save =int (len (test_loader .dataset )),device_arg =DEVICE )
        round_metrics .append (rm )
        plot_metrics (round_metrics ,out_dir )
    print ("Finished FedAvg on harmonized data.")


def make_comparison_grid_and_histograms_updated_original_vs_hm (original_dir ,hm_dir ,client_name ,out_base ,n_samples =7 ):
    base_dest =os .path .join (out_base ,"ComparisonGrid",client_name )
    ensure_dir (base_dest );diffs_dest =os .path .join (base_dest ,"diffs");ensure_dir (diffs_dest )
    fnames =sorted ([f for f in os .listdir (original_dir )if f .lower ().endswith (('.png','.jpg','.jpeg'))])[:n_samples ]
    if len (fnames )==0 :return 
    top_imgs =[];mid_imgs =[];diff_imgs =[];short_names =[]
    for fname in fnames :
        orig_p =os .path .join (original_dir ,fname )
        hm_p =os .path .join (hm_dir ,fname )
        if not os .path .exists (hm_p ):continue 
        orig =np .array (Image .open (orig_p ).convert ('RGB').resize ((224 ,224 )))
        hm =np .array (Image .open (hm_p ).convert ('RGB').resize ((224 ,224 )))
        diff =np .clip ((np .abs (orig .astype (int )-hm .astype (int ))*4 ),0 ,255 ).astype (np .uint8 )
        top_imgs .append (orig );mid_imgs .append (hm );diff_imgs .append (diff );short_names .append (fname )
        save_image (orig ,os .path .join (base_dest ,f"orig_{fname}"))
        save_image (hm ,os .path .join (base_dest ,f"hm_{fname}"))
        save_image (diff ,os .path .join (diffs_dest ,f"diff_{fname}"))
    n =len (top_imgs )
    fig ,axs =plt .subplots (3 ,n ,figsize =(3 *n ,6 ))
    if n ==1 :axs =np .array ([[axs [0 ]],[axs [1 ]],[axs [2 ]]])
    for i in range (n ):
        axs [0 ,i ].imshow (top_imgs [i ]);axs [0 ,i ].axis ('off');axs [0 ,i ].set_title (short_names [i ][:12 ])
        axs [1 ,i ].imshow (mid_imgs [i ]);axs [1 ,i ].axis ('off')
        axs [2 ,i ].imshow (diff_imgs [i ]);axs [2 ,i ].axis ('off')

    fig .suptitle (f"Harmonized (Cycle GAN.) vs. Original: {client_name}",fontsize =16 ,y =0.98 )
    fig .text (0.01 ,0.82 ,"Original\nimages",fontsize =12 ,va ='center',rotation ='vertical')
    fig .text (0.01 ,0.50 ,"Harmonized\nimages",fontsize =12 ,va ='center',rotation ='vertical')
    fig .text (0.01 ,0.18 ,"Amplified\nDifference",fontsize =12 ,va ='center',rotation ='vertical')


    plt .tight_layout ();plt .savefig (os .path .join (base_dest ,f"comparison_{client_name}.png"));plt .close ()
    print (f"Saved comparison grid for {client_name} at {base_dest}")


def get_loader (img_dir ,mask_dir ,transform ,batch_size =32 ,shuffle =True ,return_filename =True ):
    try :
        ds =CVCDataset (img_dir ,mask_dir ,transform =transform ,return_filename =return_filename )
    except TypeError :
        ds =CVCDataset (img_dir ,mask_dir ,transform =transform )
        return_filename =False 
    return DataLoader (ds ,batch_size =batch_size ,shuffle =shuffle )

if __name__ =="__main__":
    main ()
