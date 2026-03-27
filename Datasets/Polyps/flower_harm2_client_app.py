"""my-awesome-app: A Flower / PyTorch client app (with Histogram Matching options)."""
from my_awesome_app .logging_utils import setup_logging 
setup_logging ()

import logging 
import torch 
from flwr .client import ClientApp ,NumPyClient 
from flwr .common import Context 
from my_awesome_app .unet import UNET as Net 
from my_awesome_app .polyp_task_his_mat import DEFAULT_OUT_BASE 
from my_awesome_app .polyp_task_his_mat import (
get_weights ,
load_data ,
set_weights ,
train ,
evaluate_val ,
test_final ,
save_test_predictions ,
_client_name ,
)

logger =logging .getLogger (__name__ )


class FlowerClient (NumPyClient ):
    def __init__ (self ,net ,trainloader ,valloader ,testloader ,local_epochs ,
    partition_id ,num_partitions ,device ):
        self .net =net 
        self .trainloader =trainloader 
        self .valloader =valloader 
        self .testloader =testloader 
        self .local_epochs =local_epochs 
        self .partition_id =partition_id 
        self .num_partitions =num_partitions 
        self .device =device 

        self .net .to (self .device )
        print (
        f"[DEBUG] Client {self.partition_id} using device {self.device}, "
        f"visible GPUs={torch.cuda.device_count()}",
        flush =True ,
        )

    def fit (self ,parameters ,config ):
        set_weights (self .net ,parameters )
        train_loss ,train_metrics =train (
        self .net ,self .trainloader ,self .local_epochs ,self .device 
        )
        return (
        get_weights (self .net ),
        len (self .trainloader .dataset ),
        {"train_loss":train_loss ,**train_metrics },
        )

    def evaluate (self ,parameters ,config ):
        set_weights (self .net ,parameters )
        val_loss ,val_metrics =evaluate_val (self .net ,self .valloader ,self .device )
        test_loss ,test_metrics =test_final (
        self .net ,self .testloader ,self .device ,self .partition_id 
        )



        client_label =f"Client{self.partition_id}_{_client_name(self.partition_id)}"


        round_num =config .get ("round",None )if isinstance (config ,dict )else None 



        save_test_predictions (
        self .net ,
        self .testloader ,
        client_label ,
        out_base =DEFAULT_OUT_BASE ,
        round_num =round_num ,
        max_to_save =32 ,
        device_arg =self .device ,
        )

        logger .info (
        f"[Client {self.partition_id}] Global Model Test "
        f"Loss: {test_loss:.4f}, "
        +", ".join ([f"{k}: {v:.4f}"for k ,v in test_metrics .items ()])
        )
        combined_metrics ={f"val_{k}":v for k ,v in val_metrics .items ()}
        combined_metrics [f"client{self.partition_id}_dice_no_bg"]=test_metrics ["dice_no_bg"]
        combined_metrics [f"client{self.partition_id}_iou_no_bg"]=test_metrics ["iou_no_bg"]
        return val_loss ,len (self .valloader .dataset ),combined_metrics 



def client_fn (context :Context ):
    net =Net (in_channels =3 ,out_channels =1 )

    partition_id =context .node_config ["partition-id"]
    num_partitions =context .node_config ["num-partitions"]


    use_hist_match =bool (context .run_config .get ("use-hist-match",True ))
    reference_client_idx =int (context .run_config .get ("reference-client-idx",0 ))
    n_ref_samples =int (context .run_config .get ("n-ref-samples",64 ))


    trainloader ,valloader ,testloader =load_data (
    partition_id ,num_partitions ,
    use_hist_match =use_hist_match ,
    reference_client_idx =reference_client_idx ,
    n_ref_samples =n_ref_samples ,
    )

    local_epochs =context .run_config ["local-epochs"]

    device =torch .device ("cuda:0"if torch .cuda .is_available ()else "cpu")
    print (f"[DEBUG] Client {partition_id} sees device: {device}",flush =True )

    return FlowerClient (
    net ,trainloader ,valloader ,testloader ,
    local_epochs ,partition_id ,num_partitions ,device 
    ).to_client ()


app =ClientApp (client_fn )
