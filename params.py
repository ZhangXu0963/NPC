import argparse

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiments", type=str, default='ex0',
                        help="Experiment Name")
    
    parser.add_argument("--dataset_root", type=str,
                        default='./dataset/MSCOCO', 
                        help="Path to the dataset",)
    parser.add_argument("--num_anns", type=int, 
                        choices=[5, 1], #5 for MSCOCO and Flickr30K; 1 for CC120K
                        default=1, 
                        help="Number of annotations per image.")
    parser.add_argument("--dev_length", type=int, 
                        choices=[5000, 1000], # 5000 for MSCOCO and Flickr30K; 1000 for CC120K
                        default=1000, 
                        help="Number of samples for validation.")
    
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training.")
    parser.add_argument("--dataset", type=str, default='cc', #coco/f30k/cc
                        choices=['coco', 'f30k', 'cc'],
                        help="The name of dataset.")
    
    parser.add_argument("--noise_ratio", type=float, default=0,
                        help="Set the ratio of noise in training.")
    
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of full epochs to train for (only works if --max_steps is None).")
    
    parser.add_argument("--eval", action="store_true", default=False,
                        help="train or eval",)
    
    parser.add_argument("--eval_batch_size", type=int, default=64, 
                        help="Batch size for eval per GPU.")  
    
    parser.add_argument("--display", type=int, default=20, 
                        help="The steps interval of display.")
    
    parser.add_argument( "--checkpoint_path", type=str, default='./logs',
                        help="Path to save the model",)
    
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="The number of workers for dataloader.") 
          
    parser.add_argument("--name", type=str,default="NPC",
                        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",)
    
    parser.add_argument("--context_length", type=int, default=64, 
                        help="The maximum length of input text (include [CLS] & [SEP] tokens).")
    
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay.")
   
    parser.add_argument("--warmup", type=int, default=500, 
                        help="Number of steps to warmup for.")
    parser.add_argument("--resume", default=None, type=str,
                        help="path to latest checkpoint (default: none)",)

    parser.add_argument("--vision_model",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"],
        default="ViT-B/32",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument("--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank."
    )
    parser.add_argument("--seed", 
        type=int, 
        default=99, 
        help="Random seed."
    )
    args = parser.parse_args()

    return args
