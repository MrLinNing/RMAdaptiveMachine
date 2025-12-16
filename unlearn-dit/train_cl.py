# References:
# DiT: https://github.com/facebookresearch/DiT
# --------------------------------------------------------

import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# my
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets import ImageFolder

from tools.lora import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_lora_params(model):
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 使用 module_name.parameter_name 作为键，确保唯一性
            lora_params[f'{name}.A.weight'] = module.A.weight.data.cpu()
            lora_params[f'{name}.B.weight'] = module.B.weight.data.cpu()
    return lora_params

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, subfolders=None):
        super(CustomImageFolder, self).__init__()
        self.transform = transform
        self.subfolders = subfolders  # Now a list of subfolder names
        # Find classes and class_to_idx based on the subfolders in each class directory
        self.classes, self.class_to_idx = self._find_classes(root)
        # Make a dataset of all images in the specified subfolders
        self.samples = self._make_dataset(root, self.class_to_idx)

    def _find_classes(self, dir):
        """
        Find classes in dataset directory based on subfolders.
        Each class name is a combination of its parent and subfolder name.

        Params:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx)
        """
        classes = []
        for parent in os.scandir(dir):
            if parent.is_dir():
                parent_folder_name = os.path.basename(parent.path)
                for subfolder in os.scandir(parent.path):
                    if subfolder.is_dir() and (self.subfolders is None or subfolder.name in self.subfolders):
                        class_name = f"{parent_folder_name}-{subfolder.name}"
                        classes.append(class_name)
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        """
        Make a dataset of all images in the specified subfolders with class names as parent_subfolder.

        Params:
            dir (string): Root directory path.
            class_to_idx (dict): Dictionary mapping class names to indices.

        Returns:
            list: List of (image_path, class_index) tuples
        """
        images = []
        dir = os.path.expanduser(dir)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, *target_class.split('-'))  # Split the class name to get parent and subfolder
            for root, _, fnames in sorted(os.walk(target_dir)):
                if root.endswith(target_class.split('-')[-1]):  # Check if the directory ends with the subfolder name
                    for fname in sorted(fnames):
                        if fname.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp')):
                            path = os.path.join(root, fname)
                            item = (path, class_index)
                            images.append(item)
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class SubsetForLabels(Dataset):
    def __init__(self, dataset, label_range):
        """
        Initializes the SubsetForLabels dataset with a range of labels.

        Args:
            dataset (Dataset): The original dataset.
            label_range (range): A range of integer labels to include in the subset.
        """
        self.dataset = dataset
        self.label_range = label_range

        # Filter samples to include only those within the specified label range
        self.filtered_samples = [sample for sample in self.dataset.samples if sample[1] in self.label_range]

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its label.
        """
        # Retrieve the filtered sample
        path, target = self.filtered_samples[index]
        sample = Image.open(path)
        if self.dataset.transform is not None:
            sample = self.dataset.transform(sample)
        return sample, target

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.filtered_samples)

class NumericSortedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        super(NumericSortedImageFolder, self).__init__(root, transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        numeric_sorted_classes = sorted(self.classes, key=lambda x: int(x))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(numeric_sorted_classes)}

        self.samples = [(s[0], self.class_to_idx[self.classes[s[1]]]) for s in self.samples]
        self.targets = [s[1] for s in self.samples]

class MergedDataset(Dataset):
    def __init__(self, dataset, dataset_gen, k):
        self.dataset = dataset
        self.dataset_gen = dataset_gen
        self.k = k
        self.length = max(len(dataset), len(dataset_gen) * k)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx % (self.k + 1) < self.k:
            return self.dataset[idx % len(self.dataset)]
        else:
            return self.dataset_gen[idx % len(self.dataset_gen)]


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        # start a new wandb run to track this script
        if not args.test:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            logger = create_logger('test_log')
            logger.info(f"Test mode: no experiment directory created.")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        convnext=args.convnext,
    )

    # load checkpoint if provided:
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage.cuda(rank))
        model.load_state_dict(checkpoint["model"])

        logger.info(f"Loaded checkpoint from {args.ckpt_path}")
        dist.barrier()

    if args.lora:
        update_model(model, rank=args.lora_r, alpha=args.lora_r*2, device='cuda')
        logger.info(f"Updated model with LoRALinear layers.")

    if args.fc_only:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # load_path = "local_vae/"
    # vae = AutoencoderKL.from_pretrained(load_path).to(device)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)        


    if args.ckpt_lora_path is not None:
        lora_checkpoint = torch.load(args.ckpt_lora_path, map_location=lambda storage, loc: storage.cuda(rank))
        lora_params = lora_checkpoint["lora_params"]
        
        if getattr(args, 'multi_lora', False):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name.endswith('pretrained.weight') and name.replace('pretrained.weight', 'A.weight') in lora_params:
                        # 计算等效合并：W_merged = W_orig + B*A * (alpha/rank)
                        A = lora_params[name.replace('pretrained.weight', 'A.weight')]
                        B = lora_params[name.replace('pretrained.weight', 'B.weight')]
                        scaling = lora_checkpoint.get('alpha', args.lora_r*2) / lora_checkpoint.get('rank', args.lora_r)
                        param.add_(B @ A * scaling)
            
            logger.info(f"Merged pretrained LoRA into base weights and kept new trainable LoRA")
        else:
            for name, param in model.named_parameters():
                if name in lora_params:
                    param.data.copy_(lora_params[name])
            logger.info(f"Loaded single LoRA parameters from {args.ckpt_lora_path}")
        
        dist.barrier()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = CustomImageFolder(args.data_path, transform=transform, subfolders=args.subfolders)
    if not args.cl and not args.ul and not args.ulcl:
        dataset = SubsetForLabels(dataset, range(args.cl_num_classes_begin, args.cl_num_classes_begin + args.cl_num_classes))

    if args.cl or args.ulcl:
        dataset = SubsetForLabels(dataset, range(args.old_num_classes_begin, args.old_num_classes_begin + args.old_num_classes))
        dataset_gen = CustomImageFolder(args.data_path, transform=transform, subfolders=args.subfolders)
        dataset_gen = SubsetForLabels(dataset_gen, range(args.cl_num_classes_begin, args.cl_num_classes_begin + args.cl_num_classes))
        dataset = MergedDataset(dataset, dataset_gen, args.k_gr)

    if args.ul:
        dataset = SubsetForLabels(dataset, range(args.ul_num_classes_begin_gen, args.ul_num_classes_begin_gen + args.ul_num_classes_gen))


    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if args.ul or args.ulcl:
        # -------- Unlearning -------- #
        mu_type = 'CG'
        if args.ul_classes is not None:
            mu_classes = args.ul_classes
        logger.info(f"Machine Unlearning type: {mu_type}, classes: {mu_classes}")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            if args.ul or args.ulcl:
                loss_dict = diffusion.training_losses_ul(model, x, t, model_kwargs, mu_type=mu_type, mu_classes=mu_classes, gamma=5e-2)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()

            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.lora:
                        # get LoRA parameters
                        lora_params = get_lora_params(model)
                        
                        checkpoint = {"lora_params": lora_params}
                        
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved LoRA parameters to {checkpoint_path}")
                    else:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--ckpt-lora-path", type=str, default=None)
    parser.add_argument("--subfolders", type=str, nargs='+', default=None,
                    help="A list of subfolder names")
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--cl", action='store_true', default=False)
    parser.add_argument("--ul", action='store_true', default=False)
    parser.add_argument("--ulcl", action='store_true', default=False)
    parser.add_argument("--ul-classes", type=int, nargs='+', default=None)
    parser.add_argument("--ul-num-classes-gen", type=int, default=10)
    parser.add_argument("--ul-num-classes-begin-gen", type=int, default=None)

    parser.add_argument("--old-num-classes", type=int, default=10)
    parser.add_argument("--old-num-classes-begin", type=int, default=None)
    parser.add_argument("--cl-num-classes", type=int, default=10)
    parser.add_argument("--cl-num-classes-begin", type=int, default=None)
    parser.add_argument("--data-path-gen", default=None)
    parser.add_argument("--cl-num-classes-gen", type=int, default=10)
    parser.add_argument("--cl-num-classes-begin-gen", type=int, default=0)
    parser.add_argument("--k-gr", type=int, default=2)

    parser.add_argument("--lora", action='store_true', default=False)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-scale", action='store_true', default=False)
    parser.add_argument("--multi-lora", action='store_true', default=False)

    parser.add_argument("--convnext", action='store_true', default=False)
    parser.add_argument("--fc-only", action='store_true', default=False)

    args = parser.parse_args()

    assert not (args.cl and args.ul), "Cannot use CL and UL at the same time."
    main(args)
