# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# --------------------------------------------------------

import sys
import os
import copy
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# Imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses.vicreg import vicreg_loss
from masks.utils import apply_masks
from utils.distributed import init_distributed, AllReduce
from utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from utils.tensors import repeat_interleave_batch
from datasets.ucla import make_ucla
from helper import (
    load_checkpoint,
    init_model,
    init_opt)

# -- Configuration globale
log_timings = True
log_freq = 10
checkpoint_freq = 10 
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PARAMÈTRES DEPUIS LE FICHIER CONFIG
    # ----------------------------------------------------------------------- #
    use_bfloat16 = args['meta']['use_bfloat16']
    use_vicreg = args['meta'].get('use_vicreg', True)
    use_multiscale = args['meta'].get('use_multiscale', True)
    accumulation_steps = args['meta']['accumulation_steps']
    attn_mode = args['meta']['attn_mode']
    add_w = args['meta']['add_w']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    mask_mode = args['meta']['mask_mode']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    crop_size = args['data']['crop_size']
    gradient_csv_path = args['data']['gradient_csv_path']

    if mask_mode == 'roi_mask':
        allow_overlap = args['mask']['allow_overlap']
        patch_size = args['mask']['patch_size']
        min_keep = args['mask']['min_keep']
        enc_mask_scale = args['mask']['enc_mask_scale']
        pred_mask_R_scale = args['mask']['pred_mask_R_scale']
        pred_mask_T_scale = args['mask']['pred_mask_T_scale']
        pred_mask_T_roi_scale = args['mask']['pred_mask_T_roi_scale']
        pred_mask_R_roi_scale = args['mask']['pred_mask_R_roi_scale']

    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['ref_lr'] # Utilise ref_lr comme LR de référence
    final_lr = args['optimization']['final_lr']

    # -- Logging setup
    folder = args['logging']['folder']
    if not load_model:
        folder += '_' + datetime.now().strftime("%y%m%d-%H%M%S")
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # INITIALISATION DISTRIBUTÉE
    # ----------------------------------------------------------------------- #
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')

    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'), ('%.5f', 'mask-B'), ('%d', 'time (ms)'))

    # -- Chargement sécurisé des gradients
    def load_gradient():
        if gradient_csv_path is None or not os.path.exists(gradient_csv_path):
            logger.warning("Gradient file not found or None. Using zeros.")
            num_patches = (crop_size[0] // patch_size) * (crop_size[1] // patch_size)
            return torch.zeros(1, num_patches, pred_emb_dim)
        df = pd.read_csv(gradient_csv_path, header=None)
        return torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    
    gradient = load_gradient().to(device, non_blocking=True)
    
    # -- Initialisation Modèles
    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim, model_name=model_name,
        gradient_pos_embed=gradient, attn_mode=attn_mode, add_w=add_w)
    target_encoder = copy.deepcopy(encoder)

    # -- Masques et Loader
    from masks.spatialtemporal_multiblock import MaskCollator_fmri as MBMaskCollator
    mask_collator = MBMaskCollator(
        input_size=(crop_size[0], crop_size[1]), patch_size=patch_size,
        enc_mask_scale=enc_mask_scale, pred_mask_R_scale=pred_mask_R_scale,
        pred_mask_T_scale=pred_mask_T_scale, pred_mask_T_roi_scale=pred_mask_T_roi_scale,
        pred_mask_R_roi_scale=pred_mask_R_roi_scale, allow_overlap=allow_overlap,
        min_keep=min_keep, use_multiscale=use_multiscale)

    # On utilise le chemin relatif ou absolu configuré
    data_path = args['data'].get('data_dir', 'data/timeseries')
    _, unsupervised_loader = make_ucla(
        batch_size=batch_size, collator=mask_collator,
        num_workers=num_workers, data_dir=data_path)
    
    unsupervised_sampler = None # Défini par défaut à None pour GPU unique
    ipe = len(unsupervised_loader)
    logger.info(f'Number of batches per epoch: {ipe}')

    # -- Optimizer & DDP Fix
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor, wd=wd, final_wd=final_wd,
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr, iterations_per_epoch=ipe,
        warmup=warmup, num_epochs=num_epochs, ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16, accumulation_steps=accumulation_steps)

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file else latest_path
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device, r_path=load_path, encoder=encoder, predictor=predictor,
            target_encoder=target_encoder, opt=optimizer, scaler=scaler)

    # ----------------------------------------------------------------------- #
    # BOUCLE D'ENTRAÎNEMENT (CORRIGÉE)
    # ----------------------------------------------------------------------- #
    training_text = os.path.join(folder, 'training_log.txt')

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Starting Epoch {epoch + 1}')

        if unsupervised_sampler is not None:
            unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            
            def train_step():
                imgs = udata['fmri'].to(device, non_blocking=True)
                m_enc = [u.to(device, non_blocking=True) for u in masks_enc]
                m_pred = [u.to(device, non_blocking=True) for u in masks_pred]
                
                maskA_meter.update(len(m_enc[0][0]))
                maskB_meter.update(len(m_pred[0][0]))

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    # Target
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        h = apply_masks(h, m_pred)
                        h = repeat_interleave_batch(h, len(imgs), repeat=len(m_enc))
                    
                    # Context
                    z = encoder(imgs, m_enc)
                    z = predictor(z, m_enc, m_pred)
                    
                    # Loss
                    p_loss = F.smooth_l1_loss(z, h)
                    if use_vicreg:
                        r_loss = vicreg_loss(z, lambda_reg=0.04)
                        loss = (p_loss + r_loss) / accumulation_steps
                    else:
                        loss = p_loss / accumulation_steps

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    if (itr + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    if (itr + 1) % accumulation_steps == 0:
                        optimizer.step()

                if (itr + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                    # Momentum update
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                            pk.data.mul_(m).add_((1.-m) * pq.detach().data)
                else:
                    _new_lr = optimizer.param_groups[0]['lr']
                    _new_wd = optimizer.param_groups[0]['weight_decay']

                return float(loss * accumulation_steps), _new_lr, _new_wd

            (loss_val, current_lr, current_wd), etime = gpu_timer(train_step)
            loss_meter.update(loss_val)
            time_meter.update(etime)

            # Logging
            if (itr % log_freq == 0):
                csv_logger.log(epoch + 1, itr, loss_val, maskA_meter.val, maskB_meter.val, etime)
                msg = f'[{epoch+1}, {itr:5d}] loss: {loss_meter.avg:.3f} [lr: {current_lr:.2e}] ({etime:.1f} ms)'
                logger.info(msg)
                with open(training_text, 'a') as f:
                    f.write(msg + '\n')

        # Sauvegarde fin d'époque
        save_dict = {
            'encoder': encoder.state_dict(), 'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(), 'opt': optimizer.state_dict(),
            'epoch': epoch + 1, 'loss': loss_meter.avg
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=epoch + 1))

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cmd_args = parser.parse_args()
    with open(cmd_args.config, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    main(get_args())