"""FALCON-aware `train_one_epoch` replacement.

For each local search space of size |M|:
  1. Inference-forward to cache (vision_cls, language_cls) features for M
     samples. No grad.
  2. Compute symmetric similarity S = sim + sim.T. Build sorted, softmax-
     normalized quantile features S̃ (M × m).
  3. Scheduler πϕ(S̃) → (α, β) per anchor → sample q. Keep log_p(q | α, β)
     as a tensor with grad wrt scheduler parameters.
  4. Construct |M|/B mini-batches via GRIT-VLP-style quantile sampling.
  5. For each minibatch: ΔITC reward, VLP step, accumulate REINFORCE loss.
  6. After the chunk: single backward + step for the scheduler. Using the
     original log_p (computed at the start of the chunk) is correct policy
     gradient — the q's were sampled from that policy.
"""
import math
import sys
import time

import torch

from .batch_construction import build_falcon_minibatches, build_quantile_features
from .scheduler import FalconScheduler


def _collate_from_dataset(dataset, indices):
    items = [dataset[i] for i in indices]
    out = {}
    for k, v0 in items[0].items():
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items], dim=0)
        else:
            out[k] = torch.tensor([it[k] for it in items])
    return out


def _move_to(batch, device, fp16=False):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
            if fp16 and k.startswith("image"):
                batch[k] = batch[k].half()
    return batch


@torch.no_grad()
def _encode_chunk(model, dataset, chunk_indices, device, micro_batch: int = 64):
    was_training = model.training
    model.eval()
    img_feats, txt_feats = [], []
    try:
        for s in range(0, len(chunk_indices), micro_batch):
            sub = chunk_indices[s:s + micro_batch]
            batch = _collate_from_dataset(dataset, sub)
            batch = _move_to(batch, device)
            with torch.amp.autocast("cuda"):
                vc, _ = model(image=batch["image"], only_infer=True)
                _, lc = model(text_description=batch["language_tokens"],
                              padding_mask=batch["padding_mask"], only_infer=True)
            img_feats.append(vc.float())
            txt_feats.append(lc.float())
    finally:
        model.train(was_training)
    return torch.cat(img_feats, dim=0), torch.cat(txt_feats, dim=0)


def falcon_train_one_epoch(
    model, dataset, optimizer, device, handler, epoch, start_steps,
    lr_schedule_values, loss_scaler, max_norm,
    scheduler_module: FalconScheduler, scheduler_optimizer,
    reward_fn,
    M: int = 1024, B: int = 184, m_bins: int = 16,
    update_freq: int = 1, model_ema=None, log_writer=None,
    reward_every: int = 1,
    log_every: int = 10,
):
    """Drop-in replacement for `engine_for_finetuning.train_one_epoch` when
    `--use_falcon` is enabled. `dataset` is passed directly (not a DataLoader)
    so mini-batch construction can index arbitrary samples.
    """
    model.train(True)

    n = len(dataset)
    perm = torch.randperm(n).tolist()
    total_chunks = n // M
    global_step = start_steps
    log_loss_sum, log_loss_n = 0.0, 0
    log_delta_sum, log_delta_n = 0.0, 0
    epoch_loss_sum, epoch_loss_n = 0.0, 0

    for chunk_i in range(total_chunks):
        chunk_indices = perm[chunk_i * M: (chunk_i + 1) * M]

        # 1. Inference forward of M samples
        t_chunk = time.time()
        img_M, txt_M = _encode_chunk(model, dataset, chunk_indices, device)
        sim = img_M @ txt_M.t()                                            # (M, M)
        S = (sim + sim.t()).detach()                                       # symmetric

        # 2. Scheduler → (α, β); sample q + log_p
        S_tilde = build_quantile_features(S, m_bins=m_bins)
        alpha, beta = scheduler_module(S_tilde)                            # (M,), (M,)
        q, log_p = FalconScheduler.sample_q(alpha, beta)                   # log_p ∝ ∇ϕ

        # 3. Build mini-batches in local index space
        local_batches = build_falcon_minibatches(S, q, batch_size=B)

        # 4. Train + accumulate REINFORCE loss
        sched_loss_accum = torch.zeros((), device=device)
        n_reward = 0

        for b_local in local_batches:
            global_indices = [chunk_indices[i] for i in b_local]
            batch = _collate_from_dataset(dataset, global_indices)
            batch = _move_to(batch, device, fp16=(loss_scaler is None))

            # LR schedule
            if lr_schedule_values is not None and global_step < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[global_step] * pg["lr_scale"]

            do_reward = (reward_fn is not None) and (global_step % reward_every == 0)
            if do_reward:
                itc_before = reward_fn.loss(model)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                results = handler.train_batch(
                    model,
                    image=batch["image"],
                    language_tokens=batch["language_tokens"],
                    padding_mask=batch["padding_mask"],
                    image_id=batch["image_id"],
                )
            loss = results.pop("loss")
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=False, update_grad=True)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if model_ema is not None:
                model_ema.update(model)

            if do_reward:
                itc_after = reward_fn.loss(model)
                delta = (itc_before - itc_after).detach()                 # positive ⇒ improved
                # REINFORCE: maximize E[Δ · log π] ⇒ minimize -Δ · log π.
                idx = torch.tensor(b_local, device=device)
                sched_loss_accum = sched_loss_accum + (-delta * log_p[idx].sum())
                log_delta_sum += delta.item(); log_delta_n += 1
                n_reward += 1

            log_loss_sum += loss_value; log_loss_n += 1
            epoch_loss_sum += loss_value; epoch_loss_n += 1
            global_step += 1

            if global_step % log_every == 0:
                d_avg = (log_delta_sum / log_delta_n) if log_delta_n else 0.0
                tl_avg = log_loss_sum / max(1, log_loss_n)
                lr_cur = optimizer.param_groups[0]["lr"]
                print(f"[FALCON] step={global_step} chunk={chunk_i + 1}/{total_chunks} "
                      f"loss={tl_avg:.4f} ΔITC={d_avg:+.5f} lr={lr_cur:.2e} "
                      f"loss_scale={loss_scale_value:.0f} grad_norm={grad_norm:.3f}")
                if log_writer is not None:
                    log_writer.update(head="train", loss=tl_avg,
                                      delta_itc=d_avg, lr=lr_cur)
                log_loss_sum, log_loss_n = 0.0, 0
                log_delta_sum, log_delta_n = 0.0, 0

        # 5. Apply REINFORCE update once per chunk
        if n_reward > 0:
            sched_loss_accum = sched_loss_accum / n_reward
            scheduler_optimizer.zero_grad()
            sched_loss_accum.backward()
            torch.nn.utils.clip_grad_norm_(scheduler_module.parameters(), 1.0)
            scheduler_optimizer.step()

        # Free per-chunk tensors
        del img_M, txt_M, sim, S, S_tilde, alpha, beta, q, log_p, sched_loss_accum
        torch.cuda.empty_cache()

        chunk_time = time.time() - t_chunk
        if chunk_i % 5 == 0:
            print(f"[FALCON] chunk {chunk_i + 1}/{total_chunks} took {chunk_time:.1f}s")

    return {
        "loss": epoch_loss_sum / max(1, epoch_loss_n),
        "epoch_global_step": global_step,
    }
