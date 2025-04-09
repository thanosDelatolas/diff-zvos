import torch
import torch.nn.functional as F
import gc


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def restrict_neighborhood(h, w, cfg):
    size_mask_neighborhood = int(cfg.propagation_params.size_mask_neighborhood)
   
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def label_propagation_feat_cosine_similarity(cfg, feat_tar, h, w, list_frame_feats, list_segs, mask_neighborhood=None, return_aff=False):
    
    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w
    
    feat_tar = F.normalize(feat_tar, dim=1, p=2) # hw x dim
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / cfg.propagation_params.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if cfg.propagation_params.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, cfg)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

   
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    if cfg.propagation_params.topk > 0:
        tk_val, _ = torch.topk(aff, dim=0, k=cfg.propagation_params.topk)
        tk_val_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    if return_aff:
        return seg_tar, return_feat_tar, mask_neighborhood, aff
    else:
        return seg_tar, return_feat_tar, mask_neighborhood
    


def label_propagation_feat_euclidean(cfg, feat_tar, h, w, list_frame_feats, list_segs, mask_neighborhood=None):
    
    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w
    
    feat_tar = F.normalize(feat_tar, dim=1, p=2) # hw x dim
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)

    f_s = feat_sources.permute(0,2,1)
    aff = -torch.cdist(feat_tar, f_s, p=2)
    aff = torch.exp(aff / cfg.propagation_params.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if cfg.propagation_params.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, cfg)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

   
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=cfg.propagation_params.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)

    return seg_tar, return_feat_tar, mask_neighborhood


def label_propagation_feat_l1(cfg, feat_tar, h, w, list_frame_feats, list_segs, mask_neighborhood=None):
    
    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w
    
    feat_tar = F.normalize(feat_tar, dim=1, p=2) # hw x dim
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    
    f_s = feat_sources.permute(0,2,1)
    aff = -torch.cdist(feat_tar, f_s, p=1)
    aff = torch.exp(aff / cfg.propagation_params.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if cfg.propagation_params.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, cfg)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

   
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=cfg.propagation_params.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)

    return seg_tar, return_feat_tar, mask_neighborhood


def aggregate(prob):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=1, keepdim=True),
        prob
    ], 1).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))
    return logits



def label_propagation_oracle(cfg, feat_tar, h, w, list_frame_feats, list_segs, mask_neighborhood=None, return_aff=False, used_gt_segs=None, curr_gt_seg=None):
    """
    filters out fg-bg correspondences by oracle.
    """
    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w
    
    feat_tar = F.normalize(feat_tar, dim=1, p=2) # hw x dim
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / cfg.propagation_params.temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if cfg.propagation_params.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, cfg)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

   
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)

    # filter out fg-bg correspedneces by oracle.
    mem_masks = []
    for i in range(len(used_gt_segs)):
        mem_m = used_gt_segs[i]
        mem_m = mem_m[:,1] # fg
        mem_masks.append(mem_m.flatten())
    mem_masks = torch.cat(mem_masks,dim=0)
    q_mask = curr_gt_seg.flatten()
    
    # mask_affinity: 1 if mem_masks[i] == q_mask[j] else 0
    mask_affinity = (mem_masks[:, None] == q_mask[None, :]).float()
    mask_affinity = mask_affinity.to(aff.device)
    aff = aff * mask_affinity


    if cfg.propagation_params.topk > 0:
        tk_val, _ = torch.topk(aff, dim=0, k=cfg.propagation_params.topk)
        tk_val_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    if return_aff:
        return seg_tar, return_feat_tar, mask_neighborhood, aff
    else:
        return seg_tar, return_feat_tar, mask_neighborhood