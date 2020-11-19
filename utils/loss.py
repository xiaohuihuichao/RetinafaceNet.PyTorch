import torch
import torch.nn.functional as F


if __name__ == "__main__":
    from match import match
    from prior import get_prior
    from iou import jaccard_iou
    from box_transform import point_form, center_form, encode
else:
    from .match import match
    from .prior import get_prior
    from .iou import jaccard_iou
    from .box_transform import point_form, center_form, encode, encode_landmark
    

class Cls_Box_Landmark_Loss():
    def __init__(self, config):
        self.config = config
        self.priors = None
        self.priors_point = None
    
    def __call__(self, preds, gts, neg_rate=3):
        # cls list len 3, [N, h, w, num_prior_per_point, num_classes+1]
        # bbox list len 3, [N, h, w, num_prior_per_point, 4]
        # landmark list len 3, [N, h, w, num_prior_per_point, num_landmark_points]
        cls_out, bbox_out, landmark_out = preds
        
        cls_gt, bbox_gt, landmark_gt = gts
        # [N, n], n 为 obj 的个数
        obj_mask = [i>0 for i in cls_gt]
        # cls list len N
        cls_gt = [i[mask] for i, mask in zip(cls_gt, obj_mask)]
        
        # [N, n, 4]
        bbox_gt = [i[mask] for i, mask in zip(bbox_gt, obj_mask)]
        
        # [N, n, num_points*2]
        landmark_gt = [i[mask] for i, mask in zip(landmark_gt, obj_mask)]
        
        if self.priors == None or self.priors_point == None:
            feature_map_sizes = [i.shape[1:3] for i in cls_out]
            priors = [get_prior(feature_map_size, self.config["image_size"], min_size, max_size, ratio)for feature_map_size, min_size, max_size, ratio in zip(feature_map_sizes, self.config["min_sizes"], self.config["max_sizes"], self.config["ratios"])]
            self.priors = torch.cat(priors, dim=0)
            self.priors_point = point_form(self.priors)
            if self.config["cuda"]:
                self.priors = self.priors.cuda()
                self.priors_point = self.priors_point.cuda()
                
        batch_size = len(cls_out[0])
        # [batch_size, num_prior, 4]
        bbox_out = torch.cat([i.reshape(batch_size, -1, 4) for i in bbox_out], dim=1)
        landmark_out = torch.cat([i.reshape(batch_size, -1, self.config["num_landmark_points"]*2) for i in landmark_out], dim=1)
        cls_out = torch.cat([i.reshape(batch_size, -1) for i in cls_out], dim=1)
        cls_out = cls_out.reshape(batch_size, landmark_out.shape[1], -1) # [B, num_prior, num_classes+1]
        
        cls_loss, bbox_loss, landmark_loss = 0, 0, 0
        for bbox_out_i, cls_out_i, landmark_out_i, bbox_gt_i, cls_gt_i, landmark_gt_i in zip(bbox_out, cls_out, landmark_out, bbox_gt, cls_gt, landmark_gt):
            pos_priors_mask_i, idx_obj_i = match(self.priors_point, bbox_gt_i)
            if pos_priors_mask_i == None or idx_obj_i == None:
                batch_size -= 1
                continue
            
            # bbox loss
            match_prior = self.priors[pos_priors_mask_i]
            match_bbox_gt_center_i = center_form(bbox_gt_i[idx_obj_i])
            # match_gt_i = bbox_gt_center_i
            
            match_bbox_gt_center_encode_i = encode(match_bbox_gt_center_i, match_prior, self.config)
            bbox_loss += F.smooth_l1_loss(match_bbox_gt_center_encode_i, bbox_out_i[pos_priors_mask_i])
            
            # landmark loss, [num_obj, num_point*2]
            match_landmark_gt_i = landmark_gt_i[idx_obj_i]
            landmark_mask = match_landmark_gt_i.sum(1) > 0
            if landmark_mask.sum() > 0:
                match_landmark_gt_encode_i = encode_landmark(match_landmark_gt_i[landmark_mask], match_prior[landmark_mask], self.config)
                landmark_loss += F.smooth_l1_loss(match_landmark_gt_encode_i, landmark_out_i[pos_priors_mask_i])
            
            # cls loss
            num_pos = match_prior.shape[0]
            cls_t = torch.zeros(cls_out_i.shape[0]).type(torch.int64)
            if self.config["cuda"]:
                cls_t = cls_t.cuda()
            cls_t[pos_priors_mask_i] = cls_gt_i[idx_obj_i].reshape(-1)
            
            ce_loss = F.cross_entropy(cls_out_i, cls_t, reduction="none")
            neg_loss = ce_loss[~pos_priors_mask_i]
            neg_loss_sort = torch.sort(neg_loss, descending=True)[0]
            num_neg = round(min(num_pos*neg_rate, neg_loss.shape[0]))
            cls_loss += (ce_loss[pos_priors_mask_i].sum() + neg_loss_sort[0:num_neg].sum()) / (num_pos + num_neg)
        return bbox_loss / batch_size, landmark_loss / batch_size, cls_loss / batch_size
