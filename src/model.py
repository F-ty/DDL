import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math

class ITFD(nn.Module):
    def __init__(self, args, hidden_dim=1024, dropout = 0.5):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pt_path)
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer(args.backbone)
        
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        
        self.combiner_fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                         nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())

        self.del_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.prs_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.new_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.trip = nn.TripletMarginLoss(margin=1.0)
        self.cos = nn.CosineSimilarity()
    
    def extract_img_fea(self, x):
        image_features = self.clip.encode_image(x)
        return image_features

    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).cuda(0)
        text_features = self.clip.encode_text(txt)
        return text_features

    def extract_query(self, textual_query, visual_query):
        textual_query = F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)

        del_mask = self.del_proj(textual_query)
        prs_mask = self.prs_proj(textual_query)
        new_mask = self.new_proj(textual_query)

        prs_ref = F.normalize(prs_mask * visual_query, p=2, dim=-1)
        new_text = F.normalize(new_mask * textual_query, p=2, dim=-1)

        combined_feature = self.combiner_fc(torch.cat([new_text, prs_ref], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * new_text + (1 - dynamic_scaler) * prs_ref

        # id_only
        # combined_feature = self.combiner_fc(torch.cat([textual_query, prs_ref], dim=-1))
        # dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        # query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * prs_ref

        # td_only
        # combined_feature = self.combiner_fc(torch.cat([new_text, visual_query], dim=-1))
        # dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        # query = dynamic_scaler * new_text + (1 - dynamic_scaler) * visual_query

        return F.normalize(query, p=2, dim=-1), textual_query, visual_query, del_mask, prs_mask, new_mask
    

    def extract_target(self, target_img):
        target_img_fea = self.extract_img_fea(target_img)
        return F.normalize(target_img_fea, p=2, dim=-1)
    

    def compute_loss(self, textual_query, visual_query, target_img):
        query, mod, ref, del_mask, prs_mask, new_mask = self.extract_query(textual_query, visual_query) 
        target = self.extract_target(target_img)  

        loss = {}  
        loss['ranking'] = self.ranking_nce_loss(query, mod, ref, target, del_mask, prs_mask, new_mask)                                                                                    
        
        return loss

    def ranking_nce_loss(self, query, mod, ref, target, del_mask, prs_mask, new_mask) :
        x = torch.mm(query, target.t())
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda(0)
        loss = F.cross_entropy(self.loss_weight * x, labels)

        del_text = F.normalize(del_mask * mod, p=2, dim=-1)
        new_text = F.normalize(new_mask * mod, p=2, dim=-1)
        con1 = self.trip(del_text, ref, target)
        con2 = self.trip(new_text, target, ref)
        loss2 = con1 + con2

        prs_ref = F.normalize(prs_mask * ref, p=2, dim=-1)
        prs_tar = F.normalize(prs_mask * target, p=2, dim=-1)
        loss3 = torch.mean(1.0 - self.cos(prs_ref, prs_tar))

        # return loss
        # return loss + 0.1 * loss2
        # return loss + 0.5 * loss3
        return loss + 0.2 * loss2 + 0.7 * loss3
