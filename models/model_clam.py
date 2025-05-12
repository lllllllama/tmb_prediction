import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import sys
import os

# 导入本地CLIP模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip.clip import tokenize, load

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
    use_clip: whether to use CLIP for text feature extraction
    fusion_method: method to fuse image and text features ('concat', 'add', 'attention')
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
                 use_clip=True, fusion_method='add', clip_model_name="ViT-B/32"):
        super().__init__()
        self.use_clip = use_clip
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化CLIP模型用于文本特征提取
        if self.use_clip:
            # 加载本地CLIP模型
            self.clip_model, self.preprocess = load(clip_model_name, device=self.device)
            
            # 冻结CLIP模型参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # 获取CLIP文本特征维度
            self.text_feature_dim = self.clip_model.text_projection.shape[1]  # 通常是512
            
            # 特征融合层
            if fusion_method == 'concat':
                # 如果使用拼接，需要调整后续网络的输入维度
                self.size_dict = {"small": [embed_dim + self.text_feature_dim, 512, 256], 
                                 "big": [embed_dim + self.text_feature_dim, 512, 384]}
                # 特征融合投影层
                self.fusion_projection = nn.Identity()
            elif fusion_method == 'add':
                # 如果使用加法融合，需要将文本特征投影到与图像特征相同的维度
                self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
                self.fusion_projection = nn.Linear(self.text_feature_dim, embed_dim)
            elif fusion_method == 'attention':
                # 如果使用注意力融合，需要额外的注意力层
                self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
                self.fusion_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
                self.fusion_norm = nn.LayerNorm(embed_dim)
        else:
            # 不使用CLIP时保持原有尺寸
            self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    def extract_text_features(self, text_descriptions):
        """
        使用CLIP提取文本特征（在CPU上运行以避免CUDA问题）
        Args:
            text_descriptions: 文本描述列表
        Returns:
            text_features: 文本特征张量 [1, D_txt]
        """
        with torch.no_grad():
            # 保存当前设备
            original_device = next(self.clip_model.parameters()).device
            # print(f"CLIP running on: CPU")
            
            # 将CLIP模型暂时移到CPU
            self.clip_model = self.clip_model.to("cpu")
            
            # 在CPU上处理文本
            text_tokens = tokenize(text_descriptions).to("cpu")
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 将模型移回原设备（如果需要）
            if original_device != torch.device("cpu"):
                self.clip_model = self.clip_model.to(original_device)
            
            # 将特征移回GPU设备用于后续处理
            text_features = text_features.to(self.device)
            
            # 转换为Float类型
            text_features = text_features.float()
        return text_features
    
    def fuse_features(self, image_features, text_features):
        """
        融合图像和文本特征
        Args:
            image_features: 图像特征张量 [N, D_img]，其中N是切片数量，D_img是特征维度(1024)
            text_features: 文本特征张量 [1, D_txt]，其中D_txt是CLIP文本特征维度(512)
        Returns:
            fused_features: 融合后的特征张量
        """
        # 将文本特征扩展到与图像特征相同的批次大小
        batch_size = image_features.shape[0]  # N
        expanded_text_features = text_features.expand(batch_size, -1)  # [1, D_txt] -> [N, D_txt]
        
        if self.fusion_method == 'concat':
            # 拼接特征: [N, D_img] + [N, D_txt] -> [N, D_img+D_txt]
            fused_features = torch.cat([image_features, expanded_text_features], dim=1)  # [N, 1536]
            
        elif self.fusion_method == 'add':
            # 投影文本特征: [N, D_txt] -> [N, D_img]
            projected_text = self.fusion_projection(expanded_text_features)  # [N, 1024]
            # 相加: [N, D_img] + [N, D_img] -> [N, D_img]
            fused_features = image_features + projected_text  # [N, 1024]
            
        elif self.fusion_method == 'attention':
            # 调整形状为注意力层所需: [N, D] -> [N, 1, D]
            img_feats = image_features.unsqueeze(1)  # [N, 1, D_img]
            text_feats = expanded_text_features.unsqueeze(1)  # [N, 1, D_txt]
            
            # 多头注意力: [N, 1, D_img] + [N, 1, D_txt] -> [N, 1, D_img]
            attn_output, _ = self.fusion_attention(img_feats, text_feats, text_feats)  # [N, 1, D_img]
            # 残差连接和归一化: [N, 1, D_img] + [N, 1, D_img] -> [N, 1, D_img]
            fused_features = self.fusion_norm(img_feats + attn_output).squeeze(1)  # [N, D_img]
        
        return fused_features
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, text_descriptions=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        前向传播
        Args:
            h: 图像特征 [N, D_img]，其中N是切片数量，D_img是特征维度(1024)
            text_descriptions: 文本描述，字符串或字符串列表
            label: 标签
            instance_eval: 是否进行实例级评估
            return_features: 是否返回特征
            attention_only: 是否只返回注意力权重
        """
        # 如果启用CLIP且提供了文本描述
        if self.use_clip and text_descriptions is not None:
            # 提取文本特征
            text_features = self.extract_text_features(text_descriptions)
            # 融合图像和文本特征
            h = self.fuse_features(h, text_features)
        
        # 以下与原始CLAM_SB相同
        A, h = self.attention_net(h)  # A: [N, 1], h: [N, 512]        
        A = torch.transpose(A, 1, 0)  # A: [1, N]
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h)  # M: [1, 512]
        logits = self.classifiers(M)  # logits: [1, n_classes]
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
                use_clip=True, fusion_method='add', clip_model_name="ViT-B/32"):
        
        # 先初始化基类之外的属性
        nn.Module.__init__(self)
        self.use_clip = use_clip
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化CLIP模型
        if self.use_clip:
            self.clip_model, self.preprocess = load(clip_model_name, device=self.device)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.text_feature_dim = self.clip_model.text_projection.shape[1]
            
            # 特征融合层
            if fusion_method == 'concat':
                self.size_dict = {"small": [embed_dim + self.text_feature_dim, 512, 256], 
                                "big": [embed_dim + self.text_feature_dim, 512, 384]}
                self.fusion_projection = nn.Identity()
            elif fusion_method == 'add':
                self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
                self.fusion_projection = nn.Linear(self.text_feature_dim, embed_dim)
            elif fusion_method == 'attention':
                self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
                self.fusion_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
                self.fusion_norm = nn.LayerNorm(embed_dim)
        else:
            self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        
        # 初始化原有部分
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, text_descriptions=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        # 处理文本特征
        if self.use_clip and text_descriptions is not None:
            text_features = self.extract_text_features(text_descriptions)
            h = self.fuse_features(h, text_features)
        
        # 以下是原始的多分支CLAM前向传播逻辑
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


