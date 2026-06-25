import torch
import torch.nn as nn
import clip
from config import CLIP_MODEL

def _class_token_ids(name: str) -> torch.Tensor:
    tokens  = clip.tokenize(name)[0]        
    eos_pos = tokens.argmax().item()        
    return tokens[1:eos_pos]               

class LearnablePrompt(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx: int = 16, device="cpu"):
        super().__init__()
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx
        dtype   = clip_model.dtype                              
        ctx_dim = clip_model.ln_final.weight.shape[0]           
        with torch.no_grad():
            init_tok = clip.tokenize("a photo of a").to(device)
            init_emb = clip_model.token_embedding(init_tok).squeeze(0)  
            init_ctx = init_emb[1:5].clone()                             
        ctx = torch.zeros(n_ctx, ctx_dim, dtype=dtype, device=device)
        n_init = min(4, n_ctx)
        ctx[n_ctx - n_init:] = init_ctx[:n_init]   
        self.ctx = nn.Parameter(ctx)
        ids_list = [_class_token_ids(n) for n in class_names]
        lengths  = [ids.shape[0] for ids in ids_list]
        max_K    = max(lengths)
        padded_ids = torch.zeros(self.n_cls, max_K, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            padded_ids[i, :ids.shape[0]] = ids
        with torch.no_grad():
            padded_ids = padded_ids.to(device)
            cls_emb = clip_model.token_embedding(
                padded_ids.reshape(-1)                      
            ).reshape(self.n_cls, max_K, ctx_dim)   
        self.register_buffer("class_emb",     cls_emb.clone())                   
        self.register_buffer("token_lengths", torch.tensor(lengths, dtype=torch.long))  
        with torch.no_grad():
            dummy     = clip.tokenize("").to(device)            
            dummy_emb = clip_model.token_embedding(dummy)[0]   
        self.register_buffer("sos_emb", dummy_emb[0].clone())  
        self.register_buffer("eos_emb", dummy_emb[1].clone())
        self.register_buffer(
            "positional_embedding", clip_model.positional_embedding.data.clone()   
        )
        self.register_buffer(
            "text_projection", clip_model.text_projection.data.clone()             
        )
        self.transformer = clip_model.transformer
        self.ln_final    = clip_model.ln_final
        self.dtype       = dtype

    def build_prompt_embeddings(self):
        n_ctx, n_cls = self.n_ctx, self.n_cls
        ctx          = self.ctx                         
        dtype, dev   = self.dtype, ctx.device
        dim          = ctx.shape[-1]
        emb = torch.zeros(n_cls, 77, dim, dtype=dtype, device=dev)
        emb[:, 0] = self.sos_emb.to(dtype)
        emb[:, 1:1 + n_ctx] = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        cls_emb = self.class_emb.to(dtype)             
        K_lens  = self.token_lengths            
        for i in range(n_cls):
            K_i = K_lens[i].item()
            emb[i, 1 + n_ctx: 1 + n_ctx + K_i] = cls_emb[i, :K_i]
            emb[i, 1 + n_ctx + K_i]             = self.eos_emb.to(dtype)

        eot_pos = (1 + n_ctx + K_lens).to(dev)        
        return emb, eot_pos

    def encode_text(self):
        emb, eot_pos = self.build_prompt_embeddings()         
        x = emb + self.positional_embedding.to(emb.dtype)     
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                                
        x = self.ln_final(x).to(self.dtype)
        feat = x[torch.arange(self.n_cls, device=x.device), eot_pos]  
        feat = feat @ self.text_projection.to(feat.dtype)           
        feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.float()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:

        text_features = self.encode_text()                       
        return image_features.float() @ text_features.T           

class CoOpCLIP(nn.Module):

    def __init__(self, class_names, n_ctx: int = 16, device="cpu"):
        super().__init__()
        clip_model, self.preprocess = clip.load(CLIP_MODEL, device=device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad_(False)

        self.image_encoder = clip_model.visual
        self.dtype         = clip_model.dtype
        self.register_buffer("logit_scale", clip_model.logit_scale.data.clone())

        self.prompt = LearnablePrompt(
            clip_model, class_names, n_ctx=n_ctx, device=device
        )

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.image_encoder(images.type(self.dtype))
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(images)          
        logit_scale    = self.logit_scale.exp().float()
        return self.prompt(image_features) * logit_scale   
