import torch
import torch.nn as nn

from config.base_config import Config
from modules.text_transformer import text_transformer
from modules.video_transfomer import video_transformer
class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        self.n_text_samples = self.config.n_text_samples
        self.n_video_samples = self.config.n_video_samples

        '''
            self.alpha = self.config.alpha ==>> 1e-1
            self.beta = self.config.beta   ==>> 1e-4
        '''
        self.alpha = self.config.alpha
        self.beta = self.config.beta

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        self.text_transformer = text_transformer(config)
        self.video_transformer = video_transformer(config)

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        word_features, text_features = self.clip.get_text_features(**text_data)
        _, video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        if is_train:

            text_pool = self.text_transformer(text_features, video_features)
            text_pooled = torch.diagonal(text_pool, dim1=0, dim2=1).permute(1, 0)
            text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

            video_pooled = self.video_transformer(text_features, video_features)

            return text_features, text_pooled, video_features, video_pooled
        else:
            return text_features, video_features

