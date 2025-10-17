# import torch
# import torch.nn as nn
# # from mld.models.operator.style_blocks import *

# """
# discriminator
# 1) conv w/o acti or norm, keeps dims
# 2) [disc_down_n] *
#         (ActiFirstResBlk(channel[i], channel[i])
#         + ActiFirstResBlk(channel[i], channel[i + 1])
#         + AvgPool(pool_size, pool_stride))
# 3) 2 ActiFirstResBlks that keep dims(channel[-1])
# 4) conv, [channel[-1] -> num_classes]

# """

# # class StyleClassifier(nn.Module):
# #     def __init__(self,
# #         disc_channels: int = [263, 256, 128],
# #         disc_down_n: int = 2,
# #         disc_kernel_size: int = 6,
# #         disc_stride: int = 1,
# #         disc_pool_size: int = 3,
# #         disc_pool_stride: int = 2,
# #         num_classes: int = 100,
# #         **kwargs) -> None:
# #         super(StyleClassifier, self).__init__()

# #         channels = disc_channels
# #         down_n = disc_down_n
# #         ks = disc_kernel_size
# #         stride = disc_stride
# #         pool_ks = disc_pool_size
# #         pool_stride = disc_pool_stride

# #         assert down_n + 1 == len(channels)

# #         cnn_f = ConvLayers(kernel_size=ks, in_channels=channels[0], out_channels=channels[0])

# #         for i in range(down_n):
# #             cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i], stride=stride, acti='lrelu', norm='none')]
# #             cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i + 1], stride=stride, acti='lrelu', norm='none')]
# #             cnn_f += [get_conv_pad(pool_ks, pool_stride)]
# #             cnn_f += [nn.AvgPool1d(kernel_size=pool_ks, stride=pool_stride)]

# #         cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]
# #         cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]

# #         self.cnn_f = nn.Sequential(*cnn_f)
        
# #         self.linear = nn.Linear(in_features=128, out_features=num_classes)
        
# #         self.dropout = nn.Dropout(0.2, inplace=True)

# #     def forward(self, x):
        
# #         x = x.permute(0, 2, 1)
        
# #         feat = self.cnn_f(x)
        
# #         # bs, channel, conv_dims。conv_dims随输入序列长度不同而不同，所以做mean
# #         feat2 = torch.mean(feat, dim=2)
        
# #         pred = self.linear(feat2)

# #         pred = self.dropout(pred)
        
# #         debug = False
# #         if debug:
# #             print('x', x.shape)
# #             print('feat', feat.shape) # bs, channel, conv_dims。conv_dims随输入序列长度不同而不同
# #             print('feat2', feat2.shape) # bs, channel, conv_dims。conv_dims随输入序列长度不同而不同
# #             print('pred', pred.shape)
        
# #         return pred



# class StyleClassifier(nn.Module):
#     def __init__(self,
#         input_dim: list = 263,
#         latent_dim: list = 256,
#         ff_size: int = 1024,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#         activation: str = "gelu",
#         num_classes: int = 100,
#         **kwargs) -> None:
#         super(StyleClassifier, self).__init__()
        
#         self.linear0 = nn.Linear(in_features=input_dim, out_features=latent_dim)

#         encoder_layer = nn.TransformerEncoderLayer(
#                     d_model=latent_dim,
#                     nhead=num_heads,
#                     dim_feedforward=ff_size,
#                     dropout=dropout,
#                     activation=activation)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                
#         self.linear = nn.Linear(in_features=latent_dim, out_features=num_classes)
        

#     def forward(self, x):

#         x = self.linear0(x)
        
#         feat = self.encoder(x)
        
#         # T随输入序列长度不同而不同，所以做mean
#         feat2 = torch.mean(feat, dim=1)
        
#         pred = self.linear(feat2)
        
#         debug = False
#         if debug:
#             print('x', x.shape)
#             print('feat', feat.shape) # bs, channel, conv_dims。conv_dims随输入序列长度不同而不同
#             print('feat2', feat2.shape) # bs, channel, conv_dims。conv_dims随输入序列长度不同而不同
#             print('pred', pred.shape)
        
#         return pred

