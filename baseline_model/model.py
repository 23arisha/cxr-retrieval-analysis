import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import BertModel

# ====================================================================
# image_encoder
# ====================================================================
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SoftAttention(nn.Module):
    def __init__(self, in_groups, m_heads, in_channels):
        super(SoftAttention, self).__init__()
        self.learnable_scalar = nn.Parameter(torch.rand(1))
        self.conv3d = nn.Conv3d(
            in_channels  = in_groups,
            out_channels = m_heads,
            kernel_size  = (in_channels, 1, 1),
            stride       = (in_channels, 1, 1)
        )
        self.lrelu   = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        c = torch.unsqueeze(x, 1)
        c = self.conv3d(c)
        c = self.lrelu(c)
        c = c.squeeze(2)
        c = c.view(c.shape[0], c.shape[1], h * w)
        c = self.softmax(c)
        c = c.view(c.shape[0], c.shape[1], h, w)
        attn_maps  = torch.unsqueeze(c.sum(1), 1)
        importance = x * attn_maps
        out = x + importance * self.learnable_scalar.expand_as(importance)
        return out, attn_maps, self.learnable_scalar


class ImageEncoder(nn.Module):
    def __init__(self, output_channels=512, pretrained=True):
        super(ImageEncoder, self).__init__()

        weights = 'IMAGENET1K_V2' if pretrained else None
        resnet  = tvm.resnet50(weights=weights)

        # adapt first conv for grayscale
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride      = old_conv.stride,
            padding     = old_conv.padding,
            bias        = False
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    old_conv.weight.sum(dim=1, keepdim=True)
                )
        resnet.conv1 = new_conv

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3   # 1024 channels
        self.layer4 = resnet.layer4   # 2048 channels

        self.sa_l3        = SoftAttention(in_groups=1, m_heads=16, in_channels=1024)
        self.region        = conv1x1(1024, output_channels)
        self.avgpool        = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout        = nn.Dropout(p=0.3)
        self.global_feats   = nn.Linear(2048, output_channels)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x, attn_maps_l3, scalar_l3 = self.sa_l3(x)
        region_feat = self.region(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        z = self.dropout(torch.flatten(x, 1))
        global_feat = self.global_feats(z)

        return (region_feat, global_feat, attn_maps_l3, scalar_l3, attn_maps_l3, scalar_l3)

# ====================================================================
# text_encoder
# ====================================================================
class MLMHead(nn.Module):
    def __init__(self, bert_config):
        super(MLMHead, self).__init__()
        self.transform = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.act       = nn.GELU()
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.decoder   = nn.Linear(bert_config.hidden_size, bert_config.vocab_size - 1)

    def forward(self, seq_feat):
        return self.decoder(self.LayerNorm(self.act(self.transform(seq_feat))))


class TextEncoder(nn.Module):
    def __init__(self, bert_config, output_channels, pool='cls'):
        super(TextEncoder, self).__init__()
        self.pool    = pool
        self.bert    = BertModel(bert_config, add_pooling_layer=False)
        self.mlm     = MLMHead(bert_config)
        self.sent_fc = nn.Linear(output_channels, output_channels)
        self.word_fc = nn.Linear(output_channels, output_channels)

    def forward(self, x, mask, task='itm'):
        seq_feat = self.bert(input_ids=x, attention_mask=mask)[0]
        if task == 'mlm':
            return self.forward_mlm(seq_feat)
        elif task == 'itm':
            return self.forward_itm(seq_feat, mask)
        else:
            raise ValueError(f'Invalid task: {task}')

    def forward_mlm(self, seq_feat):
        pred = self.mlm(seq_feat)
        return pred.transpose(1, 2)

    def forward_itm(self, seq_feat, mask):
        if self.pool == 'cls':
            sent_feat = self.sent_fc(seq_feat[:, 0])
        else:
            sent_feat = (
                (seq_feat * mask.unsqueeze(-1)).sum(1)
                / mask.sum(1).unsqueeze(-1)
            )
            sent_feat = self.sent_fc(sent_feat)
        word_feat = self.word_fc(seq_feat)
        return word_feat.transpose(1, 2), sent_feat

# ====================================================================
# attention
# ====================================================================
def func_attention(query, context, gamma1):
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw  = context.size(2), context.size(3)
    sourceL = ih * iw

    context  = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query)
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(-1)(attn)

    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = attn * gamma1
    attn = nn.Softmax(-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)

    attnT           = torch.transpose(attn, 1, 2).contiguous()
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm           = nn.Softmax(-1)
        self.mask         = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        ih, iw     = input.size(2), input.size(3)
        queryL     = ih * iw
        batch_size = context.size(0)
        sourceL    = context.size(2)

        target  = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()

        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)

        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)

        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()

        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn            = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
