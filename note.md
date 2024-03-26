# PETR 代码学习笔记

## 1. 论文动机和创新点
DETR3D方法存在3个缺点：

**a. 参考点投影机制**

DETR3D首先根据object query预测 N 个3D参考点，
然后利用相机参数将参考点反投影回图像，对2D图像特征进行采样，
最后根据采样得到的2D图像特征预测3D目标信息。
如果参考点预测的不够准确，那么投影回图像的位置就有可能位于目标区域之外，
导致采样得到无效的图像特征。

**b. 单点特征采样**

DETR3D只会选取参考点反投影位置对应的图像特征，导致模型对于全局特征学习的不够充分。

**c. 流程复杂**

特征采样过程比较复杂，不利于方法落地。

**PETR想实现**：是否有可能将二维特征从多视图转换为3D感知特征（创新点）。

具体做法：PETR通过3D Position Embedding将多视角相机的2D特征转化为3D感知特征，
使得object query可以直接在3D语义环境下更新，
省去了参考点反投影以及特征采样两个步骤。
由6个视图共享的相机视锥空间，离散为3D网格坐标。
通过不同的摄像机参数进行坐标转换，得到3D世界空间的坐标。
将从主干网络提取的2D图像特征和3D坐标输入到一个简单的3D位置编码器中，
生成3D位置感知特征。

## 2.配置文件

`projects/configs/petr/petr_r50dcn_gridmask_c5.py`

`projects/configs/petr/petr_r50dcn_gridmask_p4.py`

两者主要在neck部分有变动

```
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(type='ResNet'),
    img_neck=dict(type='CPFPN'),  
    pts_bbox_head=dict(
        type='PETRHead',
        num_query=900,
        LID=True, # LID: Local Inference Decoder
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention'),
                        dict(type='PETRMultiheadAttention'),
                        ],
                    with_cp=True, 
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(type='NMSFreeCoder'), 
        positional_encoding=dict(type='SinePositionalEncoding3D'),
        loss_cls=dict(type='FocalLoss'),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
```

- CPFPN: This FPN remove the unused parameters which can be used with checkpoint (with_cp = True in Backbone)
- LID: Linearly Increasing Discretization
- with_cp: whether to use checkpoint or not, which can reduce the memory usage during training
梯度检查点是一种用训练时间换取显存的办法，其核心原理是在反向传播时重新计算神经网络的中间激活值而不用在前向时存储，torch.utils.checkpoint 包中已经实现了对应功能。简要实现过程是：在前向阶段传递到 checkpoint 中的 forward 函数会以 torch.no_grad 模式运行，并且仅仅保存输入参数和 forward 函数，在反向阶段重新计算其 forward 输出值。

```
def forward(self, x):
    def _inner_forward(x):
        out = ...(x)
        return out
    # x.requires_grad 这个判断很有必要
    if self.with_cp and x.requires_grad:
        out = cp.checkpoint(_inner_forward, x)
    else:
        out = _inner_forward(x)
    return self.relu(out)
```


### FPNs

参考：[一文看尽物体检测中的各种FPN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/148738276)

动机是保留并融合骨干网络提取的多尺度表征。在网络中一般处于骨干网络和任务头之间。
FPN的演进如下：
![FPN的演进.png](..%2F..%2FDesktop%2FFPN%E7%9A%84%E6%BC%94%E8%BF%9B.png)

举例来说：
参考： https://zhuanlan.zhihu.com/p/457154948

```faster rcnn利用了C2~C5，并且在output的P5上进行最大池化获得P6
neck=dict( type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
# start_level默认为0，end_level默认为-1，利用C2~C5
# num_outs>end_level-start_level，并且又没有指定add_extra_convs，那么将采用maxpool
```

## petr3d.py
Model的整个流程是detector设置的（在mmdet3d/models/detectors下）。模型类的定义如下
```
@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    def __init__(self,...):
        super(Petr3D, self).__init__(...)
        self.grid_mask = GridMask(...)
```
这里继承了`MVXTwoStageDetector`，主要重写了`forward_train`, `forward_test`等
Lidar only大都是继承的VoxelNet如PointPillar，而多模态的都是继承`MVXTwoStageDetector`，如DETR3D、FUTR3D、Transfusion、BEVFormer。

```
@force_fp32(apply_to=('img', 'points'))
def forward(self, return_loss=True, **kwargs):
    if return_loss:
        return self.forward_train(**kwargs)
    else:
        return self.forward_test(**kwargs)
```
这里的`forward`函数是一个分发函数，根据`return_loss`的值来调用`forward_train`或者`forward_test`函数。
force_fp32是一个装饰器，这里'img', 'points'是传入的参数的变量名，没有显示给出。

foward_train函数的主要工作如下，extract_feat 会调用img_backbone的forward函数。 
forward_pts_train会调用pts_bbox_head的forward_train函数。
```
img_feats = self.extract_feat(img=img, img_metas=img_metas)
losses_pts = self.forward_pts_train(img_feats, gt...)
```

## petr_head.py

### 3D Coordinates Generator 
projects\mmdet3d_plugin\models\dense_heads\petr_head.py :: position_embedding

这里看公示和LSS/DETR3D生成3d网格相似
```
# 论文创新点
def position_embeding(self, img_feats, img_metas, masks=None):
    eps = 1e-5
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    B, N, C, H, W = img_feats[self.position_level].shape
    # 生成像素平面内的网格 
    coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H 
    # ticks * step = (0,...,H-1) * (pad_h / H)
    coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

    if self.LID:  # 线性递增离散化（深度方向）
        # depth_num  64 
        index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
    else:  # UD   均匀离散化
        # [0,64)
        index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
        # (61.2-1)/64
        bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
        coords_d = self.depth_start + bin_size * index # 1 + 等间隔的深度  [1,65)

    D = coords_d.shape[0]  # 64
    # meshgrid 的功能是生成网格，可以用于生成坐标
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)  # 拼接一维，与内参矩阵对应 # W,H,D,4
    # 对应论文的 u_j * d, v_j*d, d, 1
    coords[..., :2] = coords[..., :2] * \
            torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

    # 处理内外参
    img2lidars = []
    for img_meta in img_metas:
        img2lidar = []
        for i in range(len(img_meta['lidar2img'])):
            img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i])) 
        img2lidars.append(np.asarray(img2lidar))
    img2lidars = np.asarray(img2lidars)
    img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)
    
    # 将coords复制 Batch_size * N 份 
    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)  # B,N,W,H,D,4,1
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # B,N,W,H,D,4,4
    
    # camera --> lidar坐标系 B,N,W,H,D 维度相乘
    # B,N,W,H,D,4,1 -> B,N,W,H,D,3
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
    
    # x,y,z 分别进行归一化 这里给定了范围 好像是 61.2, 61.2, 10.0
    coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
    coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
    coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

    # B,N,W,H,D,3
    coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # 不在范围内的元素将被mask遮住
    coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)  # B, N, W, H, sum(D * 3)
    coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # B, N, H, W # 与mask相或 ???
    
    # B,N,W,H,D,3  --->    B*N,D*3,H,W
    coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
    coords3d = inverse_sigmoid(coords3d)  # -ln((1 / x) - 1) or -ln((1 / (x + 1e-8)) - 1)
    
    # 与论文 3.3 的 3D Position Encoder对应
    # 送入几个1*1的2d卷积网络中+Relu，进一步加深编码信息，因此只变化C的维度，从  D*3--> 4*embed_dims --> embed_dims
    coords_position_embeding = self.position_encoder(coords3d)  
    
    # B*N,embed_dims,H,W
    return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask
```

### 3D Position Encoder
projects\mmdet3d_plugin\models\dense_heads\petr_head.py :: position_encoder

```
self.position_encoder = nn.Sequential(
    nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
    nn.ReLU(),
    nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
)
```



