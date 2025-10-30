import torch
# from model.SGraAGFormer import SGraAGFormer
from model.SGraDiFormer import sgraformer
from common.opt import opts
import os


# 设置参数
opt = opts().parse()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
CUDA_ID = [0]
device = torch.device("cuda")

# 创建模型实例
def create_model():
    model = sgraformer(
        num_frame=opt.frames,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2
    )
    return model

# 测试模型前向传播
def test_forward_pass():
    # 创建模型
    model = create_model()
    # 检查CUDA是否可用
    
    model = model.to(device)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    frames = opt.frames
    views = 4
    joints = 17
    channels = 2
    
    # 输入数据 [batch, frames, views, joints, channels]
    x = torch.randn(batch_size, frames, views, joints, channels).to(device)
    
    # Hop图 [batch, views, joints, joints]
    hops = torch.randn(batch_size, views, joints, joints).to(device)
    
    # 进行前向传播
    with torch.no_grad():
        output, loss_contrastive = model(x, hops)
        
    # 打印输出形状以验证
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass test completed successfully!")

if __name__ == "__main__":
    test_forward_pass()