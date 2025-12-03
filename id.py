# detailed_gpu_check.py

import subprocess
import os
import sys

def detailed_gpu_check():
    print("=== 详细GPU状态诊断 ===")
    
    # 1. 检查环境变量
    print("\n1. 环境变量检查:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    import torch
    # 2. 检查PyTorch CUDA支持
    print("\n2. PyTorch CUDA支持:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print("CUDA不可用，检查驱动安装")
        return
    
    # 3. 检查GPU数量
    print("\n3. GPU设备检查:")
    try:
        gpu_count = torch.cuda.device_count()
        print(f"检测到的GPU数量: {gpu_count}")
        
        if gpu_count == 0:
            print("警告: 未检测到GPU设备")
            return
    except Exception as e:
        print(f"获取GPU数量失败: {e}")
        return
    
    # 4. 逐个检查GPU状态
    print("\n4. 逐个GPU状态检查:")
    for i in range(gpu_count):
        try:
            print(f"\n--- GPU {i} ---")
            print(f"设备名称: {torch.cuda.get_device_name(i)}")
            print(f"计算能力: {torch.cuda.get_device_capability(i)}")
            
            # 设置当前设备
            torch.cuda.set_device(i)
            
            # 检查内存
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"已分配内存: {allocated:.2f} GB")
            print(f"保留内存: {cached:.2f} GB")
            
            # 简单测试GPU计算
            test_tensor = torch.randn(1000, 1000).cuda()
            result = test_tensor @ test_tensor.T
            print(f"GPU计算测试: 通过")
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"GPU {i} 检查失败: {e}")
    
    # 5. 检查nvidia-smi输出
    print("\n5. nvidia-smi输出:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi执行成功")
            # 只显示前几行避免信息过多
            lines = result.stdout.split('\n')[:20]
            for line in lines:
                print(line)
        else:
            print(f"nvidia-smi执行失败，返回码: {result.returncode}")
            print(f"错误信息: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("nvidia-smi执行超时")
    except Exception as e:
        print(f"执行nvidia-smi失败: {e}")

def check_system_gpu():
    print("\n=== 系统级GPU检查 ===")
    
    # 检查GPU进程
    try:
        print("\n检查GPU进程:")
        result = subprocess.run(['nvidia-smi', 'pmon', '-c', '1'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
    except:
        print("无法检查GPU进程")
    
    # 检查GPU温度
    try:
        print("\n检查GPU温度:")
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            temps = result.stdout.strip().split('\n')
            for i, temp in enumerate(temps):
                print(f"GPU {i} 温度: {temp}°C")
    except:
        print("无法检查GPU温度")

if __name__ == "__main__":
    detailed_gpu_check()
    check_system_gpu()