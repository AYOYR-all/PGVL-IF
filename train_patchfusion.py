import os
import kornia                 # 用于图像处理
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import utils
from data_RGB_LLVIP import get_training_data, get_test_data
# from data_RGB_mfnet import get_training_data, get_test_data
# from data_RGB_whu import get_training_data, get_test_data
from warmup_scheduler import GradualWarmupScheduler  #先用小学习率热身，等模型稳定了再用正常学习率
from tqdm import tqdm                                #for循环显示进度条
import torch.utils.data
from utils.seg_util import *
from utils.dice import *
import warnings
from utils.MEF_SSIM_loss import th_SSIM_LOSS,Y_Upper
import logging
import argparse
from Evaluator import *
from model_patchFusion import MMFNet as Net
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2


def show_img(img):
    """
    显示形状为 (3, 320, 320) 的图像

    参数:
    img: 可以是 numpy array 或 PyTorch Tensor
    """
    # 如果是 PyTorch Tensor，转换为 numpy
    if isinstance(img, torch.Tensor):
        if img.requires_grad:
            img = img.detach().cpu().numpy()
        else:
            img = img.cpu().numpy()

    # 确保是 numpy 数组
    img = np.asarray(img)

    # 检查形状
    if img.shape != (3, 256, 320):
        raise ValueError(f"Expected shape (3, 256, 320), got {img.shape}")

    # 转换维度: (C, H, W) -> (H, W, C)
    img_display = img.transpose(1, 2, 0)

    # 显示图像
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display)
    plt.axis('off')
    plt.title('3×320×320 Image Display')
    plt.show()
def rgb_to_ycbcr(img):
    """
    将R、G、B转换成Y、Cr、Cb
    """
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)
def rgb_to_ycbcr(img):                               
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr
def evaluation_one(ir_name, vi_name, f_name):
    """
    计算评价指标，计算的函数均来自Evaluator中创建的class Evaluator
    """
    ir = image_read_cv2(ir_name, 'GRAY')
    vi = image_read_cv2(vi_name, 'GRAY')
    fi = image_read_cv2(f_name, 'GRAY')
    if ir.shape != fi.shape:
        ir = cv2.resize(ir, (fi.shape[1], fi.shape[0]), interpolation=cv2.INTER_LINEAR)
    if vi.shape != fi.shape:
        vi = cv2.resize(vi, (fi.shape[1], fi.shape[0]), interpolation=cv2.INTER_LINEAR)
    EN = Evaluator.EN(fi) #熵
    SD = Evaluator.SD(fi) # 标准差，表示对比度，SD越高->图像越清晰，亮度变化越丰富
    SF = Evaluator.SF(fi)  # 空间频率，表征图像纹理、细节多少。SF越大->边缘、纹理越丰富
    AG = Evaluator.AG(fi)  # 平均梯度,描述图像尖锐程度，AG越大->图像越锐利（细节更清楚）
    SCD = Evaluator.SCD(fi, ir, vi)  # 结构相似度度量
    VIFF = Evaluator.VIFF(fi, ir, vi) #视觉信息忠实度
    return EN, SD, SF, AG, SCD, VIFF

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用哪块GPU运行以下程序‘0’代表第一块，‘1’代表第二块
NUM_EPOCHS = 50
BATCH_SIZE = 4 # batch_size
LR_INITIAL = 2e-5
LR_MIN = 1e-6
data = 'llvip'
save_module_path = os.path.join('./check/', data, 'models', 'patchfusion.pkl') # dt_rank = 16
model_dir = os.path.join('./check/', data, 'models')  # './checkpoints/mfnet/models'
os.makedirs(model_dir, exist_ok=True)
train_dir = os.path.join('./dataset/train/', data) 
train_ir_text_csv_path = os.path.join('./dataset/train/', data,'llvip_ir_text.csv')
train_vis_text_csv_path = os.path.join('./dataset/train/', data,'llvip_vis_text.csv')
train_pv_dir = os.path.join(train_dir, 'pv')
test_ir_text_csv_path = os.path.join('./dataset/test/', data,'llvip_ir_text.csv')
test_vis_text_csv_path = os.path.join('./dataset/test/', data,'llvip_vis_text.csv')
test_pv_dir = os.path.join('./dataset/test/', data, 'pv')
if not os.path.exists(train_ir_text_csv_path):
    print(f"Warning: train IR text csv not found, fallback to default IR text. path={train_ir_text_csv_path}")
if not os.path.exists(train_vis_text_csv_path):
    print(f"Warning: train VIS text csv not found, fallback to default VIS text. path={train_vis_text_csv_path}")
if not os.path.exists(test_ir_text_csv_path):
    print(f"Warning: test IR text csv not found, fallback to default IR text. path={test_ir_text_csv_path}")
if not os.path.exists(test_vis_text_csv_path):
    print(f"Warning: test VIS text csv not found, fallback to default VIS text. path={test_vis_text_csv_path}")

# model = Net(9, use_clip_guidance=True)  # mfnet
# model = torch.load(save_module_path, weights_only=False)
# model = Net(6)  # potsdam
#model = Net(8, use_clip_guidance=True)  # llvip: num_classes is only used when seg labels are provided.
model = Net(
    8,
    use_clip_guidance=True,
    clip_model_name="ViT-B-16",
    clip_checkpoint_path="./check/ViT-B-16.safetensors",
)
# agent = FusionAgent(state_dim=320 * 320 * 3, action_dim=3, action_bound=0.1, Loss_ratio=1)
# Segmodel = SegHead()
# ================#
#     GPU并行    #
# ===============#
# if torch.cuda.device_count() > 1:
#     print("===== Let's use", torch.cuda.device_count(), "GPUs! =====")
#     model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

model.cuda()
# model=model.cuda()
# model.eval()


optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs,
                                                        eta_min=LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

train_dataset = get_training_data(
    train_dir,
    ir_text_csv_path=train_ir_text_csv_path,
    vis_text_csv_path=train_vis_text_csv_path,
    pv_dir=train_pv_dir,
)
print(f"训练数据路径: {train_dir}")
print(f"路径是否存在: {os.path.exists(train_dir)}")
print(f"路径内容: {os.listdir(train_dir)}")
print(f"数据集长度: {len(train_dataset)}")
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,drop_last=True, pin_memory=True)  # drop_last=False
print('===> Loading datasets')
img_dir_test = os.path.join('./dataset/test/', data)
test_dataset = get_test_data(
    img_dir_test,
    ir_text_csv_path=test_ir_text_csv_path,
    vis_text_csv_path=test_vis_text_csv_path,
    pv_dir=test_pv_dir,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False,pin_memory=True)
result_dir = os.path.join('./dataset/results/', data)
eva_dir = os.path.join('./dataset/eva/', data)

l1_criterion = nn.L1Loss()
# 像素值标为255表示”无效区域/模糊区域“，不参与计算
CE_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
Dice_criterion = DiceLoss(smooth=0.05, ignore_index=255)

best_EN, best_SD, best_SF, best_AG, best_SCD, best_VIFF = 0, 0, 0, 0, 0, 0
for epoch in range(NUM_EPOCHS):
    valid_batch = iter(test_loader)  # 验证集迭代器
    for step, data in enumerate(tqdm(train_loader), 0):
        model.train()
        # agent.train()
        # Segmodel.train()
        target_ir = data[0].cuda()
        input_ir = data[1].cuda()
        target_rgb = data[2].cuda()
        input_rgb = data[3].cuda()
        target_seg = data[4].cuda()
        ir_text_prompts = data[5]
        vis_text_prompts = data[6]
        pv = data[7].cuda()

        input_ycbcr = rgb_to_ycbcr(input_rgb)
        target = Y_Upper(target_ir[:, :1, :, :], input_ycbcr[:, :1, :, :], 1.7).cuda()
        # zero_grad，清空梯度
        for param in model.parameters():
            param.grad = None
        # 返回重建的ir、重建的vir、交叉重建的ir、交叉重建的vir、输出的语义分割图、（重建的融合图像）
        res_ir, res_y, res_rec_ir, res_rec_y, res_seg, _ = model(
            input_ir[:, :1, :, :],
            input_ycbcr[:, :1, :, :],
            ir_texts=ir_text_prompts,
            vis_texts=vis_text_prompts,
            pv=pv[:, :1, :, :],
        )
        # torch.cat((A,B,C),dim=1)在第二个维度上将A，B，C拼接
        res_ycbcr = torch.cat((torch.clamp(res_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),dim=1)
        res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)
        res_rec_ycbcr = torch.cat((torch.clamp(res_rec_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),dim=1)
        res_rec_rgb = kornia.color.ycbcr_to_rgb(res_rec_ycbcr)

        loss_ir = l1_criterion(res_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_ir)
        loss_rgb = l1_criterion(res_rgb, target_rgb) + th_SSIM_LOSS(target, res_y)
        loss_rec_ir = l1_criterion(res_rec_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_rec_ir)
        loss_rec_rgb = l1_criterion(res_rec_rgb, target_rgb) + th_SSIM_LOSS(target, res_rec_y)
        if torch.any(target_seg != 255):
            loss_seg = Dice_criterion(res_seg, target_seg) + CE_criterion(res_seg, target_seg)  # to(torch.int64)
        else:
            loss_seg = res_seg.sum() * 0.0

        loss = loss_ir + loss_rgb + loss_rec_ir + loss_rec_rgb + loss_seg

        optimizer.zero_grad() # 有点重复
        loss.backward()

        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.norm().item())

        gr = torch.tensor(grad).mean().item()
        # 更新模型参数
        optimizer.step()

        model.eval()
        # agent.eval()
        # Segmodel.eval()

    if epoch > -1:
        EN_list, SD_list, SF_list, AG_list = [], [], [], []
        SCD_list, VIFF_list = [], []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            inp_ir = data_test[0].cuda()
            inp_rgb = data_test[1].cuda()
            ir_text_prompts_test = data_test[2]
            vis_text_prompts_test = data_test[3]
            pv_test = data_test[4].cuda()
            filenames = data_test[5]
            ir_paths = data_test[6]
            rgb_paths = data_test[7]
            inp_ycbcr = rgb_to_ycbcr(inp_rgb)

            _, _, _, _, _, fus = model(
                inp_ir[:, :1, :, :],
                inp_ycbcr[:, :1, :, :],
                ir_texts=ir_text_prompts_test,
                vis_texts=vis_text_prompts_test,
                pv=pv_test[:, :1, :, :],
            )
            fus = torch.clamp(fus, 0, 1)
            fus_ycbcr = torch.cat((torch.clamp(fus, 0, 1), inp_ycbcr[:, 1:2, :, :], inp_ycbcr[:, 2:, :, :]), dim=1)
            fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)

            ones = torch.ones_like(fus_rgb)
            zeros = torch.zeros_like(fus_rgb)
            fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
            fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb) # 将异常值截断
            fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # BCHW -> BHWC  因为numpy只能在CPU上调用，所以调用.cpu
            fus_rgb = (fus_rgb - np.min(fus_rgb)) / (np.max(fus_rgb) - np.min(fus_rgb))
            fus_rgb = np.uint8(255.0 * fus_rgb)

            for batch in range(len(fus)):
                fus_img = fus_rgb[batch]
        
                epoch_result_dir = os.path.join(result_dir,f'epoch_{epoch+1:03d}')
                os.makedirs(epoch_result_dir, exist_ok=True)
                
                fusion_path = os.path.join(epoch_result_dir, filenames[batch] + '.png')
                # 保存融合图像
                utils.save_img(fusion_path, fus_img)

                # 读取IR、RGB图像
                ir_path = ir_paths[batch]
                rgb_path = rgb_paths[batch]

                EN, SD, SF, AG, SCD, VIFF = evaluation_one(ir_path, rgb_path, fusion_path)
                EN_list.append(EN)
                SD_list.append(SD)
                SF_list.append(SF)
                AG_list.append(AG)
                SCD_list.append(SCD)
                VIFF_list.append(VIFF)
        # print(f"=== Average Fusion Metrics on {data} ===")
        print(f"EN:   {np.mean(EN_list):.4f}")
        print(f"SD:   {np.mean(SD_list):.4f}")
        print(f"SF:   {np.mean(SF_list):.4f}")
        print(f"AG:   {np.mean(AG_list):.4f}")
        print(f"SCD:  {np.mean(SCD_list):.4f}")
        print(f"VIFF: {np.mean(VIFF_list):.4f}")
        os.makedirs(eva_dir, exist_ok=True)
        metrics_path = os.path.join(eva_dir, 'metrics.txt')

        with open(metrics_path, 'a', encoding='utf-8') as f:
              f.write(
                     f"Epoch {epoch+1:03d} | "
                     f"EN: {np.mean(EN_list):.4f} | "
                     f"SD: {np.mean(SD_list):.4f} | "
                     f"SF: {np.mean(SF_list):.4f} | "
                     f"AG: {np.mean(AG_list):.4f} | "
                     f"SCD: {np.mean(SCD_list):.4f} | "
                     f"VIFF: {np.mean(VIFF_list):.4f}\n"
                     )
        if (np.mean(SCD_list) > best_SCD):
            best_SCD = np.mean(SCD_list)
            torch.save(model, save_module_path)
            # torch.save(agent, save_agent_path)
            # torch.save(Segmodel, save_segmodel_path)
            print("更新")
    scheduler.step()
    
    