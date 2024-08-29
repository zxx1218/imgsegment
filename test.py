# 导入必要的库
from PIL import Image
import torch
from torchvision import transforms
from IPython.display import display
from models.birefnet import BiRefNet
from models.birefnet_tiny import BiRefNet as BiRefNet_tiny
from utils import check_state_dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv") # 忽略卷积操作替代警告


# 加载 BiRefNet 模型及权重
def load_model(model_path, option, device='cuda'):
    """
    加载 BiRefNet 模型及预训练权重

    参数:
    - model_path: 模型权重的路径
    - device: 使用的设备（默认使用 'cuda'）

    返回:
    - 加载好的 BiRefNet 模型
    """
    if option == "large":
        birefnet = BiRefNet(bb_pretrained=False)  # 初始化 BiRefNet large模型
    elif option == "tiny":
        birefnet = BiRefNet_tiny(bb_pretrained=False)  # 初始化 BiRefNet tiny模型

    state_dict = torch.load(model_path, map_location='cpu')  # 加载权重
    state_dict = check_state_dict(state_dict)  # 检查并调整权重格式
    birefnet.load_state_dict(state_dict)  # 加载权重到模型中
    birefnet.to(device)  # 将模型移到指定设备上
    birefnet.eval()  # 设置模型为评估模式
    return birefnet

# 转换输入图像
def transform_input_image(image_path):
    """
    转换输入图像以适应模型输入要求

    参数:
    - image_path: 输入图像的路径

    返回:
    - 转换后的图像张量
    - 原始 PIL 图像
    """
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    image = Image.open(image_path)  # 打开图像
    return transform(image).unsqueeze(0).to('cuda'), image  # 转换图像并添加批次维度

# 进行预测
def predict(model, input_images):
    """
    使用模型对输入图像进行预测

    参数:
    - model: 经过训练的 BiRefNet 模型
    - input_images: 输入的图像张量

    返回:
    - 预测结果的张量
    """
    with torch.no_grad():  # 禁用梯度计算
        preds = model(input_images)[-1].sigmoid().cpu()  # 获取预测结果并应用 sigmoid 激活函数
    return preds[0].squeeze()  # 去除批次维度并返回

# 处理预测结果并保存
def save_results(pred, image, output_paths):
    """
    处理预测结果并保存图像

    参数:
    - pred: 预测结果的张量
    - image: 原始 PIL 图像
    - output_paths: 保存结果图像的路径字典
    """
    pred_pil = transforms.ToPILImage()(pred)  # 将预测结果转换为 PIL 图像
    scale_ratio = 1024 / max(image.size)  # 计算缩放比例
    scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))  # 计算缩放后的大小
    
    # 调整图像大小
    image_resized = image.resize((1024, 1024))  # 调整原始图像大小
    image_masked = image_resized.copy()  # 复制调整大小后的图像
    image_masked.putalpha(pred_pil)  # 将预测结果作为 alpha 通道添加到图像中
    
    # 保存图像
    image_masked.resize(scaled_size).save(output_paths['masked'])  # 保存带掩膜的图像
    image_resized.resize(scaled_size).save(output_paths['original'])  # 保存调整大小的原始图像
    pred_pil.resize(scaled_size).save(output_paths['prediction'])  # 保存预测结果图像

# 主函数
def main():
    model_path = '/home/demo/BiRefNet/pretrain/BiRefNet-portrait-TR_P3M_10k-epoch_120.pth'  # 模型权重路径
    image_path = './examples_img/1.jpg'  # 输入图像路径
    output_paths = {
        'masked': 'masked_image.png',  # 带掩膜的图像保存路径
        'original': 'original_image.png',  # 调整大小的原始图像保存路径
        'prediction': 'prediction_image.png'  # 预测结果图像保存路径
    }
    
    # 加载模型
    # birefnet = load_model(model_path, "tiny")
    birefnet = load_model(model_path, "large")
    print('BiRefNet 模型已准备好使用。')
    
    # 转换输入图像
    input_images, image = transform_input_image(image_path)
    
    # 进行预测
    pred = predict(birefnet, input_images)
    
    # 保存结果
    save_results(pred, image, output_paths)

if __name__ == '__main__':
    main()
