"""
文件: mnist_tutorial.py
功能: PyTorch MNIST手写数字识别完整教程 - 兼容CPU和GPU
创建者: KBX
创建日期: 2025.09.20
版本: 1.0
描述: 这是一个完整的MNIST手写数字识别项目，包含数据加载、模型构建、训练和测试全流程
      代码自动检测并适配CPU/GPU环境，适合PyTorch初学者学习
"""

# =================================================================
# 阶段 1：照猫画虎，运行一个完整的 MNIST 项目
# (此代码同时兼容 CPU 和 GPU 环境)
# =================================================================

# ----------------------------------------
# 第0步：导入所有需要的库
# ----------------------------------------
# 导入PyTorch深度学习框架的核心库:创建张量(tensor)和执行各种数学运算
import torch
# 导入神经网络模块(neural network:nn)，包含各种层结构和损失函数,各种模型容器
# 全连接层（nn.Linear）、卷积层（nn.Conv2d）、激活函数（nn.ReLU）等
import torch.nn as nn
# 导入优化算法模块，如Adam、SGD(随机梯度下降函数）等
import torch.optim as optim
# 从torchvision导入计算机视觉相关工具，包括数据集（常用的数据集直接下载）和数据变换
# transforms：提供了各种数据预处理的工具，比如调整图像大小、转为张量、数据标准化等
from torchvision import datasets, transforms
# 导入数据加载器，用于批量处理和数据管理、加载
from torch.utils.data import DataLoader
import time # 导入时间库，方便我们看训练时长
# 导入调度器库,学习率调节器
from torch.optim.lr_scheduler import StepLR
# 导入PyTorch的函数式接口，包含各种常用的函数，比如激活函数、损失函数等
import torch.nn.functional as F

def set_seed(seed):
    """设置随机种子以确保实验结果的可复现性"""
    # 为CPU设置随机种子
    torch.manual_seed(seed)
    # 如果你正在使用 CUDA，还需要为所有 GPU 设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保 cuDNN 每次都选择相同的卷积算法，保证结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置一个固定的种子，比如 42
SEED = 42
set_seed(SEED)
print(f"随机种子已设置为: {SEED}")

# ----------------------------------------
# 环节A：决定计算设备 (CPU or GPU)
# ----------------------------------------
# 这段代码会自动检测你是否有可用的 CUDA GPU，如果没有，它会使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"将使用 '{device}' 设备进行计算。")


# ----------------------------------------
# 环节1：数据准备 (Data Loading)
# ----------------------------------------
print("\n开始准备数据...")

# 定义数据预处理的步骤：创建容器复合操作，图像转张量并标准化。
# 0维张量：shape: torch.Size([])：torch.tensor(3.14)
# 1维张量：shape: torch.Size([3])：torch.tensor([1, 2, 3])
# 2维张量：shape: torch.Size([2, 3])：torch.tensor([[1, 2, 3], [4, 5, 6]])
# 3维张量：tensor_3d = torch.randn(1, 28, 28)  # shape: torch.Size([1, 28, 28])
# 标准化公式：normalized = (original - mean) / std，改变后的图像数据均值为0，标准差为1
# 标准化的数据像是可以帮助神经网络更快收敛，梯度一致，像是一个碗
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

# 下载并加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器 (DataLoader)
# batch_size: 每个批次加载多少样本;工程鲁棒，太大太小都不行
# 太大：显存不够，OOM，更新慢，容易陷入局部最优；可能收敛慢
# 太小：更新快，但不稳定，噪声大，可能无法收敛
# shuffle: 是否在每个epoch开始前打乱数据，训练集通常打乱
# 不打乱：可能导致模型过拟合：无法泛化；灾难性遗忘；最终偏向；训练不稳定；
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

print("数据准备完成！")

# # ----------------------------------------
# # 环节2：模型搭建 (Model Building)
# # ----------------------------------------
# print("\n开始搭建模型...")

# # 全连接层的计算：y = Wx + b 本质还是求解函数
# # W: 权重矩阵 (Weights)
# # x: 输入特征  
# # b: 偏置向量 (Bias)
# # y: 输出结果

# # 定义一个神经网络类，继承自nn.Module（所有PyTorch模型的基类）
# class SimpleNN(nn.Module):
#     # __init__方法：定义模型的结构和组件
#     def __init__(self):
#         # 调用父类nn.Module的构造函数，完成必要的初始化
#         super(SimpleNN, self).__init__()

#         # 定义展平层：将图像的空间维度展平为特征向量
#         # 作用：将4D图像张量(batch_size, channels, height, width) 
#         #       转换为2D特征矩阵(batch_size, features)
#         # 具体转换：(64, 1, 28, 28) → (64, 784)
#         # 每个样本从2D图像变为1D特征向量，便于全连接层处理
#         self.flatten = nn.Flatten()

#         # 定义一个简单的三层全连接网络
#         # 使用Sequential容器按顺序组合多个网络层
#         # Sequential让网络层按定义顺序依次执行，简化前向传播代码        
#         self.linear_relu_stack = nn.Sequential(
            
#             # 第一个全连接层（输入层到隐藏层1）
#             # nn.Linear(in_features=28*28, out_features=512)
#             # - 输入特征数：784 (28x28像素展平后)
#             # - 输出特征数：512 (隐藏层神经元数量)
#             # - 这一层有784*512个权重 + 512个偏置 = 401,920个可训练参数            
#             nn.Linear(28*28, 512), # 输入层 (784) -> 隐藏层1 (512)
#             nn.BatchNorm1d(512),

#             # ReLU激活函数（整流线性单元）
#             # 只有线性层：网络只是线性变换的组合
#             # 输出 = W3(W2(W1*x + b1) + b2) + b3
#             # 这等价于：输出 = W_total*x + b_total （仍然是一个线性函数！）
#             # 公式：f(x) = max(0, x)
#             # 作用：引入非线性，让网络能够学习复杂模式
#             # 优点：计算简单，缓解梯度消失问题            
#             # nn.ReLU(),
#             nn.LeakyReLU(0.01),

#             nn.Dropout(0.5), # Dropout层，防止过拟合，训练时随机丢弃50%的神经元

#             # 第二个全连接层（隐藏层1到隐藏层2）
#             # nn.Linear(in_features=512, out_features=512)
#             # - 输入特征数：512
#             # - 输出特征数：512  
#             # - 参数数量：512*512 + 512 = 262,656个可训练参数
#             # nn.Linear(512, 512),   # 隐藏层1 (512) -> 隐藏层2 (512)

#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512), 

#             # 第二个ReLU激活函数
#             # nn.ReLU(),
#             nn.LeakyReLU(0.01),

#             nn.Dropout(0.5), # Dropout层，防止过拟合，训练时随机丢弃50%的神经元

#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),

#             # 输出层（隐藏层2到输出层）
#             # nn.Linear(in_features=512, out_features=10)
#             # - 输入特征数：512
#             # - 输出特征数：10 (对应0-9十个数字类别)
#             # - 参数数量：512*10 + 10 = 5,130个可训练参数
#             # nn.Linear(512, 10)     # 隐藏层2 (512) -> 输出层 (10, 对应0-9十个数字)
#             nn.LeakyReLU(0.01),
#             nn.Dropout(0.5),
#             nn.Linear(256, 10)
#         )

#     # forward方法：定义数据的前向传播路径
#     # 输入x：一个批次的图像数据，形状为(batch_size, 1, 28, 28)
#     def forward(self, x):
#         # 步骤1：将输入图像展平
#         # 输入x形状：假设为(64, 1, 28, 28) - 64张28x28的灰度图
#         # 展平后形状：变为(64, 784) - 64个样本，每个样本784个特征        
#         x = self.flatten(x)

#         # 步骤2：将展平后的数据通过线性层和激活函数的堆叠
#         # 数据流：x → Linear(784,512) → ReLU → Linear(512,512) → ReLU → Linear(512,10)
#         # 输出logits形状：(64, 10) - 64个样本，每个样本10个类别的原始分数
#         logits = self.linear_relu_stack(x)

#         # 返回未经过softmax的原始输出（logits）
#         # 注意：CrossEntropyLoss会自动应用softmax，所以这里不需要额外处理
#         # Logits：未经过softmax的原始输出分数
#         # 比如10个类别的输出可能是：[2.1, -1.3, 0.8, ..., 5.2]
#         # Softmax：将logits转换为概率分布
#         # softmax([2.1, -1.3, 0.8]) = [0.85, 0.05, 0.10]        
#         return logits

# # 实例化模型，并将其移动到我们选择的设备上 (CPU 或 GPU)
# # SimpleNN()：创建模型的实例
# # .to(device)：将模型的所有参数和缓冲区移动到指定的计算设备
# # - 如果device='cuda'：模型转移到GPU内存
# # - 如果device='cpu'：模型保留在CPU内存
# model = SimpleNN().to(device)
# print("模型搭建完成！")

# # 打印模型结构，方便查看每层的参数和输出形状
# print(model)

# ----------------------------------------
# 环节2：模型搭建 (使用CNN)
# ----------------------------------------
print("\n开始搭建CNN模型...")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积模块
        # 输入通道=1 (灰度图), 输出通道=32, 卷积核大小=3x3, 填充=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积模块
        # 输入通道=32, 输出通道=64, 卷积核大小=3x3, 填充=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 用于正则化的Dropout层
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        # 图像尺寸因两次池化减半：28x28 -> 14x14 -> 7x7。
        # 最终的特征图有64个通道，因此展平后的尺寸为 64 * 7 * 7。
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入x的形状: (批次大小, 1, 28, 28)
        # 第一个卷积模块
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # 形状: (批次大小, 32, 14, 14)
        
        # 第二个卷积模块
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # 形状: (批次大小, 64, 7, 7)
        
        # 将特征图展平为向量，以送入全连接层
        x = x.view(-1, 64 * 7 * 7) # 形状: (批次大小, 64 * 7 * 7)
        
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # 输出的是原始的logits分数，因为CrossEntropyLoss会自动处理softmax
        return x

# 实例化新的CNN模型并将其移动到设备上
model = CNN().to(device)
print("模型搭建完成！")
print(model)


# ----------------------------------------
# 环节3：定义损失函数和优化器
# ----------------------------------------
# 交叉熵衡量两个概率分布之间的差异
# 公式：Loss = -Σ(y_true * log(y_pred))
# 1. Softmax: 将logits转换为概率分布
# 2. Log: 对概率取对数
# 3. NLLLoss: 负对数似然损失

# 损失函数：交叉熵损失，适用于多分类问题
loss_fn = nn.CrossEntropyLoss()
# 优化器：Adam，一种常用的优化算法，lr 是学习率
# 学习率对深度学习的模型训练也至关重要
# adam，SGD，RMSprop等优化算法之后再研究
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 定义调度器：每 5 个 epoch，学习率乘以 0.1
# 比如：epochs 1-5, lr=0.001; epochs 6-10, lr=0.0001; epochs 11-15, lr=0.00001
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# ----------------------------------------
# 环节4：训练循环 (Training Loop)
# ----------------------------------------
print("\n开始训练...")
num_epochs = 25 # 我们只训练 15 个周期作为演示

start_time = time.time() # 记录开始时间

for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    
    # 将模型设置为训练模式,有些网络层在不同模式下表现会不一样
    model.train()
    
    # 遍历训练数据加载器，data：输入图像张量，target：对应标签张量
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到指定设备
        data, target = data.to(device), target.to(device)
        
        # 1. 清空过往梯度，PyTorch 的 .backward() 会累积梯度（默认是加在已有 .grad 上）
        # 所以每个 batch 更新前要把 .grad 清零。
        optimizer.zero_grad()
        
        # 2. 前向传播：计算模型预测值
        pred = model(data)
        
        # 3. 计算损失
        loss = loss_fn(pred, target)
        
        # 4. 反向传播：计算梯度
        loss.backward()
        
        # 5. 更新模型参数
        optimizer.step()
        
        # 每 100 个批次打印一次训练信息
        if batch_idx % 100 == 0:
            print(f"批次 {batch_idx}/{len(train_loader)} | 损失: {loss.item():.6f}")

    # 在每个 epoch 的末尾调用 scheduler.step()
    scheduler.step()
    print(f"Epoch {epoch+1}, Current LR: {scheduler.get_last_lr()[0]}")

end_time = time.time() # 记录结束时间
print(f"训练完成！总耗时: {end_time - start_time:.2f} 秒")

# ================== 模型保存 ==================
MODEL_PATH = "checkpoints/mnist_model_25.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"模型已成功保存到: {MODEL_PATH}")
# =================================================================

# ----------------------------------------
# 环节5：测试模型 (Evaluation)
# ----------------------------------------
print("\n开始测试模型...")

# 将模型设置为评估模式，这会关闭 Dropout 等只在训练时使用的层
model.eval()

# 初始化两个累加器：test_loss 用来累加（累计）损失，correct 用来累加预测正确的样本数
test_loss = 0
correct = 0

# 在测试时，我们不需要计算梯度，这可以节省计算资源
with torch.no_grad():
    # 遍历测试数据加载器
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 前向传播：计算模型输出
        output = model(data)
        
        # 累加测试损失
        loss = loss_fn(output, target)
        batch_size = data.size(0)
        # 把 batch 的平均损失还原为该 batch 的总损失
        test_loss += loss.item() * batch_size   
        
        # 获取预测结果中概率最大的那个类别
        pred = output.argmax(dim=1, keepdim=True)
        
        # 累加预测正确的数量
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print(f"\n测试结果: \n 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")