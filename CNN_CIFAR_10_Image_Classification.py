"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/15
@Desc    : 使用 PyTorch 通过卷积神经网络(CNN)进行 CIFAR-10 图像分类
@License : MIT License (MIT)
@Version : 1.0

"""

#### 导入库
# 第三方库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# PyTorch 核心库
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch 视觉相关库
import torchvision
import torchvision.transforms as transforms

#### 数据处理
def data_processing():
    ## 数据预处理:进行归一化预处理（将像素值缩放到 [0,1] 或 [-1,1]）
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ## 加载数据集:使用 torchvision.datasets.CIFAR10 加载数据集。
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    ## 采用 torch.utils.data.DataLoader 进行批量加载
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader,train_dataset.classes

## 可视化部分数据
# 可视化数据
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 取出一个batch的图像数据并可视化
def show_a_batch(train_loader):
    # 获取一个批次的数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # 显示部分图像
    imshow(torchvision.utils.make_grid(images[:]))

#### 模型构建
### CNN模型
class SimpleCNN(nn.Module):
    def __init__(self,
                 num_classes=10,          # CIFAR10对应10类，默认设为10
                 in_channels=3,           # 输入通道数，RGB为3
                 conv_channels=[32, 64, 128],  # 卷积层输出通道数
                 kernel_size=3,           # 卷积核大小
                 padding=1,               # 卷积填充
                 pool_size=2,             # 池化窗口大小
                 img_size=(32, 32),       # 新增：图像尺寸参数，默认(32,32)对应CIFAR10
                 fc_hidden_dim=512):      # 新增：全连接隐藏层维度参数，默认512
        super(SimpleCNN, self).__init__()

        # 1. 动态构建特征提取层（借鉴MLP的layers列表思路）
        layers = []
        ## 至少包含 2 个卷积层
        total_cnn_layers = len(conv_channels)
        if total_cnn_layers < 2:
            raise ValueError(f"卷积层数量需至少为2层，当前为{total_cnn_layers}层")

        current_in = in_channels  # 初始输入通道数（RGB为3）

        # 遍历卷积通道数列表，逐个构建「卷积+BN+ReLU+池化」组合
        for out_channels in conv_channels:
            # 卷积层：当前输入通道 → 目标输出通道
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=kernel_size, padding=padding))
            # 批归一化层：对应输出通道数
            layers.append(nn.BatchNorm2d(out_channels))
            # ReLU激活（原地操作节省内存）
            layers.append(nn.ReLU(inplace=True))
            # 最大池化层：缩小特征图尺寸
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            # 更新下一层的输入通道数为当前输出通道数
            current_in = out_channels

        # 把动态构建的层封装成Sequential
        self.features = nn.Sequential(*layers)

        # 2. 动态计算展平后的特征维度（使用传入的img_size，不再写死224）
        # 用一个随机张量模拟输入，自动计算输出维度
        with torch.no_grad():
            # dummy_input的尺寸：(batch_size=1, 通道数, 图像高度, 图像宽度)
            dummy_input = torch.randn(1, in_channels, img_size[0], img_size[1])
            dummy_output = self.features(dummy_input)
            self.flatten_dim = dummy_output.numel()  # 计算总元素数（展平后的维度）

        # 3. 构建分类器（全连接层，隐藏层维度使用传入的fc_hidden_dim）
        classifier_layers = []
        # 第一层：展平后的特征 → 传入的fc_hidden_dim维隐藏层
        classifier_layers.append(nn.Linear(self.flatten_dim, fc_hidden_dim))
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(0.5))
        # 第二层：fc_hidden_dim维 → 分类数
        classifier_layers.append(nn.Linear(fc_hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平（兼容动态维度，不用固定写死）
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x

#### 模型训练
def train(model, device ,train_loader, test_loader , optimizer_choice ,epochs=20,lr=0.001):
    ## 采用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    ## 使用 SGD 或 Adam 进行优化
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)           # Adam 优化器
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)            # SGD 优化器
    else:
        raise ValueError("优化器选择无效！请选择 'Adam' 或 'SGD' .")

    ## 训练至少 10 轮（Epochs）
    # 校验epochs并抛出异常
    if epochs < 10:
        # 使用raise抛出ValueError，附带指定错误信息
        raise ValueError("训练至少 10 轮（Epochs）")

    ## 训练多个 epoch，并在测试集上评估准确率
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失，outputs是向量；labels是一个0-9的数，表示哪一类
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        ## 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return train_losses, test_accuracies

#### 结果可视化
### 绘制训练损失曲线
def plot_training_loss(train_losses ,epochs=10):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

### 绘制测试集准确率曲线
def plot_test_accuracy(test_accuracies ,epochs=10):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.show()

### 计算并绘制混淆矩阵
def plot_confusion_matrix(model,test_loader,device,classes):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

### 随机选取部分测试集样本，进行预测，并可视化预测结果与真实标签对比
def plot_test_images_results(test_loader, model, device,classes):
    # 取一个批次的测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    # 只取前10张（2行5列刚好展示，避免数量超出子图数量）
    images, labels = images[:10].to(device), labels[:10].to(device)

    # 模型预测（加torch.no_grad()避免计算梯度，节省资源）
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 2行5列展示图像
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # 调整画布大小，避免标签拥挤
    for i, ax in enumerate(axes.flat):
        # 复用imshow的核心逻辑：反归一化 + 维度转换
        img = images[i].cpu()  # 转到cpu，避免GPU张量无法转numpy
        img = img / 2 + 0.5  # 反归一化（和imshow里的逻辑完全一致）
        npimg = img.numpy()
        img_show = np.transpose(npimg, (1, 2, 0))  # 维度转换（和imshow里的逻辑完全一致）

        # 显示彩色图像
        ax.imshow(img_show)
        # 显示真实标签和预测标签
        ax.set_title(f'Label: {classes[labels[i].item()]}\nPred: {classes[predicted[i].item()]}')
        ax.axis('off')  # 隐藏坐标轴，更美观

    plt.tight_layout()  # 自动调整子图间距，避免标签重叠
    plt.show()

#### 主函数
def main():
    epochs = 20
    conv_channels = [64, 128, 256]
    kernel_size = 3
    padding = 1
    pool_size = 2
    fc_hidden_dim = 256
    optimizer_choice = 'Adam'
    lr = 0.001

    ## 主体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # 设置设备（如果有 GPU 可用则使用 GPU）
    train_loader, test_loader,classes = data_processing()                       # 数据预处理
    show_a_batch(train_loader)                                                  # 可视化部分样本图像
    model = SimpleCNN(
                 conv_channels=conv_channels,  # 卷积层输出通道数
                 kernel_size=kernel_size,           # 卷积核大小
                 padding=padding,               # 卷积填充
                 pool_size=pool_size,             # 池化窗口大小
                 fc_hidden_dim=fc_hidden_dim)      .to(device)                  # 模型构建
    train_losses, test_accuracies = train(model, device,
                                          train_loader, test_loader,
                                          optimizer_choice, epochs, lr)         # 模型训练
    plot_training_loss(train_losses=train_losses, epochs=epochs)                # 绘制训练损失曲线
    plot_test_accuracy(test_accuracies=test_accuracies, epochs=epochs)          # 绘制测试集准确率曲线
    plot_confusion_matrix(model=model, test_loader=test_loader,
                          device=device, classes=classes)                       # 绘制混淆矩阵
    plot_test_images_results(test_loader=test_loader, model=model,
                             device=device, classes=classes)                    # 显示部分测试图像及其预测结果

#### 运行
if __name__ == "__main__":
    main()