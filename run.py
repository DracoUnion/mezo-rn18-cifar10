import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 超参数
lr = 1e-2
epsilon = 1e-3
grad_clip_norm = 10.0
batch_size = 128
num_epochs = 10
print_step = 20
n_queries = 32
weight_decay = 0.1
sparsity_ratio = 0.0
bn_update_freq = 50  # 每 50 个 batch 更新一次 BN running stats

# -------------------- MeZO 优化器实现 --------------------
class MeZO:
    """
    Memory-Efficient Zeroth-order Optimizer (MeZO)
    使用单次随机扰动估计梯度，并进行参数更新（ZO-SGD）
    """
    def __init__(self, 
        model: nn.Module, 
        lr: float = 1e-3, 
        epsilon: float = 1e-3, 
        weight_decay = 0.1, 
        grad_clip_norm = 1.0, 
        n_queries=3,
        sparsity_ratio=0.5,
    ):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.n_queries = n_queries
        self.weight_decay = weight_decay
        self.sparsity_ratio = sparsity_ratio

    def get_mask(self, param_dict):
        masks = {}
        for n, p in param_dict.items():
            # 计算当前层参数的绝对值
            param_abs = p.data.abs()
            # 计算分位数阈值：保留最小 sparsity_ratio 比例的参数
            k = max(1, int(self.sparsity_ratio * param_abs.numel()))
            # 如果 k 等于参数总数，则保留全部，避免分位数计算错误
            if k >= param_abs.numel():
                mask = torch.ones_like(p, dtype=torch.bool)
            else:
                # 计算第 k 小的值作为阈值
                threshold = torch.kthvalue(param_abs.view(-1), k).values
                # 生成掩码：绝对值 <= 阈值的参数为 True (将被更新)
                mask = param_abs <= threshold
            masks[n] = mask
        return masks

    def get_single_ests(self, param_dict, loss_fn, masks=None):
        # 为每个参数生成相同形状的随机扰动
        z = {n:torch.randn_like(p) for n, p in param_dict.items()}
        if masks:
            for n, zn in z.items():
                zn.masked_fill_(~masks[n], 0.0)

        # ---- 正向扰动 (θ + εz) ----
        for n, p in param_dict.items():
            p.data.add_(self.epsilon * z[n])
        loss_pos = loss_fn()  # 第一次前向传播

        # ---- 负向扰动 (θ - εz) ----
        for n, p in param_dict.items():
            p.data.sub_(2 * self.epsilon * z[n])  # 从 +εz 变为 -εz
        loss_neg = loss_fn()  # 第二次前向传播

        # ---- 恢复原始参数 ----
        for n, p in param_dict.items():
            p.data.add_(self.epsilon * z[n])   # 回到 θ

        # ---- 梯度估算与参数更新 ----
        grad_ests = {
            n: (loss_pos - loss_neg) / (2 * self.epsilon) * z[n] + self.weight_decay * p.data 
            for n, p in param_dict.items()
        }

        return grad_ests, loss_pos, loss_neg

    def step(self, loss_fn):
        """
        执行一步 MeZO 更新：
        1. 保存当前参数并生成随机扰动 z ~ N(0, I)
        2. 计算 L(θ + εz) 和 L(θ - εz)
        3. 估算梯度并更新参数
        """
        
        # 获取所有可训练参数
        param_dict = {
            n:p for n, p in self.model.named_parameters() if p.requires_grad
        }
        masks = self.get_mask(param_dict) if self.sparsity_ratio else None

        with torch.no_grad():
            grad_ests = {
                n:torch.zeros_like(p) 
                for n, p in param_dict.items()
            }
            for _ in range(self.n_queries):
                single_ests, loss_pos, loss_neg = \
                    self.get_single_ests(param_dict, loss_fn, masks) 
                for n, g in single_ests.items():
                    grad_ests[n] += g
            for g in grad_ests.values():
                g /= self.n_queries
            
            total_norm = sum([
                grad_est.norm().item() ** 2
                for grad_est in grad_ests.values()
            ]) ** 0.5
            scale = min(1.0, self.grad_clip_norm / (total_norm + 1e-6))
            
            for n, p in param_dict.items():
                p.data.sub_(self.lr * grad_ests[n] * scale)   # ZO-SGD 更新

        return loss_pos.item(), loss_neg.item()


# -------------------- 主程序 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据加载与预处理（CIFAR-10）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. 模型定义（ResNet18，输出为10类）
    model = resnet18(pretrained=False, num_classes=10).to(device)

    # 冻结 backbone，只训练最后 fc 层（MeZO 适用场景：低维参数微调）
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    criterion = nn.CrossEntropyLoss()

    # 3. 初始化 MeZO 优化器（超参数可根据需要调整）
    optimizer = MeZO(
        model, 
        lr=lr, 
        epsilon=epsilon, 
        n_queries=n_queries, 
        weight_decay=weight_decay,
        sparsity_ratio=sparsity_ratio,
    )

    # 4. 训练与测试循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 定期用 train 模式跑几个 batch 更新 BN running stats
            if batch_idx % bn_update_freq == 0 and batch_idx > 0:
                model.train()
                with torch.no_grad():
                    for bn_inputs, bn_labels in trainloader:
                        bn_inputs = bn_inputs.to(device)
                        _ = model(bn_inputs)
                        break  # 只跑 1 个 batch 更新 stats
                model.eval()

            # 定义损失函数（闭包，捕获当前 batch 的数据）
            # 关键：ZO 扰动估计必须冻结 BN running stats，否则
            # loss_pos/loss_neg 的差会混入 BN 漂移，污染梯度估计。
            model.eval()

            def loss_fn():
                with torch.no_grad():
                    outputs = model(inputs)
                return criterion(outputs, labels)

            # 执行一步 MeZO 更新（两次前向，无反向传播）
            loss_pos, loss_neg = optimizer.step(loss_fn)
            total_loss += (loss_pos + loss_neg) / 2

            # 每 print_step 个 batch 打印一次进度
            if batch_idx % print_step == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | Loss_pos: {loss_pos:.4f} | Loss_neg: {loss_neg:.4f}")

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1} finished. Average Loss (positive): {avg_loss:.4f}")

        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {acc:.2f}%\n")


if __name__ == "__main__":
    main()