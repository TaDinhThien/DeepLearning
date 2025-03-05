import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số
z_dim = 100
lr = 0.00002
batch_size = 1500
epochs = 36
num = 0  # chỉ huấn luyện trên ảnh của số 9 (giống với X_train_copy trong phiên bản Keras)

# Chuẩn bị dữ liệu MNIST
# Lưu ý: Trong Keras bạn dùng X_train / 255 nên giá trị nằm trong [0, 1]. Vì vậy, ta chỉ dùng ToTensor() không Normalize.
transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# Lọc chỉ lấy những ảnh có nhãn là num
X_train = torch.stack([img for img, label in dataset if label == num]).reshape(-1, 784).to(device)

# Tạo DataLoader (lưu ý: cách này không đảm bảo batch liên tục giống như việc lấy mẫu ngẫu nhiên mỗi lần như Keras,
# nhưng về cơ bản vẫn đảm bảo độ ngẫu nhiên và số lượng batch trung bình giống nhau)
dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

# Định nghĩa Generator (giống hệt cấu trúc trong Keras)
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 784),
            nn.Sigmoid()  # để đảm bảo đầu ra nằm trong [0, 1]
        )
    
    def forward(self, z):
        return self.model(z)

# Định nghĩa Discriminator (cấu trúc khớp với phiên bản Keras)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # đầu ra là xác suất
        )
    
    def forward(self, x):
        return self.model(x)

# Khởi tạo mô hình
g = Generator(z_dim).to(device)
d = Discriminator().to(device)

# Định nghĩa hàm mất mát và tối ưu hóa (giống cấu hình ban đầu)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

def save_generated_images(epoch, folder="my_gen_imgs", num_imgs=8):
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(num_imgs, z_dim, device=device)
        gen_images = g(noise).view(num_imgs, 1, 28, 28).cpu()
    vutils.save_image(gen_images, f"{folder}/number_{num}_epoch_{epoch}.png", nrow=num_imgs, normalize=True)

def train():
    os.makedirs("my_models", exist_ok=True)
    for epoch in range(1, epochs + 1):
        # Với mỗi epoch, lặp qua từng batch như phiên bản Keras (lấy mẫu ngẫu nhiên từ X_train_copy)
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            current_bs = real_imgs.size(0)
            
            # Huấn luyện Discriminator
            d_optimizer.zero_grad()
            # Nhãn thật được gán là 0.9 để sử dụng kỹ thuật label smoothing (giống Keras)
            real_labels = torch.full((current_bs, 1), 0.9, device=device)
            fake_labels = torch.zeros((current_bs, 1), device=device)
            
            # Tính mất mát trên ảnh thật
            real_loss = criterion(d(real_imgs), real_labels)
            # Sinh ảnh giả từ noise
            noise = torch.randn(current_bs, z_dim, device=device)
            fake_imgs = g(noise).detach()  # detach để không lan truyền gradient vào Generator khi huấn luyện Discriminator
            fake_loss = criterion(d(fake_imgs), fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Huấn luyện Generator
            g_optimizer.zero_grad()
            noise = torch.randn(current_bs, z_dim, device=device)
            fake_imgs = g(noise)
            # Trong bước này, chúng ta gán nhãn thật (0.9) cho ảnh giả để "lừa" Discriminator như trong phiên bản Keras
            g_loss = criterion(d(fake_imgs), real_labels)
            g_loss.backward()
            g_optimizer.step()
        
        print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss.item():.4f} - G Loss: {g_loss.item():.4f}")
        # Lưu mô hình sau mỗi epoch (giống Keras)
        torch.save(g.state_dict(), f"my_models/generator_num_{num}_epoch_{epoch}.pth")
        torch.save(d.state_dict(), f"my_models/discriminator_num_{num}_epoch_{epoch}.pth")
        # Lưu ảnh được tạo ra sau mỗi 2 epoch
        if epoch % 2 == 0:
            save_generated_images(epoch)

train()