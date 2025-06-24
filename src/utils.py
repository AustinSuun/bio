import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import os


class DataProcessor:
    """数据预处理"""

    def __init__(self, cancer_name, indicator):
        path = os.path.join("..", "dataset", cancer_name, indicator)
        self.data = pd.read_csv(path, sep="\t", index_col=0).T
        self.cancer_name = cancer_name
        self.indicator = indicator
        print(self.data.shape)

    def std_filter(self, number):
        """
        Select the top 'number' columns based on their standard deviation.
        基于标准差选择出差异化最大的前 number 列
        """
        index = list(self.data.std().sort_values(ascending=False)[0:number].index)
        return self.data[index]

    def MinmaxVARIABLES(self, number):
        """
        Normalize the data to be between 0 and 1.
        标准化处理数据在 0-1 之间
        """
        data = self.std_filter(number)
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return data

    def sort_corr(self, number):
        """
        Sort columns based on their cumulative correlation coefficients.
        计算相关系数，并排序，选出最高的number个特征数据
        """
        data = self.MinmaxVARIABLES(int(number * 1.2))
        abs_data = data.corr().abs()
        self.cumprod_data = abs_data.sort_values(ascending=False).cumprod() ** (
            1 / len(abs_data)
        )
        cumprod_sort_index = pd.Series(
            self.cumprod_data.iloc[:, -1].sort_values(ascending=False).index
        )
        return data[cumprod_sort_index]

    def sort_corr_(self, threshold=(0.8,)):
        """全局 corr 选择 去除特征"""

        data = self.MinmaxVARIABLES(min(int(self.data.shape[1]), 25000))
        # 计算相关矩阵
        self.corr_matrix = data.corr().abs()

        # 取上三角矩阵，避免重复
        upper = self.corr_matrix.where(
            np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool)
        )
        return_list = []
        # 设置阈值，找出相关度高于阈值的特征
        for thre in threshold:
            print(
                "{}_{} 当前阈值: {}, 原始数据形状{}".format(
                    self.cancer_name, self.indicator, thre, data.shape
                ),
                end="",
            )
            to_drop = [col for col in upper.columns if any(upper[col] > thre)]
            data_drop = data.drop(columns=to_drop)
            path = os.path.join(
                "..",
                "dataset",
                self.cancer_name,
                self.indicator + "_" + str(data.shape[1]) + "_" + str(thre),
            )
            data_drop.to_csv(path, index=True)
            print("筛选出的特征 {}".format(data_drop.shape))
            return_list.append((data_drop.shape[1], thre))

        # print(f"需要剔除的高度相关特征有：{to_drop}")

        # 剔除这些特征
        return return_list


def print_shape_hook(module: nn.Module, input, output):
    print(f"{module.__class__.__name__}: output shape = {output.shape}")


class ProgCAE1D(nn.Module):
    def __init__(
        self,
        input_len,
        kernel_size1=24,
        kernel_size2=12,
        kernel_size3=6,
        stride1=5,
        stride2=5,
        stride3=4,
        channels1=32,
        channels2=64,
        channels3=128,
        latent_dim=30,
        lr=5e-4,
        epochs=500,
        batch_size=32,
        val_ratio=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.lr = lr

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1,
                channels1,
                kernel_size=kernel_size1,
                stride=stride1,
                padding=kernel_size1 // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels1,
                channels2,
                kernel_size=kernel_size2,
                stride=stride2,
                padding=kernel_size2 // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels2,
                channels3,
                kernel_size=kernel_size3,
                stride=stride3,
                padding=kernel_size3 // 2,
            ),
            nn.ReLU(),
        )

        # Calculate shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            conv_out = self.encoder(dummy)
        self.conv_shape = conv_out.shape  # (1, C, L)
        self.flat_dim = conv_out.numel()

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.flat_dim),
            nn.ReLU(),
            nn.Unflatten(1, self.conv_shape[1:]),  # (C, L)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                channels3,
                channels2,
                kernel_size=kernel_size3,
                stride=stride3,
                padding=kernel_size3 // 2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                channels2,
                channels1,
                kernel_size=kernel_size2,
                stride=stride2,
                padding=kernel_size2 // 2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                channels1,
                1,
                kernel_size=kernel_size1,
                stride=stride1,
                padding=kernel_size1 // 2,
            ),
            nn.Sigmoid(),  # 原版是relu ，这里修改
        )

        self.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

    def train_model(self, data):
        x = data.unsqueeze(1).to(self.device)  # shape: [B, 1, input_len]
        dataset = TensorDataset(x, x)
        val_len = int(len(x) * self.val_ratio)
        train_len = len(x) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.train()
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                preds = self.forward(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                self.eval()
                with torch.no_grad():
                    val_loss = sum(
                        self.loss_fn(self(xb), yb) for xb, yb in val_loader
                    ) / len(val_loader)
                print(f"Epoch {epoch:3d}: val_loss = {val_loss.item():.6f}")

    def encode(self, x):
        x = x.unsqueeze(1).to(self.device)
        with torch.no_grad():
            x = self.encoder(x)
            x = x.flatten(start_dim=1)
            z = self.bottleneck[1](x)
        return z.cpu()

    def decode(self, z):
        with torch.no_grad():
            x = self.bottleneck[2](z)
            x = self.bottleneck[3](x)
            x = self.bottleneck[4](x)
            x = self.decoder(x)
        return x.cpu()

    def reconstruct(self, x):
        x = x.unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self.forward(x)
        return out.squeeze(1).cpu()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        self.eval()


class ProgCAE1D_2(nn.Module):
    """两层版本的prog"""

    def __init__(
        self,
        input_len,
        latent_dim=30,
        channels1=16,
        channels2=32,
        kernel_size1=5,
        stride1=2,
        kernel_size2=5,
        stride2=2,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        val_ratio=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1,
                channels1,
                kernel_size=kernel_size1,
                stride=stride1,
                padding=kernel_size1 // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels1,
                channels2,
                kernel_size=kernel_size2,
                stride=stride2,
                padding=kernel_size2 // 2,
            ),
            nn.ReLU(),
        )

        # --- 计算卷积后尺寸 ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len).to(self.device)
            conv_out = self.encoder(dummy)
        self.conv_shape = conv_out.shape  # (1, C, L)
        self.flat_dim = conv_out.numel()

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.flat_dim),
            nn.ReLU(),
            nn.Unflatten(1, self.conv_shape[1:]),  # (C, L)
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                channels2,
                channels1,
                kernel_size=4,
                stride=stride2,
                padding=kernel_size2 // 2 - 1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                channels1,
                1,
                kernel_size=4,
                stride=stride1,
                padding=kernel_size1 // 2 - 1,
            ),
            nn.Sigmoid(),  # 或者 ReLU
        )

        self.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

    def train_model(self, data):
        x = data.unsqueeze(1).to(self.device)  # shape: [B, 1, input_len]
        dataset = TensorDataset(x, x)
        val_len = int(len(x) * self.val_ratio)
        train_len = len(x) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.train()
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                preds = self.forward(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                self.eval()
                with torch.no_grad():
                    val_loss = sum(
                        self.loss_fn(self(xb), yb) for xb, yb in val_loader
                    ) / len(val_loader)
                print(f"Epoch {epoch:3d}: val_loss = {val_loss.item():.6f}")

    def encode(self, x):
        x = x.unsqueeze(1).to(self.device)
        with torch.no_grad():
            x = self.encoder(x)
            x = x.flatten(start_dim=1)
            z = self.bottleneck[1](x)
        return z.cpu()

    def decode(self, z):
        with torch.no_grad():
            x = self.bottleneck[2](z)
            x = self.bottleneck[3](x)
            x = self.bottleneck[4](x)
            x = self.decoder(x)
        return x.cpu()

    def reconstruct(self, x):
        x = x.unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self.forward(x)
        return out.squeeze(1).cpu()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        self.eval()


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        bottleneck_dim=30,
        nhead=4,
        num_layers=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        # 输入先映射到 hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 瓶颈层降维
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)

        # 瓶颈层升维
        self.bottleneck_inv = nn.Linear(bottleneck_dim, hidden_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层映射回输入维度
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [batch, input_dim]

        # 投影输入到 hidden_dim 并作为序列输入 Transformer（这里把每个样本看成长度1的序列）
        x_proj = self.input_proj(x).unsqueeze(1)  # [B, 1, hidden_dim]

        # Encoder 编码
        memory = self.encoder(x_proj)  # [B, 1, hidden_dim]

        # 瓶颈降维，先 squeeze 序列长度维度（1）
        bottleneck_vec = self.bottleneck(memory.squeeze(1))  # [B, bottleneck_dim]

        # 瓶颈升维回 hidden_dim，作为 decoder 的 tgt 输入
        tgt = self.bottleneck_inv(bottleneck_vec).unsqueeze(1)  # [B, 1, hidden_dim]

        # Decoder 重建
        decoded = self.decoder(tgt, memory)  # [B, 1, hidden_dim]

        # 输出映射回输入维度
        output = self.output_proj(decoded.squeeze(1))  # [B, input_dim]

        return output, bottleneck_vec

    def train(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, features = self(data)  # data shape [B, 100]
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        self.eval()


def train_model(path_dict: dict, model: nn.Module, latent_dim=30):
    """训练癌症的特征提取模型

    path_state：存放癌症和对应的数据名称的字典。用来生成数据的路径
    {'癌症名' : ['组学数据1, 组学数据2 ...'], ...}
    """
    for cancer in path_dict.keys():
        for indicator in path_dict[cancer]:
            path = os.path.join("..", "dataset", cancer, indicator)
            df = pd.read_csv(path, index_col=0)
            num_features = df.shape[1] // 4 * 4  # 保证 特征数 是4的倍数，符合模型的形状
            data = torch.tensor(df.iloc[:, :num_features].values, dtype=torch.float)
            model = model(num_features)
            model.train_model(data)
            model_path = os.path.join("..", "model", cancer, indicator)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
