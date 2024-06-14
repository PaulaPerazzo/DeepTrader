import numpy as np
import torch
import torch.nn as nn

class MSU(nn.Module):
    def __init__(self, in_features, window_len, hidden_dim):
        super(MSU, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, X):
        # Verificar valores 'nan' nos inputs e substituí-los por zeros
        if torch.any(torch.isnan(X)):
            # print("Valores 'nan' encontrados nos inputs antes da normalização")
            X = torch.where(torch.isnan(X), torch.zeros_like(X), X)
        
        # Normalização de dados
        X = (X - X.mean()) / (X.std() + 1e-5)
        
        if torch.any(torch.isnan(X)):
            raise ValueError("Valores 'nan' encontrados nos inputs após normalização")

        X = X.permute(1, 0, 2)
        # print("Input X2:", X)

        try:
            outputs, (h_n, c_n) = self.lstm(X)
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isnan(h_n)) or torch.any(torch.isnan(c_n)):
                outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
                h_n = torch.where(torch.isnan(h_n), torch.zeros_like(h_n), h_n)
                c_n = torch.where(torch.isnan(c_n), torch.zeros_like(c_n), c_n)
                # raise ValueError("Valores 'nan' encontrados após a LSTM")
        except ValueError as e:
            # print(f'Erro detectado: {e}')
            return outputs, h_n, c_n

        H_n = h_n.repeat((self.window_len, 1, 1))
        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))
        scores = scores.squeeze(2).transpose(1, 0)

        if torch.any(torch.isnan(scores)):
            scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
            # raise ValueError("Valores 'nan' encontrados nos scores")

        attn_weights = torch.softmax(scores, dim=1)

        if torch.any(torch.isnan(attn_weights)):
            raise ValueError("Valores 'nan' encontrados nos pesos de atenção")

        outputs = outputs.permute(1, 0, 2)
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)

        if torch.any(torch.isnan(attn_embed)):
            raise ValueError("Valores 'nan' encontrados no embed de atenção")

        embed = torch.relu(self.bn1(self.linear1(attn_embed)))

        if torch.any(torch.isnan(embed)):
            embed = torch.where(torch.isnan(embed), torch.zeros_like(embed), embed)
            # raise ValueError("Valores 'nan' encontrados após linear1 e batchnorm")

        parameters = self.linear2(embed)

        if torch.any(torch.isnan(parameters)):
            parameters = torch.where(torch.isnan(parameters), torch.zeros_like(parameters), parameters)
            # print("Valores 'nan' encontrados nos parâmetros finais")
            # raise ValueError("Valores 'nan' encontrados nos parâmetros finais")

        return parameters.squeeze(-1)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.LSTM:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    a = torch.randn((16, 20, 3))
    data = np.load("../data/ibovespa/market_data.npy", allow_pickle=True)
    data = data.astype(np.float32)
    
    # Substituir 'nan' por zeros ou outro valor adequado nos dados
    data = np.nan_to_num(data, nan=0.0)

    grouped_data = data[:16*20].reshape(16, 20, 3)
    tensor = torch.from_numpy(grouped_data)
    
    net = MSU(3, 20, 128)
    net.apply(init_weights)
    
    b = net(tensor)
    # print(b)
