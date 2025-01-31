import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
import numpy as np
from torch.optim import Adam
from einops import rearrange
from .modules import MultiHeadAttention, EmbedFeaturesFT


class UserTower(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, nr_heads):
        super(UserTower, self).__init__()
        self.embed_dim = embed_dim
        # Input Embedding Layer (projects 4D input to embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Multi-Head Attention Layer
        self.attention = MultiHeadAttention(dim_input=embed_dim, nr_heads=nr_heads)
        # Fully Connected Layer (maps back to scalar output)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, attn_mask=None):
        x = self.embedding(x)  # Project to higher dimension
        x = x.unsqueeze(1)  # Add sequence dimension: (batch, seq_len=1, embed_dim)
        attn_output = self.attention(x, x)   
        x = attn_output.mean(dim=1)  # Average over sequence dimension
        return self.fc(x) 

class ProductTower(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, nr_heads):
        super(ProductTower, self).__init__()
        self.embed_dim = embed_dim
        # Input Embedding Layer (projects 4D input to embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Multi-Head Attention Layer
        self.attention = MultiHeadAttention(dim_input=embed_dim, nr_heads=nr_heads)
        # Fully Connected Layer (maps back to scalar output)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, attn_mask=None):
        x = self.embedding(x)  # Project to higher dimension
        x = x.unsqueeze(1)  # Add sequence dimension: (batch, seq_len=1, embed_dim)
        attn_output = self.attention(x, x)   
        x = attn_output.mean(dim=1)  # Average over sequence dimension
        return self.fc(x) 


# class UserTower(nn.Module):
#     def __init__(self, categorical_features, continuous_features, dim_embedding, internal_dimension=100):
#         super(UserTower, self).__init__()

#         self.embedding_layer = EmbedFeaturesFT(
#             nr_categories=[len(cats) for cats in categorical_features],  
#             dim_embedding=dim_embedding,
#             nr_cont_features=len(continuous_features),
#             nr_cat_features=len(categorical_features),
#             internal_dimension=internal_dimension,
#         )

#         self.attention = nn.MultiheadAttention(embed_dim=dim_embedding, num_heads=4, batch_first=True)
#         self.fc = nn.Linear(dim_embedding, 1)  # Example final layer

#     def forward(self, x_cat, x_cont, x_vec=None):
#         # Embed categorical & continuous features
#         x_cat_emb, x_cont_emb, _ = self.embedding_layer(x_cat, x_cont, x_vec)

#         # Concatenate embeddings along feature axis
#         x = torch.cat([x_cat_emb, x_cont_emb], dim=1) if x_cat_emb is not None and x_cont_emb is not None else \
#             x_cat_emb if x_cont_emb is None else x_cont_emb

#         # Attention layer
#         x = self.attention(x, x, x)[0]

#         # Fully connected layer
#         x = self.fc(x.mean(dim=1))  

#         return x


# class UserTower(nn.Module):
#     def __init__(self, input_dim, embed_dim, output_dim, nr_heads, window_size=3):
#         super(UserTower, self).__init__()
#         self.window_size = window_size  # Window size for rolling
#         # Input Embedding Layer (projects 4D input to embed_dim)
#         self.embedding = nn.Linear(input_dim, embed_dim)
#         # Multi-Head Attention Layer
#         self.attention = MultiHeadAttention(dim_input=embed_dim, nr_heads=nr_heads)
#         # Fully Connected Layer (maps back to scalar output)
#         self.fc = nn.Linear(embed_dim, output_dim)

#     def rolling_window(self, x, window_size):
#         """
#         Creates overlapping sequences using a rolling window approach.
#         """
#         batch_size, embed_dim = x.shape
#         if embed_dim < window_size:
#             raise ValueError("Embed_dim should be greater than or equal to window_size")

#         return x.unfold(dimension=1, size=window_size, step=1)  # (batch_size, embed_dim-window_size+1, window_size)

#     def forward(self, x):
#         x = torch.relu(self.embedding(x))  # Apply first linear layer with activation
#         x = self.rolling_window(x, self.window_size)  # Create overlapping sequences
#         # Multihead Attention expects (batch_size, seq_len, hidden_dim) -> (seq_len, batch_size, hidden_dim)
#         x = x.permute(0, 2, 1)  
#         attn_output = self.attention(x, x)   
#         x = attn_output.permute(1, 0, 2)  # Convert back to (batch, seq_len, hidden_dim)
#         x = self.output_layer(x.mean(dim=1))  # Pooling + Final Linear Layer
#         return x
    
class UserTower_MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(UserTower_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ProductTower_MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(ProductTower_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class TwoTowerRecommendationModel_MLP(pl.LightningModule):
    """Two Tower Recommendation Model using Multi-Head Attention

    User Tower: Embedding -> Multi-Head Attention Layer -> Fully Connected Layer
    Product Tower: Embedding -> Multi-Head Attention Layer -> Fully Connected Layer
    """
    def __init__(self, user_config, product_config, learning_rate):
        super(TwoTowerRecommendationModel_MLP, self).__init__()
        self.user_tower = UserTower_MLP(**user_config)
        self.product_tower = ProductTower_MLP(**product_config)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, user_input, product_input):
        
        product_output = self.product_tower(product_input)
        user_output = self.user_tower(user_input)
        rating_pred_tensor = F.cosine_similarity(user_output, product_output)
        return rating_pred_tensor

    def training_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('test_loss', loss)
        return {'preds': rating_pred_tensor, 'target': target}

    def predict_step(self, batch, batch_idx):
        user_input, product_input, _ = batch
        rating_pred_tensor = self(user_input, product_input)
        return rating_pred_tensor

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)



class TwoTowerRecommendationModel(pl.LightningModule):
    """Two Tower Recommendation Model using Multi-Head Attention

    User Tower: Embedding -> Multi-Head Attention Layer -> Fully Connected Layer
    Product Tower: Embedding -> Multi-Head Attention Layer -> Fully Connected Layer
    """
    def __init__(self, user_config, product_config, learning_rate):
        super(TwoTowerRecommendationModel, self).__init__()
        self.user_tower = UserTower(**user_config)
        self.product_tower = ProductTower(**product_config)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, user_input, product_input):
        
        product_output = self.product_tower(product_input)
        user_output = self.user_tower(user_input)
        rating_pred_tensor = F.cosine_similarity(user_output, product_output)
        return rating_pred_tensor

    def training_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        user_input, product_input, target = batch
        rating_pred_tensor = self(user_input, product_input)
        loss = self.criterion(rating_pred_tensor, target)
        self.log('test_loss', loss)
        return {'preds': rating_pred_tensor, 'target': target}

    def predict_step(self, batch, batch_idx):
        user_input, product_input, _ = batch
        rating_pred_tensor = self(user_input, product_input)
        return rating_pred_tensor

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

