import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tabulate
from recsysNN_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
import numpy as np
from torch.optim import Adam
from einops import rearrange
pd.set_option("display.precision", 1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        dim_input: int,
        nr_heads: int = 8,
        dropout_p: float = 0.0,
        scale_factor: float = 0.5,
    ):
        """Ye olde Multihead Attention, implmented with Einstein Notation.
        Note: There' ain't no masking here, so be careful!

        Args:
            dim_input (int): The input dimension
            nr_heads (int, optional): Number of heads. Defaults to 8.
            dropout_p (float, optional): Dropout. Defaults to 0.0.
            scale_factor (float, optional): Exponent of the scaling division - default is square root. Defaults to 0.5.
        """
        super().__init__()
        self.nr_heads = nr_heads
        self.dim_input = dim_input
        self.dim_head = dim_input // nr_heads
        self.scale = self.dim_head**-scale_factor

        self.to_qkv = torch.nn.Linear(dim_input, self.dim_head * self.nr_heads * 3, bias=False)

        self.to_out = torch.nn.Linear(self.dim_head * nr_heads, dim_input)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.nr_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # Add an extra dimension for the heads (b, 1, i, j)
            sim = sim.masked_fill(attn_mask == 0, float("-inf"))

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


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
        #x = attn_output.mean(dim=self.embed_dim)  
        return self.fc(attn_output) 

class UserTower_Linear(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, nr_heads):
        super(UserTower_Linear, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ProductTower(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(ProductTower, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class TwoTowerRecommendationModel(pl.LightningModule):
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

# Generate Random Sample Data
def generate_random_sample_data(num_samples: int):
    """This function generates random sample data for the two-tower model.
    The idea is to understand how to structure the data for the model.

    Args:
        num_samples (int): number of samples

    Returns:
        _type_: _description_
    """
    user_input_dim = 6
    product_input_dim = 3
    user_data = np.random.rand(num_samples, user_input_dim)
    product_data = np.random.rand(num_samples, product_input_dim)
    target_data = np.random.randint(0, 2, size=(num_samples, 1))  # Binary targets for F1 score

    user_tensor = torch.tensor(user_data, dtype=torch.float32)
    product_tensor = torch.tensor(product_data, dtype=torch.float32)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)

    return user_tensor, product_tensor, target_tensor



if __name__ == "__main__":


    # Load Data, set configuration variables
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
    uvs = 3  # user genre vector start
    ivs = 3  # item genre vector start
    u_s = 3  # start of columns to use in training, user
    i_s = 1  # start of columns to use in training, items
    scaledata = True  # applies the standard scalar to data if true
    print(f"Number of training vectors: {len(item_train)}")
    
    use_synthetic_data = False
    if use_synthetic_data:
        # Generate synthetic data
        num_samples = 1000
        user_data, product_data, target_data = generate_random_sample_data(num_samples)
        
    # scale training data
    scaledata = True
    if scaledata:
        item_train_save = item_train
        user_train_save = user_train
        y_train_save = y_train

        scalerItem = StandardScaler()
        scalerItem.fit(item_train)
        item_train = scalerItem.transform(item_train)

        scalerUser = StandardScaler()
        scalerUser.fit(user_train)
        user_train = scalerUser.transform(user_train)
        
        targetScaler = MinMaxScaler((-1, 1))
        targetScaler.fit(y_train.reshape(-1, 1))
        y_train = targetScaler.transform(y_train.reshape(-1, 1))

        print(np.allclose(item_train_save, scalerItem.inverse_transform(item_train)))
        print(np.allclose(user_train_save, scalerUser.inverse_transform(user_train)))

    user_data_tensor = torch.tensor(user_train[:, u_s:], dtype=torch.float32)
    product_data_tensor = torch.tensor(item_train[:, i_s:], dtype=torch.float32)
    target_data_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32) 
    
    print("User Tensor", user_data_tensor.shape, 
          "Product Tensor", product_data_tensor.shape, 
          "Target Rating Tensor", target_data_tensor.shape)
    
    # ---------------------
    # Setup Model
    # ---------------------
    
    # Hyperparameters
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    use_gpu = True

    # Model configurations
    user_config = {'input_dim': user_data_tensor.shape[1], 
                'embed_dim': 128,
                'output_dim': 64, 
                'nr_heads': 8}
    product_config = {'input_dim': product_data_tensor.shape[1], 
                    'embed_dim': 128, 
                    'output_dim': 64}

    # Create DataLoader
    dataset = TensorDataset(user_data_tensor, product_data_tensor, target_data_tensor)
    num_samples = target_data_tensor.shape[0]
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = TwoTowerRecommendationModel(user_config, product_config, learning_rate)

    if not use_gpu:

        # Train the model
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(model, train_loader, test_loader)

    else:
        # Move model to GPU if available
        device = torch.device('mps' if torch.cuda.is_available() else 'mps')
        model.to(device)

        # Train the model on GPU
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
        trainer.fit(model, train_loader, test_loader)


    # ---------------------
    # Predictions
    # ---------------------
    
    new_user_id = 5000
    new_rating_ave = 1.0
    new_action = 1.0
    new_adventure = 1
    new_animation = 1
    new_childrens = 1
    new_comedy = 5
    new_crime = 1
    new_documentary = 1
    new_drama = 1
    new_fantasy = 1
    new_horror = 1
    new_mystery = 1
    new_romance = 5
    new_scifi = 5
    new_thriller = 1
    new_rating_count = 3

    user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                        new_action, new_adventure, new_animation, new_childrens,
                        new_comedy, new_crime, new_documentary,
                        new_drama, new_fantasy, new_horror, new_mystery,
                        new_romance, new_scifi, new_thriller]])

    user_vecs = gen_user_vecs(user_vec,len(item_vecs))
    
    if scaledata:
        scaled_user_vecs = scalerUser.transform(user_vecs)
        scaled_item_vecs = scalerItem.transform(item_vecs)
        user_data_tensor = torch.tensor(scaled_user_vecs[:, u_s:], dtype=torch.float32)
        product_data_tensor = torch.tensor(scaled_item_vecs[:, i_s:], dtype=torch.float32)
        y_p = model(user_data_tensor, product_data_tensor).detach().numpy()
        y_p = targetScaler.inverse_transform(y_p.reshape(-1, 1))
    else:
        y_p = model(user_vecs[:, u_s:], item_vecs[:, i_s:])
        
    if np.any(y_p < 0) : 
        print("Error, expected all positive predictions")
    
    print("Prediction Vector Shape", y_p.shape)
    
    sorted_index = np.argsort(-y_p,axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_p[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    
    y_p, user, item, movie_dict = sorted_ypu, sorted_user, sorted_items, movie_dict

    maxcount=10
    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        if movie_id in movies_listed:
            continue
        movies_listed[movie_id] = 1
        disp.append([y_p[i, 0], item[i, 0].astype(int), item[i, 2].astype(float),
                    movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow")
    
    print(table)