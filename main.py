import os
import numpy as np
import pandas as pd
import tabulate
import torch
import pytorch_lightning as pl

from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models import TwoTowerRecommendationModel
from src.utils import generate_random_sample_data
from src.movie_dataset_utils import load_data, gen_user_vecs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option("display.precision", 1)



if __name__ == "__main__":


    # Load Data, set configuration variables
    cwd = os.getcwd()
    path = os.path.join(cwd, "data/gold/movie_dataset/")
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data(path=path)

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
    epochs = 5
    learning_rate = 0.001
    use_gpu = True

    # Model configurations
    user_config = {'input_dim': user_data_tensor.shape[1], 
                'embed_dim': 128,
                'output_dim': 64, 
                'nr_heads': 8}
    product_config = {'input_dim': product_data_tensor.shape[1], 
                    'embed_dim': 128, 
                    'output_dim': 64,
                    'nr_heads': 8}

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

    df_predictions = pd.DataFrame(disp[1:], columns=disp[0])
 
    print(df_predictions)