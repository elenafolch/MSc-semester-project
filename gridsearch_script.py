import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from autoencoder_mod import FizeauDataset
from configurable_model import ConfigureModel, configAutoenc, random_indices, give_configuration
from astropy.io import fits

from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch

n_combinations = 5
random_gridsearch_indices = random_indices(0, 3, 3, n_combinations)

training_frac = 0.8
N = 1001
n_training = int(N*training_frac)
training_epochs = 100

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Path(r"simulated_images_varied_tip_and_tilt_1000")
data = sorted(data.glob("*.fits"))
print(len(data))

dataset = FizeauDataset(data, log=True)

idxs = np.arange(0, N)
np.random.shuffle(idxs)

train_idxs = idxs[:n_training]
validation_idxs = idxs[n_training:n_training + (N-n_training)//2]
test_idxs = idxs[n_training+(N-n_training)//2:]


train_dataset = Subset(dataset, train_idxs)
validation_dataset = Subset(dataset, validation_idxs)
test_dataset = Subset(dataset, test_idxs)

validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss = nn.L1Loss()
models = []
loss_conf = np.zeros((n_combinations, training_epochs))

with open("hyperparameter_grid.json", "r") as file:
    hyperparams = json.load(file)

for j, indices in enumerate(random_gridsearch_indices):
    conf_signature, lr, batch_part, enc = give_configuration(indices=indices, parameter_grid=hyperparams)

    print(f"training with configuration {j}: {conf_signature[0]}")

    configuration = {
    "model": {"n_encoder_layers": enc},
    "training": {"learning_rate": lr, "batch_part": batch_part}
    }

    json_str = json.dumps(configuration, indent=4)
    with open("model_configuration.json", "w") as f:
        f.write(json_str)

    with open("model_configuration.json", "r") as file:
        model_configuration = json.load(file)

    configuration = model_configuration["model"]

    batch_size = int(n_training*batch_part)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    model = configAutoenc(in_channels=1, in_height=60, configuration=ConfigureModel(**configuration))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    loss_per_epoch = np.zeros(training_epochs)
    validation_loss_per_epoch = []
    epochs_w_no_improvement = 0
    best_model = model
        
    for epoch in range(training_epochs):
        print(f"Epoch [{epoch + 1}/{training_epochs}]")
        model.train()
        batch_losses = []
        
        for batch_index, (data, _) in enumerate(tqdm(iter(train_loader))):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)

            # Forward pass: compute the model output
            reconstructed, bottleneck_rep, _, _ = model(data)
            batch_loss = loss(reconstructed, data)
            batch_losses.append(batch_loss.detach().numpy())

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            batch_loss.backward()
    
            # Optimization step: update the model parameters
            optimizer.step()

        #compute mean loss of current epoch
        loss_per_epoch[epoch] = np.mean(batch_losses)

        validation_loss = []
        model.eval()
        with torch.no_grad():
             
            for val_data in validation_loader:
                val_img, _ = val_data
                val_reconstructed, _, _, _ = model(val_img)
                validation_loss.append(loss(val_reconstructed, val_img))

        validation_loss_per_epoch.append(np.mean(validation_loss))

        if epoch >=1:
            if validation_loss_per_epoch[epoch] > validation_loss_per_epoch[epoch-1]:
                epochs_w_no_improvement += 1
            else: epochs_w_no_improvement = 0
            
            if epochs_w_no_improvement > 5:
                break
            else:
                best_model = model
    
    models.append(best_model) 
    loss_conf[j, :] = np.array(loss_per_epoch)

data_path = Path("/home/ipa/quanz/user_accounts/efolch/semproj/code/simulated_images_varied_tip_and_tilt_1000/")
out_root = Path("/home/ipa/quanz/user_accounts/efolch/semproj/code/reconstructions_tip_tilt_confs_100ep")

subset = test_loader.dataset

for indices, model in zip(random_gridsearch_indices, models):

    conf_signature, _, _, _ = give_configuration(indices=indices, parameter_grid=hyperparams)
    model.eval()
    conf_dir = out_root / "_".join(str(x) for x in conf_signature)
    if not conf_dir.is_dir():
        conf_dir.mkdir(parents=True)

    with torch.no_grad():
        for i, test_data in enumerate(subset):
            test_img, opd = test_data
            test_img = torch.unsqueeze(test_img, dim=0) #image needs to have shape (1, 1, H, W)
            reconstructed, bottleneck_rep, layer_reps, _ = model(test_img)

            src = Path(subset.dataset.files[subset.indices[i]])
            hdr = fits.Header()
            hdr["OPD"] = float(opd)
            hdr["SOURCE"] = src.name

            fits.writeto(conf_dir / f"{src.stem}_original.fits", test_img.detach().numpy()[0], hdr, overwrite=True)
            fits.writeto(conf_dir / f"{src.stem}_reconstructed.fits", reconstructed.detach().numpy()[0], hdr, overwrite=True)

    print(f"wrote {len(subset)} frame pairs to {conf_dir}")