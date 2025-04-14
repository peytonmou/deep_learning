from load_oxford_flowers102 import load_oxford_flowers102
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms 
from tqdm import tqdm
import numpy as np 
from torch.amp import autocast, GradScaler

# data preprocessing
transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
training_set, validation_set, test_set, class_names = load_oxford_flowers102(imsize=96, fine=True) 
train_data = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
val_data = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)
test_data = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

# Task 2a: Variant Auto-Encoder
class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim 

        self.encoder = nn.Sequential(                                           # 4 conv layers: 3x96x96 -> 512x6x6 
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),               # 48x48
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # 24x24
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # 12x12
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),            # 6x6
            nn.ReLU(),) 
        
        self.fc_mu = nn.Conv2d(512, latent_dim, kernel_size=1)                  # latent head mean [16, 512, 6, 6]
        self.fc_logvar = nn.Conv2d(512, latent_dim, kernel_size=1)              # latent head logvar [16, 512, 6, 6]
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=1),                 # 5 transposed conv layers: 512x6x6 -> 3x96x96 
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),      # 96x96
            nn.Tanh(),)

    def encode(self, x):
        h = self.encoder(x)                    # extract features
        mu = self.fc_mu(h)  
        logvar = self.fc_logvar(h)  
        return mu, logvar                      # [16, 512, 6, 6]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std                  # reparameterization trick for differentiable sampling 

    def decode(self, z):
        return self.decoder(z)                 # reconstruct input from latent space 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z) 
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):           
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]         # reconstruction error
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]     # KL divergence
    return BCE + 0.1 * KLD                
    
def train_val_test(epochs=10, lr=0.0001, load_from_file=True):
    if load_from_file:
        vae.load_state_dict(torch.load("vae_60.pth"))
        
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)

    best_loss = float('inf') 
    for epoch in range(epochs):
        # Training: 
        vae.train() 
        train_loss = 0.0 
        for batch_idx, (data, _) in enumerate(train_data):
            data = data.to(device)
            optimizer_vae.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward() 
            optimizer_vae.step()
            train_loss += loss.item() * data.size(0)                              # scale by batch size 
        
        train_loss /= len(train_data.dataset)

        # Validation:
        vae.eval() 
        val_loss = 0 
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_data):
                data = data.to(device)
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                val_loss += loss.item() * data.size(0) 
        val_loss /= len(val_data.dataset) 
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(vae.state_dict(), "new_vae.pth")
            print(f"VAE save best weights at {epoch+1}")

    # Test:
    vae.eval()  
    vae.load_state_dict(torch.load("new_vae.pth")) 
    
    recon_errors = []
    with torch.no_grad():  
        data, _ = next(iter(test_data))
        data = data.to(device)
        recon_batch, _, _ = vae(data)
        visualize(vae, data, recon_batch, device, num_images=8)
        
        for batch_idx, (data, _) in enumerate(test_data):
            data = data.to(device)
            recon_batch, _, _ = vae(data)
            error = (recon_batch - data).pow(2).mean(dim=[1,2,3])
            recon_errors.extend(error.cpu().numpy())
    
    mean_error = np.mean(recon_errors) 
    std_error = np.std(recon_errors)
    print(f"Reconstruction error mean (per pixel): {mean_error:.6f}")
    print(f"Reconstruction error standard deviation (per pixel): {std_error:.6f}")

def visualize(vae, data, recon_batch, device, num_images=8):                        # visual examples of test input and output 
    num_images = min(num_images, data.shape[0])  
    inputs = data[:num_images].cpu() * 0.5 + 0.5
    outputs = recon_batch[:num_images].cpu() * 0.5 + 0.5
    
    grid = torch.cat([inputs, outputs], dim=0)
    torchvision.utils.save_image(grid, 'vae_test_input_output.png', nrow=num_images, padding=2, normalize=True)


# Task 2b: UNet Denoiser (latent dim=512)
class pos_enc(nn.Module):                                                        # positional encoiding for denoising timestep t 
    def __init__(self, dim, max_len=1000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)                                           # initialize positional encoding matrix 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)      # create position indices vector 
        
        # calculate the divisor term for wavelength scaling  
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)                             # sin for even indices 
        pe[:, 1::2] = torch.cos(position * div_term)                             # cos for odd indices 
        self.register_buffer('pe', pe)                                           # non-trainable fixed state
    
    def forward(self, t):
        return self.pe[t.long()].unsqueeze(1)                                    # [16, 1, 512]
        
class adap_noise(nn.Module):                                                     # learns noise scaling from 1 - 64 - 512 
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid())
    
    def forward(self, t):
        return self.fc(t.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)              # [16, 512, 1, 1] 
 
class Denoiser(nn.Module):                                                      # UNet Structure
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.pos_encoding = pos_enc(512)
        self.noise_scaling = adap_noise(latent_dim)
        
        self.enc1 = self.enc_block(latent_dim, 256)                             # 3 downsampling encoder blocks: 512 - 256 - 512 - 512 
        self.enc2 = self.enc_block(256, 512)
        self.enc3 = self.enc_block(512, 512)
        
        self.bottleneck = nn.Sequential(                                        # bottlneck: 2 conv layers: 512 - 512 - 512 
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU())
        
        self.dec3 = self.dec_block(1024, 512)                                   # 3 upsampling decoder blocks with skip connections: 512+512 -> 512 -> 512+512 -> 256 -> 256+256 -> 256
        self.dec2 = self.dec_block(1024, 256)
        self.dec1 = self.dec_block(512, 256)
        
        self.final = nn.Sequential(                                             # channel adjustment 256 - 512 
            nn.Conv2d(256, latent_dim, 3, padding=1),
            nn.Tanh())
    
    def enc_block(self, in_c, out_c):                                           # to compress input and extract features             
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),                     # 1st conv layer with downsampling
            nn.GroupNorm(32, out_c),                                            # normalize groups of 32 channels for stable training
            nn.SiLU(),                                                          # sigmoid-weighted linear unit, smoother gradient than ReLU
            nn.Conv2d(out_c, out_c, 3, padding=1),                              # 2nd conv layer, no downsampling
            nn.GroupNorm(32, out_c),
            nn.SiLU())
    
    def dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # upsampling layer, 1x1 -> 2x2 
            nn.Conv2d(in_c, out_c, 3, padding=1),                                # 1st conv layer
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),                               # 2nd conv layer 
            nn.GroupNorm(32, out_c),
            nn.SiLU())
    
    def forward(self, x, t):
        t_embed = self.pos_encoding(t)
        noise_scale = self.noise_scaling(t)
        
        x1 = self.enc1(x * noise_scale)                                         # Apply adaptive noise scaling
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        b = self.bottleneck(x3)
        
        d3 = self.dec3(torch.cat([b, x3], dim=1))                               # concatenate skip connection (output from encoder block 3) 
        d2 = self.dec2(torch.cat([d3, x2], dim=1))                              # concatenate skip connection (output from encoder block 2) 
        d2 = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=False)    # to ensure d2 match x1 spatial dimensions
        d1 = self.dec1(torch.cat([d2, x1], dim=1))                              # concatenate skip connection (output from encoder block 1) 
        
        return self.final(d1)                                                   # [16, 512, 6, 6] 

def train_denoiser(vae, epochs=50, lr=0.0001, load_from_file=True): 
    if load_from_file:
        denoiser.load_state_dict(torch.load("UNet_110.pth"))  
        
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-5)
   
    # scheduled learning rate to continue loss decreasing 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_data))
    
    best_loss = float('inf')
    scaler = GradScaler()                                                 # gradient scaling 

    for epoch in range(epochs):
        denoiser.train()
        train_loss = 0.0
        progress_bar = tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)         
            with torch.no_grad():
                mu, logvar = vae.encode(data)                             # latent representation 
                z = vae.reparameterize(mu, torch.zeros_like(mu))
    
            t = torch.rand(data.size(0), device=device)                   # add noise with random timesteps t in range of [0, 1]
            noise = torch.randn_like(z)
            noisy_z = z + t.view(-1, 1, 1, 1) * noise
          
            with autocast(device_type='cuda', dtype=torch.float16):       # denoise and compute loss
                pred_z = denoiser(noisy_z, t)
                loss = F.mse_loss(pred_z, z)
            
            scaler.scale(loss).backward()                                 # backprop with gradient scaling 
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss/(batch_idx+1))
        
        val_loss = val_denoiser(vae, denoiser)
        print(f'Epoch {epoch+1} | Train Loss: {train_loss/len(train_data):.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(denoiser.state_dict(), 'new_UNet.pth') 
    return denoiser

def val_denoiser(vae, denoiser):
    denoiser.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in val_data:
            data = data.to(device)        
            mu, logvar = vae.encode(data)                                           # latent representation 
            z = vae.reparameterize(mu, torch.zeros_like(mu))

            timesteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]          # 10 fixed evaluation denoising timesteps 
            for t_val in timesteps:
                t = torch.full((data.size(0),), t_val, device=device)               # add noise at specific timestep 
                noise = torch.randn_like(z)
                noisy_z = z + t.view(-1, 1, 1, 1) * noise               
                pred_z = denoiser(noisy_z, t)
                loss = F.mse_loss(pred_z, z)
                total_loss += loss.item()    
    return total_loss / (len(val_data) * 10)                                        # normalize by samples * timesteps

def test_denoiser(vae, denoiser, num_steps=10, num_samples=5):
    denoiser.load_state_dict(torch.load('new_UNet.pth'))
    denoiser.eval()
    vae.eval() 
    
    latent_shape = (num_samples, 512, 6, 6)                               
    z = torch.randn(latent_shape, device=device)                                    # random latent noise 
    images = [] 
    with torch.no_grad():     
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)                         # from full noise to cleaner 
      
        for i, t in enumerate(timesteps):           
            current_t = t * torch.ones(num_samples, device=device) 
            z = denoiser(z, current_t)                                              # denoising latent representation
            decoded_img = vae.decode(z)                                             # generte reconstructed image
            images.append(decoded_img)              
        torchvision.utils.save_image(torch.cat(images, dim=0), 'denoising_steps.png', nrow=num_samples)
   
    total_loss = 0
    with torch.no_grad():     
        for data, _ in test_data:
            data = data.to(device)
            mu, logvar = vae.encode(data)
            z = vae.reparameterize(mu, logvar)
            
            for t_val in torch.linspace(1.0, 0.1, num_steps):                      # evaluate on test timesteps
                t = t_val * torch.ones(data.size(0), device=device)                 
                noise = torch.randn_like(z)
                noisy_z = z + t.view(-1, 1, 1, 1) * noise             
                pred_z = denoiser(noisy_z, t)
                loss = F.mse_loss(pred_z, z)
                total_loss += loss.item()
    
    avg_loss = total_loss / (len(test_data) * num_steps)
    print(f"UNet Denoiser: average test loss over {num_steps} steps: {avg_loss:.4f}")


"""
By default, the model continues training from saved weights (load_from_file = True).
To train from scratch, set load_from_file = False.

During testing, the script uses the most recently saved weights.

To test the UNet denoiser without additional training, please update the filename 
in test_denoiser() to "UNet_110.pth" 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=512).to(device)
train_val_test(epochs=1, lr=0.0001, load_from_file=True) 

denoiser = Denoiser(latent_dim=512).to(device) 
train_denoiser(vae, epochs=1, lr=0.00005, load_from_file=True)
test_denoiser(vae, denoiser, num_steps=10, num_samples=5) 
