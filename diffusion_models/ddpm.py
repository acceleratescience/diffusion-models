import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class DDPM:
    def __init__(self, model, optimizer, T: int, start: float, end: float, device: torch.device = torch.device('cpu')):
        """DDPM Scheduler

        Args:
            model (UNet.UNetSmol): The U-Net model backbone
            T (int): Total number of timesteps
            start (float): Smallest variance
            end (float): Largest variance
            device (torch.device, optional): Device. Defaults to torch.device('cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.T = T
        self.beta = torch.linspace(start, end, T).to(device)
        alpha = 1. - self.beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.noise_coefficient = (1 - alpha) / self.sqrt_one_minus_alpha_bar
        self.sqrt_alpha_inv = torch.sqrt(1 / alpha)
        

    def forward(self, x_0: torch.Tensor, t: float) -> torch.Tensor | torch.Tensor:
        """The forward diffusion process

        Args:
            x_0 (torch.Tensor): Initial input
            t (float): Current timestep

        Returns:
            torch.Tensor | torch.Tensor: The output of the forward diffusion process, the noise and
                the output of the model.
        """
        t = t.int()
        noise = torch.randn_like(x_0)
        xt = self.sqrt_alpha_bar[t, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        return xt, noise

    
    @torch.no_grad()
    def reverse(self, x_t: torch.Tensor, t: torch.Tensor, epsilon_t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion process

        Args:
            x_t (torch.Tensor): _description_
            t (torch.Tensor): _description_
            epsilon_t (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        t = torch.squeeze(t[0].int())
        u_t = self.sqrt_alpha_inv[t] * (x_t - self.noise_coefficient[t] * epsilon_t)

        if t == 0:
            return u_t
        else:
            noise = torch.randn_like(x_t)
            return u_t + torch.sqrt(self.beta[t-1]) * noise


    @torch.no_grad()
    def sample(self, label: int, batch_size: int) -> list[torch.Tensor]:
        """Sample some images from the model with a given label

        Args:
            label (int): The digit you want to generate
            batch_size (int): How many examples you want to generate

        Returns:
            list[torch.Tensor]: A list containing batch_size number of images for each step of the
                diffusion process.
        """
        x_t = torch.randn(batch_size, 1, 28, 28).to(self.device)
        labels = torch.zeros(batch_size, 10).to(self.device)
        labels[torch.arange(batch_size), label] = 1

        x_ts = []
        for i in tqdm(range(0, self.T)[::-1]):
            t = torch.full((batch_size,), i).to(self.device)
            # t = t.float()
            epsilon_t = self.model(x_t, t.float(), labels)
            x_t = self.reverse(x_t, t, epsilon_t)
            x_ts.append(x_t.detach().numpy())

        return x_ts
    


    def loss_function(self, x_0: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculates the MSE loss between the true noise and the predicted noise.

        Args:
            x_0 (torch.Tensor): _description_
            t (torch.Tensor): _description_
            label (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x_noise, noise = self.forward(x_0, t)
        noise_prediction = self.model(x_noise, t.float(), label)
        loss = F.mse_loss(noise, noise_prediction)
        return loss


    def train(self,  train_loader, epochs: int = 10) -> list[float]:
        """Train the model

        Args:
            train_loader (DataLoader): Training data loader
            epochs (int, optional): Defaults to 10.

        Returns:
            list: For some reason.
        """
        losses = []
        for epoch in range(epochs):
            running_loss = 0
            batch_size = train_loader.batch_size
            for images, targets, labels in tqdm(train_loader, desc='Training', total=len(train_loader)):
                images, targets, labels = images.to(self.device), targets.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                t = torch.randint(0, self.T, (batch_size,), device=self.device)
                t = t.float()
                loss = self.loss_function(images, t, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # print(loss.item())
                losses.append(loss.item())

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

        return losses
