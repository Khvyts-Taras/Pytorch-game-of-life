import torch
import pygame
import numpy as np

pygame.init()

window_size = [1400, 800]
display = pygame.display.set_mode(window_size)
clock = pygame.time.Clock()
device = 'cuda'

initial_state = torch.randint(0, 2, size=window_size, device=device).float()
initial_state = initial_state.unsqueeze(0)

kernel = torch.tensor([[1.0, 1.0, 1.0],
                       [1.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0]], device=device)

conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
conv.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    neighbors = conv(initial_state)
    initial_state = ((neighbors == 3) | ((neighbors == 2) & (initial_state == 1))).float()

    image = initial_state.detach().cpu().numpy().squeeze()
    scaled_image = pygame.transform.scale(pygame.surfarray.make_surface(image*255), window_size)

    display.blit(scaled_image, (0, 0))
    pygame.display.update()
    dt = clock.tick(500)
    pygame.display.set_caption(str(round(1000/dt)))

pygame.quit()
