import pygame
import sys

def init_pygame(screen_width, screen_height, scale=2, title="Ms. Pac-Man"):
    pygame.init()
    window = pygame.display.set_mode((screen_width * scale, screen_height * scale))
    pygame.display.set_caption(title)
    return window

def blit_frame(window, frame, scale=2):
    # frame: numpy array RGB (H, W, 3)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (frame.shape[1] * scale, frame.shape[0] * scale))
    window.blit(surface, (0, 0))
    pygame.display.flip()

def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

def close_pygame():
    pygame.quit()
