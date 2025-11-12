import pygame
import numpy as np
from level_manager import Level, LevelManager
import os
import time

# --- Config ---

def set_game_logo(path_to_logo = os.path.join('assets', 'logo', 'logo.png')):
    # Load the logo image (ideally a square PNG with transparency)
    logo = pygame.image.load(path_to_logo).convert_alpha()
    logo = pygame.transform.smoothscale(logo, (256, 256))
    pygame.display.set_icon(logo)  # sets taskbar + window frame icon

    # Optionally also set window caption alongside it
    pygame.display.set_caption("Sorcerer's Riddle")


def create_config(width = 900, height = 600, bottom_h = 80, fps = 60):
    CONFIG = {
        'WIDTH': width,
        'HEIGHT': height,
        'FPS': fps,
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'GRAY': (180, 180, 180),
        'BLUE': (50, 150, 255),
        'RED': (255, 80, 80),
        'SILVER' : (192, 192, 192),
        'LIGHT_SILVER' : (220, 220, 220),
        'DARK_SILVER' : (169, 169, 169),
        'GOLD' : (212, 175, 55),
        'LIGHT_GOLD' : (255, 215, 0),
        'DARK_GOLD' : (184, 134, 11),
        'LEFT_W': width // 3,
        'RIGHT_W': width - (width // 3),
        'BOTTOM_H': bottom_h,
        'PLOT_X0': width // 3,
        'PLOT_Y0': 0,
        'PLOT_W': width - (width // 3),
        'PLOT_H': height - bottom_h

    }
    return CONFIG



def load_sounds():
    return {
        'click': pygame.mixer.Sound("assets/sounds/button-press.wav"),
        'beam': pygame.mixer.Sound("assets/sounds/magic-spell.wav"),
        'level_music' : pygame.mixer.Sound("assets/sounds/background-music.wav"),
        'menu_music' : pygame.mixer.Sound("assets/sounds/menu-music.wav"),
    }

def load_ui_icons(size):
    ret = {}
    for img_name in ['sound_on', 'sound_off']:
        path = os.path.join('assets', 'buttons', f"{img_name}.png")
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (size, size))
        ret[img_name] = img
    return ret

def load_rune_assets():
    rune_images = {}
    for symb, name in [("+", "add"), ("-", "sub"), ("*", "times"), ("/", "div"), 
                       ("sin", "sin"), ("log", "log"), ("exp", "exp"), ("x", "var"), 
                       ("1", "one"), ("2", "two"), ("sqrt", "sqrt"), ("empty", "empty")]:
        path = os.path.join('assets', 'operators', f"{name}.png")
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            rune_images[symb] = img
    return rune_images

def load_button_assets():
    button_images = {}
    for name in ["cast", "retry", "restart", "next"]:
        button_images[name] = {}
        for state in ['active', 'inactive']:
            path = os.path.join('assets', 'buttons', f"btn_{name}_{state}.png")
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                button_images[name][state] = img
    return button_images

def load_enemy_assets(enemy_name, height, width, max_frames=6):
    dir_path = os.path.join('assets', 'enemies')
    file_names = [x for x in os.listdir(dir_path) if x.startswith(f'{enemy_name}_') and x.endswith('.png')]
    file_names = sorted(file_names, key=lambda x: int(x.split('_')[-1][:-4]))  # sort by frame number

    enemy_images = []
    for fn in file_names:
        path = os.path.join(dir_path, fn)
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (width, height))
        enemy_images.append(img)

    enemy_images += list(reversed(enemy_images[1:-1]))  # ping-pong animation
    return enemy_images

def load_projectile_assets(height, width):
    dir_path = os.path.join('assets', 'projectile')
    imgs = []

    file_names = [x for x in os.listdir(dir_path) if x.startswith(f'projectile_') and x.endswith('.png')]
    file_names = sorted(file_names, key=lambda x: int(x.split('_')[-1][:-4]))  # sort by frame number

    for fn in file_names:
        path = os.path.join(dir_path, fn)
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (width, height))
        imgs.append(img)
    return imgs

def load_star_assets():
    dir_path = os.path.join('assets', 'star')
    imgs = {}

    for fn, key in [('star_enabled.png', 'on'), ('star_disabled.png', 'off')]:
        path = os.path.join(dir_path, fn)
        assert os.path.exists(path), f"Star asset {path} not found"
        img = pygame.image.load(path).convert_alpha()
        imgs[key] = img
    return imgs


class Button:
    def __init__(self, images, pos, action, size=None, sound = None):
        """
        images: dict {"normal": Surface, "disabled": Surface}
        pos: (x, y) center position of the button
        action: str identifier for the button
        size: (width, height) to scale the button image (optional)
        """
        self.action = action
        self.enabled = True
        self.pos = pos
        self.sound = sound

        # If size is provided, scale both states
        if size:
            self.images = {
                "active": pygame.transform.smoothscale(images["active"], size),
                "inactive": pygame.transform.smoothscale(images["inactive"], size),
            }
        else:
            self.images = images

        self.current_image = self.images["active"]
        self.rect = self.current_image.get_rect(center=pos)

    def draw(self, screen):
        img = self.images["active"] if self.enabled else self.images["inactive"]
        self.current_image = img
        screen.blit(img, self.rect)

    def handle_event(self, event):
        if not self.enabled:
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.sound:
                    self.sound.play()
                return self.action
        return None


class MuteButton:
    def __init__(self, icons, sounds, pos=(40, 40)):
        self.icons = icons
        self.sounds = sounds
        self.volumes = {key: snd.get_volume() for key, snd in sounds.items()}
        self.is_muted = False
        self.image = self.icons['sound_on']
        self.rect = self.image.get_rect(center=pos)

    def toggle(self):
        self.is_muted = not self.is_muted
        self.image = self.icons['sound_off'] if self.is_muted else self.icons['sound_on']
        for k in self.sounds:
            self.sounds[k].set_volume(0 if self.is_muted else self.volumes[k])

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.toggle()

class Enemy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.marked = False # marked for hitting
        self.hit = False
        self.hit_time = None

class Missile:
    def __init__(self, path, x_min, x_max, y_min, y_max, config_dict, frames, color='RED', speed=200):
        self.path = path
        self.speed = speed
        self.done = False
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.config = config_dict
        self.color = self.config[color]
        self.frames = frames
        self.frame_idx = 0

        # Initial position in world coordinates
        self.pos = path[0] if path else None
        # Current x in screen pixels
        self.sx = world_to_screen(self.pos[0], 0, x_min, x_max, 0, 1, self.config)[0]

    def update(self, dt, enemies=None):
        if self.done or self.pos is None:
            self.done = True
            return
        
        t = pygame.time.get_ticks() / 1000.0  # seconds
        self.frame_idx = int((t * 10)) % len(self.frames) # 5 FPS animation
        # Advance horizontal position in screen pixels
        self.sx += min(self.speed * dt, self.config['PLOT_W']//4)
        if self.sx > world_to_screen(self.path[-1][0], 0, self.x_min, self.x_max, 0, 1, self.config)[0]:
            self.done = True
            return

        # Find nearest world x along path
        for i in range(len(self.path)-1):
            sx_i, _ = world_to_screen(self.path[i][0], 0, self.x_min, self.x_max, 0, 1, self.config)
            sx_next, _ = world_to_screen(self.path[i+1][0], 0, self.x_min, self.x_max, 0, 1, self.config)
            if sx_i <= self.sx <= sx_next:
                x0, y0 = self.path[i]
                x1, y1 = self.path[i+1]
                ratio = (self.sx - sx_i) / (sx_next - sx_i)
                nx = x0 + (x1 - x0) * ratio
                ny = y0 + (y1 - y0) * ratio
                self.pos = (nx, ny)
                
                # Check for enemy hits
                if enemies:
                    for e in enemies:
                        if not e.hit and e.marked:
                            if nx >= e.x:
                                e.hit = True  # Missed
                break


    def draw(self, screen):
        if self.done or self.pos is None:
            return
        sx, sy = world_to_screen(self.pos[0], self.pos[1], self.x_min, self.x_max, self.y_min, self.y_max, self.config)
        screen.blit(self.frames[self.frame_idx], (sx - self.frames[self.frame_idx].get_width()//2, sy - self.frames[self.frame_idx].get_height()//2))
        #pygame.draw.circle(screen, self.color, (sx, sy), 6)
        
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.rune = None
        self.left = None
        self.right = None
        self.active_left = False
        self.active_right = False

    def set_rune(self, rune):
        self.rune = rune
        # Arity rules
        if rune in ['x'] or rune.replace('.', '', 1).isdigit():  # Leaf
            self.active_left = False
            self.active_right = False
        elif rune in ['sin', 'cos', 'exp', 'log']:
            self.active_left = True
            self.active_right = False
        else:  # Binary
            self.active_left = True
            self.active_right = True


# --- Build a full binary tree of depth d ---

def build_tree(depth, x, y, dx, dy):
    """
    Build a full binary tree of given depth.

    Args:
        depth: remaining depth of the tree
        x, y: position of the current node
        dx: horizontal spacing for child nodes
        dy: vertical spacing for child nodes

    Returns:
        Node: root of the subtree
    """
    if depth == 0:
        return None

    node = Node(x, y)
    node.left = build_tree(depth-1, x - dx, y + dy, dx / 2, dy)
    node.right = build_tree(depth-1, x + dx, y + dy, dx / 2, dy)
    return node


def compute_node_size(tree_depth, config_dict, margin=10):
    """
    Compute the node box size so that leaf nodes don't overlap and fit within the panel.
    Returns (box_width, box_height)
    """
    panel_width = config_dict['LEFT_W']
    n_leaves = 2 ** (tree_depth - 1)
    # Maximum box width that fits all leaves inside panel with margin
    max_box_width = (panel_width - 2*margin) // n_leaves
    # Enforce a minimum and reasonable maximum
    box_width = min(max(20, max_box_width), panel_width // 4)
    box_height = box_width
    return box_width, box_height


def assign_leaf_positions(node, depth, max_depth, x_min, x_max, y_start, y_spacing):
    if node is None:
        return []
    
    if max_depth == 1:
        node.x = (x_min + x_max) // 2
        node.y = y_start
        return [node.x]

    if depth == max_depth:
        # Compute equally spaced x for leaves
        idx = assign_leaf_positions.counter
        total = 2**(max_depth-1)
        x = x_min + idx * (x_max - x_min) / (total - 1 if total > 1 else 1)
        y = y_start + (depth-1) * y_spacing
        node.x, node.y = int(x), int(y)
        assign_leaf_positions.counter += 1
        return [node.x]

    # Recurse
    xs_left = assign_leaf_positions(node.left, depth+1, max_depth, x_min, x_max, y_start, y_spacing)
    xs_right = assign_leaf_positions(node.right, depth+1, max_depth, x_min, x_max, y_start, y_spacing)

    # Place this node in between children
    if xs_left and xs_right:
        node.x = (min(xs_left) + max(xs_right)) // 2
    elif xs_left:
        node.x = xs_left[0]
    elif xs_right:
        node.x = xs_right[0]
    else:
        node.x = (x_min + x_max) // 2

    node.y = y_start + (depth-1) * y_spacing
    return xs_left + xs_right


def layout_tree_bottom_up(root, max_depth, config_dict, y_start=50, bottom_margin=20, margin=10):
    """
    Assign positions to all nodes in the tree so they fit within the left panel.

    - Uses compute_node_size to ensure boxes fit.
    """
    panel_width = config_dict['LEFT_W']

    # Compute vertical spacing
    tree_height_available = config_dict['HEIGHT'] - config_dict['BOTTOM_H'] - y_start - bottom_margin
    y_spacing = tree_height_available / max_depth

    # Compute box width to avoid leaf overlap
    box_width, _ = compute_node_size(max_depth, config_dict, margin)

    # Adjust x_min and x_max so leaves stay inside panel
    x_min = box_width // 2 + margin
    x_max = panel_width - box_width // 2 - margin

    assign_leaf_positions.counter = 0
    assign_leaf_positions(root, 1, max_depth, x_min, x_max, y_start, y_spacing)

    return box_width  # return box width for drawing


# --- Tree evaluation ---
def eval_tree(node, x):
    if node is None or node.rune is None:
        return np.full_like(x, np.nan, dtype=float) if isinstance(x, np.ndarray) else None
    r = node.rune

    try:
        if r == 'x':
            return x
        if r.replace('.', '', 1).isdigit():
            val = float(r)
            return np.full_like(x, val, dtype=float) if isinstance(x, np.ndarray) else val
        if r == 'sin':
            return np.sin(eval_tree(node.left, x))
        if r == 'exp':
            return np.exp(eval_tree(node.left, x))
        if r == 'log':
            return np.log(eval_tree(node.left, x))
        if r == 'sqrt':
            return np.sqrt(eval_tree(node.left, x))
        if r == '+':
            return eval_tree(node.left, x) + eval_tree(node.right, x)
        if r == '-':
            return eval_tree(node.left, x) - eval_tree(node.right, x)
        if r == '*':
            return eval_tree(node.left, x) * eval_tree(node.right, x)
        if r == '/':
            denom = eval_tree(node.right, x)
            with np.errstate(divide='ignore', invalid='ignore'):
                return eval_tree(node.left, x) / denom
    except Exception:
        return np.full_like(x, np.nan, dtype=float) if isinstance(x, np.ndarray) else None
    return np.full_like(x, np.nan, dtype=float) if isinstance(x, np.ndarray) else None

# --- Coordinate transform ---
def world_to_screen(x, y, x_min, x_max, y_min, y_max, config_dict):

    nx = (x - x_min) / (x_max - x_min)
    ny = (y - y_min) / (y_max - y_min)
    sx = int(config_dict['PLOT_X0'] + nx * config_dict['PLOT_W'])
    sy = int(config_dict['PLOT_Y0'] + (1 - ny) * config_dict['PLOT_H'])
    return sx, sy

# --- Draw functions ---

def show_instructions(screen, config_dict, sounds, mute_button):
    """
    Display instructions before starting the game.
    Waits for the player to click "Start Game".
    """
    sounds['menu_music'].play(-1)
    overlay = pygame.Surface((config_dict['WIDTH'], config_dict['HEIGHT']))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))  # semi-transparent dark background
    screen.blit(overlay, (0, 0))
    mute_button.draw(screen)

    menu_img = pygame.image.load(os.path.join('assets', 'backgrounds', 'bg_menu.png')).convert()
    h, w = menu_img.get_height(), menu_img.get_width()
    # rescale height to fit screen height
    new_h = config_dict['HEIGHT']
    new_w = int(w * (new_h / h))
    menu_img = pygame.transform.smoothscale(menu_img, (new_w, new_h))
    screen.blit(menu_img, ((config_dict['WIDTH']-new_w)//2, 0))

    btn_width, btn_height = int(0.57*new_w), int(0.1*new_h)
    start_button = pygame.Rect((config_dict['WIDTH']//2 - int(0.5*btn_width), config_dict['HEIGHT'] - int(2.3*btn_height)), (btn_width, btn_height))
    # just for debugging
    #pygame.draw.rect(screen, config_dict['BLACK'], start_button, 2)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mute_button.handle_event(event)
                
                mute_button.draw(screen)
                if start_button.collidepoint(event.pos):
                    sounds['click'].play()
                    waiting = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting = False

    
        pygame.display.flip() # update the full display
    sounds['menu_music'].stop()
    


def draw_pulsing_magic_line(screen, start, end, color, tick, thickness=1):
    """Draw a magical glowing line that pulses with time."""
    # Pulse offset (oscillates smoothly)
    pulse = 2 + int(2 * np.sin(tick / 100.0))  # adjust divisor for slower/faster pulse

    # Aura layers (big soft glow)
    for i in range(4, 0, -2):
        pygame.draw.line(screen, color, start, end, thickness + i + pulse)

    # Core line
    pygame.draw.line(screen, color, start, end, thickness + pulse // 2)

def draw_transparent_line(screen, color, start, end, width, alpha=100):
    """
    Draws a line with transparency by rendering on a temporary Surface.
    """
    # Create a surface big enough for the line
    x1, y1 = start
    x2, y2 = end
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    surf_w = max_x - min_x + width*2
    surf_h = max_y - min_y + width*2

    temp_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)

    # Shift coordinates relative to temp surface
    start_rel = (x1 - min_x + width, y1 - min_y + width)
    end_rel = (x2 - min_x + width, y2 - min_y + width)

    # Apply alpha to color
    rgba = (*color[:3], alpha)

    pygame.draw.line(temp_surf, rgba, start_rel, end_rel, width)

    # Blit to main screen
    screen.blit(temp_surf, (min_x - width, min_y - width))




def draw_tree(screen, node, config_dict, rune_images, box_size=(40, 30), tick=None):
    if node is None:
        return
    w, h = box_size

    if tick is None:
        tick = pygame.time.get_ticks()

    # --- Draw connections first ---
    if node.left:
        if node.active_left:
            base = config_dict['DARK_GOLD']
            draw_pulsing_magic_line(screen, (node.x, node.y), (node.left.x, node.left.y),
                                base, tick)
        else:
            base = config_dict['SILVER']
            draw_transparent_line(screen, base, (node.x, node.y), (node.left.x, node.left.y), 4, alpha=100)
        draw_tree(screen, node.left, config_dict, rune_images, box_size, tick)

    if node.right:
        if node.active_right:
            base = config_dict['DARK_GOLD']
            draw_pulsing_magic_line(screen, (node.x, node.y), (node.right.x, node.right.y),
                                base, tick)
        else:
            base = config_dict['SILVER']
            base = tuple(list(base) + [20])
            draw_transparent_line(screen, base, (node.x, node.y), (node.right.x, node.right.y), 4, alpha=100)
        draw_tree(screen, node.right, config_dict, rune_images, box_size, tick)

    # --- Draw node icon ---
    img_key = node.rune if node.rune else 'empty'
    icon = rune_images[img_key]
    icon = pygame.transform.smoothscale(icon, (w, h))
    screen.blit(icon, (node.x - w//2, node.y - h//2))


def draw_beam(screen, root, frame, x_min, x_max, y_min, y_max, config_dict):
    xs = np.linspace(x_min, x_max, 2000)
    ys = eval_tree(root, xs)

    # Colors
    core_color = config_dict['GOLD']
    aura_colors = [config_dict['LIGHT_GOLD'], (255, 200, 100)]
    base_thickness = 4
    pulse = 1 + (frame % 10) / 5.0

    if ys is not None:
        mask = (ys <= y_max) & (ys >= y_min)
        xs, ys = xs[mask], ys[mask]
        if len(xs) > 1:
            points = [world_to_screen(x, y, x_min, x_max, y_min, y_max, config_dict) 
                      for x, y in zip(xs, ys)]

            # Create playfield-sized surface
            beam_surf = pygame.Surface((config_dict['PLOT_W'], config_dict['PLOT_H']), pygame.SRCALPHA)

            # Shift points into local coords for beam_surf
            shifted = [(px - config_dict['PLOT_X0'], py - config_dict['PLOT_Y0']) for px, py in points]

            # Draw glow + core on the beam surface
            for i, col in enumerate(aura_colors, start=1):
                pygame.draw.lines(beam_surf, col, False, shifted, base_thickness + i*3 + int(pulse))
            pygame.draw.lines(beam_surf, core_color, False, shifted, base_thickness + int(pulse))

            # Blit clipped surface into screen
            screen.blit(beam_surf, (config_dict['PLOT_X0'], config_dict['PLOT_Y0']))


def draw_function_preview(screen, root, gt_func, x_min, x_max, y_min, y_max, config_dict):
    xs = np.linspace(x_min, x_max, 2000)  # fewer points needed
    ys = eval_tree(root, xs)

    if ys is not None:
        mask = (ys <= y_max) & (ys >= y_min)
        xs, ys = xs[mask], ys[mask]
        if len(xs) > 1:
            # Convert to screen coords
            points = [world_to_screen(x, y, x_min, x_max, y_min, y_max, config_dict) 
                      for x, y in zip(xs, ys)]

            # Create playfield-sized surface with alpha
            preview_surf = pygame.Surface((config_dict['PLOT_W'], config_dict['PLOT_H']), pygame.SRCALPHA)

            # Shift points into local coords
            shifted = [(px - config_dict['PLOT_X0'], py - config_dict['PLOT_Y0']) for px, py in points]

            # Dash parameters
            dash_length = 50
            gap_length = 10
            dist_accum = 0
            draw_on = True
            c = tuple(list(config_dict['GOLD']) + [150])

            for i in range(len(shifted) - 1):
                p1, p2 = shifted[i], shifted[i+1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len = (dx**2 + dy**2) ** 0.5

                while seg_len > 0:
                    remain = (dash_length if draw_on else gap_length) - dist_accum
                    step = min(seg_len, remain)

                    ratio = step / seg_len
                    mid_x = p1[0] + dx * ratio
                    mid_y = p1[1] + dy * ratio

                    if draw_on:
                        pygame.draw.line(preview_surf, c, p1, (mid_x, mid_y), 5)

                    seg_len -= step
                    dist_accum += step
                    p1 = (mid_x, mid_y)

                    if dist_accum >= (dash_length if draw_on else gap_length):
                        dist_accum = 0
                        draw_on = not draw_on

            # Blit the clipped preview back to screen
            screen.blit(preview_surf, (config_dict['PLOT_X0'], config_dict['PLOT_Y0']))


def draw_axes(screen, x_min, x_max, y_min, y_max, config_dict):
    """
    Draw golden coordinate ticks + labels at the edges of the playfield.
    Adds a glowing magical frame around the playfield instead of partial lines.
    """
    font = pygame.font.SysFont("georgia", 20, bold=True)

    # Helper: glowing text
    def draw_glow_text(surf, text, pos, color, edge_color, font):
        """
        Draw text with a colored edge.
        
        surf: target surface
        text: string
        pos: top-left position
        color: main text color
        edge_color: color for the edge (e.g., black)
        font: pygame font
        """
        base = font.render(text, True, color)
        edge = font.render(text, True, edge_color)
        
        # Draw edge by offsetting in all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    surf.blit(edge, (pos[0] + dx, pos[1] + dy))
        
        # Draw main text on top
        surf.blit(base, pos)

    # --- Draw frame ---
    rect = pygame.Rect(config_dict['PLOT_X0'], config_dict['PLOT_Y0'],
                       config_dict['PLOT_W'], config_dict['PLOT_H'])
    pygame.draw.rect(screen, config_dict['DARK_GOLD'], rect, 3)

    # Choose ticks (skip first and last so they don’t clash with corners)
    xticks = np.linspace(x_min, x_max, 6)[1:-1]
    yticks = np.linspace(y_min, y_max, 6)[1:-1]

    # X ticks (top edge)
    for t in xticks:
        sx, sy = world_to_screen(t, y_min, x_min, x_max, y_min, y_max, config_dict)
        label = f"{t:.1f}"
        lx = sx - font.size(label)[0] // 2
        ly = config_dict['PLOT_Y0'] + 5  # just inside top edge

        draw_glow_text(screen, label, (lx, ly),
                       config_dict['GOLD'], config_dict['BLACK'], font)

        pygame.draw.line(screen, config_dict['DARK_GOLD'],
                         (sx, config_dict['PLOT_Y0']),
                         (sx, config_dict['PLOT_Y0'] + 5), 2)

    # Y ticks (right edge)
    for t in yticks:
        sx, sy = world_to_screen(x_min, t, x_min, x_max, y_min, y_max, config_dict)
        label = f"{t:.1f}"
        lx = config_dict['PLOT_X0'] + config_dict['PLOT_W'] - font.size(label)[0] - 5
        ly = sy - font.size(label)[1] // 2

        draw_glow_text(screen, label, (lx, ly),
                       config_dict['GOLD'], config_dict['BLACK'], font)

        pygame.draw.line(screen, config_dict['DARK_GOLD'],
                         (config_dict['PLOT_X0'] + config_dict['PLOT_W'] - 5, sy),
                         (config_dict['PLOT_X0'] + config_dict['PLOT_W'], sy), 2)




def draw_enemies(screen, enemies, enemy_frames, x_min, x_max, y_min, y_max, config_dict):
    t = pygame.time.get_ticks() / 1000.0  # seconds

    for i, e in enumerate(enemies):

        if e.hit:
            continue  # don't draw hit enemies
        # Base position
        sx, sy = world_to_screen(e.x, e.y, x_min, x_max, y_min, y_max, config_dict)

        # Add floating offset (each enemy gets unique phase)
        offset = 5 * np.sin(t * 2 + i)   # 5 px amplitude, speed ~2Hz
        sy += offset

        # Select animation frame
        frame_idx = int((t * 5 + i)) % len(enemy_frames) # 5 FPS animation
        enemy_img = enemy_frames[frame_idx]
        screen.blit(enemy_img, (sx - enemy_img.get_width()//2, int(sy) - enemy_img.get_height()//2))



def cast_spell(root, enemies, threshold=1e-2):
    """
    Check which enemies are hit by the player's function.
    
    - root: root of the expression tree
    - enemies: list of (x, y) positions
    - threshold: how close the function value must be to hit
    """
    hits = []
    for ex, ey in enemies:
        y_val = eval_tree(root, ex)  # evaluate function at enemy's x
        if y_val is not None and not np.isnan(y_val):
            if abs(y_val - ey) <= threshold:
                hits.append((ex, ey))
    return hits


def draw_results_popup(screen, score, has_next_level, config_dict, btn_imgs, star_imgs, sounds):
    """
    Draws the results popup with a star rating instead of % score.
    star_imgs: dict with {"on": enabled_star_surface, "off": disabled_star_surface}
    """
    overlay = pygame.Surface((config_dict['WIDTH'], config_dict['HEIGHT']))
    overlay.set_alpha(180)
    overlay.fill((50, 50, 50))  # semi-transparent dark background
    screen.blit(overlay, (0, 0))

    font_big = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 32)

    # --- Star Rating ---
    # Determine how many stars to light up
    if score <= 25:
        stars = 0
    elif score <= 50:
        stars = 1
    elif score <= 75:
        stars = 2
    else:
        stars = 3

    # Position stars at the top of the popup
    star_size = 64  # adjust depending on your sprite resolution
    spacing = 20
    total_width = 3 * star_size + 2 * spacing
    start_x = config_dict['WIDTH']//2 - total_width//2
    y_pos = config_dict['HEIGHT']//2 - 180

    for i in range(3):
        img = star_imgs["on"] if i < stars else star_imgs["off"]
        img_scaled = pygame.transform.smoothscale(img, (star_size, star_size))
        screen.blit(img_scaled, (start_x + i*(star_size+spacing), y_pos))

    # --- Buttons ---
    padding = 100
    btn_width = (config_dict['WIDTH'] - 4*padding)//3
    btn_height = int(btn_width*0.3)

    pos = (padding + btn_width//2, config_dict['HEIGHT']//2)
    retry_button = Button(btn_imgs['retry'], pos, 'retry', (btn_width, btn_height), sound = sounds['click'])
    retry_button.draw(screen)

    pos = (2*padding + int(1.5*btn_width), config_dict['HEIGHT']//2)
    restart_button = Button(btn_imgs['restart'], pos, 'restart', (btn_width, btn_height), sound = sounds['click'])
    restart_button.draw(screen)

    pos = (3*padding + int(2.5*btn_width), config_dict['HEIGHT']//2)
    next_button = Button(btn_imgs['next'], pos, 'next', (btn_width, btn_height), sound = sounds['click'])
    if not has_next_level:
        next_button.enabled = False
    next_button.draw(screen)

    if has_next_level:
        return retry_button, restart_button, next_button
    else:
        return retry_button, restart_button, None



def build_level(level, config_dict):
    """
    Build level configuration, ensuring that plotting area
    excludes the bottom panel, so missiles start within visible region.
    """
    ret = {}
    gt_func = level.gt_func
    gt_xs = level.gt_xs
    hit_thresh = level.hit_thresh
    tree_depth = level.depth
    runes = level.spellbook

    # Compute x range
    x_min, x_max = min(gt_xs) - 1, max(gt_xs) + 1
    # Compute y range from function samples
    y_vals = [gt_func(x) for x in np.linspace(x_min, x_max, 200)]
    y_min, y_max = min(y_vals) - 1, max(y_vals) + 1

    # Ensure y_min/y_max are within plotting area (leave bottom panel for spellbook)
    # We just leave a small margin at top/bottom
    #margin = 0.05 * (y_max - y_min)
    #y_min += margin
    #y_max -= margin

    # Build tree
    root = build_tree(tree_depth, config_dict['LEFT_W'] // 2, 100, config_dict['LEFT_W'] // 4, 100)
    layout_tree_bottom_up(root, tree_depth, config_dict)
    box_size = compute_node_size(tree_depth, config_dict)

    ret['gt_func'] = gt_func
    ret['gt_xs'] = gt_xs
    ret['hit_thresh'] = hit_thresh
    ret['tree_depth'] = tree_depth
    ret['runes'] = runes
    ret['x_min'] = x_min
    ret['x_max'] = x_max
    ret['y_min'] = y_min
    ret['y_max'] = y_max
    ret['root'] = root
    ret['box_size'] = box_size

    return ret


def main(lvl_manager):

    # --- Game Setup ---
    pygame.init()
    #screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((900, 600))

    set_game_logo()

    w, h = screen.get_size()  # adapt to screen resolution
    config_dict = create_config(w, h, h//7)
    pygame.display.set_caption("Sourcerer's Riddle")
    clock = pygame.time.Clock()

    # load sounds
    ui_icons = load_ui_icons(w//20)
    sounds = load_sounds()
    sounds['click'].set_volume(0.7)
    sounds['beam'].set_volume(0.7)
    sounds['level_music'].set_volume(0.3)
    sounds['menu_music'].set_volume(0.3)
    mute_button = MuteButton(ui_icons, sounds = sounds, pos=(w-30, 30))

    
    show_instructions(screen, config_dict, sounds, mute_button)

    # --- Initial Level definition ---
    level = lvl_manager.get_current_level()
    lvl_config = build_level(level, config_dict)

    # setup game state
    gt_points = [Enemy(x, lvl_config['gt_func'](x)) for x in lvl_config['gt_xs']]
    selected_rune = None
    state = 'building'  # can be 'building', 'casting', 'results' or 'paused'
    pause_time = None
    try_count = 0

    # assets
    rune_imgs = load_rune_assets()
    spellbook_bg = pygame.image.load(os.path.join("assets", 'backgrounds', "bg_dark_wood.png")).convert()
    spellbook_bg = pygame.transform.smoothscale(spellbook_bg, (config_dict['WIDTH'], config_dict['BOTTOM_H']))
    tree_bg = pygame.image.load(os.path.join("assets", 'backgrounds', "bg_stone.png")).convert()    
    tree_bg = pygame.transform.smoothscale(tree_bg, (config_dict['LEFT_W'], config_dict['HEIGHT'] - config_dict['BOTTOM_H']))

    arena_bg = pygame.image.load(os.path.join("assets", 'backgrounds', "bg_arena.png")).convert()
    arena_bg = pygame.transform.smoothscale(arena_bg, (config_dict['WIDTH']-config_dict['LEFT_W'], config_dict['HEIGHT'] - config_dict['BOTTOM_H']))

    button_imgs = load_button_assets()
    wisp_frames = load_enemy_assets('wisp', 40, 40)

    star_imgs = load_star_assets()

    # Cast button
    btn_width = config_dict['LEFT_W']//2
    btn_width -= btn_width%2 # make sure its even
    btn_height = int(btn_width*0.25)
    pos = (config_dict['LEFT_W']//2, config_dict['HEIGHT'] - config_dict['BOTTOM_H'] - btn_height//2)
    # center position!
    size = (btn_width, btn_height)
    cast_button = Button(button_imgs['cast'], pos, 'cast', size, sound = sounds['click'])
    cast_frame = 0  # for beam animation
    score = 0

    
    sounds['level_music'].play(-1)  # loop background music

    

    running = True
    while running:
        dt = clock.tick(config_dict['FPS']) / 1000

        # --- Compute spellbook rects (centered) ---
        box_w, box_h = config_dict['BOTTOM_H'] - 15, config_dict['BOTTOM_H'] - 15
        spacing = 10
        total_w = len(lvl_config['runes']) * (box_w + spacing) - spacing
        start_x = (config_dict['WIDTH'] - total_w) // 2
        y = config_dict['HEIGHT'] - config_dict['BOTTOM_H'] + int(1.0*(config_dict['BOTTOM_H'] - box_h))

        spellbook_rects = []
        for i, r in enumerate(lvl_config['runes']):
            rect = pygame.Rect(start_x + i*(box_w+spacing), y, box_w, box_h)
            spellbook_rects.append((rect, r))



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False


            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                mute_button.handle_event(event)

                if state == 'building':
                    # Pick rune from spellbook
                    for rect, r in spellbook_rects:
                        if rect.collidepoint(mx, my):
                            selected_rune = r
                    # Place rune on tree
                    bw, bh = lvl_config['box_size']
                    def check_click(node):
                        if node is None: return
                        rect = pygame.Rect(node.x-bw//2, node.y-bh//2, bw, bh)
                        if rect.collidepoint(mx, my) and selected_rune:
                            sounds['click'].play()
                            node.set_rune(selected_rune)
                        check_click(node.left)
                        check_click(node.right)
                    check_click(lvl_config['root'])

                    # Cast button
                    if cast_button.rect.collidepoint(mx, my):
                        preds = eval_tree(lvl_config['root'], np.array(lvl_config['gt_xs']))
                        mask = (preds <= lvl_config['y_max']) & (preds >= lvl_config['y_min'])
                        if np.any(mask):
                            state = 'casting'
                            sounds['beam'].play()
                            try_count += 1

                            count = 0
                            for e, p in zip(gt_points, preds):
                                if not np.isnan(p) and abs(p - e.y) <= lvl_config['hit_thresh'] and not e.hit:
                                    e.marked = True
                                    count += 1
                                else:
                                    e.marked = False
                            cast_frame = 0
                            cast_button.enabled = False  # disable during casting

                elif state == 'results':
                    retry_btn, restart_btn, next_btn = draw_results_popup(
                        screen, score,
                        lvl_manager.has_next_level(), config_dict, button_imgs, star_imgs, sounds
                    )
                    if retry_btn.rect.collidepoint(mx, my):
                        # Retry current level
                        sounds['click'].play()
                        level = lvl_manager.get_current_level()
                        lvl_config = build_level(level, config_dict)
                        gt_points = [Enemy(x, lvl_config['gt_func'](x)) for x in lvl_config['gt_xs']]
                        selected_rune = None
                        pause_time = None
                        try_count = 0
                        state = 'building'
                    elif restart_btn.rect.collidepoint(mx, my):
                        # Restart from first level
                        sounds['click'].play()
                        lvl_manager.current = 0
                        level = lvl_manager.get_current_level()
                        lvl_config = build_level(level, config_dict)
                        gt_points = [Enemy(x, lvl_config['gt_func'](x)) for x in lvl_config['gt_xs']]
                        selected_rune = None
                        pause_time = None
                        try_count = 0
                        state = 'building'
                    elif next_btn and next_btn.rect.collidepoint(mx, my):
                        if lvl_manager.next_level():
                            sounds['click'].play()
                            level = lvl_manager.get_current_level()
                            lvl_config = build_level(level, config_dict)
                            gt_points = [Enemy(x, lvl_config['gt_func'](x)) for x in lvl_config['gt_xs']]
                            selected_rune = None
                            pause_time = None
                            try_count = 0
                            state = 'building'
                        else:
                            print("All levels completed!")
                            running = False



        screen.fill(config_dict['WHITE'])

        # Panels background
        screen.blit(tree_bg, (0, 0))
        #pygame.draw.rect(screen, (240,240,240), (0,0,config_dict['LEFT_W'],config_dict['HEIGHT']-config_dict['BOTTOM_H']))  # left
        
        screen.blit(arena_bg, (config_dict['LEFT_W'], 0))
        #pygame.draw.rect(screen, (250,250,250), (config_dict['LEFT_W'],0,config_dict['RIGHT_W'],config_dict['HEIGHT']-config_dict['BOTTOM_H']))  # right
        screen.blit(spellbook_bg, (0, config_dict['HEIGHT'] - config_dict['BOTTOM_H']))



        # Draw tree in left panel
        draw_tree(screen, lvl_config['root'], config_dict, rune_imgs, box_size=lvl_config['box_size'])

        # Cast button
        cast_button.draw(screen)
        mute_button.draw(screen)
        # Draw axes, preview, enemies
        
        if state != 'casting':
            draw_function_preview(screen, lvl_config['root'], lvl_config['gt_func'],
                                lvl_config['x_min'], lvl_config['x_max'], lvl_config['y_min'], lvl_config['y_max'], config_dict)
        draw_enemies(screen, gt_points, wisp_frames,lvl_config['x_min'], lvl_config['x_max'],
                     lvl_config['y_min'], lvl_config['y_max'], config_dict)

        
        # Spellbook in bottom panel
        for rect, r in spellbook_rects:
            if r==selected_rune:
                pygame.draw.rect(screen, config_dict['DARK_GOLD'], rect)
                pygame.draw.rect(screen, config_dict['BLACK'], rect, 2)

            # Draw rune icon
            b = 5  # padding inside box
            if r in rune_imgs:
                icon = rune_imgs[r]
                icon = pygame.transform.smoothscale(icon, (rect.width-2*b, rect.height-2*b))  # fit nicely
                screen.blit(icon, (rect.x+b, rect.y+b))


        # --- Beam update & draw ---
        if state == 'casting':
            if cast_frame < 20:
                draw_beam(screen, lvl_config['root'], cast_frame, lvl_config['x_min'], lvl_config['x_max'], lvl_config['y_min'], lvl_config['y_max'], config_dict)
                cast_frame += 1
            else:
                # end of casting
                for e in gt_points:
                    if e.marked:
                        e.hit = True
                
                hit_count = sum(1 for e in gt_points if e.hit)
                if len(gt_points) > 1:
                    score = (1 - (try_count - 1)/(len(gt_points) - 1)) * 100
                else:
                    score = 100 if hit_count == 1 else 0
                if hit_count == len(gt_points):
                    state = 'paused'
                    pause_time = pygame.time.get_ticks()
                else:
                    state = 'building'
                cast_button.enabled = True  # re-enable button

        if state == 'paused':
            if pygame.time.get_ticks() - pause_time > 1000:  # 2 second pause
                state = 'results'
        
        # draw here so they appear above beam and preview
        draw_axes(screen, lvl_config['x_min'], lvl_config['x_max'], lvl_config['y_min'], lvl_config['y_max'], config_dict)


        # Results popup
        if state == 'results':
            retry_btn, prev_btn, next_btn = draw_results_popup(
                        screen, score,
                        lvl_manager.has_next_level(), config_dict, button_imgs, star_imgs, sounds
                    )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # Define levels
    lm = LevelManager()
    lm.add_level(Level(2, lambda x: x**2, [-1, 0, 1], 1e-2, ['x', '1', '+', '-', '*']))
    lm.add_level(Level(3, lambda x: np.sin(x**2), [-2, -1, 0, 1, 2], 1e-3, ['x', '1', '2', '+', '-', '*', 'sin', 'exp', 'log']))

    main(lm)

