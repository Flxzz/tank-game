import pygame
import random
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Initialize Pygame
pygame.init()

# 在pygame.init()后添加音效加载
pygame.mixer.init()
laser_sound = pygame.mixer.Sound('laser.wav')

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TANK_SIZE = 30
BULLET_SIZE = 5
POWERUP_SIZE = TANK_SIZE // 2
WALL_THICKNESS = 4  # Make walls thinner
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tank Battle")
clock = pygame.time.Clock()

# 加载并缩放坦克图片为20%
red_tank_img_raw = pygame.image.load('red_tank.png').convert_alpha()
black_tank_img_raw = pygame.image.load('black_tank.png').convert_alpha()
red_size = (int(red_tank_img_raw.get_width() * 0.15), int(red_tank_img_raw.get_height() * 0.15))
black_size = (int(black_tank_img_raw.get_width() * 0.15), int(black_tank_img_raw.get_height() * 0.15))
red_tank_img = pygame.transform.smoothscale(red_tank_img_raw, red_size)
black_tank_img = pygame.transform.smoothscale(black_tank_img_raw, black_size)

@dataclass
class Wall:
    start: Tuple[int, int]
    end: Tuple[int, int]
    rect: pygame.Rect  # For collision detection

class Map:
    def __init__(self):
        self.walls: List[Wall] = []
        self.generate_map()
    
    def generate_map(self):
        self.walls.clear()
        # 迷宫参数
        cell_size = 100  # 增大格子间距
        rows = SCREEN_HEIGHT // cell_size
        cols = SCREEN_WIDTH // cell_size
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        # 记录每个格子的墙体（上右下左）
        walls_grid = [[[True, True, True, True] for _ in range(cols)] for _ in range(rows)]
        stack = [(0, 0)]
        visited[0][0] = True
        # DFS生成迷宫
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc, wall_idx, opp_idx in [(-1,0,0,2),(0,1,1,3),(1,0,2,0),(0,-1,3,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
                    neighbors.append((nr, nc, wall_idx, opp_idx))
            if neighbors:
                nr, nc, wall_idx, opp_idx = random.choice(neighbors)
                walls_grid[r][c][wall_idx] = False
                walls_grid[nr][nc][opp_idx] = False
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()
        # 画外边框
        self.walls.append(Wall((0,0),(SCREEN_WIDTH,0),pygame.Rect(0,0,SCREEN_WIDTH,WALL_THICKNESS)))
        self.walls.append(Wall((0,0),(0,SCREEN_HEIGHT),pygame.Rect(0,0,WALL_THICKNESS,SCREEN_HEIGHT)))
        self.walls.append(Wall((0,SCREEN_HEIGHT-1),(SCREEN_WIDTH,SCREEN_HEIGHT-1),pygame.Rect(0,SCREEN_HEIGHT-1,SCREEN_WIDTH,WALL_THICKNESS)))
        self.walls.append(Wall((SCREEN_WIDTH-1,0),(SCREEN_WIDTH-1,SCREEN_HEIGHT),pygame.Rect(SCREEN_WIDTH-1,0,WALL_THICKNESS,SCREEN_HEIGHT)))
        # 画迷宫细线
        for r in range(rows):
            for c in range(cols):
                x, y = c*cell_size, r*cell_size
                if walls_grid[r][c][0]:  # 上
                    self.walls.append(Wall((x, y), (x+cell_size, y), pygame.Rect(x, y, cell_size, WALL_THICKNESS)))
                if walls_grid[r][c][1]:  # 右
                    self.walls.append(Wall((x+cell_size, y), (x+cell_size, y+cell_size), pygame.Rect(x+cell_size, y, WALL_THICKNESS, cell_size)))
                if walls_grid[r][c][2]:  # 下
                    self.walls.append(Wall((x, y+cell_size), (x+cell_size, y+cell_size), pygame.Rect(x, y+cell_size, cell_size, WALL_THICKNESS)))
                if walls_grid[r][c][3]:  # 左
                    self.walls.append(Wall((x, y), (x, y+cell_size), pygame.Rect(x, y, WALL_THICKNESS, cell_size)))

    def draw(self, surface):
        for wall in self.walls:
            # Draw thin black line
            pygame.draw.line(surface, BLACK, wall.start, wall.end, 2)

def get_nearest_intersection(x, y, dx, dy, walls):
    if not walls:
        return None, None, None
    min_dist = float('inf')
    hit_point = None
    hit_wall = None
    hit_normal = None
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        # 射线: (x, y) + t*(dx, dy)
        # 墙体: (x1, y1) + s*((x2-x1), (y2-y1)), 0<=s<=1
        wx, wy = x2 - x1, y2 - y1
        denom = dx * wy - dy * wx
        if abs(denom) < 1e-8:
            continue  # 平行
        t = ((x1 - x) * wy - (y1 - y) * wx) / denom
        s = ((x1 - x) * dy - (y1 - y) * dx) / denom
        if t > 1e-6 and 0 <= s <= 1:
            ix = x + t * dx
            iy = y + t * dy
            dist = math.hypot(ix - x, iy - y)
            if dist < min_dist:
                min_dist = dist
                hit_point = (ix, iy)
                hit_wall = wall
                # 法线向量
                wall_dx, wall_dy = x2 - x1, y2 - y1
                norm = math.hypot(wall_dx, wall_dy)
                nx, ny = wall_dy / norm, -wall_dx / norm  # 逆时针90度
                # 判断法线朝向（应指向射线外侧）
                if (dx * nx + dy * ny) > 0:
                    nx, ny = -nx, -ny
                hit_normal = (nx, ny)
    return hit_point, hit_wall, hit_normal

class Laser:
    def __init__(self, x: float, y: float, angle: float):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 30
        self.active = True
        self.segments = []
        self.max_bounce = 10
        self.bounce_count = 0

    def update(self, walls: List[Wall]):
        if not self.active or self.bounce_count > self.max_bounce:
            self.active = False
            return
        dx = math.sin(math.radians(self.angle))
        dy = -math.cos(math.radians(self.angle))
        hit_point, hit_wall, hit_normal = get_nearest_intersection(self.x, self.y, dx, dy, walls)
        if hit_point is None:
            next_x = self.x + dx * self.speed
            next_y = self.y + dy * self.speed
            self.segments.append(((self.x, self.y), (next_x, next_y)))
            self.x, self.y = next_x, next_y
        else:
            dist = math.hypot(hit_point[0] - self.x, hit_point[1] - self.y)
            if dist > self.speed:
                next_x = self.x + dx * self.speed
                next_y = self.y + dy * self.speed
                self.segments.append(((self.x, self.y), (next_x, next_y)))
                self.x, self.y = next_x, next_y
            else:
                self.segments.append(((self.x, self.y), hit_point))
                in_vec = (dx, dy)
                nx, ny = hit_normal
                dot = in_vec[0]*nx + in_vec[1]*ny
                rx = in_vec[0] - 2*dot*nx
                ry = in_vec[1] - 2*dot*ny
                self.x = hit_point[0] + rx * 0.5
                self.y = hit_point[1] + ry * 0.5
                self.angle = (math.degrees(math.atan2(ry, rx)) + 90) % 360  # 修正反射角度
                self.bounce_count += 1

    def draw(self, surface):
        for seg in self.segments:
            pygame.draw.line(surface, RED, seg[0], seg[1], 2)

class Tank:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int], is_player: bool = False):
        self.x = x
        self.y = y
        self.color = color
        self.angle = 0
        self.speed = 3
        self.is_player = is_player
        if is_player:
            self.image = red_tank_img
            self.size = red_tank_img.get_width()
        else:
            self.image = black_tank_img
            self.size = black_tank_img.get_width()
        self.collision_scale = 1.3
        self.collision_size = int(self.size * self.collision_scale)
        self.rect = pygame.Rect(self.x - self.collision_size//2, self.y - self.collision_size//2, self.collision_size, self.collision_size)
        self.can_shoot = False if is_player else True
        self.hp = 3

    def update_rect(self):
        self.rect = pygame.Rect(self.x - self.collision_size//2, self.y - self.collision_size//2, self.collision_size, self.collision_size)

    def move(self, dx: float, dy: float, walls: List[Wall]):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        new_rect = pygame.Rect(new_x - self.collision_size//2, new_y - self.collision_size//2, self.collision_size, self.collision_size)
        can_move = True
        for wall in walls:
            if new_rect.colliderect(wall.rect):
                can_move = False
                break
        if can_move:
            self.x = new_x
            self.y = new_y
            self.rect = new_rect

    def rotate(self, angle_change: float, walls: List[Wall] = None):
        old_angle = self.angle
        self.angle = (self.angle + angle_change) % 360
        if walls is not None:
            test_rect = pygame.Rect(self.x - self.collision_size//2, self.y - self.collision_size//2, self.collision_size, self.collision_size)
            for wall in walls:
                if test_rect.colliderect(wall.rect):
                    self.angle = old_angle
                    break

    def draw(self, surface, walls):
        rotated_img = pygame.transform.rotate(self.image, -self.angle)
        rect = rotated_img.get_rect(center=(self.x, self.y))
        surface.blit(rotated_img, rect.topleft)
        if self.is_player and self.can_shoot:
            self.draw_aim_line(surface, walls)

    def get_barrel_tip(self):
        # 炮管末端坐标
        barrel_length = self.size // 2
        tip_x = self.x + math.sin(math.radians(self.angle)) * barrel_length
        tip_y = self.y - math.cos(math.radians(self.angle)) * barrel_length
        return tip_x, tip_y

    def draw_aim_line(self, surface, walls: List[Wall]):
        if not walls:
            return
        max_bounce = 10
        dash_length = 10
        dash_gap = 5
        bounces = 0
        x, y = self.get_barrel_tip()
        angle = self.angle
        while bounces < max_bounce:
            dx = math.sin(math.radians(angle))
            dy = -math.cos(math.radians(angle))
            hit_point, hit_wall, hit_normal = get_nearest_intersection(x, y, dx, dy, walls)
            if hit_point is None:
                end_x = x + dx * 2000
                end_y = y + dy * 2000
                self._draw_dashed_line(surface, (x, y), (end_x, end_y), dash_length, dash_gap)
                break
            else:
                self._draw_dashed_line(surface, (x, y), hit_point, dash_length, dash_gap)
                in_vec = (dx, dy)
                nx, ny = hit_normal
                dot = in_vec[0]*nx + in_vec[1]*ny
                rx = in_vec[0] - 2*dot*nx
                ry = in_vec[1] - 2*dot*ny
                x, y = hit_point
                angle = (math.degrees(math.atan2(ry, rx)) + 90) % 360  # 修正反射角度
                bounces += 1

    def _draw_dashed_line(self, surface, start, end, dash_length, dash_gap):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        dx /= length
        dy /= length
        dist = 0
        is_dash = True
        while dist < length:
            seg_len = min(dash_length if is_dash else dash_gap, length - dist)
            nx1 = x1 + dx * dist
            ny1 = y1 + dy * dist
            nx2 = x1 + dx * (dist + seg_len)
            ny2 = y1 + dy * (dist + seg_len)
            if is_dash:
                pygame.draw.line(surface, RED, (nx1, ny1), (nx2, ny2), 2)
            dist += seg_len
            is_dash = not is_dash

    def move_forward(self, walls: List[Wall]):
        dx = math.sin(math.radians(self.angle))
        dy = -math.cos(math.radians(self.angle))
        self.move(dx, dy, walls)
    def move_backward(self, walls: List[Wall]):
        dx = -math.sin(math.radians(self.angle))
        dy = math.cos(math.radians(self.angle))
        self.move(dx, dy, walls)

    def ai_reset(self):
        r = random.random()
        if r < 0.7:
            self.ai_mode = 'turn'
            self.ai_timer = 30
            self.ai_turn_dir = random.choice([-1, 1])  # -1左转, 1右转
        elif r < 0.85:
            self.ai_mode = 'forward'
            self.ai_timer = 60
        else:
            self.ai_mode = 'backward'
            self.ai_timer = 60

class Bullet:
    def __init__(self, x: float, y: float, angle: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.angle = angle
        self.color = color
        self.speed = 5
        self.creation_time = pygame.time.get_ticks()
        self.lifetime = 10000  # 10 seconds in milliseconds
        self.rect = pygame.Rect(x - BULLET_SIZE//2, y - BULLET_SIZE//2, BULLET_SIZE, BULLET_SIZE)

    def move(self, walls: List[Wall]) -> bool:
        remaining = self.speed
        x, y = self.x, self.y
        angle = self.angle
        max_bounce = 3  # 防止极端情况下无限反弹
        bounces = 0
        while remaining > 1e-3 and bounces < max_bounce:
            dx = math.cos(math.radians(angle))
            dy = -math.sin(math.radians(angle))
            hit_point, hit_wall, hit_normal = get_nearest_intersection(x, y, dx, dy, walls)
            if hit_point is None:
                # 没有碰撞，直接移动剩余距离
                x += dx * remaining
                y += dy * remaining
                break
            else:
                dist = math.hypot(hit_point[0] - x, hit_point[1] - y)
                if dist > remaining:
                    # 碰撞点在本帧外，直接移动
                    x += dx * remaining
                    y += dy * remaining
                    break
                # 先移动到碰撞点
                x, y = hit_point
                remaining -= dist
                # 反射
                in_vec = (dx, dy)
                nx, ny = hit_normal
                dot = in_vec[0]*nx + in_vec[1]*ny
                rx = in_vec[0] - 2*dot*nx
                ry = in_vec[1] - 2*dot*ny
                angle = math.degrees(math.atan2(-ry, rx))
                bounces += 1
        self.x = x
        self.y = y
        self.angle = angle
        self.rect = pygame.Rect(self.x - BULLET_SIZE//2, self.y - BULLET_SIZE//2, BULLET_SIZE, BULLET_SIZE)
        return False

    def should_destroy(self) -> bool:
        return pygame.time.get_ticks() - self.creation_time > self.lifetime

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), BULLET_SIZE)

class PowerUp:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.randint(WALL_THICKNESS + POWERUP_SIZE, SCREEN_WIDTH - WALL_THICKNESS - POWERUP_SIZE)
        self.y = random.randint(WALL_THICKNESS + POWERUP_SIZE, SCREEN_HEIGHT - WALL_THICKNESS - POWERUP_SIZE)
        self.rect = pygame.Rect(self.x - POWERUP_SIZE//2, self.y - POWERUP_SIZE//2, POWERUP_SIZE, POWERUP_SIZE)
        self.spawn_time = pygame.time.get_ticks()

    def draw(self, surface):
        pygame.draw.rect(surface, RED, self.rect)

def get_safe_spawn(walls, collision_size, margin=40):
    while True:
        x = random.randint(margin, SCREEN_WIDTH - margin)
        y = random.randint(margin, SCREEN_HEIGHT - margin)
        rect = pygame.Rect(x - collision_size//2, y - collision_size//2, collision_size, collision_size)
        if not any(rect.colliderect(wall.rect) for wall in walls):
            return x, y

class Game:
    def __init__(self):
        self.map = Map()
        # 先用图片尺寸创建坦克对象以获得collision_size
        temp_player = Tank(0, 0, RED, True)
        temp_enemy = Tank(0, 0, BLACK)
        px, py = get_safe_spawn(self.map.walls, temp_player.collision_size)
        ex, ey = get_safe_spawn(self.map.walls, temp_enemy.collision_size)
        self.player = Tank(px, py, RED, True)
        self.player.speed = 4
        self.enemy = Tank(ex, ey, BLACK)
        self.bullets: List[Bullet] = []
        self.laser: Optional[Laser] = None
        self.power_up = PowerUp()
        self.last_powerup_spawn_time = pygame.time.get_ticks()
        self.powerup_on_field = True
        self.last_enemy_shot = pygame.time.get_ticks()
        self.enemy_shot_delay = 2000
        self.running = True

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.player.move_forward(self.map.walls)
        if keys[pygame.K_s]:
            self.player.move_backward(self.map.walls)
        if keys[pygame.K_a]:
            self.player.rotate(-4, self.map.walls)  # 左转更快
        if keys[pygame.K_d]:
            self.player.rotate(4, self.map.walls)   # 右转更快
        # 玩家激光发射
        if keys[pygame.K_SPACE] and self.player.can_shoot and self.laser is None:
            tip_x, tip_y = self.player.get_barrel_tip()
            self.laser = Laser(tip_x, tip_y, self.player.angle)
            laser_sound.play()
            self.player.can_shoot = False

    def update_enemy(self):
        # 黑色坦克AI
        if not hasattr(self.enemy, 'ai_mode'):
            self.enemy.ai_reset()
        if self.enemy.ai_mode == 'turn':
            self.enemy.rotate(3 * self.enemy.ai_turn_dir)
        elif self.enemy.ai_mode == 'forward':
            self.enemy.move_forward(self.map.walls)
        elif self.enemy.ai_mode == 'backward':
            self.enemy.move_backward(self.map.walls)
        self.enemy.ai_timer -= 1
        if self.enemy.ai_timer <= 0:
            self.enemy.ai_reset()
        # 保持原有的射击逻辑
        current_time = pygame.time.get_ticks()
        if current_time - self.last_enemy_shot > self.enemy_shot_delay and len(self.bullets) < 2:
            self.bullets.append(Bullet(self.enemy.x, self.enemy.y, self.enemy.angle, BLACK))
            self.last_enemy_shot = current_time

    def update_power_up(self):
        current_time = pygame.time.get_ticks()
        # 物资每30秒刷新一次（如果场上没有物资才生成）
        if not self.powerup_on_field and current_time - self.last_powerup_spawn_time > 30000:
            self.power_up.respawn()
            self.powerup_on_field = True
            self.last_powerup_spawn_time = current_time
        # 玩家收集物资
        if self.powerup_on_field and self.player.rect.colliderect(self.power_up.rect):
            self.player.can_shoot = True
            self.powerup_on_field = False
        # 开局物资30秒后消失并刷新
        if self.powerup_on_field and current_time - self.last_powerup_spawn_time > 30000:
            self.powerup_on_field = False

    def update(self):
        self.handle_input()
        self.update_enemy()
        self.update_power_up()
        # 更新激光
        if self.laser and self.laser.active:
            self.laser.update(self.map.walls)
            for seg in self.laser.segments[-2:]:
                if self.enemy.rect.clipline(seg[0], seg[1]):
                    self.running = False  # 玩家胜利
                    self.laser.active = False
        elif self.laser and not self.laser.active:
            self.laser = None
        # 更新子弹
        for bullet in self.bullets[:]:
            bullet.move(self.map.walls)
            if bullet.rect.colliderect(self.player.rect):
                self.player.hp -= 1
                self.bullets.remove(bullet)
                if self.player.hp <= 0:
                    self.running = False
                break
            if bullet.should_destroy():
                self.bullets.remove(bullet)

    def draw(self):
        screen.fill(WHITE)
        self.map.draw(screen)
        self.player.draw(screen, self.map.walls)
        self.enemy.draw(screen, self.map.walls)
        if self.powerup_on_field:
            self.power_up.draw(screen)
        if self.laser:
            self.laser.draw(screen)
        for bullet in self.bullets:
            bullet.draw(screen)
        # 显示血量
        font = pygame.font.SysFont(None, 36)
        hp_text = font.render(f"HP: {self.player.hp}", True, RED)
        screen.blit(hp_text, (10, 10))
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.update()
            self.draw()
            clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit() 