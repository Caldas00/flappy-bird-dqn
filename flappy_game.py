import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self, width=400, height=600, render=True):
        self.width, self.height = width, height
        self.render_enabled = render
        
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy RL v3 - Enhanced")
            self.clock = pygame.time.Clock()
            
        # Physics parameters
        self.gravity = 0.5
        self.flap_strength = -8
        self.bird_x = 50
        self.bird_radius = 15
        
        # Pipe parameters
        self.pipe_gap = 150
        self.pipe_width = 60
        self.pipe_speed = 4
        
        # Enhanced reward system
        self.alive_reward = 0.1
        self.pipe_reward = 10.0
        self.death_penalty = -10.0
        self.distance_reward_scale = 0.01  # Small reward for getting closer to gap
        
        # Visual elements
        if render:
            self.bird_img = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(self.bird_img, (255, 255, 0), (15, 15), 15)
            pygame.draw.circle(self.bird_img, (255, 200, 0), (15, 15), 15, 2)
            
        # Game state
        self.reset()
    
    def reset(self):
        self.bird_y = self.height // 2
        self.bird_vel = 0.0
        self.pipe_x = self.width + 50  # Start pipe further away
        self.pipe_height = random.randint(100, self.height - self.pipe_gap - 100)
        self.score = 0
        self.frames_alive = 0
        self.high_score = getattr(self, 'high_score', 0)
        self.last_pipe_passed = False
        return self.get_state()
    
    def get_state(self):
        """Enhanced state representation with 6 features"""
        # Normalize bird position and velocity
        bird_y_norm = self.bird_y / self.height
        bird_vel_norm = np.clip(self.bird_vel / 15.0, -1, 1)
        
        # Horizontal distance to pipe
        pipe_x_norm = (self.pipe_x - self.bird_x) / self.width
        
        # Vertical distances to pipe openings
        pipe_center = self.pipe_height + self.pipe_gap / 2
        pipe_top_dist = (self.pipe_height - self.bird_y) / self.height
        pipe_bottom_dist = (self.bird_y - (self.pipe_height + self.pipe_gap)) / self.height
        
        # Distance to pipe center (normalized)
        pipe_center_dist = (pipe_center - self.bird_y) / self.height
        
        return np.array([
            bird_y_norm,
            bird_vel_norm,
            pipe_x_norm,
            pipe_top_dist,
            pipe_bottom_dist,
            pipe_center_dist
        ], dtype=np.float32)
    
    def step(self, action):
        # Action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird_vel = self.flap_strength
        
        # Physics update
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        self.frames_alive += 1
        
        # Pipe movement
        self.pipe_x -= self.pipe_speed
        passed_pipe = False
        
        # Check if pipe passed
        if not self.last_pipe_passed and self.pipe_x + self.pipe_width < self.bird_x:
            self.last_pipe_passed = True
            passed_pipe = True
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
        
        # Reset pipe when it goes off screen
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_height = random.randint(100, self.height - self.pipe_gap - 100)
            self.last_pipe_passed = False
        
        # Collision detection
        done = False
        
        # Ground and ceiling collision
        if self.bird_y <= self.bird_radius or self.bird_y >= self.height - 30 - self.bird_radius:
            done = True
        
        # Pipe collision - more precise
        if (self.bird_x + self.bird_radius > self.pipe_x and 
            self.bird_x - self.bird_radius < self.pipe_x + self.pipe_width):
            if (self.bird_y - self.bird_radius < self.pipe_height or 
                self.bird_y + self.bird_radius > self.pipe_height + self.pipe_gap):
                done = True
        
        # Enhanced reward calculation
        reward = self.alive_reward
        
        if passed_pipe:
            reward = self.pipe_reward
        elif done:
            reward = self.death_penalty
        else:
            # Small reward for getting closer to the gap center
            pipe_center = self.pipe_height + self.pipe_gap / 2
            distance_to_center = abs(self.bird_y - pipe_center)
            max_distance = self.pipe_gap / 2
            if distance_to_center < max_distance:
                distance_reward = (1 - distance_to_center / max_distance) * self.distance_reward_scale
                reward += distance_reward
        
        return self.get_state(), reward, done
    
    def render(self):
        if not self.render_enabled:
            return
            
        # Background
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Draw pipes with gradient effect
        pipe_color = (34, 139, 34)
        pipe_highlight = (50, 205, 50)
        
        # Top pipe
        pygame.draw.rect(self.screen, pipe_color, 
                        (self.pipe_x, 0, self.pipe_width, self.pipe_height))
        pygame.draw.rect(self.screen, pipe_highlight, 
                        (self.pipe_x, 0, 5, self.pipe_height))
        
        # Bottom pipe
        bottom_pipe_y = self.pipe_height + self.pipe_gap
        pygame.draw.rect(self.screen, pipe_color, 
                        (self.pipe_x, bottom_pipe_y, self.pipe_width, self.height - bottom_pipe_y))
        pygame.draw.rect(self.screen, pipe_highlight, 
                        (self.pipe_x, bottom_pipe_y, 5, self.height - bottom_pipe_y))
        
        # Draw bird
        self.screen.blit(self.bird_img, (self.bird_x - 15, int(self.bird_y) - 15))
        
        # Ground
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.height - 30, self.width, 30))
        
        # UI elements
        font = pygame.font.SysFont('Arial', 24, bold=True)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        high_text = font.render(f'High: {self.high_score}', True, (255, 255, 255))
        frames_text = pygame.font.SysFont('Arial', 16).render(f'Frames: {self.frames_alive}', True, (255, 255, 255))
        
        # Add shadow effect to text
        shadow_offset = 2
        score_shadow = font.render(f'Score: {self.score}', True, (0, 0, 0))
        high_shadow = font.render(f'High: {self.high_score}', True, (0, 0, 0))
        
        self.screen.blit(score_shadow, (12, 12))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_shadow, (12, 42))
        self.screen.blit(high_text, (10, 40))
        self.screen.blit(frames_text, (10, 70))
        
        # Debug info (velocity)
        vel_text = pygame.font.SysFont('Arial', 14).render(f'Vel: {self.bird_vel:.1f}', True, (200, 200, 200))
        self.screen.blit(vel_text, (self.width - 80, 10))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def get_score(self):
        return self.score
        
    def get_high_score(self):
        return self.high_score