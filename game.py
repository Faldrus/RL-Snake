import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize the pygame
pygame.init()

# Set the font
font = pygame.font.Font('arial.ttf', 25)

# Reset function such that agent can start a new game
# Reward function
# Play function to get a direction given an action play(action) -> direction
# Game iteration function to play the game and return the new state, reward, and game over
# Change in is_collision function
 

# Set directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
# Define point
Point = namedtuple('Point', 'x, y')

# Define RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Set block size and speed
BLOCK_SIZE = 20
SPEED = 30

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        # Set clock
        self.clock = pygame.time.Clock()
        
        self.reset()
    
    # Define reset function
    def reset(self):
        # Initialize the game state
        self.direction = Direction.RIGHT
        
        # Initialize the snake
        self.head = Point(self.w/2, self.h/2)   # Set head snake in the middle of the screen
        # Define the snake body with 3 blocks lenght
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        # Place food
        self._place_food()
        self.frame_iteration = 0 # To keep track of which game iteration we are
    
    # Define the method to place the food
    def _place_food(self):
        # Place food in random position
        # self.w - BLOCK_SIZE: to avoid food to be placed outside the screen
        # (self.w - BLOCK_SIZE) // BLOCK_SIZE: returns the number of blocks that fits the screen and where the food can be placed
        # * BLOCK_SIZE: to get the position in pixels coordinates
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        
        self.food = Point(x, y)
        # If food is placed in the snake body, place it again
        if self.food in self.snake:
            self._place_food()
            
    # Define the method to play the game
    def play_step(self, action):
        # Update the frame iteration at each game iteration
        self.frame_iteration += 1
        # 1. Collect and map user input to direction
        for event in pygame.event.get():
            # If user clicks the close button, quit the game
            if event.type == pygame.QUIT:
                pygame.quit()   # Free all resources used by pygame
                quit()          # Exit the program
            
    
        # 2. Move the snake
        # Update the head direction
        self._move(action) 
        # Insert the new head position at the beginning of the snake list
        self.snake.insert(0, self.head)
        
        # 3. Check if game over and rewards
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):    # Stop the game if collision occurs or at some point in time
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food 
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        # Else just move the snake and pop the last block to ensure snake's lenth is maintained
        else:
            self.snake.pop()
            
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return reward, game_over, self.score
    
    # Define the method to check collision, it needs to be public so also the agent can access it
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
            
        # If snake hits the wall
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # If snake hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    # Define the update of graphics
    def _update_ui(self):
        # Fill the display with BLACK
        self.display.fill(BLACK)
        
        # Draw the snake
        for pt in self.snake:
            # External rectangle
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Smaller internal rectangle
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw the score
        text = font.render('Score: ' + str(self.score), True, WHITE)
        # Displays the score at the top left corner on the display
        self.display.blit(text, [0, 0])
        
        game_n = font.render('Game: ' + str(self.frame_iteration), True, WHITE)
        self.display.blit(game_n, [0, 20])
        # Update the whole display
        pygame.display.flip()
        
    # Define the method to move the snake
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] # Define the clockwise direction
        idx = clock_wise.index(self.direction)  # Get the index of the current direction
        
        if np.array_equal(action, [1, 0, 0]):       # Move straight
            new_dir = clock_wise[idx]               # Keep the same direction
        if np.array_equal(action, [0, 1, 0]):       # Move right
            new_dir = clock_wise[(idx + 1) % 4]     # Move to the right with respect to the current direction    
        if np.array_equal(action, [0, 0, 1]):       # Move left
            new_dir = clock_wise[(idx - 1) % 4]     # Move to the left with respect to the current direction
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        # Move the snake in the direction
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        
        self.head = Point(x, y)