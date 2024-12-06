import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize the pygame
pygame.init()

# Set the font
font = pygame.font.Font('arial.ttf', 25)

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
SPEED = 10

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        # Set clock
        self.clock = pygame.time.Clock()
        
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
    def play_step(self):
        # 1. Collect and map user input to direction
        for event in pygame.event.get():
            # If user clicks the close button, quit the game
            if event.type == pygame.QUIT:
                pygame.quit()   # Free all resources used by pygame
                quit()          # Exit the program
            # If user presses a key
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. Move the snake
        # Update the head direction
        self._move(self.direction) 
        # Insert the new head position at the beginning of the snake list
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        
        # 4. Place new food 
        if self.head == self.food:
            self.score += 1
            self._place_food()
        # Else just move the snake and pop the last block to ensure snake's lenth is maintained
        else:
            self.snake.pop()
            
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return game_over, self.score
    
    # Define the method to check collision
    def _is_collision(self):
        # If snake hits the wall
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        
        # If snake hits itself
        if self.head in self.snake[1:]:
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
        # Update the whole display
        pygame.display.flip()
        
    # Define the method to move the snake
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        # Move the snake in the direction
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        
        self.head = Point(x, y)
            
# Main function
if __name__ == '__main__':
    game = SnakeGame()
    
    # Game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            break
        
    print('Final score', score)
    
    pygame.quit()