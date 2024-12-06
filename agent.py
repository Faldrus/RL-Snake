import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                            # Parameter to control randomness
        self.gamma = 0.9                            # Discount factor (0,1)
        self.memory = deque(maxlen = MAX_MEMORY)    # Memory to store the state, action, reward, next_state, and game_over. When full it pop the leftmost element
        self.model = Linear_QNet(11, 256, 3)        # Neural network model. 11 states as input, 256 hidden neurons, and 3 actions as output
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)    # Optimizer
        
    
    def get_state(self, game):
        head = game.snake[0]                        # Get snake head
        # Points around the snake head to check for collision
        point_l = Point(head.x - 20, head.y)        
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or       # If snake is heading to RIGHT and the point to the RIGHT is a collision -> Danger in the straight direction
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Move direction (only one of them is True)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,                      # Food is to the left of the snake head
            game.food.x > game.head.x,                      # Food is to the right of the snake head
            game.food.y < game.head.y,                      # Food is above the snake head
            game.food.y > game.head.y                       # Food is below the snake head
        ]
        
        return np.array(state, dtype=int)                   # Convert the boolean values to integers
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))      # Append to the memory one single tuple
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)                    # Sample a batch of BATCH_SIZE from the memory consisting of a list of tuples
        else:
            mini_sample = self.memory                                               # If the memory is smaller than the batch size, use the whole memory
            
        # Call the training step on mini_sample
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)       # Unzip the mini_sample into different lists (it compresses all states into one list etc.)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)  # Since we have more than one tuple, they are called with plural names
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)   # Train the model on only one game step
    
    def get_action(self, state):
        # Random moves at the beginning to explore
        self.epsilon = 80 - self.n_games                        # Epsilon decreases as the number of games increases (80 is hardcoded)
        final_move = [0, 0, 0]                                  # Initialize the final move
        
        # As epsilon decreases, the agent becomes more greedy
        if random.randint(0, 200) < self.epsilon:               # More games -> Smaller Epsilon -> Less frequently random will be less than epsilon
            move = random.randint(0, 2)                         # Random move
            final_move[move] = 1                                # Set the random move to 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)     # Convert state to tensor
            # Prediction that calls forward method of the model
            prediction = self.model(state0)                     # Model predicts the best action given the state (It can be an array of 3 probabilities -> Just pick the highest)
            # Move is again a tensor
            move = torch.argmax(prediction).item()              # Get the index of the highest probability
            final_move[move] = 1                                # Set the move to 1
            
        return final_move
            
    
def train():
    plot_scores = []        # List to track scores
    plot_mean_scores = []
    total_score = 0
    record = 0              # Record the highest score
    agent = Agent()         # Initialize agent
    game = SnakeGameAI()    # Initialize game
    
    while True:             # Training loop
        # Get old state
        state_old = agent.get_state(game)
        
        # Get move
        final_move = agent.get_action(state_old)    # Get the best action given the state

        # Perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory of agent (Only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        
        # Remember what learnt in the short memory and store in the long memory
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        if game_over:
            # Train the long memory on everything happened in the game iteration
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
                    

if __name__ == '__main__':
    train()