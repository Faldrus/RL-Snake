import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os                           # To save the model

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()                                      # Call the constructor of the parent class (nn.Module)
        # First layer
        self.linear1 = nn.Linear(input_size, hidden_size)       
        # Second layer
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))                             # Apply ReLU activation function to the first layer
        x = self.linear2(x)                                     # Apply the second layer (no need to apply activation function because we get the output)
        return x
        
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)                # Save the model state dictionary to the file
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)     # Adam optimizer to update the model parameters with given lr
        self.criterion = nn.MSELoss()                                   # Loss function
        
    def train_step(self, state, action, reward, next_state, game_over):
        # Convert the numpy arrays to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Check if we only have 1 dimension -> Need to add a dimension 
        if len(state.shape) == 1:       # If this is true, we would have (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )    # Tuple with one element
            
        # 1. Get the predicted Q values with the current state
        pred = self.model(state)
        
        # 2. Get the target Q values with the next state
        #    r + y * max(next_predicted Q value) -> only onc if not done
        # Since pred has 3 Q values for each action, we need to get the Q value for the action taken
        # pred.clone()
        # preds[argmax(action)] = Q_new
        
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        # 3. Calculate the loss
        self.optimizer.zero_grad()              # Reset the gradients
        loss = self.criterion(target, pred)     # Calculate the loss
        loss.backward()                         # Backpropagation
        self.optimizer.step()                   # Update the weights
        
            
            