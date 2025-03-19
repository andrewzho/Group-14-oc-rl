import torch
import torch.nn as nn
import numpy as np
import os

# Add global debug flag that respects verbosity setting
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'  # Set DEBUG=1 in environment to enable debug output

def debug_print(*args, **kwargs):
    """Print only when debug mode is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-5)  # Lower momentum for more stable statistics
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-5)  # Lower momentum for more stable statistics
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    """Channel attention module to focus on important features"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        return x * attention_map

class RecurrentPPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions, lstm_hidden_size=256, use_lstm=True):
        super(RecurrentPPONetwork, self).__init__()
        # Input shape is (12, 84, 84) for 4 stacked frames (4 * 3 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),  # [12, 84, 84] -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32, 20, 20] -> [64, 9, 9]
            nn.ReLU(),
            ResidualBlock(64),  # Add residual block for better gradient flow
            ChannelAttention(64, reduction=8),  # Enhanced attention with stronger reduction
            SpatialAttention(kernel_size=5),  # Add spatial attention
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [64, 9, 9] -> [64, 7, 7]
            nn.ReLU(),
            ResidualBlock(64),  # Another residual block
            ChannelAttention(64, reduction=8),  # Enhanced attention
            SpatialAttention(kernel_size=3),  # Add spatial attention with smaller kernel
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # [64, 7, 7] -> [128, 5, 5]
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),  # Extra 1x1 conv for more features
            nn.ReLU(),
            nn.Flatten()  # [128, 5, 5] -> [3200]
        )
        conv_out_size = self._get_conv_output(input_shape)
        
        # LSTM for temporal memory
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        
        if use_lstm:
            # Compress CNN features before LSTM to reduce parameters
            self.feature_compressor = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.LayerNorm(512)  # Add layer norm before LSTM
            )
            
            # LSTM layer with increased complexity
            self.lstm = nn.LSTM(
                input_size=512,
                hidden_size=lstm_hidden_size,
                num_layers=2,  # Increased to 2 layers for more memory capacity
                batch_first=True,
                dropout=0.1  # Add dropout for regularization
            )
            
            # Feature projection after LSTM
            self.post_lstm = nn.Sequential(
                nn.Linear(lstm_hidden_size, 512),
                nn.ReLU(),
                nn.LayerNorm(512)  # Normalize LSTM outputs for stable training
            )
            
            # Advantage stream uses LSTM outputs
            self.advantage_stream = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),  # Added extra layer for more capacity
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
            
            # Value stream uses LSTM outputs with more capacity
            self.value_stream = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),  # Added extra layer
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Original architecture without LSTM (for backward compatibility)
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_actions)
            )
            
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            
            # Layer norm for better training stability
            self.layer_norm_advantage = nn.LayerNorm(512)
            self.layer_norm_value = nn.LayerNorm(512)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv(input)
            return int(np.prod(output.size()))

    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization with proper scaling."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            gain = 1.0
            if module.weight.shape[0] == 1:  # Value stream final layer
                gain = 0.01
            elif len(module.weight.shape) >= 2 and module.weight.shape[1] > 0:
                # For the policy head, use smaller gain
                if module.weight.shape[0] > 10:  # Likely the policy output layer
                    gain = np.sqrt(0.1)
                else:
                    gain = np.sqrt(2)
            
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            # Special initialization for LSTM
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 (helps with long-term memory)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def init_lstm_state(self, batch_size=1, device=None):
        """Initialize LSTM hidden and cell states."""
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
            
        # Create hidden states matching the number of LSTM layers (2)
        num_layers = 2  # Match the num_layers parameter in the LSTM constructor
        h0 = torch.zeros(num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, x, lstm_state=None, episode_starts=None, sequence_length=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width] or 
               [batch_size, sequence_length, channels, height, width] if using LSTM
            lstm_state: Optional tuple of (h0, c0) for LSTM hidden and cell states
            episode_starts: Boolean tensor indicating which environments are starting new episodes
            sequence_length: The length of sequences for processing (for LSTM)
            
        Returns:
            policy_logits: Action logits
            value: Value function estimate
            next_lstm_state: Next LSTM state (if use_lstm=True)
        """
        # Handle input dimensionality
        batch_size = x.size(0)
        original_shape = x.shape
        seq_len = 1  # Default for non-sequence inputs
        
        # Use debug_print instead of print
        try:
            debug_print(f"Forward input shape: {x.shape}, ep_starts: {episode_starts}, seq_length: {sequence_length}")
        except:
            pass  # Debug print might not be defined
        
        # Use provided sequence_length if available
        if sequence_length is not None:
            # Store the sequence_length as an attribute for future reference
            self.sequence_length = sequence_length
        elif hasattr(self, 'sequence_length'):
            # Use previously stored sequence_length
            sequence_length = self.sequence_length
        else:
            # Default sequence length is 1
            sequence_length = 1
            
        # Handle different input formats
        if x.dim() == 5 and self.use_lstm:
            # Input is [batch_size, sequence_length, channels, height, width]
            # Reshape to process with CNN
            seq_len = x.size(1)
            sequence_length = seq_len
            x = x.view(batch_size * seq_len, *x.shape[2:])
        elif x.dim() == 4 and self.use_lstm and sequence_length > 1:
            # Input is [batch_size, channels, height, width] but we need to treat it as a sequence
            # We'll repeat the same input for the sequence (used when we don't have actual sequence data)
            x = x.unsqueeze(1).expand(-1, sequence_length, -1, -1, -1)
            x = x.reshape(batch_size * sequence_length, *x.shape[2:])
        elif x.dim() == 3:
            # Single observation, add batch dimension
            x = x.unsqueeze(0)
            
        # Normalize input if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # Check for NaN values in input
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        if torch.isinf(x).any():
            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
            
        # Process through CNN
        features = self.conv(x)
        
        # Process with LSTM if enabled
        if self.use_lstm:
            # Compress features before LSTM
            features = self.feature_compressor(features)
            
            # Determine actual sequence length
            if len(original_shape) == 5:
                # We have explicit sequence dimension
                actual_seq_len = original_shape[1]
            else:
                # Infer sequence length based on input/output relationship
                actual_seq_len = sequence_length if sequence_length > 1 else 1
                
            # Reshape to sequence for LSTM [batch_size, seq_len, features]
            features = features.view(batch_size, actual_seq_len, -1)
                
            # Initialize LSTM state if not provided
            if lstm_state is None:
                lstm_state = self.init_lstm_state(batch_size, device=x.device)
            
            # Handle episode starts by zeroing out states where episodes are starting
            if episode_starts is not None and self.use_lstm:
                # Create a mask for resetting states
                mask = (~episode_starts).float().view(1, -1, 1)
                # Apply the mask - multiply by 0 for new episodes, keep for continuing episodes
                h_masked = lstm_state[0] * mask
                c_masked = lstm_state[1] * mask
                # Use the masked states
                lstm_state = (h_masked, c_masked)
                
            # Process with LSTM
            lstm_out, next_lstm_state = self.lstm(features, lstm_state)
            
            # Use the last output of the sequence for policy and value
            # But return the full sequence if needed
            if actual_seq_len > 1:
                # Extract features from full sequence - shape will be [batch_size, seq_len, hidden_size]
                features = lstm_out
                # Also keep last state features for policy heads if we're predicting a single action
                last_features = lstm_out[:, -1]
            else:
                features = lstm_out.squeeze(1)
                last_features = features
                
            # Post-LSTM processing based on sequence needs
            if actual_seq_len > 1:
                # Process full sequence - reshape to [batch_size*seq_len, hidden_size]
                flat_features = features.reshape(-1, features.size(-1))
                processed_features = self.post_lstm(flat_features)
                
                # Reshape back to [batch_size, seq_len, features]
                processed_features = processed_features.view(batch_size, actual_seq_len, -1)
                
                # Apply policy and value networks to each timestep
                # Reshape to [batch_size*seq_len, feature_dim] for the advantage networks
                flat_processed = processed_features.reshape(-1, processed_features.size(-1))
                
                # Get policy logits and values for all timesteps
                policy_logits = self.advantage_stream(flat_processed)
                values = self.value_stream(flat_processed)
                
                # Reshape back to include sequence dimension
                policy_logits = policy_logits.view(batch_size, actual_seq_len, -1)
                values = values.view(batch_size, actual_seq_len)
            else:
                # Single timestep prediction
                processed_features = self.post_lstm(last_features)
                policy_logits = self.advantage_stream(processed_features)
                values = self.value_stream(processed_features).squeeze(-1)
            
            # Apply safety clipping
            if policy_logits.dim() == 3:
                # For sequences, clip each logit
                policy_logits = torch.clamp(policy_logits, -20.0, 20.0)
            else:
                # For single predictions
                policy_logits = torch.clamp(policy_logits, -20.0, 20.0)
            
            return policy_logits, values, next_lstm_state
        else:
            # Original forward pass without LSTM
            # Apply advantage stream with layer normalization
            adv_features = self.advantage_stream[0](features)  # First linear layer
            adv_features = self.layer_norm_advantage(adv_features)
            adv_features = self.advantage_stream[1](adv_features)  # ReLU
            adv_features = self.advantage_stream[2](adv_features)  # Dropout
            adv_features = self.advantage_stream[3](adv_features)  # Second linear layer
            adv_features = self.advantage_stream[4](adv_features)  # ReLU
            adv_features = self.advantage_stream[5](adv_features)  # Dropout
            policy_logits = self.advantage_stream[6](adv_features)  # Output layer
            
            # Apply value stream with layer normalization
            val_features = self.value_stream[0](features)  # First linear layer
            val_features = self.layer_norm_value(val_features)
            val_features = self.value_stream[1](val_features)  # ReLU
            val_features = self.value_stream[2](val_features)  # Dropout
            val_features = self.value_stream[3](val_features)  # Second linear layer
            val_features = self.value_stream[4](val_features)  # ReLU
            val_features = self.value_stream[5](val_features)  # Dropout
            value = self.value_stream[6](val_features)  # Output layer
            
            # Check for NaN or Inf values in outputs and replace with safe values
            if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                policy_logits = torch.where(torch.isnan(policy_logits) | torch.isinf(policy_logits), 
                                        torch.zeros_like(policy_logits), policy_logits)
            
            return policy_logits, value, None

    def get_action_and_value(self, x, lstm_state=None, deterministic=False):
        """Convenience method for getting actions and values in one call."""
        if self.use_lstm:
            policy_logits, value, next_lstm_state = self(x)
        else:
            policy_logits, value, _ = self(x)
            next_lstm_state = None
            
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        if deterministic:
            action = torch.argmax(policy_logits, dim=1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value, next_lstm_state

# For backward compatibility, keep the old class name as an alias to the new class
class PPONetwork(RecurrentPPONetwork):
    """
    Original PPONetwork class, now an alias to RecurrentPPONetwork with use_lstm=False.
    This maintains backward compatibility with existing code.
    """
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__(input_shape, num_actions, use_lstm=False)
        
    def forward(self, x):
        # Override to maintain original signature
        policy_logits, value, _ = super().forward(x)
        return policy_logits, value

if __name__ == "__main__":
    # Test the recurrent network
    recurrent_net = RecurrentPPONetwork(input_shape=(12, 84, 84), num_actions=54, use_lstm=True)
    print(recurrent_net)
    
    # Test single step forward pass
    x = torch.randn(4, 12, 84, 84)  # Batch of 4 examples
    policy, value, lstm_state = recurrent_net(x)
    print(f"Policy output shape: {policy.shape}")  # Should be [4, 54]
    print(f"Value output shape: {value.shape}")    # Should be [4]
    print(f"LSTM state shapes: {lstm_state[0].shape}, {lstm_state[1].shape}")  # Should be [1, 4, 256]
    
    # Test sequential forward pass
    seq_x = torch.randn(2, 3, 12, 84, 84)  # Batch of 2, sequence length 3
    seq_policy, seq_value, seq_lstm_state = recurrent_net(seq_x, sequence_length=3)
    print(f"Sequential policy output shape: {seq_policy.shape}")  # Should be [2, 54]
    print(f"Sequential value output shape: {seq_value.shape}")    # Should be [2]
    
    # Test backward compatibility with old interface
    net = PPONetwork(input_shape=(12, 84, 84), num_actions=54)
    print(net)
    
    # Test forward pass with old interface
    policy, value = net(x)
    print(f"Old interface policy output shape: {policy.shape}")  # Should be [4, 54]
    print(f"Old interface value output shape: {value.shape}")    # Should be [4]