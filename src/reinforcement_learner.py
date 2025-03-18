import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .helper_funcs import atomic_save

from torch.utils.tensorboard import SummaryWriter



class ProximalPolicyTrainer:
    """
    A base implementation of Proximal Policy Optimization.
    See: https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, model, epsilon=0.2, gamma=0.99, lam=0.95, lr=1e-4, ent_reg=0.001):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.ent_reg = ent_reg

    def outer_loop(self, data_collector, save_path='save.pkl', **kwargs):
        """
        Run training indefinitely, saving periodically.
        """
        for i in itertools.count():
            terms, last_terms = self.inner_loop(data_collector.trajectory_store(), **kwargs)
            self.print_outer_loop(i, terms, last_terms)
            atomic_save(self.model.state_dict(), save_path)

    def print_outer_loop(self, i, terms, last_terms):
        # Original metrics
        print('step %d: clipped=%f entropy=%f explained=%f' % 
            (i, last_terms['clip_frac'], terms['entropy'], terms['explained']))
        
        # Add floor information if available
        if hasattr(self, 'max_floor_reached'):
            print('  max floor: %d, avg floor: %.2f' % 
                (self.max_floor_reached, self.avg_floor))

    def inner_loop(self, trajectory_store, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = trajectory_store.num_steps * trajectory_store.batch_size
        advs = trajectory_store.advantages(self.gamma, self.lam)
        targets = advs + trajectory_store.value_predictions()[:-1]
        advs = (advs - np.mean(advs)) / (1e-8 + np.std(advs))
        actions = trajectory_store.actions()
        log_probs = trajectory_store.log_probs()
        firstterms = None
        lastterms = None
        for entries in trajectory_store.batches(batch_size, num_steps):
            def choose(values):
                return self.model.tensor(np.array([values[t, b] for t, b in entries]))
            terms = self.terms(choose(trajectory_store.memory_state),
                               choose(trajectory_store.obses),
                               choose(advs),
                               choose(targets),
                               choose(actions),
                               choose(log_probs))
            self.optimizer.zero_grad()
            terms['loss'].backward()
            self.optimizer.step()
            lastterms = {k: v.item() for k, v in terms.items() if k != 'model_outs'}
            if firstterms is None:
                firstterms = lastterms
            del terms
        return firstterms, lastterms

    def terms(self, memory_state, obses, advs, targets, actions, log_probs):
        model_outs = self.model(memory_state, obses)

        vf_loss = torch.mean(torch.pow(model_outs['critic'] - targets, 2))
        variance = torch.var(targets)
        explained = 1 - vf_loss / variance

        new_log_probs = -F.cross_entropy(model_outs['actor'], actions.long(), reduction='none')
        ratio = torch.exp(new_log_probs - log_probs)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        pi_loss = -torch.mean(torch.min(ratio * advs, clip_ratio * advs))
        clip_frac = torch.mean(torch.gt(ratio * advs, clip_ratio * advs).float())

        all_probs = torch.log_softmax(model_outs['actor'], dim=-1)
        neg_entropy = torch.mean(torch.sum(torch.exp(all_probs) * all_probs, dim=-1))
        ent_loss = self.ent_reg * neg_entropy

        return {
            'explained': explained,
            'clip_frac': clip_frac,
            'entropy': -neg_entropy,
            'vf_loss': vf_loss,
            'pi_loss': pi_loss,
            'ent_loss': ent_loss,
            'loss': vf_loss + pi_loss + ent_loss,
            'model_outs': model_outs,
        }
    

class LoggingProximalPolicyTrainer(ProximalPolicyTrainer):
    """ProximalPolicyTrainer implementation with TensorBoard logging."""
    
    def __init__(self, model, epsilon=0.2, gamma=0.99, lam=0.95, lr=1e-4, ent_reg=0.001, log_dir='runs/tower_agent'):
        super().__init__(model, epsilon, gamma, lam, lr, ent_reg)
        self.writer = SummaryWriter(log_dir)
        self.max_floor_reached = 0
        self.floors_completed = []
        self.episode_count = 0
        
    def outer_loop(self, data_collector, save_path='save.pkl', **kwargs):
        """Run training indefinitely, saving periodically and logging to TensorBoard."""
        for i in itertools.count():
            terms, last_terms = self.inner_loop(data_collector.trajectory_store(), **kwargs)
            self.print_outer_loop(i, terms, last_terms)
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Training/Loss', last_terms['loss'], i)
            self.writer.add_scalar('Training/ClipFraction', last_terms['clip_frac'], i)
            self.writer.add_scalar('Training/Entropy', last_terms['entropy'], i)
            self.writer.add_scalar('Training/ExplainedVariance', last_terms['explained'], i)
            
            # Log floor progress if available
            if hasattr(data_collector, 'floor_info') and data_collector.floor_info:
                self.max_floor_reached = max(self.max_floor_reached, data_collector.floor_info.get('max_floor', 0))
                if data_collector.floor_info.get('new_floors', 0) > 0:
                    self.floors_completed.append(data_collector.floor_info['new_floors'])
                    self.episode_count += data_collector.floor_info.get('episodes', 0)
                
                self.writer.add_scalar('Environment/MaxFloor', self.max_floor_reached, i)
                
                if self.floors_completed:
                    avg_floors = sum(self.floors_completed) / len(self.floors_completed)
                    self.writer.add_scalar('Environment/AvgFloorsPerEpisode', avg_floors, i)
                    self.writer.add_scalar('Environment/EpisodeCount', self.episode_count, i)
            
            atomic_save(self.model.state_dict(), save_path)
    
    def print_outer_loop(self, i, terms, last_terms):
        """Enhanced printing with floor information."""
        print('step %d: clipped=%f entropy=%f explained=%f' %
              (i, last_terms['clip_frac'], terms['entropy'], terms['explained']))
        
        if hasattr(self, 'max_floor_reached') and self.floors_completed:
            avg_floors = sum(self.floors_completed) / len(self.floors_completed)
            print('  max floor: %d, avg floors per episode: %.2f, episodes: %d' % 
                  (self.max_floor_reached, avg_floors, self.episode_count))