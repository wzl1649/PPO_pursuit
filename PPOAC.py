import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Categorical, Normal
from tqdm import tqdm
from pettingzoo.sisl import pursuit_v4

class ActorCritic(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()
        #TODO Edit the network structure
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2,64),
            nn.ReLU()
        )
        self.actor_net = nn.Sequential(
            nn.Linear(64,num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic_net = nn.Linear(64,1)

    def forward(self,x):
        if len(x.shape)==3:
            # transform shape[7,7,3] to [1,3,7,7] to fit the covl2d
            x =torch.tensor(x, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        else:
            x =torch.tensor(x, dtype=torch.float).permute(0,3,1,2)
        hidden = self.network(x)
        #return both actor and critic value
        return self.actor_net(hidden), self.critic_net(hidden)

class PPO():
    def __init__(self, lr, gamma, clip_ratio):
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
    def update(self, rollouts):
        obs, act, rew, logp_old= rollouts[:4]
        #Transform list to array
        obs = np.array(obs)

        #The discounted rewawrd to approximate the Q function at each given state and action
        returns = np.zeros_like(rew)
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t+1]

        #Cuculate advantage function

        #The value function given by critic network of each past observations
        values = self.actor_critic(torch.tensor(obs).float())[1].detach().numpy()
        #The advantage function defined to be the difference of Q and V
        adv = returns - np.sum(values, axis=1)#(n,) -(n,1)


        #Actor Loss
        act = torch.tensor(act).long()

        #The probability of choosing an act given obs by old network
        #The "Old" means the prob from which the act samples from
        #Which means the last batch size of the ratio will be identical to 1
        logp_old = torch.tensor(logp_old).float()

        #The probability of choosing an act given obs by current network
        pi_old = self.actor_critic(obs)[0].gather(1, act.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(torch.log(pi_old) - logp_old)
        surr1 = ratio * torch.from_numpy(adv).float()
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.from_numpy(adv).float()
        #Take the minus since min(surr1,surr2) is we want to maximize
        actor_loss = -torch.min(surr1, surr2).mean()

        #Critic loss
        #The current critic value of each obs state
        val = self.actor_critic(torch.tensor(obs).float())[1]
        #minmize the mle of empirical value and current predicted values

        #TODO There may be bugs here
        #TODO torch.tensor(returns) + torch.roll(val,-1)
        critic_loss = nn.MSELoss()(val.squeeze(-1), torch.tensor(returns).float()+torch.roll(val.squeeze(-1),-1))

        #Back Prop
        #TODO We may change the coefficients of the combination
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(env, epochs, steps_per_epoch, batch_size, lr,gamma, clip_ratio):
    ppo = PPO(lr=lr,gamma=gamma,clip_ratio=clip_ratio)
    ep_reward = deque(maxlen=10)
    for epoch in range(epochs):
        obs_buf,act_buf,rew_buf,logp_buf = [],[],[],[]
        for _ in tqdm(range(steps_per_epoch)):
            env.reset(seed=42)
            obs,_,_,_,_ = env.last()
            #One step means one round, horizon =steps_per_epoch
            #Remain the last ten rounds
            ep_reward.append(0)
            for t in range(batch_size):
                #In the loop, the agent act at most batch_size times
                probs = ppo.actor_critic(obs)[0]
                m = Categorical(probs)
                act = m.sample()
                logp = m.log_prob(act)
                obs_buf.append(obs)
                act_buf.append(act)
                rew_buf.append(0)
                logp_buf.append(logp)
                env.step(act.item())
                obs,rew,done,_,_= env.last()
                ep_reward[-1]+=rew
                rew_buf[-1]+=rew
                if done:
                    break
            ppo.update((obs_buf,act_buf,rew_buf,logp_buf))
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch+1,np.mean(ep_reward)))

    torch.save({
        'epoch':epoch,
        'model_state_dict':ppo.actor_critic.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'ep_reward':ep_reward,

    },"checkpoint.pth")


if __name__ == '__main__':
    n_evaders =1
    n_pursuers = 2
    n_catch =2
    max_cycles = 200
    env = pursuit_v4.env(n_evaders = n_evaders, n_pursuers = n_pursuers,max_cycles =max_cycles,n_catch=n_catch,tag_reward=2)
    env.reset(seed=42)
    train(env,epochs=500,steps_per_epoch=max_cycles//5,batch_size=128,lr=0.01,gamma=0.99,clip_ratio=0.20)
