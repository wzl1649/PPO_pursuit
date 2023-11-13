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
            x =torch.tensor(x, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) # transform shape[7,7,3] to [1,3,7,7]
        else:
            x =torch.tensor(x, dtype=torch.float).permute(0,3,1,2)
        hidden = self.network(x)

        return self.actor_net(hidden), self.critic_net(hidden)

class PPO():
    def __init__(self, lr, gamma, clip_ratio):
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
    def update(self, rollouts):
        obs, act, rew, logp_old, val_old = rollouts[:5]
        obs = np.array(obs)

        #Cumulative reward
        returns = np.zeros_like(rew)
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t+1]

        #Cuculate advantage function
        values = self.actor_critic(torch.tensor(obs).float())[1].detach().numpy()
        adv = returns - np.sum(values, axis=1)


        #Actor Loss
        act = torch.tensor(act).long()  # 将动作转换为Tensor类型
        logp_old = torch.tensor(logp_old).float()  # 将对数概率转换为Tensor类型
        pi_old = self.actor_critic(obs)[0].gather(1, act.unsqueeze(-1)).squeeze(-1)  # 得到旧策略的动作概率
        ratio = torch.exp(torch.log(pi_old) - logp_old)  # 计算比率
        surr1 = ratio * torch.from_numpy(adv).float()  # 第一项损失
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.from_numpy(adv).float()  # 第二项损失
        actor_loss = -torch.min(surr1, surr2).mean()

        #Critic loss
        val_old = torch.tensor(val_old).float()
        val = self.actor_critic(torch.tensor(obs).float())[1]
        critic_loss = nn.MSELoss()(val.squeeze(-1), torch.tensor(returns).float())

        #Back Prop
        loss = actor_loss + 0.5 * critic_loss
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
            ep_reward.append(0)
            for t in range(batch_size):
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
            ppo.update((obs_buf,act_buf,rew_buf,logp_buf,np.zeros_like(rew_buf)))
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch+1,np.mean(ep_reward)))

#if __name__ == '__main__':
   # n_evaders =2
    #n_pursuers = 1
   # n_catch =1
   # max_cycles = 200
   # env = pursuit_v4.env(n_evaders = n_evaders, n_pursuers = n_pursuers,max_cycles =max_cycles,n_catch=n_catch)
   # env.reset(seed=42)
  #  train(env,epochs=20,steps_per_epoch=max_cycles//5,batch_size=128,lr=0.02,gamma=0.99,clip_ratio=0.2)