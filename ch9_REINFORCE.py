import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0 #여기에 보상의 합 저장
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]: 
            R = r + gamma * R
            loss = -torch.log(prob) * R #gradient accent
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample() #확률이 높은 액션은 더 자주, 낮은 액션은 덜 뽑히게 함
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a])) #action-value값과 reward를 저장
            s = s_prime
            score += r
            
        pi.train_net() #하나의 에피소드 동안 모은 데이터를 이용해 실제 업데이트
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()