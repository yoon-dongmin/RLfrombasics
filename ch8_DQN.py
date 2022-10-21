import gym
import collections #deque를 이용하여 리플에이 버퍼 구현
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


#5만개의 데이터를 가지고 있다가 필요할 때마다 batch_size만큼의 데이터를 뽑아서 제공
#하나의 데이터는 (s,a,r,s_prime,done_mask)로 구성되어 있습니다.
#done_mask는 종료 상태의 밸류를 마스킹해주기 위해 만든 변수 종료:0,아님:1
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    #데이터를 버퍼에 넣어줌
    def put(self, transition):
        self.buffer.append(transition)
    
    #버퍼에서 랜덤하게 32개의 데이터를 뽑아서 미니 배치를 구성해주는 함수
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) #n=32
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        # tensor로 변환
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) #카트폴의 액션이 2개

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    #실제로 행할 액션을 정해주는 역할
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1) #랜덤하게
        else : 
            return out.argmax().item() #밸류가 큰 액션


            
def train(q, q_target, memory, optimizer): #에피소드가 끝날때마다 호출
    for i in range(10): #10개의 미니 배치를 뽑아 총 10번 업데이트 #총 320개 데이터 사용
        s,a,r,s_prime,done_mask = memory.sample(batch_size) #32개의 데이터를 가져옴

        q_out = q(s) #q네트워크에 대해서
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) #정답지를 계산할 때 쓰이는 네트워크
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict()) #q네트워크 파라미터 값들을 그대로 q_target 네트워크로 복사
    memory = ReplayBuffer() 

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) #q_target은 학습의 대상이 아니기 때문에 optimizer에게 넘겨주지 않음

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1% 0.08 -> 0.01
        s = env.reset()
        done = False
        i = 0
        while not done:
            #print(s,123123)
            if i == 0:
                s = s[0]
            #print(torch.from_numpy(s[0]).float(),5555)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)  #tensor로 바꾸어서 값 넣어줌
            #a = q.sample_action([0.04267543 -0.21853022 -0.02136922  0.2739788 ], epsilon)  
            #print(a,333)
            s_prime, r, done, info, c = env.step(a)
            #c = env.step(a)
            #print(c,555)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask)) #보상 scale 조절
            s = s_prime
            i += 1
            score += r
            if done:
                break
            
        if memory.size()>2000: #2000개 이상 쌓였을 때 학습 진행
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0: #10개의 에피소드가 끝날 때마다 가장 최근 10개 보상 총합의 평균을 출력
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()