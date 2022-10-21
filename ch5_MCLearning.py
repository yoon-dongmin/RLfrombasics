import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    #액션을 받아서 상태 변이를 일으키고 보상을 정해주는 함수
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        # list로 원소를 접근하기 위해 다음과 같이 설정
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1 # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    # y축 +1
    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
    # y축 -1
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
    # x축 -1
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
    # x측 +1
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass        

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    for k in range(50000):
        done = False
        history = []

        #에이전트가 경험 쌓음 (한번의 에피소드 진행)
        while not done:  #true가 될 때까지
            action = agent.select_action() #0,1,2,3중 하나 선택
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        cum_reward = 0
        
        for transition in history[::-1]: #경험을 이용해 테이블 업데이트(뒤에서 부터)
            x, y, reward = transition 
            data[x][y] = data[x][y] + alpha*(cum_reward-data[x][y]) #state값 update
            cum_reward = reward + gamma*cum_reward  # 책에 오타가 있어 수정하였습니다
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()
