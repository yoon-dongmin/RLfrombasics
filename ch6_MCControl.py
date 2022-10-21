import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1  # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_left(self):
        if self.y==0:
            pass
        elif self.y==3 and self.x in [0,1,2]: #벽이 있는 경우
            pass
        elif self.y==5 and self.x in [2,3,4]: #벽이 있는 경우
            pass
        else:
            self.y -= 1

    def move_right(self):
        if self.y==1 and self.x in [0,1,2]: #벽이 있는 경우
            pass
        elif self.y==3 and self.x in [2,3,4]: #벽이 있는 경우
            pass
        elif self.y==6:
            pass
        else:
            self.y += 1
      
    def move_up(self):
        if self.x==0:
            pass
        elif self.x==3 and self.y==2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x==4:
            pass
        elif self.x==1 and self.y==4:
            pass
        else:
            self.x+=1

    def is_done(self):
        if self.x==4 and self.y==6: # 목표 지점인 (4,6)에 도달하면 끝난다
            return True
        else:
            return False
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.eps = 0.9 
        self.alpha = 0.01
        
    #상태 s를 입력으로 받아 s에서 알맞은 액션을 입실론 그리디 방식을 통해 선택
    def select_action(self, s):
        # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random()
        if coin < self.eps: #처음에 0.1
            action = random.randint(0,3)  #Random한 action 선택
        else: #처음에 0.9 점점 증가 
            action_val = self.q_table[x,y,:] #해당 state에 해당하는 action들 가져옴
            action = np.argmax(action_val) #가장 큰 action 선택 
        return action

    '''
    a = list(range(10))
    >>> a
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> a[:-1]
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> a[::-1]
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    >>> a[:-1:-1]
    []
    '''
    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]: #뒤에서부터
            s, a, r, s_prime = transition
            x,y = s
            # 몬테 카를로 방식을 이용하여 업데이트. action값도 추가적으로 입력으로 넣음
            self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (cum_reward - self.q_table[x,y,a])
            cum_reward = cum_reward + r 

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1) #0.9에서 0.01까지 줄어듬

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table.tolist() #numpy -> list
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)): #5
            row = q_lst[row_idx] #7x4 matrix
            for col_idx in range(len(row)): #7
                col = row[col_idx] #1x4 matrix
                action = np.argmax(col) #가장 큰 값(action)을 가져옴
                data[row_idx, col_idx] = action # data에 저장
        print(data)
      
def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 총 1,000 에피소드 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done: # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a) #s_prime:(x_prime,y_prime)
            history.append((s, a, r, s_prime)) # history에 값 저장
            s = s_prime #x,y값 update
        agent.update_table(history) # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps() #eps값 조절 

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()