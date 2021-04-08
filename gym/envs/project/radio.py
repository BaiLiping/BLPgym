import numpy as np
import math
import sys
import gym
import matplotlib.pyplot as plt
from gym import spaces, logger
from gym.utils import seeding


wave_length=0.07
division=40
vertical_distance=2
weight=100
omega=math.pi/(division/2)
distance_magnification=1
nA=6
phase=math.pi/(nA-1)
length=8
width=6

def generate_target():
    x_axis=np.random.choice(range(length))
    y_axis=np.random.choice(range(width))
    position=(x_axis, y_axis)
    return position
    
def compute_additive_wave(x,y,p,wave,target_location):
    distance=math.sqrt(pow((distance_magnification*(x-target_location[0])),2)+pow((distance_magnification*(y-target_location[1])),2)+pow(vertical_distance,2))
    for t in range(division):
        #impact_t=(1/(distance+vertical_distance))*math.cos(distance/wave_length*(2*math.pi)+omega*t+p*phase)
        impact_t=math.cos(distance/wave_length*(2*math.pi)+omega*t+p*phase)
        wave[t]+=impact_t*weight
    magnitute=np.max(wave)
    return wave,magnitute

class RadioEnv(gym.Env):
    def __init__(self):
        self.nA=nA
        self.x=0
        self.y=0
        self.nRadio=10
        self.state=np.zeros(2*width*length)
        self.mag=0
        self.wave=np.zeros(division)
        self.target_location=generate_target()
        self.nS=len(self.state)
        self.nA=16

        lower_bound=np.zeros(self.nRadio)
        upper_bound=np.ones(self.nRadio)*10

        self.seed()


        self.action_space=spaces.Discrete(self.nA)
        self.observation_space=spaces.Box(lower_bound,upper_bound,dtype=np.float32)

    def reset(self):
        self.target_location=generate_target()
        self.x=0
        self.y=0
        self.mag=0
        self.state=np.zeros(2*width*length)
        self.wave=np.zeros(division)
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_nA(self):
        return self.nA

    def get_nS(self):
        return self.nS

    def step(self,action):
        old_mag=self.mag
        self.wave,self.mag=compute_additive_wave(self.x,self.y,action,self.wave,self.target_location)
        index=np.ravel_multi_index((self.y,self.x),(width,length))
        self.state[index]=action
        increase=self.mag-old_mag
        self.state[index+width*length]=increase

        if self.x==7:
            self.x=0
            self.y+=1
        else:
            self.x+=1
        
        return self.state,increase

    def plot_wave(self,label):
        x=np.arange(division)
        plt.plot(x,self.wave,label=label)
        
    
    def render(self):
        outfile=sys.stdout
    
        for i in range(width*length):
            position=np.unravel_index(i,(width,length))
            output='  '
            output+=str(int(self.state[i]))
            output+='  '
    
            if position[1]==7:
                output=output.rstrip()
                output+='\n'
            outfile.write(output)
        outfile.write('\n')