# A Sample Carrom Agent to get you started. The logic for parsing a state
# is built in

from thread import *
import time
import socket
import sys
import argparse
import random
import ast
import numpy as np
import math
# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument('-np', '--num-players', dest="num_players", type=int,
                    default=1,
                    help='1 Player or 2 Player')
parser.add_argument('-p', '--port', dest="port", type=int,
                    default=12121,
                    help='port')
parser.add_argument('-rs', '--random-seed', dest="rng", type=int,
                    default=0,
                    help='Random Seed')
parser.add_argument('-c', '--color', dest="color", type=str,
                    default="Black",
                    help='Legal color to pocket')
args = parser.parse_args()


host = '127.0.0.1'
port = args.port
num_players = args.num_players
random.seed(args.rng)  # Important
color = args.color



def relu(array):
    return array * (array > 0)

relu=np.vectorize(relu)


def parse(state):
    print(state)
    White_Locations = state['White_Locations']
    Black_Locations = state['Black_Locations']
    Red_Location = state['Red_Location']
    while(len(White_Locations) < 9):
        White_Locations.append((800,800))
    while(len(Black_Locations) < 9):
        Black_Locations.append((800,800))
    if len(Red_Location)==0:
        Red_Location.append((800,800))
    temp_state=[]
    temp_state.append(Red_Location)
    temp_state.append(White_Locations)
    temp_state.append(Black_Locations)
    x_test=[]

    x_test.append(map_two_to_one(temp_state[0][0][0],temp_state[0][0][1]))
    for i in range(1,10):
        if(map_two_to_one(temp_state[1][i-1][0],temp_state[1][i-1][1]) <=640800.0):
            x_test.append(map_two_to_one(temp_state[1][i-1][0],temp_state[1][i-1][1]))
        else:
            x_test.append(640800.0)

    for i in range(10,19):
        if(map_two_to_one(temp_state[2][i-10][0],temp_state[2][i-10][1])<640800.0):
            x_test.append(map_two_to_one(temp_state[2][i-10][0],temp_state[2][i-10][1]))
        else:
            x_test.append(640800.0)

    x_test=np.matrix(x_test)
    x_test=normalize(x_test)
    new_state=x_test
    
    return new_state;

# return 0 if states are different 
def compare_states(state1,state2):
    flag=1
    
    for i in range(0,19):
        if state1[0,i]!=state2[0,i] :
            flag = flag *0
            break
    return flag;


def map_two_to_one(x,y):
    t=x*800 +y
    return t


def normalize(array):
    return (array/640800.0);

def sigmoid(value):
    return (1.0)/(1.0+math.exp(value*-1.0));

def terminal(curr_state):
    #print(curr_state[0][0][1])
    for i in range(19):
        if(curr_state[0,i]!=1):
            return 0;
    return 1

sigmoid=np.vectorize(sigmoid)

theta=[([[-0.12193246,  0.08013936, -0.03308708],
        [-1.77647961,  0.41888133,  0.16590416],
        [ 0.30998893,  0.15809814, -0.42502962],
        [-0.00438511, -0.19455479, -0.17821374],
        [ 0.13825903, -0.09885452,  0.15682647],
        [-0.27653693,  0.35694126, -0.19741868],
        [-0.17873267,  0.41047164, -0.3158905 ],
        [ 0.0307289 ,  0.28492194,  0.30546157],
        [-0.48633847, -0.24285433,  0.22725044],
        [-0.14870554,  0.30272861,  0.1865813 ],
        [-0.30293244, -0.47043567, -0.40087159],
        [ 0.69206972, -0.83313709, -0.86565066],
        [ 0.42977051,  0.36024943, -0.464808  ],
        [ 0.58423404, -0.44441677, -0.01168581],
        [-0.07925532, -0.36901296,  0.09792349],
        [-0.07717392,  0.07779756, -0.00504603],
        [ 0.21766105, -0.29514865,  0.18926293],
        [-0.16377835,  0.17437244,  0.32055984],
        [-0.03813413,  0.04769719,  0.44059425]]), ([[ -3.02789841e-01,  -1.18477970e-02,   2.55658911e-01,
          -3.75902115e-01,   2.30908284e+00],
        [  1.50164802e+00,   7.69093210e-01,  -1.48773135e-01,
           2.12060779e-01,  -6.41450848e-01],
        [  5.25765058e-01,   7.76603892e-01,  -2.00591543e-03,
          -7.31212084e-01,  -1.05369960e+00]]), ([[-0.96875394,  0.12022671, -1.11776833,  0.37469915],
        [-0.32253494,  0.54477944,  0.1425303 ,  0.70360183],
        [-0.44804012, -0.6505594 , -0.19930048, -0.488737  ],
        [ 0.63899034,  0.29401209, -0.72886456,  0.2151764 ],
        [ 1.5929987 , -0.04245807, -1.2822212 , -1.00044011]]), ([[-0.37651907,  0.24132435,  0.4507306 , -2.46696843, -0.38555237,
         -0.28475254,  0.15722587, -1.58438923,  0.10890268,  0.07493319,
          0.15865568, -0.79030586,  0.33507529,  0.04765446,  0.00635586,
         -0.55451957],
        [-0.50004725,  0.60369518,  0.8508224 ,  0.75552844, -0.06056064,
          0.2620329 , -0.25162114,  0.82051573,  0.40657632, -0.83522682,
          0.1081564 , -0.95466547,  0.14418082, -0.18727758,  0.04198036,
          0.07326011],
        [ 0.04198181,  0.45868611,  0.54743206,  0.63400017,  0.43350842,
          0.55546273,  0.34569478,  0.61504838,  0.14789921, -0.31982642,
          0.03184864,  1.62040422, -0.37124795, -0.4828266 , -0.39049208,
          0.97891972],
        [-0.27911637, -0.24317825, -0.15486053, -0.24724935, -0.37658176,
         -0.45378295,  0.34463891,  1.52542388,  0.01227173,  0.70384393,
          0.2436553 ,  0.48509243,  0.36922902,  0.240027  ,  0.05770346,
         -0.27925045]])]


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.connect((host, port))


# Given a message from the server, parses it and returns state and action
classes=16
features=19
hidden_layers=3
hidden_units=[features,3,5,4,classes]


b=[]
for i in range(0,hidden_layers+1):
    #theta.append(np.random.randn(hidden_units[i],hidden_units[i+1]) / np.sqrt(hidden_units[i]))
    b.append(np.zeros((1,hidden_units[i+1])))



INITIAL_STATE = {'White_Locations': [(400, 368), (437, 420), (372, 428), (337, 367), (402, 332),
                                     (463, 367), (470, 437), (405, 474), (340, 443)],
                 'Red_Location': [(400, 403)],
                 'Score': 0,
                 'Black_Locations': [(433, 385), (405, 437), (365, 390), (370, 350), (432, 350),
                                     (467, 402), (437, 455), (370, 465), (335, 406)]}
pockets=[(44.1, 44.1), (755.9, 44.1), (755.9, 755.9), (44.1, 755.9)]
count=1
curr_score=0
cr=15.01
def is_free(coin,pocket,state):
    return true
def line(a,b):
    if(a[0]==b[0]):
        return ("INF",b[0])
    m=(b[1]-a[1])/(b[0]-a[0])
    c=b[1]-m*b[0]
    return (m,c)


def parse_state_message(msg):
    s = msg.split(";REWARD")
    s[0] = s[0].replace("Vec2d", "")
    reward = float(s[1])
    state = ast.literal_eval(s[0])
    return state, reward

prev_state=0
def agent_1player(state):

    if(state):
        f=open("epispode.txt","a")
        global count
        global curr_score
        global prev_state
        prev_queen=0.5
        flag = 1
        # print state
        try:
            state, reward = parse_state_message(state)  # Get the state and reward
        except:
            pass

        # Assignment 4: your agent's logic should be coded here
        a = str(random.random()) + ',' + \
            str(random.randrange(-45, 225)) + ',' + str(random.random())
        if(state==INITIAL_STATE):
            a=str(1)+','+str(135)+','+str(0.5)
            current_state=parse(state)
            #f.write(str(current_state)+'\n')
            prev_state=INITIAL_STATE
      
        else:

            if(count!=1):
                curr=parse(prev_state)
                prev_queen=curr[0,0]
            
            current_state=parse(state)
         
            curr_queen=current_state[0,0]
            new_score=state["Score"]
            tmp=(new_score-curr_score)*10
            if(prev_queen==1.0 and curr_queen==0.5):
                tmp=-20
            if(prev_queen==1.0 and curr_queen==1.0):
                tmp=30
            if(prev_queen!=1.0 and curr_queen==1.0):
                tmp=20
            if(state==prev_state):
                tmp=-30

            if(count!=1):
                f.write(str(tmp)+'\n')
            curr_score=new_score
            for k in range(0,19):
                f.write(str(current_state[0,k])+'\n')
            f.write('\n')
            y_pred=[]
            '''
            a=[]
            a.append(current_state.T)
            #print(current_state)
            for j in range(0,hidden_layers+1):
                a.append(sigmoid(theta[j]*(a[j])))
                if(j!=hidden_layers):
                    a[j+1]=np.vstack([[1],a[j+1]])

            y_pred.append(a[hidden_layers+1])           
            y_pred=np.array(y_pred[0])
            '''
            a=[]
            z=[]
          

            a.append(current_state)
            #print(current_state)
           
            for j in range(0,hidden_layers+1):
                z.append(a[j].dot(theta[j])+b[j])       
                a.append(np.tanh(z[j]))

            y_pred=(a[hidden_layers+1]+1)/2
            y_pred=np.array(y_pred[0])
            



            #Calculating q values
            q=[]
            for i in range(3,16,4):
                q.append(y_pred[0,i])


            q=np.array(q)
            
            for it in range(0,16):
                f.write(str(y_pred[0,it])+'\n')
            f.write('\n')


            argmax_q=np.argmax(q)
            #print(argmax_q)

            optimum_pos=y_pred[0,argmax_q*4]
            optimum_angle=y_pred[0,argmax_q*4+1]
            optimum_force=y_pred[0,argmax_q*4+2]

            optimum_angle_final=(optimum_angle*270-45)


            action=[]
            action.append(optimum_pos)
            action.append(optimum_angle_final)
            action.append(optimum_force)
            a=str(action[0])+','+str(action[1])+','+str(action[2])
            if count%6==0:
                a=str(0.5)+','+str(135+random.randrange(-20,20))+','+str(1.0)
            prev_state=state
            count=count+1




        

        f.close()

        try:
            s.send(a)
          
        except Exception as e:
            print "Error in sending:",  a, " : ", e
            print "Closing connection"
            flag = 0
    else:
        flag=0
   
    return flag


def agent_2player(state, color):

    flag = 1

    # Can be ignored for now
    a = str(random.random()) + ',' + \
        str(random.randrange(-45, 225)) + ',' + str(random.random())

    try:
        s.send(a)
    except Exception as e:
        print "Error in sending:",  a, " : ", e
        print "Closing connection"
        flag = 0

    return flag


while 1:
    state = s.recv(1024)  # Receive state from server
    print(state)
    if num_players == 1:
        if agent_1player(state) == 0:
            break
    elif num_players == 2:
        if agent_2player(state, color) == 0:
            break
s.close()


