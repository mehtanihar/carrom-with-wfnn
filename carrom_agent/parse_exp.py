import numpy as np

X=[]
Y=[]
classes=16
features=19
hidden_layers=3
hidden_units=[features,3,5,4,classes]
reg_lambda=0.1
alpha=0.00001

target = open("data_accumulating.txt", "w")


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
best_theta=theta
norm=10000

b=[]
for i in range(0,hidden_layers+1):
    #theta.append(np.random.randn(hidden_units[i],hidden_units[i+1]) / np.sqrt(hidden_units[i]))
    b.append(np.zeros((1,hidden_units[i+1])))


def parse_exp():
    count =0
    print "hello"
    global X
    global Y

    exp=[]
    c=0.9
    num_lines = sum(1 for line in open('epispode.txt'))
   
    epsilon=0.01
    f = open("epispode.txt",'r')
    
    temp4=[]
    while(count<num_lines-37):
        temp1=[]
        temp2=[]
        #state initial
        current_state=[]
        if(count==0):                    
            for i in range(0,19):
                temp2.append(float(f.readline()))
                count=count+1
            f.readline()
            count = count+1
            current_state=temp2
            temp1.append(temp2)
            
        else:
            temp1.append(temp4)
            current_state=temp4

        #ypred
        temp3=[]
        for i in range(0,16):
            temp3.append(float(f.readline()))
            count = count+1
        f.readline()
        count = count +1
        temp1.append(temp3)

        #reward 
        reward = float(f.readline())
        temp1.append(reward)
        count = count +1
        print temp1
        print(num_lines-count)
        
        #next state
        if(num_lines-count>18):
            temp4=[]
            for i in range(0,19):
                temp4.append(float(f.readline()))
                count=count+1
            temp1.append(temp4)
            temp1.append(f.readline())
            count =count +1

        exp.append(temp4)
        
        q=[]
       
        y_pred = temp3
       
        for i in range(3,16,4):
            q.append(y_pred[i])

        action=[]
        temp5=[]
        for i in range(0,16):
            
            if i%4 !=3 :
                temp5.append(y_pred[i])
            else:
                action.append(temp5)
                temp5=[]

        q=np.array(q)
        argmax_q=np.argmax(q)
        #print(argmax_q)

        optimum_pos=y_pred[argmax_q*4]
        optimum_angle=y_pred[argmax_q*4+1]
        optimum_force=y_pred[argmax_q*4+2]

        optimal_action=[]
        optimal_action.append(optimum_pos)
        optimal_action.append(optimum_angle)
        optimal_action.append(optimum_force)
        optimal_action=np.array(optimal_action)
        action=np.array(action)

        gamma =0.9
        
        Q_new=(reward/50.0)+gamma*q.max()
        num=0
        den=0
        deriv_q=[]
        deriv_u0=[]
        deriv_u1=[]
        deriv_u2=[]
        for it in range(0,4):
            print optimal_action
            print action[it]
            weight=np.linalg.norm(optimal_action-action[it])+c*(q.max()-q[it]+epsilon)
            den=den+(1.0/weight)
            num=num+(q[it]/weight)
            deriv_q.append((den*(weight+q[it]*c)-num*c)/pow((weight*den),2))
            deriv_u0.append(((num-den*q[it])*2*(action[it][0]-optimal_action[0]))/(pow(weight*den,2)))
            deriv_u1.append(((num-den*q[it])*2*(action[it][1]-optimal_action[1]))/(pow(weight*den,2)))
            deriv_u2.append(((num-den*q[it])*2*(action[it][2]-optimal_action[2]))/(pow(weight*den,2)))
        Q_dash=num/den
        error=Q_new-Q_dash
        for it in range(0,4):
            q[it]=q[it]+error*deriv_q[it]
            action[it][0]=action[it][0]+error*deriv_u0[it]
            action[it][1]=action[it][1]+error*deriv_u1[it]
            action[it][2]=action[it][2]+error*deriv_u2[it]

 
        y=[]
        for it in range(0,4):
            y.append(action[it][0]*2-1)
            y.append(action[it][0]*2-1)
            y.append(action[it][0]*2-1)
            y.append(q[it]*2-1)

        X.append(current_state)
        Y.append(y)

    X=np.matrix(X)
    Y=np.matrix(Y)
    print(X.shape)
    print(Y.shape)

        

def backprop():
    global X
    global Y
    global norm


    deriv_theta=[]
    deriv_b=[]

    for i in range(0,hidden_layers+1):
        deriv_theta.append(np.zeros((hidden_units[i],hidden_units[i+1])))
        deriv_b.append(np.zeros((1,hidden_units[i+1])))
        b[i]=np.matrix(b[i])
        theta[i]=np.matrix(theta[i])


    for i in range(0,50000):

        y_pred=[]
        
        a=[]
        z=[]

      

        a.append(X)
        
       
        for j in range(0,hidden_layers+1):
            z.append(a[j].dot(theta[j])+b[j])       
            a.append(np.tanh(z[j]))


        y_pred=a[hidden_layers+1]

        delta=[]
        for p in range(0,hidden_layers+2):
            delta.append([])
        ##priny[i].T)
        #print(y_pred[i])
        

        delta[hidden_layers+1]=y_pred-Y


        #np.multiply((y_pred[i]-y[i].T),np.multiply(a[hidden_layers+1],(1-a[hidden_layers+1])))
        #print(delta[2])
        delta[hidden_layers+1]=np.matrix(delta[hidden_layers+1])

        curr_norm=np.linalg.norm(delta[hidden_layers+1])
        print(curr_norm)
        if(curr_norm<norm):
            best_theta=theta
            norm=curr_norm

        #print(i)


        for l in range(hidden_layers,-1,-1):
            deriv_theta[l]=(a[l].T).dot(delta[l+1])+reg_lambda*theta[l]
        
            deriv_b[l]=np.sum(delta[l+1], axis=0, keepdims=True)
            if(l!=0):
                delta[l]=np.multiply(delta[l+1].dot(theta[l].T),(1-np.power(a[l],2)))
            delta[l]=np.matrix(delta[l])
            theta[l]=theta[l]-alpha*deriv_theta[l]
            b[l]=b[l]-alpha*deriv_b[l]
    target.write(str(best_theta))

       
        

expp = parse_exp()
backprop()
print expp


