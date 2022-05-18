# -*- coding: utf-8 -*-
"""
Created on Sun May  8 09:04:11 2022

@author: htand_000
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:47:36 2022

@author: htand_000
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




# Create a Peg Solitaire Table

def create_a_table(a=2,b=0):
    r1 = np.append(np.append(np.repeat(a,2),np.repeat(1,3)),np.repeat(a,2))
    r3 = np.tile(r1,(2,1))
    r6 = np.ones((3,7),dtype=int)
    r6 = np.vstack((r3,r6))
    r  = np.vstack((r6,r3))
    r[3,3] = b
    return r


    
def create_scatter(table,c1 = "blue",c2 = "red"):
    fig,ax = plt.subplots(figsize = (5,5))
    x = np.where(table==1)[1]
    y = -np.where(table==1)[0] 

    ax.scatter(x, y,color = c1,s=60,marker ="o")
    a = np.where(table==0)[1]
    b = -np.where(table==0)[0]
   
    ax.scatter(a, b,color = c2,s=60,marker ="o")
    plt.show()
    
# Remove motions ending outside of the table i.e. less than 0 or greater than 6  

def remove_lt_z(result):
    to_remove = np.array(-100,dtype=int)
    if np.any(result < 0) :
        ind_0 = np.argwhere(result < 0)
        
        for i in ind_0:
            to_remove = np.hstack((to_remove,np.array(i[0])))
        to_remove = np.unique(to_remove)
        
    if np.any(result > 6):
        ind_6 = np.argwhere(result > 6)
    
        for j in ind_6:
            to_remove = np.hstack((to_remove,np.array(j[0])))
        to_remove = np.unique(to_remove)
        
    
    to_remove = to_remove[to_remove >= 0]
    result = np.delete(result,to_remove,0)
    return result
    
# Remove motions which are not legitimate i.e does not respect the pattern 0,1,1

def remove_unplayable(result,table):
    unplay = False
    to_remove = np.array(-100,dtype=int)
    i = 0
    for group in result:
        
        l=[]
        for coor in group:
            
            l.append(table[tuple(coor)])
        if l != [0,1,1]:
            to_remove = np.vstack((to_remove,np.array(i)))
            unplay = True
        i += 1
    if unplay:
        to_remove = to_remove[to_remove >= 0]
        result = np.delete(result,to_remove,0)
    return result


        
# Find out all possible motions for a given table 

def possible_motion(table):
    zeros = np.argwhere(table == 0)
    position = np.array([[0,0],[0,1],[0,2],[0,0],[0,-1],[0,-2],[0,0],[1,0],[2,0],[0,0],[-1,0],[-2,0]],dtype = int).reshape((4,3,2))
    result = np.empty_like(position)
    for zero in zeros:
        z = np.full((4,3,2),(np.repeat(zero,3).reshape(2,3).T),dtype =int)    
        result = np.vstack((result,np.subtract(z,position)))
    
    result = result[4:,:,:]

    
   
    result = remove_lt_z(result)
           
    
    
    result = remove_unplayable(result,table)
        
    return result

# As there is a perfect symetry the first motion choice is compulsory

def first_move():
    table = create_a_table(a=2,b=0)  
    motion = possible_motion(table)[1]
    
    return motion 

# For a given table, when you play a motion, you have a new table

def move_it(table,motion):
    new_table = np.copy(table)
    for index in motion:
        new_table[tuple(index)] ^= True
    return new_table





# Choose randomly a motion within possible motions

def random_move(motions):
    sample = np.random.randint(motions.shape[0])
    return motions[sample,:,:]  

# Let's play randomly until depleting all possible motions.
# Meanwhile create a directory to stack motions in order, last table and 
# the number of move

def simulate_move_randomly():
    table = create_a_table(a=2,b=0)  
    n=0
    motion_dir = {}
    motion = first_move()
    table = move_it(table,motion)
    motion_dir[n] = motion
    while True:
        n += 1
        motions = possible_motion(table)
        if motions.size == 0:
            motion_dir['length'] = n
            motion_dir['table'] = table
            return motion_dir
            
        motion = random_move(motions)
        table = move_it(table,motion)
        motion_dir[n] = motion

# Play 500 times and store all motion directories in a learning list

def learn(episode= 500):
    learning_list = []

    for play in range(episode):
        motion_dir = simulate_move_randomly()
        learning_list.append(motion_dir)
    #length_sorted_learning_list = sorted(learning_list,key=itemgetter('length'))
    
    return learning_list

# Ones a learning list is built, statsiticaly investigate every position cumulative failure at the end of each play

def learning_result(learning_list):
    ones = np.zeros((7,7))
    for dic in learning_list:
        ones += dic['table'] 
    ones[ones == 2*len(learning_list)] = 0
    
    return ones

# As the aim of the game is to eliminate as many pawn as possible; 
# now we can calculate a score for every possible motion.
# Eliminated pawn positions' score are positive, the residual pawn position score is negative

def score(motion,ones):
    s = ones[tuple(motion[1])] + ones[tuple(motion[2])] - ones[tuple(motion[0])]
    return s

# For any instance of the game we can calculate best score motion.

def best_score_motion(motions,ones):
    scores = []
    for motion in motions:
        s = score(motion,ones)
        scores.append(s)
    ind = np.argmax(np.array(scores))
    return motions[ind]

# Now let's look at the axes in rows and columns, 
# in order to reveal if all axes scores are close to each other.
# To do that for every direction of action, for exemple row wise from left to right,
# after normalizing each raw I calculate average normalized score and
# I select axes giving scores higher than the average to classify as an 
# underlying axe. To be greateer than the average on one direction is granted.

# First create a helper procedure
def normalise(a):
    norm = np.zeros((1,7))
    for item in a:
        norm = np.vstack((norm,item/min(item)))
    
    return norm[1:,:]

def weak_link(ones):
    z_x_left = []
    z_y_down = []
    z_x_right = []
    z_y_up = []
    
    # row wise average score from left to right
    for x in range(7):
        zx = 0
        n = 0
        for y in range(5):
            if ones[x,y] > 0 and ones[x,y + 2] > 0:
                z = ones[x,y] + ones[x,y + 1] - ones[x,y + 2]
                zx += z
                n += 1
        z_x_left.append(zx/n)
        
    # row wise average score from right to left
    for x in range(7):
        zx = 0
        n = 0
        for y in range(5):
            if ones[x,y] > 0 and ones[x,y + 2] > 0:
                z = -ones[x,y] + ones[x,y + 1] + ones[x,y + 2]
                zx += z
                n += 1
        z_x_right.append(zx/n)
        
    # column wise average score from top to down
    for y in range(7):
        zy = 0
        n = 0
        for x in range(5):
            if ones[x,y] > 0 and ones[x + 2,y] > 0:
                z = ones[x,y] + ones[x + 1,y] - ones[x + 2,y]
                zy += z
                n += 1
        z_y_down.append(zy/n)
        
    # column wise average score from down to up
    for y in range(7):
        zy = 0
        n = 0
        for x in range(5):
            if ones[x,y] > 0 and ones[x + 2,y] > 0:
                z = - ones[x,y] + ones[x + 1,y] + ones[x + 2,y]
                zy += z
                n += 1
        z_y_up.append(zy/n)
    result = normalise((z_x_left,z_x_right,z_y_down,z_y_up))
    
    average = np.mean(result,axis = 1)
    row_wise = np.zeros((1,7),dtype = bool)
    for i in range(2):
        comp = result[i,:] > average[i]
        row_wise = np.vstack((row_wise,comp))
    row_wise = row_wise[1:,:]
    
    underlying_axe_row_wise = np.argwhere(np.any(row_wise,axis=0) > 0).reshape((4,))
    
    column_wise = np.zeros((1,7),dtype = bool)
    for i in range(2,4):
        comp = result[i,:] > average[i]
        column_wise = np.vstack((column_wise,comp))
    column_wise = column_wise[1:,:]
    
    underlying_axe_column_wise = np.argwhere(np.any(column_wise,axis=0) > 0).reshape((4,))
    

    return underlying_axe_row_wise,underlying_axe_column_wise

# When I run weak_link procedure, I end up in all direction, right to left,
# left to right, up to down, down to up row wise underlying axes are 0,2,4,6 
# and column wise underlying axes are 0,2,4,6. 


# Now let's define a procedure to come out with underlying axes motions

def possible_motion_with_underlying_axes(table,underlying_axe_row_wise,underlying_axe_column_wise ):
    authorized_motions = np.zeros((1,6),dtype = int)
    pm = possible_motion(table)
    #axes = [0,2,4,6]
    for m in pm:
        mf = m.reshape(1,6)
        for axe in underlying_axe_row_wise: #was axe
            if np.all(mf[0,0:6:2] == axe): #or np.all(mf[0,1:6:2] == axe):
                authorized_motions = np.vstack((authorized_motions,mf))
                break
        for axe in underlying_axe_column_wise: #was axe
            if np.all(mf[0,1:6:2] == axe):
                authorized_motions = np.vstack((authorized_motions,mf))
                break
                
    r = authorized_motions.shape[0]
    authorized_motions = authorized_motions.reshape(r,3,2)[1:,:]
    return authorized_motions


def simulate_move_randomly_learned(ones,prob,underlying_axe_row_wise,underlying_axe_column_wise ):
    table = create_a_table(a=2,b=0)  
    n=0
    motion_dir = {}
    motion = first_move()
    table = move_it(table,motion)
    motion_dir[n] = motion

   
    while True:
        
        orienter = np.random.choice(2,p=[prob,1 - prob])
        n += 1
        motions = possible_motion(table)
        motions_underlying = possible_motion_with_underlying_axes(table,underlying_axe_row_wise,underlying_axe_column_wise )
        
        if motions_underlying.size != 0:
            
            if orienter: 
                motion = random_move(motions_underlying)
            else:
                motion = best_score_motion(motions_underlying,ones)

        else:
            
            if motions.size == 0:
                motion_dir['length'] = n
                motion_dir['table'] = table
                return motion_dir
                
            if n == 30:
                for m in motions:
                    t = move_it(table,m)
                    if t[3,3] == 1:
                        motion = m
                        break
            elif orienter:
                motion = random_move(motions)
            else:
                motion = best_score_motion(motions,ones)



        
        table = move_it(table,motion)
        motion_dir[n] = motion

def play_until_success(ones,prob,underlying_axe_row_wise,underlying_axe_column_wise ):
    n = 0
    while True:
        
        motion_dir = simulate_move_randomly_learned(ones,prob,underlying_axe_row_wise,underlying_axe_column_wise )
        if motion_dir['length'] == 31:
            return n,motion_dir
        else:
            n += 1
            if n % 200 == 0:
                print('n = ',n)

# We can now focus on probability parameter
# up to 10 success with various probability gives me a lot of information
# about optiimized probability. The graph is saved as 'probability_optimisation.png'
    
def optimize_prob(episode,ones,underlying_axe_row_wise,underlying_axe_column_wise):
    success = []
    prob_success_dir = {}
    p = np.linspace(0.40,0.80,9)       
    for prob in p:
        print('prob = ',prob)
        sum_of_n = 0
        prob_success_dir[prob] = []
        for i in range(episode):
            print('episode = ',i)
            n,motion_dir = play_until_success(ones,prob,underlying_axe_row_wise,underlying_axe_column_wise)
            prob_success_dir[prob].append(n)
            sum_of_n += n
        success.append(sum_of_n/episode)   
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.scatter(p,success)
    plt.xlabel('random play probability')
    plt.ylabel('mean to succeed over ' + str(episode) + ' plays') 
    plt.title('Probability Optimization')
    plt.show() 
    fig.savefig('probability_optimisation.png')
    return success,prob_success_dir

# Boxplot graphs gives much more information
# Prob_succes_dir is also saved with the procedure below

def optimazation_boxplot(prob_success_dir):
    import pandas as pd
    df = pd.DataFrame(prob_success_dir)
    column_indices = [4,6]
    new_names = [0.60,0.70]
    old_names = df.columns[column_indices]
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    df.to_csv('prob_success_dir.csv',index=False)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.boxplot(df,positions=list(df.columns),widths=0.02)
    plt.xlim(0.35,0.85)
    plt.xlabel('random play probability')
    plt.ylabel('number of play to succeed over 30 plays') 
    plt.title('Probability Optimization')
    plt.show()
    fig.savefig('probability_optimisation_boxplot.png')
    

def confirm_the_probability(ones,prob,episode,underlying_axe_row_wise,underlying_axe_column_wise):
    filename = 'confirmation_over_'+str(episode)+'_plays.png'
    success = []
    for i in range(episode):
        n,motion_dir = play_until_success(ones,prob,underlying_axe_row_wise,underlying_axe_column_wise)
        
        success.append(n)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.hist(success,bins = 10)
    plt.xlabel('number of play until success')
    plt.ylabel('total number of success')
    plt.axvline(x=sum(success)/episode,color='k', linestyle='--')
    plt.title('Histogram of imposed play with p = '+ str(prob)+' over '+str(episode)+' plays')
    plt.show() 
    fig.savefig(filename)
    return success



def create_scatter_line(table,motion_dir,c1 = "blue",c2 = "red"):
    fig,ax = plt.subplots(figsize = (5,5))
    x = np.where(table==1)[1]
    y = -np.where(table==1)[0] 

    ax.scatter(x, y,color = c1,s=60,marker ="o")
    a = np.where(table==0)[1]
    b = -np.where(table==0)[0]

    ax.scatter(a, b,color = c2,s=60,marker ="o")
    for i in range(len(motion_dir)):
        ax.plot([motion_dir[i][0][1],motion_dir[i][2][1]],[-motion_dir[i][0][0],-motion_dir[i][2][0]],linewidth=1)
    plt.show()
    


def plot_a_succesfull_game(game,c1 = "blue",c2 = "red"):
    table = create_a_table()
    for i in range(game['length']):
        fig,ax = plt.subplots(figsize = (5,5))
        x = np.where(table==1)[1]
        y = -np.where(table==1)[0] 

        ax.scatter(x, y,color = c1,s=60,marker ="o")
        a = np.where(table==0)[1]
        b = -np.where(table==0)[0]

        ax.scatter(a, b,color = c2,s=60,marker ="o")
        
        ax.plot([game[i][0][1],game[i][2][1]],[-game[i][0][0],-game[i][2][0]],linewidth=1)
        plt.show()
        table = move_it(table, game[i])
    fig,ax = plt.subplots(figsize = (5,5))
    x = np.where(table==1)[1]
    y = -np.where(table==1)[0] 

    ax.scatter(x, y,color = c1,s=60,marker ="o")
    a = np.where(table==0)[1]
    b = -np.where(table==0)[0]
    ax.scatter(a, b,color = c2,s=60,marker ="o")  
    plt.show()
#`np.save('ones.npy',ones)

with open('ones.npy', 'rb') as f:
    ones = np.load(f)
    
    


def create_a_gif(motion_dir):
    game = motion_dir
    import matplotlib.animation 
    global x_1,x_2,y_1,y_2
    x_1 = game[0][0][1]
    x_2 = game[0][2][1]
    y_1 = -game[0][0][0]
    y_2 = -game[0][2][0]
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(300*px, 300*px))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True,linewidth=1,linestyle='--',color = "k")
    table = create_a_table()
    x = np.where(table==1)[1]
    y = -np.where(table==1)[0] 
    c1 = "blue"   
    

    scat = ax.scatter(x, y,color = c1,s=150,marker ="o")
    ax.plot([x_1,x_2],[y_1,y_2],linewidth=2,linestyle=':',color = "r")

    tables = []

    tables.append(table)
    for i in range(31):
        table = move_it(table, game[i])
        tables.append(table)

    def update(frame_number):
        global x_1,x_2,y_1,y_2
        if frame_number <= game['length']:

            ax.plot([x_1,x_2],[y_1,y_2],linewidth=2,color = "w")
            ax.plot([x_1,x_2],[y_1,y_2],linewidth=1,linestyle='--',color = "k")
            table = tables[frame_number]
            x = np.where(table==1)[1]
            y = -np.where(table==1)[0]

            if frame_number <= 30:
                x_1 = game[frame_number][0][1]
                x_2 = game[frame_number][2][1]
                y_1 = -game[frame_number][0][0]
                y_2 = -game[frame_number][2][0]
                ax.plot([x_1,x_2],[y_1,y_2],linewidth = 2,linestyle=':',color = "r")
            scat.set_offsets(np.array((x,y)).T)
    ani = matplotlib.animation.FuncAnimation(fig,update,interval=500)
    ani.save("play_game.gif", writer='imagemagick',fps=1)
    
    
def density_hist_comparison_with_gamma_disribution(success,c1,c2,a):
    filename ='Density_histogram.png'
    data = np.array(success)
    avg = data.mean()
    gamma_data = np.random.gamma(1,avg,1000)
    fig,ax = plt.subplots()
    ax.hist(data,density = True,bins = 10,color = c1,alpha=a)
    ax.hist(gamma_data,density = True,bins=10,color=c2,alpha = a)
    ax.title.set_text('Density_histogram comparison with a gamma RV')
    ax.legend(['Real_data','Gamma_data'])
    plt.show()
    fig.savefig(filename)
    
