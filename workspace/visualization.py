import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def save_img(x, y_mean, y_min, y_max, x_org, y_org, loop_num):
    plt.figure(figsize=(6,4))
    
    plt.plot(x, y_mean, linestyle="solid", color='b')
    plt.plot(x, y_min, linestyle="dashed", color='lightskyblue')
    plt.plot(x, y_max, linestyle="dashed", color='lightskyblue')
    plt.scatter(x_org, y_org, linestyle="solid", color='r')

    plt.xlim([np.min(x), np.max(x)])
    plt.ylim([0, 0.6])
    plt.savefig('./results/num_{}.jpg'.format(str(loop_num).zfill(3)))
    plt.close()
    
def save_gif(): 
    picList = glob.glob('./results/' + "*.jpg")
    fig = plt.figure(figsize=(6.4,4.8))  
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)

    ims = []
    for i in range(len(picList)):
        tmp = Image.open(picList[i])
        ims.append([plt.imshow(tmp)])
    
    # ファイルがないと保存時に怒られるため，予め作成
#    f = open('result.gif','a')
#    f.close()
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=3000)
    ani.save('./results/' + 'result.gif', writer='pillow')