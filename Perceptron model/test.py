#感知机对偶形式
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

data_set=np.array([[3,3,1],[4,3,1],[1,1,-1]])
# data_set=np.array([[3,3,1],[4,3,1],[1,1,-1],[1,2,1],[5,4,1],[4,1,-1],[6,2,-1],[0,3,1]])

datax=data_set[:,0:2]   #x数据集
datay=data_set[:,2]#标记-1/+1
print("datay shape", np.shape(datay))
lenth=data_set.shape[0] #数据集长度

#构建Gram矩阵
Gram=np.zeros([lenth,lenth])
for i in range(lenth):
    for j in range(lenth):
        Gram[i][j]=np.dot(datax[i],datax[j])
print('输出格拉姆矩阵：\n',Gram)


#建立alpha系数矩阵
alpha=np.zeros([1,data_set.shape[0]])[0]
print("alpha shape", np.shape(alpha))
b=0

#添加列表存储系数
history=[]

def ganzhiji2():
    global alpha,b
    w1,w2=np.dot(alpha*datay,datax)
    history.append([[w1,w2],b])

    for i in range(data_set.shape[0]):
        print("alpha*datay", alpha*datay, np.shape(alpha), np.shape(datay), type(alpha), type(datay))
        if((np.dot(alpha*datay,Gram[:,i])+b)*datay[i]<=0):
            alpha[i]=alpha[i]+1
            b=b+datay[i]
            print(alpha[i], b)
            ganzhiji2()
    return alpha,b

if __name__=='__main__':
    s=ganzhiji2()
    print(history)

    # fig=plt.figure()
    # ax=plt.axes(xlim=(-6, 6), ylim=(-6, 6))
    # line,=ax.plot([],[],'g',lw=2)
    # label=ax.text([],[],'')

    # #显示测试数据的开始框架
    # def show():
    #
    #     x, y, x_, y_= [],[],[],[]
    #     for i in range(data_set.shape[0]):
    #         if(data_set[i,2]>0):
    #             x.append(data_set[i,0])
    #             y.append(data_set[i,1])
    #         else:
    #             x_.append(data_set[i,0])
    #             y_.append(data_set[i,1])
    #     #对中文显示进行支持设置中文字体
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #     plt.plot(x,y,'bo',x_,y_,'rx')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title('感知机模型')
    #     return line,label
    #
    #
    # #历史参数的动态显示
    # def animate(i):
    #     w=history[i][0]
    #     b=history[i][1]
    #     if(w[1]==0):return line,label
    #     x1=-6
    #     y1=-(b+w[0]*x1)/w[1]
    #     x2=6
    #     y2=-(b+w[0]*x2)/w[1]
    #     line.set_data([x1,x2],[y1,y2])
    #     x0=0
    #     y0=-(b+w[0]*x0)/w[1]
    #     str1=[w[0],w[1]],b
    #     label.set_text(str1)
    #     label.set_position([x0,y0])
    #     return line,label
    # print('hi')
    # for i in history:
    #     print(i)
    # #ImageMagick-7.0.1-Q16解码器需要提前装入才能生成动态视频
    # plt.rcParams["animation.convert_path"]="C:\ImageMagick-7.0.1-Q16\magick.exe"
    # anim=animation.FuncAnimation(fig,animate,init_func=show,frames=len(history),interval=500, repeat=True,blit=True)
    # plt.show()
    # anim.save('对偶型.gif', fps=2, writer='imagemagick')