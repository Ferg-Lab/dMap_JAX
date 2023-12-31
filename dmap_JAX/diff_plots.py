import numpy as np

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

showPlots = True

def plotmaps(psi, eps, plot_stride=1, select_axis_2d=[1, 2], skip_lead=True, colorMap='blue', cmap='Accent', ticks=[1, 2, 3, 4, 5, 6], colorbar=True, colorlabel='Traj', skip_3D=False, figSuffix=''):
    
    if skip_lead:
        lead = 1
    else:
        lead = 0
        
    _x = select_axis_2d[0] + lead
    _y = select_axis_2d[1] + lead
    
    
    print("chosen x axis:{0} and y axis:{1}".format(_x, _y))
        
    # plotting embedding

    #plot_stride = 100
    fig = plt.figure(figsize = (12,8))
    
    ax = fig.add_subplot(111)
    im = ax.scatter(psi[::plot_stride,_x], psi[::plot_stride,_y], c=colorMap, cmap=cmap)
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=ticks)
        #im.set_clim(model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].min(), model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].max())
        cbar.set_label(colorlabel, size=18)
    
    
    ax.set_xlabel('$\psi$$_{'+str(_x+lead)+'}$')
    ax.set_ylabel('$\psi$$_{'+str(_y+lead)+'}$')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlim([np.min(psi[:,1])*1.1,np.max(psi[:,1])*1.1])
    #plt.ylim([np.min(psi[:,2])*1.1,np.max(psi[:,2])*1.1])
    
    if showPlots:
        plt.draw()
        plt.show()
        
    fig.savefig('evecs_'+str(lead+1)+str(lead+2)+'_' + str(eps) +  '_' + figSuffix + '.png', dpi=300)
    plt.close()

    if not skip_3D:
        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(psi[::plot_stride,lead], psi[::plot_stride,lead+1], psi[::plot_stride,lead+2], c=colorMap, cmap=cmap)
        
        if colorbar:
            if ticks:
                cbar = plt.colorbar(im, ax=ax, ticks=ticks)
            else:
                cbar = plt.colorbar(im, ax=ax)
            #im.set_clim(model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].min(), model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].max())
            cbar.set_label(colorlabel, size=18)
        
        ax.set_xlabel('$\psi$$_{'+str(lead+1)+'}$', labelpad=20)
        ax.set_ylabel('$\psi$$_{'+str(lead+2)+'}$', labelpad=20)
        ax.set_zlabel('$\psi$$_{'+str(lead+3)+'}$', labelpad=30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_box_aspect(aspect=(4,4,4), zoom=0.8)
    
        # disable auto rotation
        ax.zaxis.set_tick_params(labelsize=15)
        ax.zaxis.set_rotate_label(False)
        
        if showPlots:
            plt.draw()
            plt.show()
            
        fig.tight_layout()
        fig.savefig('evecs_'+str(lead+1)+str(lead+2)+str(lead+3)+'_'+str(eps)+ '_' + figSuffix + '.png', dpi=300)
        plt.close()
    return

def plot2Dmaps(psi, eps, plot_stride=1, select_max_vecs=4, skip_lead=True, colorMap='blue', cmap='Accent', ticks=None, colorbar=True,  colorlabel='Traj', figSuffix='', cticklabels=None, offset=0, singlecolorbar=False):
    
    if skip_lead:
        lead = 1
    else:
        lead = 0
        
    nplots = int((select_max_vecs+1-lead)*(select_max_vecs+1-lead-1)*.5)
    nplots_split = int(nplots*.5)
        
    print("nplots_split:{0}".format(nplots_split))
        
    fig = plt.figure(figsize = (16,12))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.375,
                    hspace=0.375)
    
    count = 1
    for i in range(lead, select_max_vecs + 1):
        for j in range(i+1, select_max_vecs + 1):
    
            ax = fig.add_subplot(nplots_split,nplots_split,count)
        
            im = ax.scatter(psi[::plot_stride,i], psi[::plot_stride,j], c=colorMap, cmap=cmap)
        
            # Set scientific notation for the x-axis
            plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            
            # Set scientific notation for the y-axis
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            
            if colorbar:
                if ticks:
                    if singlecolorbar:
                        if i == 2 and j == 4:
                            cbar_ax = fig.add_axes([0.25, 0.29, 0.5, 0.1]) 
                            cbar_ax.set_axis_off()
                            
                            cbar = plt.colorbar(im, ax=cbar_ax, ticks=ticks, location='bottom')
                            cbar.set_ticklabels(cticklabels)
                            cbar.set_label(colorlabel, size=25)
                            #plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                            #plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    else:
                        cbar = plt.colorbar(im, ax=ax, ticks=ticks)
                        cbar.set_ticklabels(cticklabels)
                        cbar.set_label(colorlabel, size=18)
                else:
                     cbar = plt.colorbar(im, ax=ax, ticks=ticks)
                     #cbar.set_ticklabels(cticklabels)
                     cbar.set_label(colorlabel, size=18)
                    
                #im.set_clim(model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].min(), model_loss_eigen['feat_evecs_SRV_unbiased'][:,ii+1].max())
                    
            
        
            ax.set_xlabel('$\psi$$_{'+str(i+1+offset)+'}$')
            ax.set_ylabel('$\psi$$_{'+str(j+1+offset)+'}$')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

        
            count += 1
    print(count)
    if showPlots:
        plt.draw()
        plt.show()
    fig.tight_layout(pad=.5)
    #fig.savefig('evecs_'+str(lead+1)+str(lead+2)+'_' + str(eps) +  '_' + figSuffix + '.png', dpi=300)
    plt.close()

    return


def plotspect(lamb, eps, alpha, skip_lead=True, figSuffix=''):
    
    if skip_lead:
        lead = 1
    else:
        lead = 0
    
    # plotting eval spectrum
    fig = plt.figure(figsize = (8,4))
    ax = fig.add_subplot(111)
    
    #ax.scatter(np.arange(lead,len(lamb)), lamb[lead:], color='deepskyblue', s=40)
    ax.bar(np.arange(lead,len(lamb)), lamb[lead:], color='deepskyblue', label=r'log($\epsilon$)='+str(eps)+r' $\alpha$= '+str(alpha))

    ax.set_xlabel('eval idx')
    ax.set_ylabel('eval')
    
    plt.xticks(np.arange(lead, len(lamb), step=1), list(np.arange(lead+1, len(lamb)+1)), fontsize=15)
    #plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],fontsize=10)
    plt.legend(frameon=False,fontsize=12)
    
    if showPlots:
        plt.draw()
        plt.show()
    fig.tight_layout()
    #fig.savefig('evals_' +str(eps) + '_'+ figSuffix+'.png', dpi=300)
    plt.close()
    return
