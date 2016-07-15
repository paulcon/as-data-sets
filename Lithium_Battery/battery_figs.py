import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pdb


def get_coefficients(X, f):
    M, m = X.shape
    
    A = np.hstack((np.ones((M, 1)), X)) 
    u = np.linalg.lstsq(A, f)[0]
    w = u[1:].reshape((m, 1))
    return w / np.linalg.norm(w)
    
def get_coefficients_boot(X, f, nboot=100):
    M, m = X.shape
    
    a = np.zeros((m, nboot))
    for i in range(nboot):
        ind = np.random.randint(M, size=(M, ))
        X0 = X[ind,:].copy()
        f0 = f[ind,:].copy()
        a[:,i] = get_coefficients(X0, f0).reshape((m, ))
        
    se = np.std(a, axis=1)
    ind = se < 1e-15
    se[ind] = 1e-15
    return se

class BatteryData():
    
    CAPFAC025 = 7.5/36000.
    CAPFAC1 = 30./36000.
    CAPFAC4 = 120./36000.
    
    def __init__(self, filename, vfilename, cflag):
        """
        cflag: 0 is 0.25C, 1 is 1C, 2 is 4C
        """
        
        df = pn.DataFrame.from_csv(filename)
        data = df.as_matrix()
        labels = df.keys()
        
        # inputs, normalized to [-1,1]
        if cflag==2:
            self.X = np.delete(data[:,:19], 1537, 0)
        else:
            self.X = data[:,:19]
            
        self.in_labels = labels[:19]
        
        # voltage
        if cflag==2:
            self.voltage = np.delete(data[:,19:69], 1537, 0)
        else:
            self.voltage = data[:,19:69]
        
        # capacity
        if cflag==0:
            self.capacity = self.CAPFAC025*data[:,69:119]
        elif cflag==1:
            self.capacity = self.CAPFAC1*data[:,69:119]
        elif cflag==2:
            self.capacity = self.CAPFAC4*np.delete(data[:,69:119], 1537, 0)
        else:
            self.capacity = []
        
        # concentration anode
        if cflag==2:
            self.concentration_anode = np.delete(data[:,119:169], 1537, 0)
        else:
            self.concentration_anode = data[:,119:169]  
        
        # concentration separator
        if cflag==2:
            self.concentration_separator = np.delete(data[:,169:219], 1537, 0)
        else:
            self.concentration_separator = data[:,169:219]  
        
        # concentration cathode
        if cflag==2:
            self.concentration_cathode = np.delete(data[:,219:269], 1537, 0)
        else:
            self.concentration_cathode = data[:,219:269]            
        
        # time range
        self.tstar = np.linspace(1, 99, 50)[::-1]
        
        # voltage as time
        self.vtime = np.loadtxt(vfilename)
    
    
def input_labels():
    # to get tex in matplotlib: 
    # http://matplotlib.org/users/usetex.html
    
    in_labels = [r'$\epsilon_a$', 
        r'$\epsilon_s$', 
        r'$\epsilon_c$', 
        r'brugg$_a$', 
        r'brugg$_s$', 
        r'brugg$_c$', 
        r'$t_+^0$', 
        r'$D$', 
        r'$D_a$', 
        r'$D_c$', 
        r'$\sigma_a$', 
        r'$\sigma_c$', 
        r'$k_a$', 
        r'$k_c$', 
        r'$r_a$', 
        r'$r_c$', 
        r'$L_a$', 
        r'$L_s$', 
        r'$L_c$']
        
    linestyles = ['-', 
        '-',
        '-',
        '-',
        '-',
        '-',
        '-',
        ':',
        '-.',
        '-.',
        ':',
        ':',
        '-.',
        '-.',
        '--',
        '--',
        '--',
        '--',
        '--']
        
    colors = ['b',
        'g',
        'r',
        'b',
        'g',
        'r',
        'm',
        'm',
        'b',
        'r',
        'b',
        'r',
        'b',
        'r',
        'b',
        'r',
        'b',
        'g',
        'r']
        
    linewidths = [3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        1,
        1,
        3,
        3,
        3]
        
    
    anode_index = [0, 3, 16, 8, 10, 12, 14]
    separator_index = [1, 4, 17]
    cathode_index = [2, 5, 18, 9, 11, 13, 15]
    misc_index = [6, 7]
    indices = [anode_index, cathode_index, separator_index, misc_index]
    """
    comp_ind = [0, 3, 16, 8, 10, 12, 14, 2, 5, 18, 9, 11, 13, 15, 1, 4, 17, 6, 7]
    """
    return in_labels, linestyles, colors, linewidths, indices

def Cap_025C():

    bd = BatteryData('0.25C.txt', 'V_025C.txt', 0)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('$\mathbf{w}$ components, Capacity, 0.25C')
    ax.fill_between([bd.vtime[1], bd.vtime[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[16], bd.vtime[18]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[37], bd.vtime[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.148, 0.85, 'A')
    fig.text(0.386, 0.85, 'B')
    fig.text(0.719, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)
   
    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()
    
    # summary plots 95%
    ax = axes[0]
    ff = bd.capacity[:,2].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 0.25C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('A, V = {:2.1f}'.format(bd.vtime[2]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 0.25C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.capacity[:,17].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('B, V = {:2.1f}'.format(bd.vtime[17]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.capacity[:,38].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('C, V = {:2.1f}'.format(bd.vtime[38]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()
    
def Cap_1C():
    
    bd = BatteryData('1C.txt', 'V_1C.txt', 1)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('$\mathbf{w}$ components, Capacity, 1C')
    
    # 5, 34, 48
    ax.fill_between([bd.vtime[4], bd.vtime[6]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[33], bd.vtime[35]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[47], bd.vtime[49]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.196, 0.85, 'A')
    fig.text(0.655, 0.85, 'B')
    fig.text(0.876, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)
    
    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()
    
    # summary plots 95%
    ax = axes[0]
    ff = bd.capacity[:,5].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 1C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('A, V = {:2.1f}'.format(bd.vtime[5]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 1C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.capacity[:,34].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('B, V = {:2.1f}'.format(bd.vtime[34]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.capacity[:,48].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('C, V = {:2.1f}'.format(bd.vtime[48]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()

def Cap_4C():
    
    bd = BatteryData('4C.txt', 'V_4C.txt', 2)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
     
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('$\mathbf{w}$ components, Capacity, 4C')
    
    # 5, 34, 48
    ax.fill_between([bd.vtime[4], bd.vtime[6]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[33], bd.vtime[35]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[47], bd.vtime[49]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.197, 0.85, 'A')
    fig.text(0.655, 0.85, 'B')
    fig.text(0.876, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)
    
    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()
    
    # summary plots 95%
    ax = axes[0]
    ff = bd.capacity[:,5].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 4C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('A, V = {:2.1f}'.format(bd.vtime[5]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Capacity, 4C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.capacity[:,34].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('B, V = {:2.1f}'.format(bd.vtime[34]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.capacity[:,48].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0, 1.05*np.max(bd.capacity)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title('C, V = {:2.1f}'.format(bd.vtime[48]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()

def Vol_025C():
    bd = BatteryData('0.25C.txt', 'V_025C.txt', 0)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('$\mathbf{w}$ components, Voltage, 0.25C')
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.148, 0.85, 'A')
    fig.text(0.719, 0.85, 'B')
    fig.text(0.86, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)
  
    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()

    # summary plots 95%
    ax = axes[0]
    ff = bd.voltage[:,2].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 0.25C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'A, $t^*$ = {:2.1f}'.format(bd.tstar[2]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 0.25C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.voltage[:,38].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'B, $t^*$ = {:2.1f}'.format(bd.tstar[38]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.voltage[:,47].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'C, $t^*$ = {:2.1f}'.format(bd.tstar[47]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()

def Vol_1C():
    bd = BatteryData('1C.txt', 'V_1C.txt', 1)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('$\mathbf{w}$ components, Voltage, 1C')
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.148, 0.85, 'A')
    fig.text(0.719, 0.85, 'B')
    fig.text(0.86, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)

    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()

    # summary plots 95%
    ax = axes[0]
    ff = bd.voltage[:,2].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 1C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'A, $t^*$ = {:2.1f}'.format(bd.tstar[2]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 1C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.voltage[:,38].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'B, $t^*$ = {:2.1f}'.format(bd.tstar[38]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.voltage[:,47].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'C, $t^*$ = {:2.1f}'.format(bd.tstar[47]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()

def Vol_4C():
    bd = BatteryData('4C.txt', 'V_4C.txt', 2)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
    
    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, ylim=(-1, 1))
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('$\mathbf{w}$ components, Voltage, 4C')
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    fig.text(0.148, 0.85, 'A')
    fig.text(0.719, 0.85, 'B')
    fig.text(0.86, 0.85, 'C')
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)        
        
    leg0 = plt.legend(handles=[linez[i] for i in indices[0]], labels=[in_labels[i] for i in indices[0]], \
        bbox_to_anchor=(1.06, 1), loc=2, borderaxespad=0., fontsize=16, title='Anode', handlelength=2)
    plt.gca().add_artist(leg0)

    leg3 = plt.legend(handles=[linez[i] for i in indices[3]], labels=[in_labels[i] for i in indices[3]], \
        bbox_to_anchor=(1.265, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Miscellany', handlelength=2, borderpad=0.8)
    plt.gca().add_artist(leg3)
    
    leg2 = plt.legend(handles=[linez[i] for i in indices[2]], labels=[in_labels[i] for i in indices[2]], \
        bbox_to_anchor=(1.06, 0.235), loc=2, borderaxespad=0., fontsize=16, title='Separator', handlelength=2, borderpad=0.42)
    plt.gca().add_artist(leg2)
    
    leg1 = plt.legend(handles=[linez[i] for i in indices[1]], labels=[in_labels[i] for i in indices[1]], \
        bbox_to_anchor=(1.265, 1), loc=2, borderaxespad=0., fontsize=16, title='Cathode', handlelength=2)

    fig, axes = plt.subplots(2, 3, figsize=(3*3.5, 2*3.5)); axes = axes.reshape(6).squeeze()

    # summary plots 95%
    ax = axes[0]
    ff = bd.voltage[:,2].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 4C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'A, $t^*$ = {:2.1f}'.format(bd.tstar[2]))
    
    # ZOOM
    ax = axes[3]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.set_ylabel('Voltage, 4C')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 65%
    ax = axes[1]
    ff = bd.voltage[:,38].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'B, $t^*$ = {:2.1f}'.format(bd.tstar[38]))
    
    ax = axes[4]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    # summary plots 23%
    ax = axes[2]
    ff = bd.voltage[:,47].reshape((M, 1))
    aa = get_coefficients(bd.X, ff).reshape((m, ))
    ax.set_xlim((-2, 2)); ax.set_ylim((0.95*np.min(bd.voltage), 1.05*np.max(bd.voltage)))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    ax.set_title(r'C, $t^*$ = {:2.1f}'.format(bd.tstar[47]))
    
    ax = axes[5]
    ax.set_xlim((-2, 2))
    ax.grid(True)
    ax.set_xticks([-2,0,2])
    ax.locator_params(axis='y',nbins=4)
    ax.set_xlabel('$\mathbf{w}^T\mathbf{x}$')
    ax.plot(np.dot(bd.X, aa), ff, 'ko')
    
    fig.tight_layout()
        
def bootstrap_SE():    

    plt.rc('text', usetex=False)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 16}
    plt.rc('font', **font)    
    
    fig, axes = plt.subplots(3, 2, figsize=(2*6, 3*3)); axes = axes.reshape(6).squeeze()
    
    ax = axes[0]
    
    bd = BatteryData('0.25C.txt', 'V_025C.txt', 0)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Capacity, 0.25C')
    ax.set_ylabel('Bootstrap SE')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.vtime[1], bd.vtime[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[16], bd.vtime[18]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[37], bd.vtime[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.025, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.325, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.755, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
    
    
    ax = axes[2]
    
    bd = BatteryData('1C.txt', 'V_1C.txt', 1)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Capacity, 1C')
    ax.set_ylabel('Bootstrap SE')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.vtime[4], bd.vtime[6]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[33], bd.vtime[35]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[47], bd.vtime[49]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.085, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.670, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.960, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
        
    ax = axes[4]
        
    bd = BatteryData('4C.txt', 'V_4C.txt', 2)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime))
    for i in range(ntime):
        ff = bd.capacity[:,i].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.vtime[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.vtime[tind]])
    ax.invert_xaxis()
    ax.set_xlabel('Capacity, 4C')
    ax.set_ylabel('Bootstrap SE')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.vtime[4], bd.vtime[6]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[33], bd.vtime[35]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.vtime[47], bd.vtime[49]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.085, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.670, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.960, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.vtime, a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
        
    ax = axes[1]
        
    bd = BatteryData('0.25C.txt', 'V_025C.txt', 0)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage, 0.25C')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.024, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.758, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.941, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
        
    ax = axes[3]
        
    bd = BatteryData('1C.txt', 'V_1C.txt', 1)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage, 1C')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.024, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.758, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.941, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
        
    ax = axes[5]
        
    bd = BatteryData('4C.txt', 'V_4C.txt', 2)
    ntime = bd.tstar.shape[0]
    M, m = bd.X.shape
    
    a = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a[:,i] = get_coefficients(bd.X, ff).reshape((m, ))
            
    a_br = np.zeros((m, ntime-2))
    for i in range(ntime-2):
        ff = bd.voltage[:,i+1].reshape((M, 1))
        a_br[:,i] = get_coefficients_boot(bd.X, ff).reshape((m, ))
    
    ax.grid(True)
    tind = [0,12,25,37,49]
    ax.set_xticks(bd.tstar[tind])
    ax.set_xticklabels(['{:2.1f}'.format(x) for x in bd.tstar[tind]])
    ax.set_xlim([np.min(bd.tstar), np.max(bd.tstar)])
    ax.invert_xaxis()
    ax.set_xlabel('Voltage, 4C')
    ax.set_ylim([0, 0.01])
    ax.fill_between([bd.tstar[1], bd.tstar[3]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[37], bd.tstar[39]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.fill_between([bd.tstar[46], bd.tstar[48]], [-1, -1], [1, 1], facecolor=[0.7, 0.7, 0.7])
    ax.text(0.024, 0.90, 'A', transform=ax.transAxes)
    ax.text(0.758, 0.90, 'B', transform=ax.transAxes)
    ax.text(0.941, 0.90, 'C', transform=ax.transAxes)    
    
    in_labels, linestyles, colors, linewidths, indices = input_labels()
    linez = []
    for i in range(m):
        ll, = ax.plot(bd.tstar[1:-1], a_br[i,:], linestyle=linestyles[i], color=colors[i], linewidth=linewidths[i], label=in_labels[i])
        linez.append(ll)
        
    fig.tight_layout()

if __name__ == '__main__':
    
    plt.close('all')
    
    Cap_025C()
    Cap_1C()
    Cap_4C()
    Vol_025C()
    Vol_1C()
    Vol_4C()
    
    plt.show()

    