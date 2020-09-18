import numpy as np

# manhattan distance map on flat torus geometry
# https://stackoverflow.com/a/62524083
def mdc(x, y, size_x, size_y):
    a, b = divmod(size_x, 2)
    x_template = np.r_[:a+b, a:0:-1] # [0 1 2 1] for size_x == 4 and [0 1 2 2 1] for size_x == 5
    x_template = np.roll(x_template, x) # for x == 2, size_x == 8: [2 1 0 1 2 3 4 3]
    a, b = divmod(size_y, 2)
    y_template = np.r_[:a+b, a:0:-1]
    y_template = np.roll(y_template, y)
    return np.add.outer(x_template, y_template)

def dm_from_sys(arr_sy):
    # return normalised manhattan distance to nearest shipyard from an array of shipyards
    dms = []
    r, c = np.nonzero(arr_sy)
    for i in range(len(r)):
        dms.append(mdc(r[i],c[i],21,21))
    if len(dms)>0: 
        return np.amin(dms,axis=0)
    else:
        return np.ones((21,21), dtype = np.float32)

def randomtoroidalcrop_single(mat, x, y):
    return mat.take(range(mat.shape[0]), mode="wrap", axis=0).take(range(x,x+32), mode="wrap", axis=1).take(range(y,y+32), mode="wrap", axis=2)

def flood_dist(x, y, size_x, size_y):
    a, b = divmod(size_x, 2)
    x_template = np.r_[:a+b, a:0:-1] # [0 1 2 1] for size_x == 4 and [0 1 2 2 1] for size_x == 5
    x_template = np.roll(x_template, x) # for x == 2, size_x == 8: [2 1 0 1 2 3 4 3]
    a, b = divmod(size_y, 2)
    y_template = np.r_[:a+b, a:0:-1]
    y_template = np.roll(y_template, y)
    return np.add.outer(x_template, y_template)

def xy(n):
    return n % 21, n // 21

def c(n):
    return n % 21

# Big kludge
def padit(canvas):
    
    canvas2 = np.zeros((canvas.shape[0],32,32),dtype=np.float32)
    canvas2[:,5:26,5:26] = canvas

    canvas2[:,:5,5:26] = canvas[:,-5:,:]
    canvas2[:,-6:,5:26] = canvas[:,:6,:]
    canvas2[:,5:26,:5] = canvas[:,:,-5:]
    canvas2[:,5:26,-6:] = canvas[:,:,:6]

    canvas2[:,:5,:5] = canvas2[:,21:26,:5]
    canvas2[:,:5,26:32] = canvas2[:,:5,5:11]
    canvas2[:,26:32,:5] = canvas2[:,26:32,21:26]
    canvas2[:,26:32,26:32] = canvas2[:,26:32,5:11]

    return canvas2
