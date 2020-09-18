## Halite 4 doesn't allow internet access. All external libraries must be included as code
## Credits - thank you to you all!
## - EfficientNet from https://github.com/lukemelas/EfficientNet-PyTorch
## - Unet from https://github.com/qubvel/segmentation_models.pytorch
## - Pytorch starter from https://www.kaggle.com/david1013/pytorch-starter
## - Optimus mine agent from https://www.kaggle.com/solverworld/optimus-mine-agent

RESOURCE_DIR = '/kaggle_simulations/agent/'

WEIGHTS_FILE = RESOURCE_DIR + 'weights.pth' 

#-------------- IMPORTS SECTION ------------------#

from time import perf_counter
start_time = perf_counter()

import sys
sys.path.append(RESOURCE_DIR)

import torch
from torch.nn import Softmax 
import numpy as np
import os
import random
import scipy.ndimage
import scipy.optimize
from scipy.special import softmax as softmax_scipy
from heapq import nsmallest
from kaggle_environments.envs.halite.helpers import *

from halite4.network import Unet
from halite4.utils import dm_from_sys, randomtoroidalcrop_single, flood_dist, xy, c, padit

#-------------- PARAMETERS SECTION ------------------#

N_INPUTS = 29 # number of input channels

# Do TTA only if step >= TTA_START_STEP and at least the last TTA_STABLE_STEPS steps have taken < TTA_SLOW_EPOCH_TRIGGER ms
TTA_SLOW_EPOCH_TRIGGER = 2500 #ms
TTA_START_STEP = 20
TTA_STABLE_STEPS = 3

AVOID_STUPID_MOVES = True
BASE_RAID_AT_GAME_END = True
STEP_BASE_RAID = 360
STEP_TO_STOP_SPAWN = 290

device = 'cpu'

#-------------- CONSTANTS SECTION ------------------#

size = 21
me = None
num_converts = 0
prev_board = None

HALITE_DIV = 500
CARGO_DIV = 500
PH_DIV = 5000

HALITE = 0
BASES = 1
SHIPS = 2

POSITION = 0
CARGO = 1

PRED_PWR = 2.7

offsets = [(0,1),(0,-1),(1,0),(-1,0)] + [(1,1), (1,-1), (-1,1), (-1,-1), (2,0), (0,2), (-2,0),(0,-2)]
ship_filter=np.array([[0,1,0],[1,1,1],[0,1,0]])

NORTH, SOUTH, EAST, WEST = ShipAction.NORTH, ShipAction.SOUTH, ShipAction.EAST, ShipAction.WEST
CONVERT, SPAWN = ShipAction.CONVERT, ShipyardAction.SPAWN
action_map = {'': 0, 'NORTH': 1, 'EAST': 2, 'SOUTH': 3, 'WEST': 4, 'CONVERT': 5, 'SPAWN': 6}
action_map2 = {None: 0, NORTH: 1, EAST: 2, SOUTH: 3, WEST: 4, CONVERT: 5, SPAWN: 6}

print("{:.0f} ms - load to point F".format((perf_counter() - start_time)*1e3))

#-------------- Data/Model SECTION ------------------#

def convolve_norm(mat, filt=ship_filter, mult=1):
    for _ in range(mult):
        mat += scipy.ndimage.convolve(mat, filt, mode='wrap',cval=0.0)
    return (mat-mat.min())/(mat.max()-mat.min())

def getInputStack3(obs, step, halite, ship, base, cargo, ph, threat, action_map, 
                      first_player):

    first_player += 1 # from 0-3 to 1-4
    other_players = [p for p in [1, 2, 3, 4] if p != first_player] # shuffle other players
    random.shuffle(other_players)   

    input_stack = np.zeros((N_INPUTS, size, size), dtype = np.float32)
    
    # board halite
    input_stack[0] = halite
    
    # mine per xy
    input_stack[1] = (ship==first_player) * halite
    input_stack[2] = (ship==first_player) * cargo
    input_stack[3] = (ship == first_player)  
    input_stack[4] = (base == first_player)  
    input_stack[5] = (ship==first_player) * ph
    input_stack[6] = np.logical_and(input_stack[3],(cargo==0))

    # theirs per xy
    for i in range(3): 
        input_stack[7] += ((ship==other_players[i]))
        input_stack[8+i] = ((ship==other_players[i])) 

    for i in range(3): 
        input_stack[11] += ((ship==other_players[i]) * halite)
        input_stack[12] += ((ship==other_players[i]) * cargo)
        input_stack[13] += (base == other_players[i])
        input_stack[14] += ((ship==other_players[i]) * ph)
        input_stack[15] += np.logical_and(input_stack[8+i],(cargo==0))
        
    # Current step. Let the last 12 steps be 1.
    input_stack[16] = np.min([step/387,1]) 

    # Base distance maps
    input_stack[17] = dm_from_sys(input_stack[4,:,:])
    input_stack[18] = dm_from_sys(input_stack[13,:,:])

    # Vicinity maps
    input_stack[19] = 1*(scipy.ndimage.convolve(input_stack[3,:,:].copy(), ship_filter, mode='wrap',cval=0.0) - \
                            input_stack[7])>0
    input_stack[20] = 1*(scipy.ndimage.convolve(input_stack[7,:,:].copy(), ship_filter, mode='wrap',cval=0.0) - \
                            input_stack[3])>0
    
    # Dominance maps
    if np.sum(input_stack[3,:,:]) > 0:
        input_stack[21] = convolve_norm(input_stack[3,:,:].copy(), mult=3)
    if np.sum(input_stack[7,:,:]) > 0:
        input_stack[22] = convolve_norm(input_stack[7,:,:].copy(), mult=3)

    # Previous actions 
    input_stack[23:28] = getPreviousActions(obs, step, first_player-1)
    
    # Number of lighter enemy ships within manhattan<=2
    input_stack[28] = threat
        
    return input_stack 

def getPreviousActions(obs, step, first_player):
    action_ship = np.zeros((size,size), dtype = np.int8)
    # ship actions
    for player in [first_player]: # range(4):
        for k, v in obs['players'][player][SHIPS].items():
            x, y = xy(v[POSITION])
            if k in turn.last_action.keys():
                action_ship[x, y] = turn.last_action[k]
     
    output_stack = np.zeros((5, size, size), dtype = np.float32)
    for i in range(1,5): # NESW
        output_stack[i,:,:] = 1*action_ship==i

    output_stack[0,:,:] = 1*action_ship==-1 # STILL
        
    return output_stack

def getHalite(obs):
    return (np.fromiter(obs['halite'], dtype = np.float32).reshape(size, size).T / HALITE_DIV).astype(np.half)

def getAllObjects3(obs, conf, config):
    ships = {};                                                # List of xy of all ships (not bases)
    ship = np.zeros((conf.size, conf.size), dtype=np.int8)     # MxM of ships for player P, at ship locations
    base = np.zeros((conf.size, conf.size), dtype=np.int8 )    # MxM of bases for player P, at base locations
    cargo = np.zeros((conf.size, conf.size), dtype=np.half)    # MxM of cargo per ship, at ship locations
    ph = np.zeros((conf.size, conf.size), dtype=np.half)       # MxM of overall halite per player, at ship locations
    threat = np.zeros((conf.size, conf.size), dtype=np.half)
    board = Board(obs, config)
    
    for pidx, p in enumerate(obs['players']):                  # for each player (0-3)
        for b_idx, b in enumerate(p[BASES].values()):          #   for each base
            x, y = xy(b)                                       #     set base and ph       
            base[x, y] = pidx + 1                              #   for each ship
            ph[x, y] = p[HALITE] / PH_DIV                      #     set ship, cargo, ph, ships

        for k, v in p[SHIPS].items():
            x, y = xy(v[POSITION])
            ship[x, y] = pidx + 1
            cargo[x, y] = v[CARGO] / CARGO_DIV 
            ph[x, y] = p[HALITE] / PH_DIV
            ships[k] = v[POSITION]

            # Threat count
            sh = board.ships[k]
            count_threat = 0
            for j, oset in enumerate(offsets):
                nei = sh.cell.neighbor(oset).ship
                if nei is None: continue
                if nei.player_id == sh.player_id: continue
                if nei.halite >= sh.halite: continue
                if j<=3: count_threat += 0.1
                else: count_threat += 0.07

            threat[x, y] = count_threat 
            
    return ships, ship, base, cargo, ph, threat

def processStep3(obs, conf, config):   
    step = obs['step']
    halite = getHalite(obs)
    ships, ship, base, cargo, ph, threat  = getAllObjects3(obs, conf, config)

    return (step, halite, ship, base, cargo, ph, threat)

#-------------- UTILITIES SECTION ------------------#

def act(a):
    dictt = {'NORTH':NORTH, 'SOUTH':SOUTH, 'EAST':EAST, 'WEST':WEST, 
    'CONVERT':CONVERT, None:None, 'SPAWN':SPAWN}
    return dictt[a]

def dist(p1, p2):
    _, __, d = dirs_to(p1, p2)
    return d

def dirs_to(p1, p2, size=size):
    dX, dY = p2 - p1
    if abs(dX) > size/2:
        if dX<0: dX += size
        elif dX > 0: dX-=size
    if abs(dY)>size/2:
        if dY<0: dY+=size
        elif dY>0: dY-=size
    ret=[]
    if dX>0: ret.append(EAST)
    if dX<0: ret.append(WEST)
    if dY>0: ret.append(NORTH)
    if dY<0: ret.append(SOUTH)
    if abs(dX)<abs(dY): ret = ret[::-1] # prioritize longer path
    if len(ret)==0: return [None], (dX, dY), abs(dX) + abs(dY)

    # Sample an alternative action here
    acts = [NORTH, SOUTH, EAST, WEST, None]
    prob = None
    if dX!=0 and dY==0: prob = [0.25, 0.25, 0, 0, 0.5]
    elif dX==0 and dY!=0: prob = [0, 0, 0.25, 0.25, 0.5]
    if prob is not None: 
        alternate = np.random.choice(acts, 1, p=prob)[0]
        ret.append(alternate)
    ret.append(None)
    return ret, (dX, dY), abs(dX) + abs(dY) # action, step, distance

def nearest_shipyard(pos, yards):
    if len(yards)==0: return 100, None
    d = [dist(pos,yard.position) for yard in yards]
    return min(d), yards[np.argmin(d)].position

def shipyard_actions(board):
    #spawn a ship as long as there is no ship already moved to this shipyard
    conv = convolve(turn.halite_matrix, filt='gaussian') # select the best shipyard to spawn (most halite around)
    sy_rank = [conv[sy.position] for sy in me.shipyards]
    order = np.argsort(sy_rank)[::-1]
    if board.step < 80: order = list(range(len(me.shipyards)))[::-1] # spawn from latest built yard
    limit = 1000 if len(turn.convert_spots) > 0 else 500

    # if we want to convert, it's better to not spawn until we reach 1000 halite, then the...
    # new base will immediately have remain 500 halite to spawn a new ship to guard it
    for j in order:
        sy = me.shipyards[j]
        cond1 = turn.num_ships < turn.max_ships and me.halite>=limit
        if cond1: #or cond2:
            if not turn.taken[sy.position] and board.step<STEP_TO_STOP_SPAWN:
                sy.next_action = SPAWN
                turn.taken[sy.position] = 1
                turn.num_ships += 1
                turn.total_halite -= 500
                break

def gen_matrix_one_enemy(board, enemy_id):
    EP, EH, ES = np.zeros((size,size)), np.zeros((size,size)), np.zeros((size,size))
    for id,ship in board.ships.items():
        if ship.player_id == enemy_id: 
            EH[ship.position]=ship.halite
            EP[ship.position]=1
    for id, sy in board.shipyards.items():
        if sy.player_id == enemy_id: ES[sy.position]=1
    return EP,EH,ES

def gen_matrix(board, player_id):
    EP, EH, ES = np.zeros((size,size)), np.zeros((size,size)), np.zeros((size,size))
    MP, MH, MS = np.zeros((size,size)), np.zeros((size,size)), np.zeros((size,size))
    for id,ship in board.ships.items():
        if ship.player_id != player_id: 
            EH[ship.position]=ship.halite
            EP[ship.position]=1
        else: 
            MH[ship.position]=ship.halite
            MP[ship.position]=1
    for id, sy in board.shipyards.items():
        if sy.player_id != player_id: ES[sy.position]=1
        else: MS[sy.position]=1
    return EP,EH,ES,MP,MH,MS

def compute_max_ships(board):
  #using Logistic regression analyzer
  min_ships=5
  max_ships=50
  nships = turn.num_ships

  coeff=np.array([1.61,-.53,6.58,4.87,-6.41,3.52,-9.18])
  intercept=-1.9
  mean=np.array([9.44404679e+03, 6.48383313e+01, 1.59664290e+02, 1.82539408e+02,
        8.57612171e+04, 8.23785001e+07, 3.36728076e+04])
  scale=np.array([6.61922007e+03, 1.51620419e+01, 9.04440277e+01, 2.28999086e+02,
        3.31893927e+05, 6.17634535e+08, 2.99437250e+04])

  #h=board_halite/num_total_ships
  #values are: halite, num_ships, step, h, h**2, h**3, step**2
  s=board.step
  h=turn.total_avail_halite/(1+turn.num_total_ships)
  data=np.array([turn.total_avail_halite, turn.num_total_ships, s, h, h**2, h**3, s**2])
  v=((data-mean)/scale).dot(coeff)+intercept
  num = nships
  if v>0:
    num = nships+3
  if num > max_ships:
    num = max_ships
  if board.step < 180:
    return max(turn.op_nships.max() + 10, num)
  if board.step > 300:
    return 2
  if nships < min_ships:
    return 8
  if (board.step < 280) and (num < nsmallest(2, turn.op_nships)[-1]): # keep up with 2nd smallest
      return nsmallest(2, turn.op_nships)[-1]
  return num

def make_dominance_matrix(EP, MP):
    mat1 = convolve(EP, filt=np.ones((3,3)))
    mat2 = convolve(EP, filt=np.ones((5,5)))
    mat3 = convolve(EP, filt=np.ones((7,7)))
    mat = mat1 + mat2 + mat3
    mat = (mat-mat.min())/(mat.max()-mat.min()) if mat.max()>mat.min() else mat
    return mat

def make_avoidance_equal_matrix(myship_halite):
    bad_ship = np.logical_and(turn.EH == myship_halite, turn.EP) # IMPORTANT!!! <= CHANGED TO <
    avoid = convolve(bad_ship)
    return avoid

def make_avoidance_matrix(myship_halite):
    bad_ship = np.logical_and(turn.EH < myship_halite, turn.EP) # IMPORTANT!!! <= CHANGED TO <
    avoid = convolve(bad_ship)
    return avoid

def ship_converts(board):
    #if no shipyard, convert the ship with least dominance
    if turn.num_shipyards == 0 and not turn.last_episode and not board.step<=12:
        dom = [turn.dominance[ship.position] for ship in me.ships]
        order = np.argsort(dom)
        for od in order:
            ship = me.ships[od]
            if ship.halite + turn.total_halite > 500 and board.step < 375:
                ship.next_action = CONVERT
                turn.taken[ship.position]=1
                turn.num_shipyards+=1
                turn.total_halite-=500
                break

    if board.step < 200: # only spawn 2nd shipyard when step < 200
        for pos in turn.convert_spots:
            if board.cells[pos].ship is not None and board.cells[pos].ship.player.is_current_player \
                and (turn.total_halite + board.cells[pos].ship.halite)>1000:
                all_d = [nearest_shipyard(pos, player.shipyards)[0] for pid, player in board.players.items()] 
                if min(all_d) >= 5:
                    board.cells[pos].ship.next_action = CONVERT
                    turn.taken[pos] = 1
                    break

def convolve(mat, filt=np.array([[0,1,0],[1,1,1],[0,1,0]])):
    if isinstance(filt, str):
        if filt=='gaussian': return scipy.ndimage.gaussian_filter(mat, mode='wrap',sigma=1)
    else: return scipy.ndimage.convolve(mat, filt, mode='wrap',cval=0.0)

def move(pos, action):
   if action==NORTH: ret=pos+Point(0,1)
   elif action==SOUTH: ret=pos+Point(0,-1)
   elif action==EAST: ret=pos+Point(1,0)
   elif action==WEST: ret=pos+Point(-1,0)
   else: ret = pos
   return ret % size

def deduct_move(pos, last_pos):
   if move(last_pos, NORTH) == pos: return NORTH
   elif move(last_pos, SOUTH) == pos: return SOUTH
   elif move(last_pos, EAST) == pos: return EAST
   elif move(last_pos, WEST) == pos: return WEST
   else: return None

def set_turn_data(board):
    global num_converts
    global prev_board
    turn.num_ships = len(me.ships)
    turn.total_halite = me.halite
    turn.cargo = sum([sh.halite for sh in me.ships])
    turn.halite_matrix = np.flip( np.reshape(board.observation['halite'], (size,size)), axis=0 ).T
    turn.num_total_ships=len(board.ships)
    turn.steps_remaining=399-board.step
    turn.num_halite_cells=np.sum(turn.halite_matrix>0)
    turn.total_avail_halite=np.sum(turn.halite_matrix)
    turn.avg_halite=turn.total_avail_halite/(turn.num_halite_cells+1)
    turn.halite_per_ship=turn.total_avail_halite/(turn.num_total_ships+1)    

    pos, hals = [], []
    for pt in board.cells.keys():
        _h = turn.halite_matrix[pt]
        if _h > 0:
            pos.append(pt)
            hals.append(_h)
    turn.num_shipyards = len(me.shipyards)
    turn.EP, turn.EH, turn.ES, turn.MP, turn.MH, turn.MS = gen_matrix(board, board.current_player.id) 
    turn.taken = np.zeros((size,size))
    turn.last_episode = (board.step == (board.configuration.episode_steps-2))
    turn.dominance = make_dominance_matrix(turn.EP, turn.MP)

    filt = np.array([[0, 1, 2, 3, 2, 1, 0], [1, 2, 3, 4, 3, 2, 1], [2, 3, 4, 5, 4, 3, 2], [3, 4, 5, 0, 5, 4, 3], 
                     [2, 3, 4, 5, 4, 3, 2],  [1, 2, 3, 4, 3, 2, 1], [0, 1, 2, 3, 2, 1, 0] ])
    conv_halite = convolve(turn.halite_matrix, filt=filt)
    conv_EP = convolve(turn.EP, filt=filt)
    conv_ES = convolve(turn.ES, filt=filt)
    
    turn.convert_spots = []
    if turn.num_shipyards*10 < turn.num_ships and num_converts<=3:
        if board.step!=0:
            conv_avoid = convolve(turn.ES)
            conv = conv_halite - conv_EP*10 - conv_ES*20
            pts, score = [], []
            for pt, c in board.cells.items():
                if conv_avoid[pt]: continue
                if c.ship is not None and c.ship.player.is_current_player:
                    if (me.halite + c.ship.halite) > 500:
                        pts.append(pt)
                        score.append(conv[pt])
            order = np.argsort(score)[::-1]
            turn.convert_spots = [pts[ii] for ii in order]

        else: turn.convert_spots = [me.ships[0].position]
            
    turn.op_halites = np.array([op.halite for op in board.opponents])
    turn.op_nships = np.array([len(op.ships) for op in board.opponents])
    turn.op_nyards = np.array([len(op.shipyards) for op in board.opponents])
    turn.op_ids = np.array([op.id for op in board.opponents])
    turn.op_cargos = np.array([sum([sh.halite for sh in op.ships]) for op in board.opponents])
    a = (board.step/400)*0.6 + 0.2
    op_rank = a*(turn.op_halites + turn.op_cargos) + (1-a)*turn.op_nships*500
    
    _idx = np.argmax(op_rank)
    turn.best_enemy_idx = _idx
    _h, _s, _y, _id, _c = turn.op_halites[_idx], turn.op_nships[_idx], turn.op_nyards[_idx], turn.op_ids[_idx], turn.op_cargos[_idx]
    turn.EPbest, turn.EHbest, turn.ESbest = gen_matrix_one_enemy(board, _id)     
    turn._h, turn._s, turn._y, turn._id, turn._c = _h, _s, _y, _id, _c
    
    turn.avoid, turn.attack, turn.avoid_equal = {}, {}, {}
    for ship in me.ships:
        turn.avoid[ship.id] = make_avoidance_matrix(ship.halite)
        turn.avoid_equal[ship.id] = make_avoidance_equal_matrix(ship.halite)
                
    turn.max_ships = compute_max_ships(board)
    turn.last_action = {}
    for shipid, ship in board.ships.items():
        if shipid not in prev_board.ships.keys(): continue
        turn.last_action[shipid] = action_map2 [ deduct_move(ship.position, prev_board.ships[shipid].position) ]

def find_steps_to(point1: Point, point2: Point) -> List[Point]:
    dx, dy = point2 - point1
    result = []

    if 0 < dx <= 11 or dx < -11:
        result.append(EAST)

    if -11 <= dx < 0 or dx > 11:
        result.append(WEST)

    if 0 < dy <= 11 or dy < -11:
        result.append(SOUTH)

    if -11 <= dy < 0 or dy > 11:
        result.append(NORTH)
        
    return result

print("{:.0f} ms - load to point G".format((perf_counter() - start_time)*1e3))

#-------------- CREATE MODEL SECTION ------------------#

model = Unet(encoder_name="efficientnet-b0", classes=5, encoder_depth=2, decoder_channels=(64, 32), in_channels=64, encoder_weights=None)

print("{:.0f} ms - load to point H".format((perf_counter() - start_time)*1e3))

model.load_state_dict(torch.load(WEIGHTS_FILE, torch.device(device))['model_state_dict'])
model.to(device)
model.eval();
torch.no_grad();
torch.set_num_threads(os.cpu_count())

class Turn_Info:
    pass

turn = Turn_Info()

print("{:.0f} ms - load time before 1st step".format((perf_counter() - start_time)*1e3))

steps_since_slow_epoch = 0

#-------------- MAIN LOOP SECTION ------------------#

def main(obs, config):
    start_time = perf_counter()
    global size
    global me
    global num_converts
    global prev_board
    global do_tta
    global steps_since_slow_epoch
    
    conf = config 
        
    board = Board(obs, config)
    if board.step == 0: prev_board=board
    me = board.current_player
    set_turn_data(board)
    ship_converts(board)            
    
    # convert to game format
    step, halite, ship, base, cargo, ph, threat = processStep3(obs, conf, config)

    print('Step',step)

    # check if we will do TTA this step
    do_tta = (step >= TTA_START_STEP) and (steps_since_slow_epoch >= TTA_STABLE_STEPS)

    print("{:.0f} ms - model about to predict".format((perf_counter() - start_time)*1e3))

    # featurise
    input_stack = getInputStack3(obs, step, halite, ship, base, cargo, ph, threat, action_map, first_player = obs['player'])
    #input_stack = randomtoroidalcrop_single(input_stack, 0, 0)
    input_stack = padit(input_stack[:,:,:])

    if not do_tta:
    
        input_stack = torch.as_tensor(input_stack).unsqueeze(0).to(device)  
        policy_output = model(input_stack)[:,:5,:,:].to(device) # Discard Spawn/Convert predictions
        nn_ship_actions = Softmax(1)(policy_output).cpu().detach().numpy()[:,:,5:26,5:26]
    
    else: # build TTA batch

        input_stack = np.repeat(input_stack[np.newaxis, :, :], 5, axis=0)
        input_stack[1] = randomtoroidalcrop_single(input_stack[1], 0, 10)
        input_stack[2] = randomtoroidalcrop_single(input_stack[2], 10, 0)
        input_stack[3] = randomtoroidalcrop_single(input_stack[3], 10, 10)
        input_stack[4] = randomtoroidalcrop_single(input_stack[4], 5, 5)
        input_stack = torch.as_tensor(input_stack).to(device)  
        
        # predict
        policy_output = model(input_stack)[:,:5,:,:].to(device) # Discard Spawn, Convert predictions

        # revert TTA back
        nn_ship_actions = policy_output.cpu().detach().numpy()
        nn_ship_actions[1] = randomtoroidalcrop_single(nn_ship_actions[1], 0, -10)
        nn_ship_actions[2] = randomtoroidalcrop_single(nn_ship_actions[2], -10, -0)
        nn_ship_actions[3] = randomtoroidalcrop_single(nn_ship_actions[3], -10, -10)
        nn_ship_actions[4] = randomtoroidalcrop_single(nn_ship_actions[4], -5, -5)
            
        nn_ship_actions = softmax_scipy(nn_ship_actions, axis=1)[:,:,5:26,5:26] 

    print("{:.0f} ms - model predicted".format((perf_counter() - start_time)*1e3))
    
    actions = {}
    
    # my assets -- to predict which step to take
    my_ships = obs['players'][obs['player']][SHIPS]
    my_bases = obs['players'][obs['player']][BASES]
    my_halite = obs['players'][obs['player']][HALITE]
    all_ships = {}
    for player in range(len(obs['players'])):
        all_ships.update(obs['players'][player][SHIPS])
    ship_list = list(my_ships.items())
    base_list = list(my_bases.items())
    print('Base list: ',base_list)
    ship_id_to_idx = {ship_key: ship_idx for ship_idx, (ship_key, ship_info) in enumerate(ship_list)}

    # score matrix -- can only pick valid actions
    C = -10000* np.ones((len(my_ships) + len(my_bases), size * size + len(my_ships) + len(my_bases)))
    
    # add ships to scoring matrix
    for ship_idx, (ship_key, ship_info)  in enumerate(ship_list):        
        x, y = xy(ship_info[POSITION])
        ship_pred_actions = nn_ship_actions[0, :, x, y]
            
        raw_ship_pred_actions = np.copy(ship_pred_actions)
            
        restore_sum = np.sum(ship_pred_actions[1:5])
        ship_pred_actions[1:5] = np.where(ship_pred_actions[1:5] > 0., ship_pred_actions[1:5], 0)
        if np.sum(ship_pred_actions[1:5]) > 0:
            ship_pred_actions[1:5] *= restore_sum / sum(ship_pred_actions[1:5])

        restore_sum = np.sum(ship_pred_actions[1:5])
        if restore_sum > 1e-6:
            ship_pred_actions[1:5] = ship_pred_actions[1:5] ** PRED_PWR
            ship_pred_actions[1:5] *= restore_sum / sum(ship_pred_actions[1:5])

        ship_ranked_actions = np.zeros((5,), dtype = np.float32)
        for rank in range(0, np.sum(ship_pred_actions > 1e-6) ):
            while True:
                action = int(random.choice(np.flatnonzero(ship_pred_actions)))
                    
                if random.random() < ship_pred_actions[action]:
                    ship_ranked_actions[action] = 5 - rank + raw_ship_pred_actions[action]
                    ship_pred_actions[action] = 0
                    if np.sum(ship_pred_actions) > 0:
                        ship_pred_actions = ship_pred_actions / np.sum(ship_pred_actions) 
                    else:
                        ship_pred_actions = 0
                    break
                
        C[ship_idx, x + size*y] = ship_ranked_actions[0]  
        C[ship_idx, x + size*c(y - 1)] = ship_ranked_actions[1] # NORTH
        C[ship_idx, c(x + 1) + size*y] = ship_ranked_actions[2] # EAST 
        C[ship_idx, x + size*c(y + 1)] = ship_ranked_actions[3] # SOUTH
        C[ship_idx, c(x - 1) + size*y] = ship_ranked_actions[4] # WEST
        
        if AVOID_STUPID_MOVES: # Prevent stupid moves. Sometimes helps, but probably not needed
            pt = board.ships[ship_key].position
            if turn.avoid[ship_key][pt]: C[ship_idx, x + size*y] -= 1500
            if turn.avoid[ship_key][move(pt, NORTH)]: C[ship_idx, x + size*c(y - 1)] -= 1000
            if turn.avoid[ship_key][move(pt, EAST)]: C[ship_idx, c(x + 1) + size*y] -= 1000
            if turn.avoid[ship_key][move(pt, SOUTH)]: C[ship_idx, x + size*c(y + 1)] -= 1000
            if turn.avoid[ship_key][move(pt, WEST)]: C[ship_idx, c(x - 1) + size*y] -= 1000

        if BASE_RAID_AT_GAME_END: # Kamikaze into enemy bases
            ship = board.ships[ship_key]
            if board.step > STEP_BASE_RAID and ship.halite==0:
                if turn.num_ships > 5:
                    d, sy_pt = nearest_shipyard(ship.position, board.players[turn._id].shipyards)
                    if d < 20:
                        acts, steps, distance = dirs_to(ship.position, sy_pt)
                        C[ship_idx, x + size*y] -= 1500 # not idle
                        if NORTH in acts: C[ship_idx, x + size*c(y - 1)] += 2500 # NORTH
                        if EAST in acts:C[ship_idx, c(x + 1) + size*y] += 2500 # EAST
                        if SOUTH in acts:C[ship_idx, x + size*c(y + 1)] += 2500 # SOUTH
                        if WEST in acts:C[ship_idx, c(x - 1) + size*y] += 2500 # WEST

    shipyard_actions(board) 
    
    # add bases to scoring matrix
    for base_idx, (base_key, base_info) in enumerate(base_list):
        x, y = xy(base_info)                
        spawn_yesno = False
        for sy in me.shipyards:
            if sy.position[0] == x and (20-sy.position[1]) == y:
                if sy.next_action == SPAWN:
                        spawn_yesno = True

        if my_halite >= conf.spawnCost:
            C[len(my_ships) + base_idx, x + size*y] = spawn_yesno * 10
            C[len(my_ships) + base_idx, size*size + len(my_ships) + base_idx] = 5
        else:
            C[len(my_ships) + base_idx, size*size + len(my_ships) + base_idx] = 10
 
    #-------------- PROTECTOR SECTION ------------------#
    # In each step, try to have 1 ship closer than the closest enemy ship. If not, we 
    # move towards our shipyard. This allows protectors to still mine when no threat

    # For each shipyard, get closest friendly & enemy ship
    protectors = {}
    attackers = {}
    assigned_ships = set()
    for sy in me.shipyards:

        ship = board.cells[sy.position].ship
        if ship is not None and ship.player_id==me.id and board.step > 150:
            continue

        # If the action is SPAWN then we can skip
        if sy.next_action == SPAWN:
            continue

        syx, syy = sy.position.x, 20 - sy.position.y
        sy_dists = flood_dist(syx, syy, size, size)

        protector_dists = {}
        attacker_dists = {}
        for ship_key, ship_info in all_ships.items():
            if ship_key in assigned_ships:
                continue
            shipx, shipy = xy(ship_info[POSITION])
            if ship_key in my_ships: # For our own ships, use A* distance
                # Prefer 0 ship first, then the heaviest ship, to protect, it will
                # deposit its ore first and become a 0-ship.
                d = sy_dists[shipx, shipy]
                protector_dists[ship_key] = (d, -ship_info[CARGO])
            else: # For enemies, use L1 distance
                d = sy_dists[shipx, shipy]
                attacker_dists[ship_key] = (d, -ship_info[CARGO])

        if len(protector_dists) > 0 and len(attacker_dists) > 0:
            sorted_protector_dist = sorted(protector_dists.items(), key=lambda x: x[1])
            protector = min(protector_dists.items(), key=lambda x: x[1])
            attacker = min(attacker_dists.items(), key=lambda x: x[1])

            # If the second-closet ship is 1 move away (and closest enemy  > 1), then he probably wants to 
            # deposit, so we make it the protector (HACK TO AVOID DEADLOCK AT OWN BASE)
            if len(sorted_protector_dist) > 1 and sorted_protector_dist[0][1][0] == 0 and sorted_protector_dist[1][1][0] == 1 and sorted_protector_dist[1][1][-1] != 0 and attacker[1][0] > 1:
                protector = sorted_protector_dist[1]
            protectors[sy] = protector
            assigned_ships.add(protector[0])
            attackers[sy] = attacker

    for sy in attackers:
        # If closest is further than d=1 but closer than friendly ship
        #  with buffer), than we need to move towards our SY.
        syx, syy = sy.position.x, 20 - sy.position.y
        for ship_idx, (ship_key, ship_info)  in enumerate(ship_list):        
            if ship_key == protectors[sy][0]:
                x, y = xy(ship_info[POSITION])
                break

        if (protectors[sy][1][0] < attackers[sy][1][0] < protectors[sy][1][0] + 3) or (protectors[sy][1][0] == attackers[sy][1][0] and -protectors[sy][1][-1] <= -attackers[sy][1][-1]):
            print(f'{attackers[sy][0]} is close to {sy.id} at ({syx, syy}) (d={attackers[sy][1][0]}). Sending {protectors[sy][0]} (d={protectors[sy][1][0]})')
            acts = find_steps_to(Point(x,y), Point(syx, syy))
            print(f'Current position = ({x}, {y}), actions = {acts}')
            ship_idx = ship_id_to_idx[protectors[sy][0]]
            if len(acts) > 0:
                C[ship_idx, x + size*y] -= 20000 # not idle
            else:
                C[ship_idx, x + size*y] += 20000 # idle

            if NORTH in acts: C[ship_idx, x + size*c(y - 1)] += 20000 # NORTH
            if EAST in acts:  C[ship_idx, c(x + 1) + size*y] += 20000 # EAST
            if SOUTH in acts: C[ship_idx, x + size*c(y + 1)] += 20000 # SOUTH
            if WEST in acts:  C[ship_idx, c(x - 1) + size*y] += 20000 # WEST

    entity_idxs, assignments = scipy.optimize.linear_sum_assignment(C, maximize=True)
    
    # iterate over ships, assign them action
    assigned = dict(zip(entity_idxs, assignments))
    
    remaining_str = ""
    converting_str = ""
    moving_str = ""
    protecting_str = ""
    protecting_list = []
    moving_list = []

    if turn.last_episode: # Last step, convert laden ships to bases
        for ship_idx, (ship_key, ship_info) in enumerate(ship_list):        
            x, y = xy(ship_info[POSITION]) 

            if ship_info[CARGO] > conf.convertCost:
                actions[ship_key] = 'CONVERT'
    else:

        # Ship_converts() has already decided conversion. 
        # Need to read it back as ML bot is driving action thrugh another mechanism.
        ships_to_convert = []
        for ship in me.ships:
            if ship.next_action == CONVERT:
                ships_to_convert.append(ship.position)

        for ship_idx, (ship_key, ship_info) in enumerate(ship_list):        
            x, y = xy(ship_info[POSITION]) 

            # TEAM XY and MY XY are different: MY Y = 20 - TEAM Y
            converting = False
            for s_to_c in ships_to_convert:
                if s_to_c[0] == x and (20-s_to_c[1]) == y:
                    if (my_halite+ship_info[CARGO]) >= conf.convertCost: # just in case
                        actions[ship_key] = 'CONVERT'
                        my_halite -= conf.convertCost
                        converting_str += '{} at ({},{}), '.format(ship_key, x, y)
                        converting = True
                        turn.taken[s_to_c] = 1
                        num_converts += 1
            
            if not converting:

                xt, yt = xy(assigned[ship_idx])
                if x == xt and y == yt:
                    remaining_str += '{} at ({},{}), '.format(ship_key, x, y) 
                else:
                    moving_list.append([ship_key, x, y, xt, yt])
                    a = None
                    if   c(xt-x) == 1: a = 'EAST'
                    elif c(yt-y) == 1: a = 'SOUTH'
                    elif c(x-xt) == 1: a = 'WEST'
                    elif c(y-yt) == 1: a = 'NORTH'
                    if a is not None:
                        actions[ship_key] = a
                    else:
                        print('   says to move but where???')
                    turn.taken[move(board.ships[ship_key].position, act(a))] = 1 


    #--------------- SPAWN SECTION --------------------#
    spawning_str = ""
    for base_idx, (base_key, base_info) in enumerate(base_list):
        if assigned[len(my_ships) + base_idx] >= size * size:
            continue; # no spawn
        else:

            if my_halite >= conf.spawnCost:
                my_halite -= conf.spawnCost
                actions[base_key] = 'SPAWN'
                spawning_str += '({},{})'.format(*xy(base_info))
                turn.taken[board.shipyards[base_key].position] = 1
                #print('spawning a ship at {},{}'.format(*xy(base_info)))
            else:
                print('assigned to spawn at {}, {} but no cash on hand'.format(*xy(base_info)))
    # -------------------------------------------------------#

    #--------------- PROTECTOR SECTION --------------------#
    # # Find protectors (currently on own shipyards)
    protectors = {}
    for sy in me.shipyards:
        ship = board.cells[sy.position].ship
        if ship is not None and ship.player_id==me.id:
            protectors[sy.position] = ship
    # Go back to protect if there's no new ship coming in
    for pt, ship in protectors.items():
        if not turn.taken[pt] and ship.id in actions.keys(): 
            protecting_list.append(ship.id)
            del actions[ship.id] # stay idle to protect
    # -------------------------------------------------------#

    # Print actions
    for ship_key, x, y, xt, yt in moving_list:
        if ship_key in protecting_list:
            protecting_str += "{} at ({},{}), ".format(ship_key, x, y)
        else:
            moving_str += '{} from ({},{}) to ({},{}), '.format(ship_key, x, y, xt, yt)

    if len(remaining_str) > 0: print("Ships mining:", remaining_str)    
    if len(moving_str) > 0: print("Ships moving:", moving_str)
    if len(protecting_str) > 0: print("Ships protecting:", protecting_str)
    if len(converting_str) > 0: print("Ships converting to base:", converting_str)
    if len(spawning_str) > 0: print("Ships spawning:", spawning_str)

    print('Actions: ', actions)
    print("{:.0f} ms - total".format((perf_counter() - start_time)*1e3))
    print()

    prev_board = board

    if (perf_counter() - start_time)*1e3 > TTA_SLOW_EPOCH_TRIGGER:
        steps_since_slow_epoch = 0
    else:
        steps_since_slow_epoch += 1

    return actions
