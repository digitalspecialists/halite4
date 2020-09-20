# %%writefile submission.py
# =========== NOTE =============

#

# ============================


# =========== LOGS =============

# reduce atk yard dist
# call ship back
# aggressive.7 for balanced game. Let my ship perc be x. 
#     if x < .25: atk = x/.25 * .7
#     else: atk = .7 + (x-.25)

# --- old improvement ---
# end game protection < .5%
# guarantee spawn step < 180
# old halite_group
# guarding ships, improved. Endangered-ships fixed. Not swap guard if dangerous.
# Move positive ships, than guard, attack, than move 0-ships
# position to convert add ranking factor: dist, axis dist to oppo yard
# fix shipAttackShipyard
# select shipyard to spawn
# allow swap when shipyard spawn (adjust at last check)
# fix bug tryRescue tells ships to crash with new-born ships
# < ATTACK_NEAR_YARD

# ============================

# Imports helper functions
import time
import math
import random
import itertools
from collections import defaultdict, deque
import heapq
import inspect
import copy
import functools 

from kaggle_environments.envs.halite.helpers import *
import numpy as np
from scipy import stats

MAX_HALITE_ATTACK_SHIPYARD = 0
THRESHOLD_STORE = None
DECADE_THRESHOLD_STORE = None
DISTANCE_TO_ATTACK = None
REST_AFTER_ATTACK = 0
ATTACK_NEAR_YARD = None
ATTACK_STEPS_NEAR_YARD = 1

ship_states = {} # ship.id => ShipState obj
shipyard_states = {}
frey_states = {}

last_attack_time = {}
attacked_freys = {}

ship_actions = {}
shipyard_actions = {}

eligible_for_rescue = None # set of not-asking-for-rescue ships

threatening_wealthy_yards = defaultdict(lambda : [])
oppo_yard_not_resisting = set()

oppo_ships_near_shipyard = defaultdict(lambda: 0)

cnt_attack_success, cnt_attack_failure = 0, 0

CENTER = Point(0, 0)
ADD_VICINITY = 0

class ShipState:
    MINE = 0
    STORE = 1
    ATTACK_SHIP = 2
    ATTACK_YARD = 3
    CONVERT = 4
    PROTECT = 5
    GUARD = 6

    def __init__(self, state, target, timeout):
        self.state = state
        self.target = target
        self.timeout = timeout
        self.times = 0

class ShipyardState:
    STAY = 101
    SPAWN = ShipyardAction.SPAWN

    def __init__(self, state):
        self.state = state

class FreyState:
    HALITE_0_NEAR_YARD_LONG = 1001
    HALITE_POSITIVE_NEAR_YARD = 1002
    HALITE_0_RESTRICTED_MOVE = 1003
    HALITE_POSITIVE_RESTRICTED_MOVE = 1004
    HALITE_0_IN_VICINITY = 1005
    HALITE_POSITIVE_IN_VICINITY = 1006
    HALITE_0_OUT_VICINITY = 1007
    HALITE_POSITIVE_OUT_VICINITY = 1008

    @staticmethod
    def getSlots(frey_state):
        if (frey_state == FreyState.HALITE_0_NEAR_YARD_LONG or
            frey_state == FreyState.HALITE_0_IN_VICINITY):
            return 1
        if (frey_state == FreyState.HALITE_POSITIVE_NEAR_YARD or
            frey_state == FreyState.HALITE_POSITIVE_RESTRICTED_MOVE or 
            frey_state == FreyState.HALITE_POSITIVE_OUT_VICINITY or
            frey_state == FreyState.HALITE_POSITIVE_IN_VICINITY
           ):
            return 4
        
        return 0

# ==
class MyHeap(object):
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        self.index = 0
        if initial:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []
  
    def __len__(self):
        return self._data.__len__()
  
    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1
  
    def pop(self):
        return heapq.heappop(self._data)[2]


def isThereNoShip(me):
    return(len(me.ships) == 0)

def noStoreShipAround(cell, me):
    if (cell.north.ship is not None and 
        cell.north.ship.player_id == me.id and 
        ship_states[cell.north.ship.id].state == ShipState.STORE):
        return False
    if (cell.east.ship is not None and 
        cell.east.ship.player_id == me.id and 
        ship_states[cell.east.ship.id].state == ShipState.STORE):
        return False
    if (cell.south.ship is not None and 
        cell.south.ship.player_id == me.id and 
        ship_states[cell.south.ship.id].state == ShipState.STORE):
        return False
    if (cell.west.ship is not None and 
        cell.west.ship.player_id == me.id and 
        ship_states[cell.west.ship.id].state == ShipState.STORE):
        return False

    return True

def getDistance(pos1, pos2, size=21):
    x1, y1 = pos1
    x2, y2 = pos2

    delta_x, delta_y = abs(x1 - x2), abs(y1 - y2)
    
    dist_x = min(delta_x, size - delta_x)
    dist_y = min(delta_y, size - delta_y)
    
    return dist_x + dist_y

def getAxisDistance(pos1, pos2, size=21):
    diff_x = abs(pos1.x - pos2.x)
    diff_y = abs(pos1.y - pos2.y)

    return max(min(diff_x, size-diff_x), min(diff_y, size-diff_y))

def getDirection(source, dest, size=21):
    x1, y1 = source
    x2, y2 = dest    
    
    if x1 != x2:
        if ((x1 < x2) and (x2 - x1 < (size-1)//2)) or ((x1 > x2) and (x1 - x2 > (size-1)//2)):
            x_dirs = [ShipAction.EAST, ShipAction.WEST]
        else:
            x_dirs = [ShipAction.WEST, ShipAction.EAST]
    else:
        x_dirs = [ShipAction.EAST, ShipAction.WEST]
    
    if y1 != y2:
        if ((y1 < y2) and (y2 - y1 < (size-1)//2)) or ((y1 > y2) and (y1 - y2 > (size-1)//2)):
            y_dirs = [ShipAction.NORTH, ShipAction.SOUTH]
        else:
            y_dirs = [ShipAction.SOUTH, ShipAction.NORTH]
    else:
        y_dirs = [ShipAction.SOUTH, ShipAction.NORTH]
            
    delta_x, delta_y = abs(x1 - x2), abs(y1 - y2)

    if delta_x > delta_y:
        dirs = [x_dirs[0], y_dirs[0], y_dirs[1], x_dirs[1]]
    else:
        dirs = [y_dirs[0], x_dirs[0], x_dirs[1], y_dirs[1]]

    if source == dest:
        dirs = [None] + dirs
    else:
        if board_info.getHaliteAtPos(source) == 0:
            if delta_x == 0 or delta_y == 0:
                dirs = dirs[:1] + [None] + dirs[1:]
            else:
                dirs = dirs[:2] + [None] + dirs[2:]
        else:
            dirs = dirs + [None]

    return dirs

def getLocationByMove(cell, action):
    if action == ShipAction.NORTH:
        return cell.north.position
    elif action == ShipAction.EAST:
        return cell.east.position
    elif action == ShipAction.SOUTH:
        return cell.south.position
    elif action == ShipAction.WEST:
        return cell.west.position
    else:
        return cell.position

def getCellByMove(cell, action):
    if action == ShipAction.NORTH:
        return cell.north
    elif action == ShipAction.EAST:
        return cell.east
    elif action == ShipAction.SOUTH:
        return cell.south
    elif action == ShipAction.WEST:
        return cell.west
    else:
        return cell

def getCellDirections(cell, include_stay=False):
    directions = [
        (ShipAction.NORTH, cell.north), 
        (ShipAction.EAST, cell.east),
        (ShipAction.SOUTH, cell.south),
        (ShipAction.WEST, cell.west)
    ]

    if include_stay:
        directions += [(None, cell)]

    return directions

def _model_coeffs(step):
    vals={300: {'mean': np.array([3.6254e+03, 1.1610e+03, 2.3787e+00, 2.1533e+01, 1.3253e+01,
       8.6134e+01, 3.0877e+04]), 'scale': np.array([5.1599e+03, 1.3868e+03, 1.5617e+00, 1.3994e+01, 9.7139e+00,
       2.2553e+01, 1.5546e+04]), 'coeff': np.array([ 5952.9156,  3356.4129,  2714.7983,   313.886 ,  3688.9617,
       -2103.4456,  3191.3629]), 'intercept': 13267.990462733907}, 320: {'mean': np.array([5.4567e+03, 1.2589e+03, 2.3860e+00, 2.0602e+01, 1.1982e+01,
       8.2408e+01, 2.9674e+04]), 'scale': np.array([6.4673e+03, 1.5185e+03, 1.6003e+00, 1.4164e+01, 9.2298e+00,
       2.2023e+01, 1.4537e+04]), 'coeff': np.array([ 7166.4883,  3070.6126,  2481.5619,   311.3993,  2327.4121,
       -1425.5229,  2393.8341]), 'intercept': 13267.990462733842}, 340: {'mean': np.array([7.5512e+03, 1.3258e+03, 2.3740e+00, 1.9678e+01, 1.0695e+01,
       7.8710e+01, 2.7696e+04]), 'scale': np.array([8.0928e+03, 1.5882e+03, 1.6153e+00, 1.4240e+01, 8.6868e+00,
       2.1438e+01, 1.3466e+04]), 'coeff': np.array([8413.3252, 2606.9503, 2032.3257,  106.0128, 1566.67  , -871.4708,
       1598.2906]), 'intercept': 13267.990462733971}, 360: {'mean': np.array([9.9747e+03, 1.4076e+03, 2.3151e+00, 1.8293e+01, 9.1359e+00,
       7.3172e+01, 2.4319e+04]), 'scale': np.array([9.9084e+03, 1.6423e+03, 1.6201e+00, 1.3820e+01, 7.6969e+00,
       2.0205e+01, 1.2316e+04]), 'coeff': np.array([9707.964 , 2353.1533, 1140.9455,  252.2888,  218.9249, -110.0682,
        788.575 ]), 'intercept': 13267.990462734095}, 380: {'mean': np.array([1.2159e+04, 1.1018e+03, 2.1617e+00, 1.4698e+01, 7.1979e+00,
       5.8792e+01, 2.2760e+04]), 'scale': np.array([1.1565e+04, 1.4165e+03, 1.6217e+00, 1.1620e+01, 6.3689e+00,
       1.8308e+01, 1.1726e+04]), 'coeff': np.array([ 1.1479e+04,  1.2684e+03,  3.2165e+02, -3.5256e+00,  2.7987e+01,
        1.8811e+02,  6.7236e+01]), 'intercept': 13267.990462733987}}
    ret=None
    for key in sorted(vals.keys(),reverse=True):
        if step >= key:
            ret=vals[key]
            # print('winner using step key {}'.format(key))
            break
    if ret is None:
        return None
    return ret['coeff'], ret['intercept'], ret['mean'], ret['scale']

def final_halite(board):
    #returns array of 4 players predicted halite
    ret=[]
    x=_model_coeffs(board.step)
    if x is None:
        return [0,0,0,0]
    coeff,intercept,mean,scale=x
    for player_id in range(4):
        step=board.step-300
        player=board.players[player_id]
        #compute quantities
        halite=player.halite
        cargo=sum([ship.halite for ship in player.ships])
        num_shipyards=len(player.shipyards)
        num_ships=len(player.ships)
        num_zships=sum([1 if ship.halite==0 else 0 for ship in player.ships])
        total_num_ships=len(board.ships)
        #halite_step=halite*step
        #cargo_step=cargo*step
        #num_ships_step=num_ships*step
        total_board_halite=np.sum(board.observation['halite'])
        #total_board_halite_step=total_board_halite*step
        data=np.array((halite,cargo,
                                     num_shipyards,
                                     num_ships,
                                     num_zships,
                                     total_num_ships,
                                     total_board_halite))
        v=((data-mean)/scale).dot(coeff)+intercept
        ret.append(v)
    # printred('final halite prediction: {}'.format(ret))
    return ret

class BoardInfo:
    
    def __init__(self, obs, config):
        self.size = config.size
        self.board = Board(obs, config)
        self.me = self.board.current_player

        
        self.initValues()

        self.recordMyStatus()
        self.recordOppoStatus()
        self.recordHaliteMatrix()
        self.recordCellMatrix()
        self.calDistanceToShipyard()
        self.calDistanceToOpponentShip()
        self.calDistanceToOpponentShipyard()

        # self.calOppoShipFutureHalite()
        # self.calDistanceToOpponentShipFuture()
        # ----        
        self.evolutionInit()
        
        # if (self.board.step % 99 == 0):
        #     print('{}: my worth: {}'.format(self.board.step, self.my_approx_worth))
        #     print('{}: oppo worth: {}'.format(self.board.step, self.oppo_approx_worth))

        self.prepareShipHaliteGroup()
    #---------------- pre-record and calculation ----------------

    def initValues(self):
        # halite_distance_decay
        if self.board.step <= 10:
            self.halite_distance_decay = np.ones(self.size)
        else:
            self.halite_distance_decay = np.array([(.85**i) for i in range(self.size)])

        #
        self.MAX_OPPO_SHIP_DIST = 4
        
    def recordMyStatus(self):
        # halite
        self.my_halite = self.me.halite
        #ships
        self.my_ships = [ship for ship in self.me.ships]
        self.my_ship_positions = defaultdict(lambda: 0)
        for ship in self.my_ships:
            self.my_ship_positions[ship.position] += 1
        self.num_my_ships = len(self.my_ships)
        # shipyards
        self.my_shipyards = [shipyard for shipyard in self.me.shipyards]
        self.my_shipyard_positions = set([shipyard.position for shipyard in self.my_shipyards])
        self.num_my_shipyards = len(self.my_shipyards)
        # cargo
        self.my_cargo = sum([ship.halite for ship in self.my_ships])

        if self.board.step % 20 == 0:
            print('{}: player ships {}'.format(self.board.step, [len(player.ships) for player in self.board.players.values()]))
            print('{}: player halite {}'.format(self.board.step, [player.halite for player in self.board.players.values()]))
            print('{}: player cargo {}'.format(self.board.step, [sum([ship.halite for ship in player.ships]) for player in self.board.players.values()]))
    
    def recordOppoStatus(self):
        # halite
        self.oppo_halite = {oppo.id : oppo.halite for oppo in self.board.opponents}
        self.max_oppo_halite = max(self.oppo_halite.values(), default=0)
        # ships
        self.num_oppo_ships = {oppo.id : len(oppo.ships) for oppo in self.board.opponents}
        self.max_num_oppo_ships = max(self.num_oppo_ships.values(), default=0)
        # shipyards
        self.num_oppo_shipyards = {oppo.id : len(oppo.shipyards) for oppo in self.board.opponents}
        # cargo
        self.oppo_cargo = {oppo.id : sum([ship.halite for ship in oppo.ships]) for oppo in self.board.opponents}
        # halite + cargo
        self.oppo_halite_and_cargo = {oppo.id : (sum([ship.halite for ship in oppo.ships]) + oppo.halite) for oppo in self.board.opponents}
        self.max_oppo_halite_and_cargo = max(self.oppo_halite_and_cargo.values(), default=0)
        # approx worth
        if self.board.step >= 300:
            self.player_approx_worth = final_halite(self.board)
        else:
            self.player_approx_worth = {
                oppo.id : self._approxWorth(
                    self.oppo_halite[oppo.id],
                    self.num_oppo_ships[oppo.id],
                    self.num_oppo_shipyards[oppo.id],
                    self.oppo_cargo[oppo.id]
                )
                for oppo in self.board.opponents
            }
            self.player_approx_worth.update({
                self.me.id : self._approxWorth(
                    self.my_halite, 
                    self.num_my_ships, 
                    self.num_my_shipyards, 
                    self.my_cargo
                )
            })
        # eliminated
        self.oppo_eliminated = {oppo.id : (self.num_oppo_ships[oppo.id] == 0 and
                                                                ( self.oppo_halite[oppo.id] < 500 or self.num_oppo_shipyards[oppo.id] == 0))
                                                for oppo in self.board.opponents}

    #---------------- initialization -------------------------------

    def recordHaliteMatrix(self):
        self.halite_matrix = np.zeros((self.size, self.size))
        for pos, cell in self.board.cells.items():
            x, y = pos.x, pos.y
            self.halite_matrix[x, y] = cell.halite

        self.board_halite = self.halite_matrix.sum()
        self.num_positive_halite_cell = np.sum(self.halite_matrix > 0)

        if self.board.step == 0:
            print('number of positive halite cells:', np.sum(self.halite_matrix > 0))

        if self.board.step % 20 == 0:
            print('{}: board halite: {}'.format(self.board.step, self.board_halite))
            print('{}: board halite avg: {}'.format(self.board.step, self.board_halite/max(1, self.num_positive_halite_cell)))
            print('{}: around halite avg: {}'.format(self.board.step, self.getAvgCellHaliteAround()))

    def recordCellMatrix(self):
        self.cell = [[None for __ in range(self.size)] for _ in range(self.size)]
        for pos, cell in self.board.cells.items():
            x, y = pos
            self.cell[x][y] = cell

    def calDistanceToOpponentShip(self):
        self.dist_to_oppo_ship = np.full((self.size, self.size), self.size)
        self.oppo_ship_halite_in = np.full((self.size, self.size, self.MAX_OPPO_SHIP_DIST+1), 1e9)

        oppo_ships = [ship for oppo in self.board.opponents for ship in oppo.ships]

        for x in range(self.size):
            for y in range(self.size):
                point = Point(x, y)
                dist_n_halite = [(getDistance(point, ship.position), ship.halite) for ship in oppo_ships]

                if len(dist_n_halite):
                    dist_list, _ = zip(*dist_n_halite)
                    self.dist_to_oppo_ship[x, y] = min(dist_list)

                for dist, halite in dist_n_halite:
                    for d in range(self.MAX_OPPO_SHIP_DIST, -1, -1):
                        if dist > d:
                            break

                        if halite < self.oppo_ship_halite_in[x, y, d]:
                            self.oppo_ship_halite_in[x, y, d] = halite
    
    def calDistanceToOpponentShipyard(self):
        self.dist_to_oppo_shipyard = np.full((self.size, self.size), self.size)

        oppo_shipyards = [shipyard for oppo in self.board.opponents for shipyard in oppo.shipyards]

        for x in range(self.size):
            for y in range(self.size):
                point = Point(x, y)
                min_distance = min([getDistance(point, yard.position, self.size) for yard in oppo_shipyards], default=1000)
                self.dist_to_oppo_shipyard[x, y] = min_distance
    
    def calDistanceToShipyard(self):
        self.dist_to_shipyard = np.full((self.size, self.size), self.size)
        self.axis_dist_to_shipyard = np.full((self.size, self.size), self.size)
        shipyard_poss = [pos for pos in self.my_shipyard_positions]
        
        for x in range(self.size):
            for y in range(self.size):
                point = Point(x, y)
                min_distance = min([getDistance(point, yard_pos) for yard_pos in shipyard_poss], default=self.size)
                self.dist_to_shipyard[x, y] = min_distance

                min_axis_distance = min([getAxisDistance(point, yard_pos) for yard_pos in shipyard_poss], default=10)
                self.axis_dist_to_shipyard[x, y] = min_axis_distance
    
    #---------------- get halite -------------------------------

    def getMyHalite(self):
        return self.my_halite

    def getMyHaliteAndCargo(self):
        return self.my_halite + self.my_cargo

    def getMaxOppoHalite(self):
        return self.max_oppo_halite

    def getMaxOppoHaliteAndCargo(self):
        return self.max_oppo_halite_and_cargo
    
    def getBoardHalite(self):
        return self.board_halite

    def getMinHaliteMine(self):
        if not hasattr(self, 'min_halite_mine'):
            step = self.board.step

            if self.num_positive_halite_cell > 0:
                avg = self.board_halite / self.num_positive_halite_cell
            else:
                avg = 0

            if step < 50:
                self.min_halite_mine = 0
            elif step < 350:
                self.min_halite_mine = min(210, avg)
            else:
                self.min_halite_mine = avg/2

            if (board_info.getNumMyShips() > board_info.getMaxOppoNumShips() and
                board_info.getNumMyShips() >= board_info.getBoardNumShips()*.3 and
                board_info.getNumMyShips() >= 12 and
                step < 300
               ):
                self.min_halite_mine = max(self.min_halite_mine, min(100, self.min_halite_mine+20))

        return self.min_halite_mine

    def getHaliteAtPos(self, pos):
        return self.halite_matrix[pos.x, pos.y]

    def getCargo(self, oppo_id=None):
        if oppo_id is None:
            return self.my_cargo
        else:
            return self.oppo_cargo[oppo_id]

    def getWorthRank(self):
        return 1 + len([x for x in self.player_approx_worth if x > self.player_approx_worth[self.me.id]])
    
    def getApproxWorth(self, player_id=None): # player_id == None means myself
        if player_id is None:
            player_id = self.me.id

        return self.player_approx_worth[player_id]

    def _approxWorth(self, halite, nship, nyard, cargo):
        if self.board.step < 200:
            worth = halite + nship*500 + nyard*1000 + cargo*0.8
        elif self.board.step < 300:
            worth = halite + nship*400 + nyard*400 + cargo*0.8
        elif self.board.step < 370:
            worth = halite + nship*100 + nyard*100 + cargo*0.9
        else:
            worth = halite + cargo*0.9
        return worth

    def isEliminated(self, oppo_id):
        return self.oppo_eliminated[oppo_id]

    #---------------- get ship -------------------------------

    def getNumMyShips(self):
        return self.num_my_ships
    
    def getNumShipyards(self):
        return self.num_my_shipyards

    def getBoardNumShips(self):
        return self.num_my_ships + sum(self.num_oppo_ships.values())

    def getMaxOppoNumShips(self):
        return self.max_num_oppo_ships

    def getDistToShipyard(self, pos):
        return self.dist_to_shipyard[pos.x, pos.y]

    def getAxisDistToShipyard(self, pos):
        return self.axis_dist_to_shipyard[pos.x, pos.y]
    
    def getNearestShipyardPos(self, pos):
        yards = [(yard_pos, getDistance(pos, yard_pos, self.size)) 
                        for yard_pos in self.my_shipyard_positions]
        yards = sorted(yards, key=lambda x: x[1])
        if len(yards) > 0:
            return yards[0][0]
        else:
            return CENTER

    def getDistToOpponentShip(self, pos, my_ship_halite):
        if self.dist_to_oppo_ship[pos.x, pos.y] > self.MAX_OPPO_SHIP_DIST:
            return self.dist_to_oppo_ship[pos.x, pos.y]

        for d in range(self.MAX_OPPO_SHIP_DIST, -1, -1):
            if self.oppo_ship_halite_in[pos.x, pos.y, d] > my_ship_halite:
                return d+1

        return 0
    
    def getDistToOpponentShipyard(self, pos):
        return self.dist_to_oppo_shipyard[pos.x, pos.y]

    def getAxisDistToOpponentShipyard(self, pos):
        return min([
            getAxisDistance(pos, oppo_yard.position) 
                for oppo in self.board.opponents for oppo_yard in oppo.shipyards
            ], 
            default=self.size
        )

    def getOppoShipyardsAround(self, ship):
        oppo_shipyards = [yard for oppo in self.board.opponents for yard in oppo.shipyards]
        oppo_shipyards = sorted(oppo_shipyards, key=lambda x: getDistance(x.position, ship.position))
        return oppo_shipyards

    #---------------- get position -------------------------------

    def getRealSafePathAndDistance(self, source_cell, dest, my_ship_halite, max_dist):
        assert max_dist > 0
        source = source_cell.position

        # some early-return cases
        if not self.isSafeMoveTo(dest, source, my_ship_halite):
            return None, max_dist + 1
        if source == dest:
            return None, 0
        # init
        traversed = np.full((self.size, self.size), False)
        traversed[source.x, source.y] = True
        queue = deque()
        # push the first cells to queue
        for direc in getDirection(source_cell.position, dest):
            next_cell = getCellByMove(source_cell, direc)
            next_step = next_cell.position
            if self.isSafeMoveTo(next_step, source, my_ship_halite):
                if next_step == dest:
                    return direc, 1
                traversed[next_step.x, next_step.y] = True
                queue.append((next_cell, 1, direc))
        # flood to find the way
        while len(queue) > 0:
            c, dist, direc = queue.popleft()
            for d in getDirection(c.position, dest):
                next_cell = getCellByMove(c, d)
                next_step = next_cell.position
                if (self.isSafeMoveTo(next_step, c.position, my_ship_halite) and 
                    not traversed[next_step.x, next_step.y]
                    ):
                    if next_step == dest:
                        return direc, dist + 1
                    traversed[next_step.x, next_step.y] = True
                    
                    if dist + 1 < max_dist:
                        queue.append((next_cell, dist + 1, direc))

        return None, max_dist + 1

    def getMaxHarvestOneAttempt(self, ship, num_moves):
        if num_moves < self.getDistToShipyard(ship.position):
            return 0

        max_harvest = 0

        for x in range(self.size):
            for y in range(self.size):
                p = Point(x, y)
                d_ship = getDistance(p, ship.position)
                d_yard = self.getDistToShipyard(p)
                mine_time = max(0, num_moves - d_ship + d_yard)
                mine_halite = self.getHaliteAtPos(p) * (1. - .75**mine_time)
                if mine_halite > max_harvest:
                    max_harvest = mine_halite

        return max_harvest

    def myShipIn(self, pos):
        return self.my_ship_positions[pos]

    def myShipyardIn(self, pos):
        return pos in self.my_shipyard_positions

    # ### This is the faulty version but get high LB ???
    # def isPositionSafe(self, pos, my_ship_halite, safe_level=1):
    #     if  self.getDistToOpponentShipyard(pos) < safe_level:
    #         return False

    #     if safe_level == 1:
    #         for _, next_cell in getCellDirections(self.cell[pos.x][pos.y]):
    #             if next_cell.ship is not None and next_cell.ship.player_id == self.me.id:
    #                 if next_cell.ship.halite < my_ship_halite:
    #                     my_ship_halite = next_cell.ship.halite

    #     if self.getDistToOpponentShip(pos, my_ship_halite) <= safe_level:
    #         return False

    #     return True

    ### This is the fixed version
    def isPositionSafe(self, pos, my_ship_halite, safe_level=1):
        if  self.getDistToOpponentShipyard(pos) < safe_level:
            return False

        if self.getDistToOpponentShip(pos, my_ship_halite) > safe_level:
            return True

        if safe_level == 1:
            ally_halite = 1e6
            for _, next_cell in getCellDirections(self.cell[pos.x][pos.y]):
                if next_cell.ship is not None:
                    if next_cell.ship.player_id == self.me.id:
                        if next_cell.ship.halite < ally_halite:
                            ally_halite = next_cell.ship.halite
                    else:
                        if (next_cell.ship.halite <= my_ship_halite and 
                            len(board_info.getPredictedOppoMoves(next_cell.ship)) == 0
                           ):
                            return False

            if self.getDistToOpponentShip(pos, ally_halite) > safe_level:
                return True

        return False

    def isSafeMoveTo(self, pos, curr_pos, my_ship_halite, safe_level=1): # move to this pos causes no penalty
        # if (self.board.step == 34 and pos == Point(9, 12)):
            # print('Is Safe:', self.myShipIn(pos), self.getDistToShipyard(pos), self.isPositionSafe(pos, my_ship_halite, safe_level))

        if curr_pos != pos and self.myShipIn(pos) >= 1:
            return False

        if curr_pos == pos and self.myShipIn(pos) >= 2:
            return False

        if self.getDistToShipyard(pos) == 0 and my_ship_halite == 0:
            return True

        return self.isPositionSafe(pos, my_ship_halite, safe_level)

    def isSafeSwap(self, ship1_pos, ship2_pos, ship1_halite, ship2_halite, ship1, ship2, safe_level=1):
        if (self.myShipIn(ship1_pos) == 1 and 
            self.myShipIn(ship2_pos) == 1 and
            self.isPositionSafe(ship1_pos, ship2_halite, safe_level) and
            self.isPositionSafe(ship2_pos, ship1_halite, safe_level)
           ):
            return True
        # swap at yard with protector
        if (self.myShipIn(ship1_pos) == 1 and 
            self.myShipIn(ship2_pos) == 1 and            
            self.isPositionSafe(ship1_pos, ship2_halite, safe_level) and
            ship_states[ship1.id].state == ShipState.STORE and
            ship2.cell.shipyard is not None and
            ship2_halite == 0 and
            ship2.id in ship_actions and
            ship_actions[ship2.id] is None and
            self.getDistToOpponentShip(ship2.position, ship1.halite-1) > 1
           ):
            return True

        # swap with GUARD
        # if (self.myShipIn(ship1_pos) == 1 and 
        #     self.myShipIn(ship2_pos) == 1 and
        #     ( self.isPositionSafe(ship1_pos, ship2_halite, safe_level) or ship_states[ship2.id].state == ShipState.GUARD) and
        #     ( self.isPositionSafe(ship2_pos, ship1_halite, safe_level) or ship_states[ship1.id].state == ShipState.GUARD)
        #    ):
        #     return True

        return False

    #---------------- set functions -------------------------------

    def changeMyHalite(self, val):
        self.my_halite += val

    def addShip(self, ship_position):
        self.my_ship_positions[ship_position] += 1
        self.num_my_ships += 1

    def addShipyard(self, pos):
        self.my_shipyard_positions.add(pos)
        self.num_my_shipyards += 1
        self.calDistanceToShipyard()
    
    def shipMove(self, ship, new_pos):
        self.my_ship_positions[ship.position] -= 1
        self.my_ship_positions[new_pos] += 1

    #---------------- evolution -------------------------------

    def evolutionInit(self):
        # 
        global CENTER
        if len(self.me.shipyards):
            CENTER = functools.reduce(lambda a,b: a+b, [shipyard.position for shipyard in self.me.shipyards]) *(1./len(self.me.shipyards))
            CENTER = Point(round(CENTER.x), round(CENTER.y))
            # CENTER = self.me.shipyards[0].position
        # elif len(self.me.ships):
            # CENTER = self.me.ships[0].position
        #
        global ADD_VICINITY
        if ADD_VICINITY > self.getNumMyShips() // 10:
            ADD_VICINITY = self.getNumMyShips() // 10

        self.vicinity = math.ceil(((self.getNumMyShips()/max(1, self.getBoardNumShips())*400*1.5)**.5)/2)
        self.vicinity += ADD_VICINITY
        self.manhattan_vicinity = math.ceil(self.vicinity * 1.7)
        self.vicinity = max(min(self.vicinity, 10), 1)

        # record ATTACK_SHIP ships
        self.num_attack_ship = 0

        # record which opponents do not protect their shipyards if being threatened.
        global threatening_wealthy_yards, oppo_yard_not_resisting
        for yard_pos in threatening_wealthy_yards:
            x, y = yard_pos
            attackers = threatening_wealthy_yards[yard_pos]

            if (self.cell[x][y].shipyard is not None and
                self.cell[x][y].ship is None and
                all([attacker in [(player.id, ship.id) for player in self.board.players.values() for ship in player.ships] for attacker in attackers])
               ):
                if self.cell[x][y].shipyard.player_id not in oppo_yard_not_resisting:
                    oppo_yard_not_resisting.add(self.cell[x][y].shipyard.player_id)
                    print('{}: oppo {} at {} not resisting'.format(self.board.step, self.cell[x][y].shipyard.player_id, yard_pos))
        
        threatening_wealthy_yards.clear()
        for oppo_yard in [oppo_yard for oppo in self.board.opponents for oppo_yard in oppo.shipyards]:
            if oppo_yard.player.halite >= 500:
                for _, next_cell in getCellDirections(oppo_yard.cell):
                    if next_cell.ship is not None and next_cell.ship.player_id != oppo_yard.player_id:
                        threatening_wealthy_yards[oppo_yard.position].append((next_cell.ship.player_id, next_cell.ship.id))

        # record oppo ships near my shipyard
        # only attack oppo ships that are close to my shipyard for at least n consecutive steps
        global ATTACK_NEAR_YARD
        ATTACK_NEAR_YARD = int(self.getNumMyShips()**.5)
        global oppo_ships_near_shipyard
        nearby_oppo_ships = set([(oppo.id, ship.id) 
                                                    for oppo in self.board.opponents 
                                                    for ship in oppo.ships 
                                                    if self.getDistToShipyard(ship.position) < ATTACK_NEAR_YARD])
        list_oppo_ship_ids = list(oppo_ships_near_shipyard.keys())
        for ship_id in list_oppo_ship_ids:
            if ship_id not in nearby_oppo_ships:
                del oppo_ships_near_shipyard[ship_id]
        for ship_id in nearby_oppo_ships:
            oppo_ships_near_shipyard[ship_id] += 1

        # adjust THRESHOLD_STORE
        global THRESHOLD_STORE
        step = self.board.step

        if step < 40:
            # THRESHOLD_STORE = [200, 225, 250]
            THRESHOLD_STORE = [400, 500, 600]
        elif step < 100:
            THRESHOLD_STORE = [200, 300, 400]
        else:
            my_ship_perc = self.getNumMyShips() / max(1, self.getBoardNumShips())
            THRESHOLD_STORE = np.array([600, 800, 1000]) * my_ship_perc

        if self.getNumShipyards() == 0:
            for i in range(3):
                THRESHOLD_STORE[i] = max(THRESHOLD_STORE[i], 500 - self.me.halite)

        if self.board.step % 10 == 0:
            global DECADE_THRESHOLD_STORE
            DECADE_THRESHOLD_STORE = THRESHOLD_STORE

        #
        self.predict_oppo_moves = {}

    def prepareShipHaliteGroup(self):
        num_groups = 5

        self.ship_group_by_halite_threshold = []
        sorted_ship_halite = sorted([ship.halite for ship in self.me.ships])
        groups = np.array_split(sorted_ship_halite, num_groups)
        for  group in (groups):
            if len(group):
                self.ship_group_by_halite_threshold.append(group[-1])

        # num_groups = max(2, num_groups)
        # self.ship_group_by_halite_threshold = []
        # for i in range(num_groups):
        #     self.ship_group_by_halite_threshold.append(i/(num_groups-1)*DECADE_THRESHOLD_STORE[1])

        BASE_DECAY = 5
        factor = BASE_DECAY**(1. / self.vicinity)
        increasing_decay = [(factor**i) for i in range(self.vicinity+1)]
        decresing_decay = increasing_decay[-2::-1]

        self.decay_by_ship_halite_group = []
        for i in range(num_groups):
            if self.board.step < 350:
                perc = i / (num_groups-1)
                n = round(perc*self.vicinity)
                self.decay_by_ship_halite_group.append(
                    increasing_decay[n:] + decresing_decay[:n]
                )
            else:
                self.decay_by_ship_halite_group.append([1])

        if self.board.step % 20 == 0:
            if self.board.step == 0:
                print('{}: decay_by_ship_halite_group'.format(self.board.step))
                for each in self.decay_by_ship_halite_group:
                    print(each)
            print(self.board.step, 'group threshold:', self.ship_group_by_halite_threshold)
            print(self.board.step, 'ship halite:', sorted([ship.halite for ship in self.me.ships]))

    def needMoreAttackShip(self):
        my_ships_perc = self.getNumMyShips()/max(1, self.getBoardNumShips())
        if my_ships_perc > .25:
            up_atk = .7 + (my_ships_perc - .25)
        else:
            up_atk = .7 * (my_ships_perc/.25)

        if self.board.step < 50:
            attack_perc = 0
        elif self.board.step < 320:
            if ((self.board.step - 50) // 55) % 2 == 0:
                attack_perc = up_atk
            else:
                attack_perc = 0.3
        elif self.board.step < 370:
            attack_perc = 0.2
        else:
            attack_perc = 0

        return self.num_attack_ship < self.getNumMyShips()*attack_perc

    def addAttackShip(self, ship):
        self.num_attack_ship += 1

    def getShipHalitePercRank(self, my_ship_halite): # return 1. for least halite, 0.1 for most halite
        if len(self.ship_group_by_halite_threshold) <= 1:
            return 1

        if self.board.step < 75: 
            return 1

        group_id = next((i for i, val in enumerate(self.ship_group_by_halite_threshold) 
                                                   if my_ship_halite <= val), 
                                   len(self.ship_group_by_halite_threshold)-1)
        # group_id == 0 means ship has least halite
        return 1. - (group_id/(len(self.ship_group_by_halite_threshold)-1))*.9

    def getDecayByShipHaliteGroup(self, my_ship_halite, axis_dist_to_center):
        if self.board.step < 75:
            return 1

        group_id = next((i for i, val in enumerate(self.ship_group_by_halite_threshold) 
                                                   if my_ship_halite <= val), 
                                   len(self.ship_group_by_halite_threshold)-1)

        axis_dist_to_center = min(axis_dist_to_center, 
                                                  len(self.decay_by_ship_halite_group[group_id])-1)

        return self.decay_by_ship_halite_group[group_id][axis_dist_to_center]        

    def calOppoShipFutureHalite(self):
        oppo_ships = [ship for oppo in self.board.opponents for ship in oppo.ships]
        self.MAX_FUTURE = 3
        self.oppo_future_halite = []
        for f in range(self.MAX_FUTURE+1):
            self.oppo_future_halite.append([
                (ship.position, ship.halite+ship.cell.halite*(1-.75**f)) for ship in oppo_ships
            ])

    def calDistanceToOpponentShipFuture(self):
        self.dist_to_oppo_ship_future = np.full((self.MAX_FUTURE+1, self.size, self.size), self.size)
        self.oppo_ship_halite_in_future = np.full((self.MAX_FUTURE+1, self.size, self.size, self.MAX_OPPO_SHIP_DIST+1), 1e9)

        for f in range(self.MAX_FUTURE+1):
            for x in range(self.size):
                for y in range(self.size):
                    point = Point(x, y)
                    dist_n_halite = [(getDistance(point, position), halite) for position, halite in self.oppo_future_halite[f]]

                    if len(dist_n_halite):
                        dist_list, _ = zip(*dist_n_halite)
                        self.dist_to_oppo_ship_future[f, x, y] = min(dist_list)

                    for dist, halite in dist_n_halite:
                        for d in range(self.MAX_OPPO_SHIP_DIST, -1, -1):
                            if dist > d:
                                break

                            if halite < self.oppo_ship_halite_in_future[f, x, y, d]:
                                self.oppo_ship_halite_in_future[f, x, y, d] = halite

    def distToShipDecayFactor(self, my_ship_halite, dist_to_ship):
        return self.halite_distance_decay[dist_to_ship]

    def distToShipyardDecayFactor(self, my_ship_halite, axis_dist_to_shipyard):
        if self.getBoardHalite() > 12000:
            group_id = math.ceil(my_ship_halite / 40)
        elif self.getBoardHalite() > 6000:
            group_id = math.ceil(my_ship_halite / 30)
        elif self.getBoardHalite() > 2500:
            group_id = math.ceil(my_ship_halite / 20)
        else:
            group_id = math.ceil(my_ship_halite / 10)

        group_id = min(group_id, len(self.ship_dist_to_shipyard_decay)-1)

        return self.ship_dist_to_shipyard_decay[group_id][axis_dist_to_shipyard]

    def getDistToOpponentShipFuture(self, pos, my_ship_halite, my_ship_pos):
        f = min(self.MAX_FUTURE, max(0, getDistance(my_ship_pos, pos)-1))

        if self.dist_to_oppo_ship_future[f, pos.x, pos.y] > self.MAX_OPPO_SHIP_DIST:
            return self.dist_to_oppo_ship_future[f, pos.x, pos.y]

        for d in range(self.MAX_OPPO_SHIP_DIST, -1, -1):
            if self.oppo_ship_halite_in_future[f, pos.x, pos.y, d] > my_ship_halite:
                return d+1

        return 0

    def getSuggestedMinePos(self, pos, my_ship_halite, return_length, must_include_yard):

        def getCellRewardByPathLength(cell_halite, distance, min_mine_cell):
            max_mine_step = max(398 - self.board.step - distance, 0)
            if max_mine_step == 0:
                return 0

            gain = 0
            for mine_step in range(1, max_mine_step+1):
                gain += cell_halite*.25
                cell_halite *= .75
                if cell_halite < min_mine_cell:
                    break

            reward = gain / max(1, distance+mine_step)
            return reward

        def prepareMines():
            self.big_mines = []
            self.big_mines_and_yards = []

            for x in range(yard_x-self.vicinity, yard_x+self.vicinity+1):
                for y in range(yard_y-self.vicinity, yard_y+self.vicinity+1):
                    if abs(yard_x-x) + abs(yard_y-y) <= self.manhattan_vicinity:
                        p = Point(x, y) % self.size
                        if self.halite_matrix[p.x, p.y] >= self.getMinHaliteMine():
                            self.big_mines.append(p)
            self.big_mines_and_yards = self.big_mines + [yard.position for yard in self.me.shipyards]

        def getSuggestedList(pos, my_ship_halite, mines):
            cands = []
            for p in mines:
                dist_to_ship = getDistance(pos, p)
                dist_to_opponent = self.getDistToOpponentShip(p, my_ship_halite)
                # dist_to_opponent = self.getDistToOpponentShipFuture(p, my_ship_halite+self.cell[p.x][p.y].halite*.25, pos)
                axis_dist_to_shipyard = getAxisDistance(p, Point(yard_x, yard_y))
                # halite value of a cell also depends on its distance to opponent ships
                val = self.halite_matrix[p.x, p.y] / 3 * min(3, dist_to_opponent - (dist_to_opponent==1))
                # val = self.halite_matrix[p.x, p.y] * self.distToOppoShipDecayFactor(p, pos, my_ship_halite)
                # halite value of a cell depends on its distance to my ship
                val = val * self.distToShipDecayFactor(my_ship_halite, dist_to_ship)
                # halite value of a cell also depends on its distance to my shipyard
                # val = val * self.distToShipyardDecayFactor(my_ship_halite, axis_dist_to_shipyard)
                val = val * self.getDecayByShipHaliteGroup(my_ship_halite, axis_dist_to_shipyard)
                # add to list
                cands.append((self.halite_matrix[p.x, p.y] > self.getMinHaliteMine(), val, dist_to_ship, p))
                # candidates.append((val, dist_to_ship, p))
            return cands

        def getSuggestedList2(pos, my_ship_halite, mines):
            min_mine_cell = self.getMinHaliteMine()
            cands = []
            for p in mines:
                dist_to_ship = getDistance(pos, p)
                dist_to_shipyard = self.getDistToShipyard(p)
                axis_dist_to_shipyard = getAxisDistance(p, Point(yard_x, yard_y))
                dist_to_opponent = self.getDistToOpponentShip(p, my_ship_halite)
                #
                val =  getCellRewardByPathLength(self.halite_matrix[p.x, p.y], dist_to_ship + dist_to_shipyard, min_mine_cell)
                # halite value of a cell also depends on its distance to opponent ships
                val = val / 3 * min(3, dist_to_opponent - (dist_to_opponent==1))
                # halite value of a cell also depends on its distance to my shipyard
                val = val * self.getDecayByShipHaliteGroup(my_ship_halite, axis_dist_to_shipyard)
                # add to list
                if val > 1e-3 or dist_to_shipyard == 0: #i.e. val > 0 or is shipyard
                    cands.append((self.halite_matrix[p.x, p.y] > self.getMinHaliteMine(), val, dist_to_ship, p))
                # candidates.append((val, dist_to_ship, p))
            return cands
        # -------------------------------

        yard_x, yard_y = CENTER

        if not hasattr(self, 'big_mines'):
            prepareMines()

        ship_vicinity = round((self.getShipHalitePercRank(my_ship_halite)**.5) * self.vicinity)

        if must_include_yard:
            my_mines = [p for p in self.big_mines_and_yards if getAxisDistance(CENTER, p) <= ship_vicinity]
        else:
            my_mines = [p for p in self.big_mines if getAxisDistance(CENTER, p) <= ship_vicinity]

        candidates = getSuggestedList(pos, my_ship_halite, my_mines) 
        # if self.board.step == 187 and pos == Point(6, 5):
            # print('ZZZZZZZZZZZZZZZZZZ')
            # print('self.getShipHalitePercRank(my_ship_halite):', self.getShipHalitePercRank(my_ship_halite))
            # print('my_ship_halite:', my_ship_halite)
            # print('vicinity:', self.vicinity)
            # print('ship_vicinity:', ship_vicinity)
            # print('CENTER:', CENTER)
            # print('all_mines:', self.all_mines)
            # print('my_all_mines: {}\ncandidates: {}'.format(my_all_mines, candidates))

        candidates = sorted(candidates, key=lambda x: (x[0], x[1], -x[2]), reverse=True)[:return_length]
        if len(candidates) == 0:
            candidates = [(0, 0, 0, pos)]

        result = [cand[-2:] for cand in candidates]
        return result

    def isOppoYardNotResisting(self, oppo_id):
        return oppo_id in oppo_yard_not_resisting

    def getVicinity(self):
        return self.vicinity

    def addToVicinity(self, add):
        global ADD_VICINITY
        if ADD_VICINITY < self.getNumMyShips() // 10:
            ADD_VICINITY += 1
            print('{}: ADD_VICINITY: {}'.format(self.board.step, ADD_VICINITY))

    def getNearestOppoShips(self, pos, min_halite):
        oppo_ships = [ship for oppo in self.board.opponents for ship in oppo.ships 
                                       if ship.halite >= min_halite and 
                                          getAxisDistance(ship.position, CENTER) <= self.getVicinity() #and
                                          # getDistance(pos, ship.position) <= 10
                              ]
        oppo_ships = sorted(oppo_ships, 
                                          key=lambda s: getDistance(pos, s.position)
                                         )
        return oppo_ships

    def getNearestOppoShipPos(self, pos, min_halite):
        oppo_ships = [ship for oppo in self.board.opponents for ship in oppo.ships 
                                       if ship.halite >= min_halite]
        oppo_ships = sorted(oppo_ships, 
                                          key=lambda s: (self.getAxisDistToCenter(s.position) <= self.vicinity, -getDistance(s.position, pos)),
                                          reverse=True
                                         )
        if len(oppo_ships):
            return oppo_ships[0].position
        else:
            return (Point(10, 10) + CENTER) % self.size

    def getAxisDistToCenter(self, pos):
        return getAxisDistance(pos, CENTER)

    def oppoShipNearYardForLong(self, oppo_ship):
        return oppo_ships_near_shipyard[(oppo_ship.player_id, oppo_ship.id)] >= ATTACK_STEPS_NEAR_YARD

    def myShipNumsIsDominant(self):
        return (
            self.getNumMyShips() >= self.getMaxOppoNumShips()*1.25 and
            self.getNumMyShips() >= self.getMaxOppoNumShips() + 5
        )

    def getPredictedOppoMoves(self, ship):

        def isPosDangerousForOppo(cell, oppo_id, oppo_halite):
            arounds = getCellDirections(cell) + [(None, cell)]
            for _, next_cell in arounds:
                if (next_cell.ship is not None and 
                    next_cell.ship.player_id != oppo_id and
                    next_cell.ship.halite <= oppo_halite
                   ):
                    return True
            return False
        # ===

        if (ship.player_id, ship.id) not in self.predict_oppo_moves:

            p_id = ship.player_id
            is_safe = [] # safe for oppo

            all_moves = [(None, ship.cell)] + getCellDirections(ship.cell) # assume oppo will stay still first
            for direc, next_cell in all_moves:
                if not isPosDangerousForOppo(next_cell, p_id, ship.halite):
                    is_safe.append(next_cell.position)

            self.predict_oppo_moves[ship.player_id, ship.id] = is_safe
        # ---
        return self.predict_oppo_moves[ship.player_id, ship.id]

    def getRichShipsNeedGuards(self):
        if not hasattr(self, 'ships_need_guard'):
            self.ships_need_guard = [
                ship for ship in self.me.ships 
                if ship.halite > 0 and 
                   self.isPositionSafe(ship.position, ship.halite, safe_level=1) == False and
                   len(self.safeMovesForShip(ship, include_stay=True, include_swap=True)) <= 1
            ]

        return self.ships_need_guard

    # def getRichShipsNeedGuards(self):
    #     if not hasattr(self, 'ships_need_guard'):
    #         self.ships_need_guard = []

    #         for ship in me.ships:
    #             if ship.halite > 0 and self.isPositionSafe(ship.position, ship.halite, safe_level=1) == False:
    #                 safe_moves = self.safeMovesForShip(ship, include_stay=True, include_swap=True)
    #                 if len(safe_moves) <= 1:


    #         self.ships_need_guard = [
    #             ship for ship in self.me.ships 
    #             if ship.halite > 0 and 
    #                self.isPositionSafe(ship.position, ship.halite, safe_level=1) == False and
    #                len(self.safeMovesForShip(ship, include_stay=True, include_swap=True)) <= 1
    #         ]

    #     return self.ships_need_guard

    def getAvgCellHaliteAround(self, axis_dist=4):
        total_halite = 0
        cnt_halite = 0

        for x in range(CENTER.x-axis_dist, CENTER.x+axis_dist+1):
            for y in range(CENTER.y-axis_dist, CENTER.y+axis_dist+1):
                p = Point(x, y) % self.size
                if self.getHaliteAtPos(p) > 0:
                    total_halite += self.getHaliteAtPos(p)
                    cnt_halite += 1

        return total_halite / max(1, cnt_halite)

    def getTotalCellHaliteAround(self, pos, axis_dist):
        total_halite = 0

        for x in range(pos.x-axis_dist, pos.x+axis_dist+1):
            for y in range(pos.y-axis_dist, pos.y+axis_dist+1):
                p = Point(x, y) % self.size
                if self.getHaliteAtPos(p) > 0:
                    total_halite += self.getHaliteAtPos(p)

        return total_halite

    def getPositionConvertRank(self, pos) -> '0 is worst, 1 is best':
        def calcScore(p):
            max_d = 2

            score = 0
            for add_x in range(-max_d, max_d+1):
                for add_y in range(-(max_d-abs(add_x)), (max_d-abs(add_x))+1):
                    if (add_x != 0 or add_y != 0):
                        neighbor =  Point(x+add_x, y+add_y) % self.size
                        d = abs(add_x) + abs(add_y)
                        if self.getHaliteAtPos(neighbor) > 0:
                            score += (200 + self.getHaliteAtPos(neighbor)) * (.5 ** d)

            score = score * (
                ((self.getDistToOpponentShipyard(pos) + 
                  self.getAxisDistToOpponentShipyard(pos)
                 )/2
                ) **.75
            )

            return score
        # ----------------------
        if not hasattr(self, 'pos_convert_rank'):
            self.pos_convert_rank = {}

            p_list = []
            for x in range(self.size):
                for y in range(self.size):
                    p = Point(x, y)
                    # distance criteria
                    if (board_info.getDistToShipyard(p) >= 3 + self.getNumShipyards() and
                        board_info.getAxisDistToShipyard(p) <= 2 + self.getNumShipyards() and
                        board_info.getDistToOpponentShipyard(p) >= 8
                       ): 
                        p_list.append((p, calcScore(p)))

            p_list = sorted(p_list, key=lambda p: p[1])

            for i, (p, score) in enumerate(p_list):
                self.pos_convert_rank[p] = (i+1) / len(p_list)

        return self.pos_convert_rank.get(pos, 0)

    def safeMovesForShip(self, ship, include_stay, include_swap) -> 'list(tuple(move-type, detail))': 
        # WARNING: stay on top of a shipyard if the yard is spawning is currently OK
        # the lastCheck helps keep no collide in this case.
        def getSafeSwaps(ship, brave):
            safe_moves = []
            safe_swaps = self.safeSwapsForShip(ship, brave)
            for direc, next_ship in safe_swaps:
                if ship_states[ship.id].target is not None:
                    dist_to_target = getDistance(next_ship.position, ship_states[ship.id].target)
                else:
                    dist_to_target = 0
                safe_moves.append((dist_to_target, 'swap', next_ship))
            return safe_moves

        def getSafeStay(ship, brave):
            safe_moves = []
            if board_info.isSafeMoveTo(ship.position, ship.position, ship.halite-brave):
                # get distance to target to sort
                if ship_states[ship.id].target is not None:
                    dist_to_target = getDistance(ship.position, ship_states[ship.id].target)
                else:
                    dist_to_target = 0
                safe_moves.append((dist_to_target, 'normal', None))
            return safe_moves

        def getSafeMoves(ship, brave):
            safe_moves = []
            for direc, next_cell in getCellDirections(ship.cell):
                if board_info.isSafeMoveTo(next_cell.position, ship.position, ship.halite-brave):
                    # get distance to target to sort
                    if ship_states[ship.id].target is not None:
                        dist_to_target = getDistance(next_cell.position, ship_states[ship.id].target)
                    else:
                        dist_to_target = 0
                    safe_moves.append((dist_to_target, 'normal', direc))
            return safe_moves
        # -----------------------------------------------
        list_safe_moves = []

        # swap is safe
        if include_swap:
            list_safe_moves.extend(getSafeSwaps(ship, brave=0))
        # stay here is safe
        if include_stay:
            list_safe_moves.extend(getSafeStay(ship, brave=0))
        # move is safe
        list_safe_moves.extend(getSafeMoves(ship, brave=0))

        # sort and return
        if len(list_safe_moves) > 0:
            list_safe_moves = sorted(list_safe_moves, key=lambda x: x[0])
            moves = [move[1:] for move in list_safe_moves]
            return moves

        # come to here means no absolute safe moves
        # then, let's try avoiding equal-weight oppo ships

        # swap is safe
        if include_swap:
            list_safe_moves.extend(getSafeSwaps(ship, brave=1))
        # stay here is safe
        if include_stay:
            list_safe_moves.extend(getSafeStay(ship, brave=1))
        # move is safe
        list_safe_moves.extend(getSafeMoves(ship, brave=1))

        # sort and return
        if len(list_safe_moves) > 0:
            list_safe_moves = sorted(list_safe_moves, key=lambda x: x[0])
            moves = [move[1:] for move in list_safe_moves]
            return moves

        # here means no other ways
        return []

    def safeSwapsForShip(self, ship, brave):
        safe_swaps = []

        if self.isReadyForSwap(ship):
            for direc, next_cell in getCellDirections(ship.cell):
                if (next_cell.ship is not None and 
                    next_cell.ship.player_id == self.me.id and 
                    self.isReadyForSwap(next_cell.ship) and 
                    # shouldSwap(ship, next_cell.ship, brave)
                    board_info.isSafeSwap(ship.position, next_cell.ship.position, 
                                                         ship.halite-brave, next_cell.ship.halite-brave,
                                                         ship, next_cell.ship)
                   ):
                    safe_swaps.append((direc, next_cell.ship))

        return safe_swaps

    def isReadyForSwap(self, ship):

        # if (ship.cell.shipyard is not None and
        #     shipyard_actions.get(ship.cell.shipyard.id, -9) == ShipyardAction.SPAWN
        #    ):
        #     return False

        if ship.id not in ship_actions:
            return True

        if (ship.id in ship_actions
            and ship_actions[ship.id] is None 
            and ship.halite==0 
            and ship.cell.shipyard is not None
            and board_info.getDistToOpponentShip(ship.position, 0) > 1
           ): # means this is the protector
            return True

        return False

    

def updateStates(board, me):
    # clear actions
    ship_actions.clear()
    shipyard_actions.clear()

    # init state for new ships
    for ship in me.ships:
        # newly created ships are all MINE
        if ship.id not in ship_states:
            ship_states[ship.id] = ShipState(ShipState.MINE, None, None)
        # storing ships that have stored is changed to MINE
        if (ship_states[ship.id].state == ShipState.STORE and
            board_info.getDistToShipyard(ship.position) == 0):
            ship_states[ship.id] = ShipState(ShipState.MINE, None, None)

    # adjust DISTANCE_TO_ATTACK
    global DISTANCE_TO_ATTACK
    step = board.step

    if step < 75:
        DISTANCE_TO_ATTACK = 2
    else:
        DISTANCE_TO_ATTACK = 3

    # eligible_for_rescue
    global eligible_for_rescue
    eligible_for_rescue = set([ship for ship in me.ships])

    # show attack result to screen
    oppo_ships = set([(oppo.id, ship.id) for oppo in board.opponents for ship in oppo.ships])
    global cnt_attack_success, cnt_attack_failure
    for oppo_ship in attacked_freys:
        if oppo_ship not in oppo_ships:
            cnt_attack_success += 1
            print('{}: last step attack {} successfully? {}-{}'.format(
                board.step, attacked_freys[oppo_ship], cnt_attack_success, cnt_attack_failure)
            )
        else:
            cnt_attack_failure += 1
    if board.step == 398:
        print('attack result {}-{}'.format(cnt_attack_success, cnt_attack_failure))

    attacked_freys.clear()

def handleShipYards(board, me, size):
    
    def doSpawn(shipyard):
        # record action
        shipyard_actions[shipyard.id] = ShipyardAction.SPAWN
        shipyard_states[shipyard.id] = ShipyardState(ShipyardState.SPAWN)
        board_info.addShip(shipyard.position)
        board_info.changeMyHalite(-500)

    def shouldSpawn(shipyard):
        # already had too many ships
        if board_info.getNumMyShips() >= 50:
            return False

        # already too dominant
        if (board_info.getNumMyShips() >= board_info.getMaxOppoNumShips()*2 and
            board_info.getNumMyShips() >= board_info.getMaxOppoNumShips() + 10
           ):
           return False 

        # spawn to protect shipyard
        if (board_info.getDistToOpponentShip(shipyard.position, 9e5) == 1
            and shipyard.cell.ship is None
            and board_info.isPositionSafe(shipyard.position, 0, safe_level=1)
            and board.step < 392
           ): 
            return True      
        # keep some money in case the yard is destroyed in the next steps
        # if (board.step > 100 and
        #     board_info.getNumShipyards() == 1 and
        #     board_info.getMyHalite() < 500 + (500 - THRESHOLD_STORE[0])
        #    ):
        #    return False 

        safeSpawn = noStoreShipAround(shipyard.cell, me)

        if (safeSpawn
            and board.step < 180
           ):
            return True

        if (safeSpawn
            and board.step < 250
            # and board_info.getBoardHalite() >= 8000 
            and board_info.getBoardHalite() >= 6000 + 7000/70*(board.step-180)
           ):
            return True

        if (safeSpawn
            and board_info.getBoardHalite() >= 8000 
            and not (board_info.getNumMyShips() >= 20
                           and board.step > 200 
                           and board_info.getMaxOppoHalite() >= 4000
                           and board_info.getNumMyShips() >= board_info.getMaxOppoNumShips()
                         )
            and (board.step <= 200 + min(80, max(0, (board_info.getBoardHalite()-10000)/200))
                    or (400-board.step) // 20 >= board_info.getNumMyShips()
                    or board_info.getNumMyShips() < 3
                   )            
            and board.step <= 330
        ):
            return True

        # spawn if dominate
        if (safeSpawn
            and board_info.getMyHaliteAndCargo() >= board_info.getMaxOppoHaliteAndCargo() + 2000
            and board_info.getNumMyShips() < 40
            and board.step <= 330
           ):
           return True 
        
        
            # and board_info.getNumShipyards() <= 1
            # board_info.getNumMyShips() > board_info.getMaxOppoNumShips() *.7

        return False

    def canSpawn(shipyard):
        return (#shipyard.next_action is None 
                    shipyard.id not in shipyard_actions
                    and board_info.getMyHalite() >= 500)

    # If there are no ships, spawn a ship from the first shipyard.
    if isThereNoShip(me):
        for shipyard in me.shipyards:
            if canSpawn(shipyard):
                doSpawn(shipyard)
                break

    # # handle the remaining shipyards
    # for shipyard in me.shipyards:
    #     if canSpawn(shipyard):
    #         if shouldSpawn(shipyard):
    #             doSpawn(shipyard)

    # select shipyard to spawn
    yards = [shipyard for shipyard in me.shipyards if canSpawn(shipyard) and shouldSpawn(shipyard)]
    yards = sorted(yards, key=lambda yard: board_info.getTotalCellHaliteAround(yard.position, axis_dist=3), reverse=True)
    for yard in yards:
        if canSpawn(yard) and shouldSpawn(yard):
            doSpawn(yard)
        

def handleShips(board, me, size): 

    # convert
    def shipConvertToShipyard():

        def getBestShipToConvert(me):
            if board.step == 0:
                return me.ships[0]

            if board_info.getNumShipyards() == 0:
                candidates = sorted([ship for ship in me.ships if ship.id not in ship_actions and ship.halite + board_info.getMyHalite() >= 500], 
                                                   key=lambda x: getDistance(x.position, CENTER))
                if (len(candidates) and 
                    getDistance(candidates[0].position, CENTER) <= 4 and
                    getAxisDistance(candidates[0].position, CENTER) <= 3
                   ):
                    return candidates[0]

            else:
                candidates = [
                    (ship, board_info.getPositionConvertRank(ship.position))
                        for ship in me.ships 
                        if (ship.id not in ship_actions and
                            ship.halite + board_info.getMyHalite() >= 500 and 
                            board_info.getDistToOpponentShip(ship.position, ship.halite) >= 2)
                ]
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                if len(candidates):
                    candidate, score_rank = candidates[0]
                    if score_rank >= .9:
                        printviolet('{}: convert {} which has score rank {} (curr ships is {})'.
                            format(board.step, candidate.position, score_rank, board_info.getNumMyShips())
                        )
                        return candidate

            return None        

        def doConvert(ship, postpone_ship_assure=False):
            # record action
            # ship_actions[ship.id] = ShipAction.CONVERT
            moveShip(ship, ShipAction.CONVERT, ship.position, None, postpone_ship_assure)
            ship_states[ship.id] = ShipState(ShipState.CONVERT, ship.position, None)
            board_info.addShipyard(ship.position)
            board_info.changeMyHalite(-500)
        
        def shouldConvertAny():
            if board.step > 250:
                return False
            
            return (
                (board_info.getNumShipyards() == 0)
                or
                ( board_info.getNumMyShips() >= 16 and 
                  board_info.getNumShipyards() <= 1) 
                or
                (board_info.getNumMyShips() >= 30 and 
                 board_info.getNumShipyards() <= 2 and
                 board_info.myShipNumsIsDominant())
            )

        def canConvert(ship):
            return (ship is not None
                 and ship.halite + board_info.getMyHalite() >= 500)

        def run():
            # If there are no shipyards, convert the best/first ship into shipyard.
            if (board_info.getNumShipyards() == 0 and 
                ( (board.step < 250 and board_info.getNumMyShips() > 2) or 
                  (board.step < 360 and board_info.getNumMyShips() > 5) or
                  (board.step < 396 and board_info.getCargo() > 1000)
                )
               ):
                ship = getBestShipToConvert(me)
                if canConvert(ship):
                    doConvert(ship)

            # If there are many ships but too few shipyards, convert one.
            if shouldConvertAny():
                ship = getBestShipToConvert(me)
                if canConvert(ship):
                    doConvert(ship)

            # if time is almost up and the rich ships cannot be on time
            for ship in me.ships:
                if (ship.halite > 500 and 
                    board_info.getNumShipyards() > 0 and 
                    board_info.getDistToShipyard(ship.position) > (399 - board.step)
                   ):
                    doConvert(ship)

        run()

    no_collide = set()
    def protectYardEndgame():
        global frey_slots

        # attack HALITE_0_NEAR_YARD_LONG when end game
        if board.step >= 375:
            freys = [
                oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
                if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_0_NEAR_YARD_LONG
                   and 3 >= board_info.getDistToShipyard(oppo_ship.position) >= 2
            ]
            attack_queue = []
            for frey in freys:
                predicted_frey_moves = getDirection(frey.position, board_info.getNearestShipyardPos(frey.position))[:2] + [None]
                predicted_frey_poss = [getLocationByMove(frey.cell, move) for move in predicted_frey_moves]
                attack_queue.append((board_info.getDistToShipyard(frey.position), predicted_frey_poss, frey))

            attack_queue = sorted(attack_queue, key=lambda x: x[0])
            attack_queue = deque(attack_queue)
            while len(attack_queue):
                dist, poss, frey = attack_queue.popleft()
                pos = poss[0]

                if (board_info.cell[pos.x][pos.y].ship is not None and
                    board_info.cell[pos.x][pos.y].ship.player_id == me.id and
                    board_info.cell[pos.x][pos.y].ship.id not in ship_actions and
                    board_info.cell[pos.x][pos.y].ship.halite < board_info.getMyHalite()/200
                   ):
                    attacker = board_info.cell[pos.x][pos.y].ship
                    doAttackOppoShip(attacker, frey, pos)
                    frey_slots[(frey.player_id, frey.id)] -= 1
                    no_collide.add(pos)
                    # print('## {}: attack near yard end game, attacker {} moves to {} to attack {}'.format(board.step, attacker.position, pos, frey.position))

                elif board_info.myShipIn(pos):
                    pass

                else:
                    my_attackers = [
                        ship for ship in me.ships 
                            if getDistance(ship.position, pos) <= 1 and 
                            shipSuitableAttackOppoShip(ship) and
                            ship.cell.shipyard is None
                    ]
                    my_attackers = sorted(my_attackers, key=lambda ship: (getDistance(ship.position, pos), ship.halite))
                    if len(my_attackers):
                        for attacker in my_attackers:
                            if doAttackOppoShip(attacker, frey, pos):
                                frey_slots[(frey.player_id, frey.id)] -= 1
                                no_collide.add(pos)
                                # print('#### {}: attack near yard end game, attacker {} moves to {} to attack {}'.format(board.step, attacker.position, pos, frey.position))
                                break

                if len(poss) > 1:
                    attack_queue.append((dist, poss[1:], frey))

    # when game is nearly end, collide ship near shipyard to take all cargo
    def shipCollideEndGame():

        def run():
            if len(me.shipyards) == 0:
                return
            if board_info.getNumMyShips() <= 3:
                return
            if board.step < 393:
                return

            if len(me.shipyards) > 0:
                assemble_point = me.shipyards[0].position
            else:
                assemble_point = CENTER
                
            steps_remain = min(398 - board.step, 5)
            cells = [(point, cell) for point, cell in board.cells.items() if getDistance(assemble_point, point) <= steps_remain and point not in no_collide]
            # print('{}: no_collide {}'.format(board.step, [point for point, cell in board.cells.items() if point in no_collide]))
            cells = sorted(cells, key=lambda x: getDistance(assemble_point, x[0]))
            for point, cell in cells:
                if cell.ship is not None and cell.ship.player_id == me.id and ship_actions.get(cell.ship.id, None) is None:
                    cell_ship_halite = cell.ship.halite
                else:
                    cell_ship_halite = 1000000

                nearby_ships = [ship for ship in me.ships if ship.id not in ship_actions and getDistance(point, ship.position) == 1]
                nearby_ships = sorted(nearby_ships, key=lambda s: s.halite)

                min_halite = min([cell_ship_halite] + [ship.halite for ship in nearby_ships])
                if min_halite == cell_ship_halite:
                    nships_with_min_halite = 1
                else:
                    nships_with_min_halite = 0

                for ship in nearby_ships:
                    if ship.halite > min_halite:
                        moveShip(ship, getDirection(ship.position, point)[0], point, point, postpone_ship_assure=True)
                    elif ship.halite == min_halite and nships_with_min_halite == 0:
                        moveShip(ship, getDirection(ship.position, point)[0], point, point, postpone_ship_assure=True)
                        nships_with_min_halite += 1

                if cell.ship is not None and cell.ship.player_id == me.id and cell.ship.id not in ship_actions:
                    moveShip(cell.ship, None, point, point, postpone_ship_assure=True)

        run()

    # shipyard protector
    def shipProtectShipyard():

        def cmpProtector(inp): # the smaller will be the protector
            ship, dist = inp
            state = ship_states[ship.id].state
            target = ship_states[ship.id].target
            halite = ship.halite
            pos = ship.position

            crit_safe_moves = len(board_info.safeMovesForShip(ship, include_stay=True, include_swap=True)) >= 2
            # criterion 2 is by state
            if state == ShipState.STORE:
                crit_state = 0
            elif state == ShipState.ATTACK_SHIP or state == ShipState.ATTACK_YARD:
                crit_state = 1
            elif state == ShipState.MINE and pos != target: #means MINE but not HOOK
                crit_state = 2
            else:
                crit_state = 3

            return (dist, crit_safe_moves, crit_state, -halite)

        def getProtectorCandidates(shipyard, yard_protectors, need_dist):
            candidates = [(ship, getDistance(shipyard.position, ship.position, size)) 
                                   for ship in me.ships 
                                   if (ship.id not in yard_protectors and 
                                       ship.id not in ship_actions and 
                                       getDistance(shipyard.position, ship.position, size) <= need_dist)
                                  ]
            if len(candidates):
                candidates = sorted(candidates, key=cmpProtector)
                cands, _ = zip(*candidates)
                return cands

            return []

        def makeProtector(shipyard, ship, need_dist):
            # print('{}: in makeProtector, yard {}, ship {}, need_dist {}'.format(board.step, shipyard.position, ship.position, need_dist))
            if getDistance(ship.position, shipyard.position) + 1 <= need_dist:
                return True

            # change state to MINE, mine only a radius 'need_dist' around shipyard
            # move to a position that is distance 2 from ally is preferred
            def getBestMove(targets, moves):
                second_best_move = None

                for d, pos in targets:
                    for direc, next_cell in moves:
                        new_dist = getDistance(next_cell.position, pos, size)
                        if  new_dist < d:
                            return direc, next_cell, pos
                        elif new_dist == d and second_best_move is None:
                            second_best_move = (direc, next_cell, pos)

                return second_best_move
            # --------------------------------------------------
            my_other_ships = [a_ship for a_ship in me.ships if a_ship.id != ship.id]

            possible_moves = [(direc, next_cell) for direc, next_cell in (getCellDirections(ship.cell) + [(None,ship.cell)])
                                            if (getDistance(next_cell.position, shipyard.position, size) <= need_dist and
                                                board_info.isSafeMoveTo(next_cell.position, ship.position, ship.halite))
                                          ]
            preferred_moves = [(direc, next_cell) for direc, next_cell in possible_moves 
                                              if min([getDistance(next_cell.position, other_ship.position, size) for other_ship in my_other_ships], default=20) >= 2]
            # print('{}: in makeProtector, yard {}, ship {}, possible_moves {}'.format(board.step, shipyard.position, ship.position, possible_moves))
            # print('{}: in makeProtector, yard {}, ship {}, preferred_moves {}'.format(board.step, shipyard.position, ship.position, preferred_moves))
            # get new targets
            targets = board_info.getSuggestedMinePos(ship.position, ship.halite, 
                                                                                   return_length=None, must_include_yard=True)
            # print('{}: in makeProtector, yard {}, ship {}, all targets {}'.format(board.step, shipyard.position, ship.position, targets))
            targets = [(d, pos) for d, pos in targets if getDistance(pos, shipyard.position, size) <= need_dist]
            # print('{}: in makeProtector, yard {}, ship {}, filtered targets {}'.format(board.step, shipyard.position, ship.position, targets))
            # move to the best position
            best_move = getBestMove(targets, preferred_moves)
            if best_move is None:
                best_move = getBestMove(targets, possible_moves)

            if best_move is not None:
                direc, next_cell, pos = best_move
                moveShip(ship, direc, next_cell.position, pos)
                if ship.cell.shipyard is not None and direc is None:
                    ship_states[ship.id].state = ShipState.PROTECT
                else:
                    ship_states[ship.id].state = ShipState.STORE
                # print('{}: protector at {} ==> {}, need_dist {}, target {}'.format(board.step, ship.position, direc, need_dist, pos))
                return True

            return False

        def protectShipyard(shipyard, yard_protectors):
            # if game is almost end, no need to protect
            if board.step > 390:
                return None
            # if this yard is going to spawn, don't worry
            # print('need protector', shipyard.position)

            if shipyard_actions.get(shipyard.id, None) == ShipyardAction.SPAWN:
                return None

            # else, need a protector
            dist_to_oppo = min(board_info.getDistToOpponentShip(shipyard.position, 9e5),
                                            board_info.getDistToOpponentShipyard(shipyard.position) + 1)
            # after this step, my closest ship must have this dist
            need_closest_ship_dist = max(0, dist_to_oppo - 2)

            for d in [-1, 0, 1]:
                candidates = getProtectorCandidates(shipyard, yard_protectors, need_closest_ship_dist+d)
                # if board.step == 187:
                #     print('yard {}, candidates: {}'.format(shipyard.position, [s.position for s in candidates]))
                for candidate in candidates:
                    if makeProtector(shipyard, candidate, need_closest_ship_dist):
                        return candidate
            # come to here means we have no suitable protector
            # call the closest back
            my_ships = [ship for ship in me.ships if ship.id not in ship_actions and ship.id not in yard_protectors]
            my_ships = sorted(my_ships, key=lambda ship: (getDistance(ship.position, shipyard.position), ship.halite != 0, ship.halite))
            for candidate in my_ships:
                if makeProtector(shipyard, candidate, getDistance(candidate.position, shipyard.position)-1):
                    printgreen('{}: call a ship {} back to protect yard (not protectable yet)'.format(board.step, candidate.position))
                    return candidate

        def run():
            # assure my closest ship is closer to my shipyard than all enemy ships.
            yard_protectors = set()
            for shipyard in me.shipyards:
                protector = protectShipyard(shipyard, yard_protectors)
                if protector is not None:
                    yard_protectors.add(protector.id)      
                    # print('protector:', protector.position)  

        run()

    # attack opponent shipyard
    def shipAttackShipyard():

        def shipSuitableAttackShipyard(ship):
            return (ship.id not in ship_actions
                 and ship.halite <= MAX_HALITE_ATTACK_SHIPYARD
            )

        def shouldAttackShipyard(ship, oppo_shipyard):
            oppo_id = oppo_shipyard.player_id

            # basic conditions
            if (shipSuitableAttackShipyard(ship)
                and not board_info.isEliminated(oppo_id)
                and getDistance(ship.position, oppo_shipyard.position) <= 399 - board.step
                and (board_info.myShipIn(oppo_shipyard.position) == False
                    or getDistance(ship.position, oppo_shipyard.position) > 1
                       )
                and (board_info.getApproxWorth(oppo_id) > board_info.getApproxWorth()*0.8
                    or board_info.getApproxWorth(oppo_id) > board_info.getApproxWorth() - 2000
                       )
                # and board_info.getApproxWorth(oppo_id) < board_info.getApproxWorth()*1.5
               ):
                if (getDistance(ship.position, oppo_shipyard.position) == 1
                    and board_info.getWorthRank() <= 2
                    and board_info.getNumMyShips() > 20
                    and (
                        (oppo_shipyard.player.halite < 500 and board_info.getDistToOpponentShip(oppo_shipyard.position, 0) > 1)
                        or board_info.isOppoYardNotResisting(oppo_id)
                           )
                   ):
                    return True

                if (board.step >= 385 and ship.halite == 0
                    and board_info.getNumMyShips() > 3 and 
                    board_info.getMaxHarvestOneAttempt(ship, 396 - board.step) < 50
                   ):
                    return True

                if (board.step >= 370 and ship.halite == 0 and 
                    board_info.getNumMyShips() > 5 and
                    board_info.getMaxHarvestOneAttempt(ship, 396 - board.step) < 50
                   ):
                    return True

            rep_dist_to_center = ((board_info.getAxisDistToCenter(oppo_shipyard.position) + getDistance(oppo_shipyard.position, CENTER))/2)
            if (shipSuitableAttackShipyard(ship)
                and getDistance(ship.position, oppo_shipyard.position) <= 3
                and getDistance(ship.position, oppo_shipyard.position) <= 399 - board.step
                and (board_info.myShipIn(oppo_shipyard.position) == False
                    or getDistance(ship.position, oppo_shipyard.position) > 1
                       )
                and (lambda x: (x**2))(rep_dist_to_center) <= board_info.getNumMyShips()
                and board_info.getNumMyShips() > board_info.getMaxOppoNumShips()
               ):
                printgreen('{}: ship {} suitable attack {}'.format(board.step, ship.position, oppo_shipyard.position))
                return True

            return False

        def doAttackShipyard(ship, oppo_shipyard):
            # print ('doAttackShipyard', ship.position, oppo_shipyard.position)
            for direction in getDirection(ship.position, oppo_shipyard.position, size):
                if direction is None:
                    continue

                next_step = getLocationByMove(ship.cell, direction)
                if not board_info.myShipIn(next_step):
                    moveShip(ship, direction, next_step, oppo_shipyard.position)
                    ship_states[ship.id] = ShipState(ShipState.ATTACK_YARD, oppo_shipyard.position, None)
                    # print('{}: tell {} to go {} attack yard {}'.format(
                    #     board.step, ship.position, direction, oppo_shipyard.position))
                    return

        def run():
            # If spot a "suitable" enemy shipyard around, 
            # or if time is almost out, attack enemy shipyards.
            attack_list = []
            my_ships = [ship for ship in me.ships if ship.id not in ship_actions and shipSuitableAttackShipyard(ship)]
            for ship in my_ships:
                oppo_shipyards_around = board_info.getOppoShipyardsAround(ship) # already sorted by distance
                oppo_shipyard = next((yard for yard in oppo_shipyards_around if shouldAttackShipyard(ship, yard)), None)
                if oppo_shipyard is not None:
                    attack_list.append((ship, oppo_shipyard))

            attack_list = sorted(attack_list, key=lambda x: getDistance(x[0].position, x[1].position))
            for ship, oppo_shipyard in attack_list:
                if shipSuitableAttackShipyard(ship) and  shouldAttackShipyard(ship, oppo_shipyard):
                    doAttackShipyard(ship, oppo_shipyard)

        run()

    # help rescue ships that have no safe moves
    def rescueStrandedShips():
        
        def tryRescue(sad_ship):
            # early return
            if (# sad_ship.next_action is not None 
                sad_ship.id in ship_actions
                or len(board_info.safeMovesForShip(sad_ship, include_stay=True, include_swap=True)) > 0
               ):
                return True
            # init
            list_asked = {sad_ship.id}
            queue = deque([(sad_ship, [])])
            # bfs
            while len(queue):
                ship, list_movement = queue.popleft()
                safe_moves = board_info.safeMovesForShip(ship, include_stay=False, include_swap=False)
                if len(safe_moves) == 0:
                    for direc, next_cell in getCellDirections(ship.cell):
                        if (next_cell.ship is not None and 
                            next_cell.ship.player_id == me.id and 
                            next_cell.ship.id not in ship_actions and
                            next_cell.ship.id not in list_asked and
                            board_info.isPositionSafe(next_cell.position, ship.halite) and
                            board_info.myShipIn(next_cell.position) - (next_cell.shipyard is not None and shipyard_actions.get(next_cell.shipyard.id, None) == ShipyardAction.SPAWN) <= 1
                           ):
                            list_asked.add(next_cell.ship.id)
                            queue.append( (next_cell.ship, [(ship, direc)] + list_movement) )

                else: # finally find a way out                
                    # print('{}: find a way out:'.format(board.step))
                    direc = safe_moves[0][1]
                    list_movement = [(ship, direc)] + list_movement

                    for a_ship, a_move in list_movement:
                        moveShip(a_ship, a_move, getLocationByMove(a_ship.cell, a_move), None, postpone_ship_assure=True)
                        # print('{}: way: {} => {}'.format(board.step, a_ship.position, a_move))
                    return True

            # cannot rescue
            return False

        def tryMovebyPredictOppoShip(sad_ship):
            oppo_ships = [ship for oppo in board.opponents 
                                           for ship in oppo.ships 
                                           if getDistance(sad_ship.position, ship.position, size) <= 2
                                  ]
            possible_occupancy = [board_info.getPredictedOppoMoves(ship) for ship in oppo_ships]
            forbidden_poss = set().union(*possible_occupancy)
            # print('forbidden_poss:', forbidden_poss)

            for direc, cell in getCellDirections(sad_ship.cell) + [(None, sad_ship.cell)]:
                if ( cell.position not in forbidden_poss and
                     # (cell.ship is None or cell.ship.player_id != me.id or direc==None) and
                     board_info.myShipIn(cell.position) - (direc is None) == 0 and
                     (cell.shipyard is None or board_info.myShipyardIn(cell.position))
                   ):
                    moveShip(sad_ship, direc, cell.position, None, postpone_ship_assure=True)
                    # print('{}: guide ship {} to {}'.format(board.step, sad_ship.position, direc))
                    return True
            # cannot help
            return False

        def assureShipsSafety():
            global eligible_for_rescue

            while True:            
                list_ship_to_check = list(eligible_for_rescue)
                urgent_ships = []

                for ship in list_ship_to_check:
                    if (ship.id in ship_actions or # ship already moved
                        (   # board.step >= 370 and 
                            ship_states[ship.id].state == ShipState.ATTACK_YARD and 
                            ship.halite == 0 # attack oppo yard when time is up
                        ) or
                        (   ship_states[ship.id].state == ShipState.ATTACK_SHIP and 
                            ship.halite == 0 and
                            ship_states[ship.id].target is not None and
                            board_info.getDistToShipyard(ship_states[ship.id].target) < ATTACK_NEAR_YARD
                        ) or # attack oppo ship near my shipyard
                        (
                            ship_states[ship.id].state == ShipState.PROTECT
                        )
                       ):
                        # print('remove', ship.id)
                        eligible_for_rescue.remove(ship)
                    else:
                        if len(board_info.safeMovesForShip(ship, include_stay=True, include_swap=True)) <= 1:
                            urgent_ships.append(ship)

                if len(urgent_ships) == 0:
                    return

                urgent_ships = sorted(urgent_ships, key=lambda x: x.halite, reverse=True)
                # print('{}: urgent_ships: {}'.format(board.step, [s.position for s in urgent_ships]))

                for ship in urgent_ships:
                    # print('try rescue', ship.id, ship.position)
                    if ship.id not in ship_actions:
                        safe_moves = board_info.safeMovesForShip(ship, include_stay=True, include_swap=True)
                        if len(safe_moves) == 1:
                            if safe_moves[0][0] == 'normal':
                                direc = safe_moves[0][1]
                                moveShip(ship, direc, getLocationByMove(ship.cell, direc), ship_states[ship.id].target, postpone_ship_assure=True)
                                # print('{}: assure move (normal): {} {}'.format(board.step, ship.position, direc))
                            elif safe_moves[0][0] == 'swap':
                                next_ship = safe_moves[0][1]
                                swapShips(ship, next_ship, postpone_ship_assure=True)
                                # print('{}: assure move (swap): {} {}'.format(board.step, ship.position, next_ship.position))

                        elif len(safe_moves) == 0:
                            # first, try rescue by asking allies around
                            if not tryRescue(ship):
                                # if not work, try predict opponent's moves to counter
                                if not tryMovebyPredictOppoShip(ship):
                                    # if still not work, left to later arrangement in arrangeMoveShip
                                    # print('remove2', ship.id)
                                    eligible_for_rescue.remove(ship)

        def run():
            # if a ship has no safe moves, it needs rescue
            # if a ship has only 1 safe moves, move.
            assureShipsSafety()

        run()

    # swap
    def swapShips(ship, next_ship, postpone_ship_assure=False):
        if (   (ship_states[ship.id].state == ShipState.MINE and
                ship_states[next_ship.id].state == ShipState.GUARD and
                ship_states[next_ship.id].target is not None and
                ship_states[next_ship.id].target == ship.position)
               or
              (ship_states[next_ship.id].state == ShipState.MINE and
                ship_states[ship.id].state == ShipState.GUARD and
                ship_states[ship.id].target is not None and
                ship_states[ship.id].target == next_ship.position)
           ):
            print('{}: swap guard and rich ship: {} and {}'.format(board.step, ship.position, next_ship.position))

        direction_1 = getDirection(ship.position, next_ship.position, size)[0]
        moveShip(ship, direction_1, next_ship.position, ship_states[ship.id].target, 
                         postpone_ship_assure=True, override=True) # always postpone at this step

        direction_2 = getDirection(next_ship.position, ship.position, size)[0]        
        moveShip(next_ship, direction_2, ship.position, ship_states[next_ship.id].target,
                         postpone_ship_assure=postpone_ship_assure, override=True)

        # print('{}: swap 2 ships {} at {} and {} at {}'.format(board.step, ship.id, ship.position, next_ship.id, next_ship.position))
        # print(isUrgingStoringShip(ship), isRunningMiningShip(next_ship))
        # print(ship.halite, next_ship.halite)

    def shouldSwap(ship, next_ship):
        # print('{}: in shouldSwap:'.format(board.step), ship.position, next_ship.position)

        # if it is not safe for either ship, it is not ok.
        if not board_info.isSafeSwap(ship.position, next_ship.position, 
                                                      ship.halite, next_ship.halite,
                                                      ship, next_ship):
            return False
        # print('{}: in shouldSwap, pass first phase:'.format(board.step), ship.position, next_ship.position)

        # only swap if at least one ship gets benefit and the other ship does not lose benefit
        # (it seems like any ship must either gain or lose, no 'equal')
        ship_target = ship_states[ship.id].target
        next_ship_target = ship_states[next_ship.id].target

        if ship_target is None:
            ship_gain = 0
        else:
            ship_gain =  getDistance(ship.position, ship_target, size) - getDistance(next_ship.position, ship_target, size)
        
        if next_ship_target is None:
            next_ship_gain = 0
        else:
            next_ship_gain =  getDistance(next_ship.position, next_ship_target, size) - getDistance(ship.position, next_ship_target, size)

        # decide base on gains
        # ---
        if (board.step > 360 and
            ship_states[ship.id].state == ShipState.STORE and 
            ship_states[next_ship.id].state != ShipState.STORE and 
            ship_gain > 0
           ):
            return True

        if (board.step > 360 and
            ship_states[next_ship.id].state == ShipState.STORE and 
            ship_states[ship.id].state != ShipState.STORE and 
            next_ship_gain > 0
           ):
            return True
        # ---
        if (ship_states[ship.id].state == ShipState.STORE and 
            ship_states[next_ship.id].state == ShipState.PROTECT and
            ship_gain > 0
           ):
            return True

        if (ship_states[next_ship.id].state == ShipState.STORE and
            ship_states[ship.id].state == ShipState.PROTECT and
            next_ship_gain > 0
           ):
            return True
        # ---
        # # MINE ships with positive halite can swap GUARD
        # if (ship_states[ship.id].state == ShipState.MINE and 
        #     ship_states[next_ship.id].state == ShipState.GUARD and
        #     ship_gain > 0 and
        #     ship.halite > 0
        #    ):
        #     return True

        # if (ship_states[next_ship.id].state == ShipState.MINE and 
        #     ship_states[ship.id].state == ShipState.GUARD and
        #     next_ship_gain > 0 and
        #     next_ship.halite > 0
        #    ):
        #     return True

        return ship_gain + next_ship_gain > 0

    def run_SwapShips():
        # swapping ships
        for ship in me.ships:
            if board_info.isReadyForSwap(ship):
                for direc, next_cell in getCellDirections(ship.cell):
                    if (next_cell.ship is not None and 
                        next_cell.ship.player_id == me.id and 
                        board_info.isReadyForSwap(next_cell.ship) and 
                        shouldSwap(ship, next_cell.ship)
                       ):
                        swapShips(ship, next_cell.ship)
                        break
    
    # attack opponent ship
    def shipSuitableAttackOppoShip(ship):
        # if ship.next_action is not None:
        if ship.id in ship_actions:
            return False
        # if my ship has traced oppo for quite a while
        if (ship_states[ship.id].state == ShipState.ATTACK_SHIP and
            ship_states[ship.id].timeout == 0 and REST_AFTER_ATTACK > 0):
            return False
        # a ship should not attack too frequently
        if ( (ship_states[ship.id].state != ShipState.ATTACK_SHIP or ship_states[ship.id].timeout == 0) and
             ship.id in last_attack_time and 
             board.step - last_attack_time[ship.id] <= REST_AFTER_ATTACK
           ):
            return False
        # if my ship has quite a lot of halite
        if ship.halite > 200:
            return False
        # if my ship is going storing
        if ship_states[ship.id].state == ShipState.STORE:
            return False

        # if my ship is protecting: NO NEED, as PROTECt ship is assigned with an action already?
        if ship_states[ship.id].state == ShipState.PROTECT:
            return False

        return True

    def movesToAttack(ship, target_pos, oppo_ship_halite):
        def shouldEvenSuicide(next_cell):
            if oppo_ship_halite > 0:
                return 0
            if not board_info.getDistToShipyard(target_pos) < ATTACK_NEAR_YARD:
                return 0

            for _, around_cell in getCellDirections(next_cell, include_stay=True):
                if around_cell.ship is not None and around_cell.ship.player_id != me.id:
                    if around_cell.ship.halite < ship.halite:
                        return 0
                    elif around_cell.ship.halite == ship.halite:
                        if len(board_info.getPredictedOppoMoves(around_cell.ship)) <= 1:
                            return 0

            return 1

        current_distance = getDistance(ship.position, target_pos)
        moves = []

        for direc, next_cell in getCellDirections(ship.cell, include_stay=True):
            dist_to_oppo_ship = getDistance(next_cell.position, target_pos)
            even_suicide = shouldEvenSuicide(next_cell)
            if (board_info.isSafeMoveTo(next_cell.position, ship.position, ship.halite - even_suicide) and
                (dist_to_oppo_ship < current_distance or dist_to_oppo_ship <= 1)
               ):
                moves.append((direc, next_cell, dist_to_oppo_ship))

        moves = sorted(moves, key=lambda x: x[2])
        return moves

    def shouldAttackOppoShip(ship, oppo_ship):

        if getDistance(ship.position, oppo_ship.position) > DISTANCE_TO_ATTACK:
            return False

        if not shipSuitableAttackOppoShip(ship):
            return False

        if ( ( oppo_ship.halite > ship.halite or
               (oppo_ship.halite == ship.halite and oppo_ship.halite == 0)
             ) and 
             board_info.oppoShipNearYardForLong(oppo_ship)
           ):
            return True

        if oppo_ship.halite <= ship.halite:
            return False

        # if cannot attack, return False
        oppo_safe_moves = board_info.getPredictedOppoMoves(oppo_ship)
        if ( len(oppo_safe_moves) == 0 or
             (len(oppo_safe_moves) == 1 and oppo_safe_moves[0] is None)
           ):
            target_pos = oppo_ship.position
            if len(movesToAttack(ship, target_pos, oppo_ship.halite)) == 0:
                return False
        elif len(oppo_safe_moves) == 1:
            target_pos = oppo_safe_moves[0]
            if (#board_info.getBoardHalite() > 3000 or 
                #board_info.getApproxWorth(oppo_ship.player_id) < board_info.getApproxWorth()*.6 or
                len(movesToAttack(ship, target_pos, oppo_ship.halite)) == 0
               ):
                return False
        else: # oppo ship has too many safe moves
            return False        

        # print('{}: ship {} should attack oppo_ship {}'.format(board.step, ship.position, oppo_ship.position))

        return True

    def doAttackOppoShip(ship, oppo_ship, next_step=None):
        if next_step is None:
            oppo_safe_moves = board_info.getPredictedOppoMoves(oppo_ship)
            if len(oppo_safe_moves) > 0:
                target_pos = oppo_safe_moves[0]
            else:
                target_pos = oppo_ship.position

            moves = movesToAttack(ship, target_pos, oppo_ship.halite)
            # print('ship {}, moves: {}'.format(ship.position, moves))
            if len(moves) == 0:
                return False
            # chose the best move
            if moves[0][2] == 0: # prioritize take opponent's place
                best_move = moves[0]
            else:
                potential_allies = [ally for ally in me.ships if shipSuitableAttackOppoShip(ally) 
                                                                                 and shouldAttackOppoShip(ally, oppo_ship)
                                                                                 and ally.id != ship.id
                                            ]# potential allies who have not decided to attack yet
                if len(potential_allies):
                    moves = sorted(moves, key=lambda x: min([getDistance(x[1].position, ally.position) for ally in potential_allies]), reverse=True)
                best_move = moves[0]            

            direc, next_cell, _ = best_move
        
        else:
            direc = getDirection(ship.position, next_step)[0]
            next_cell = board_info.cell[next_step.x][next_step.y]

        # print('{}:doAttackOppoShip {} >> {} >> {}'.format(board.step, ship.position, direc, oppo_ship.position))
        moveShip(ship, direc, next_cell.position, oppo_ship.position)

        if ship_states[ship.id].state == ShipState.ATTACK_SHIP:
            ship_states[ship.id].timeout -= 1
        else:
            ship_states[ship.id] = ShipState(ShipState.ATTACK_SHIP, oppo_ship.position, 1)

        last_attack_time[ship.id] = board.step
        attacked_freys[oppo_ship.player_id, oppo_ship.id] = oppo_ship.position

        if ship.halite == 0:
            board_info.addAttackShip(ship)
        return True

    def oppoShipIsFrey(oppo_ship):

        if getAxisDistance(oppo_ship.position, CENTER) > board_info.getVicinity():
            return False

        oppo_safe_moves = board_info.getPredictedOppoMoves(oppo_ship)
        if (len(oppo_safe_moves) == 0):
            return True

        if (len(oppo_safe_moves) == 1):
            return True

        if board_info.oppoShipNearYardForLong(oppo_ship):
            return True

        if board_info.getDistToShipyard(oppo_ship.position) < ATTACK_NEAR_YARD and oppo_ship.halite > 0:
            return True

        return False
    
    def formListOfFreys():
        global frey_states, frey_slots
        frey_states, frey_slots = {}, {}

        for oppo in board.opponents:
            for oppo_ship in oppo.ships:
                zero_halite = (oppo_ship.halite == 0)
                in_vicinity = getAxisDistance(oppo_ship.position, CENTER) <= board_info.getVicinity()
                near_yard = board_info.getDistToShipyard(oppo_ship.position) < ATTACK_NEAR_YARD
                near_yard_long = board_info.oppoShipNearYardForLong(oppo_ship)
                move_restricted = (len(board_info.getPredictedOppoMoves(oppo_ship)) <= 1)

                if zero_halite and near_yard_long :
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_0_NEAR_YARD_LONG
                elif zero_halite and in_vicinity and move_restricted:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_0_RESTRICTED_MOVE
                elif zero_halite and in_vicinity:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_0_IN_VICINITY
                elif zero_halite:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_0_OUT_VICINITY
                elif not zero_halite and near_yard:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_POSITIVE_NEAR_YARD
                elif not zero_halite and in_vicinity and move_restricted:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_POSITIVE_RESTRICTED_MOVE
                elif not zero_halite and in_vicinity:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_POSITIVE_IN_VICINITY
                else:
                    frey_states[(oppo_ship.player_id, oppo_ship.id)] = FreyState.HALITE_POSITIVE_OUT_VICINITY

                frey_slots[(oppo_ship.player_id, oppo_ship.id)] = FreyState.getSlots(frey_states[(oppo_ship.player_id, oppo_ship.id)])

    def attackNearYardShips():
        global frey_slots

        # attack HALITE_0_NEAR_YARD_LONG
        freys = [
            oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
            if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_0_NEAR_YARD_LONG and
               len(board_info.getPredictedOppoMoves(oppo_ship)) >= 2 
        ]
        for frey in freys:
            if frey_slots[(frey.player_id, frey.id)] > 0:
                my_ships = [ship for ship in me.ships if shouldAttackOppoShip(ship, frey)]
                my_ships = sorted(my_ships, key=lambda ship: getDistance(ship.position, frey.position))
                for ship in my_ships:
                    if shouldAttackOppoShip(ship, frey):
                        if doAttackOppoShip(ship, frey):
                            frey_slots[(frey.player_id, frey.id)] -= 1
                            if frey_slots[(frey.player_id, frey.id)] == 0:
                                break
        
        # attack HALITE_POSITIVE_NEAR_YARD
        freys = [
            oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
            if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_POSITIVE_NEAR_YARD
        ]
        for frey in freys:
            if frey_slots[(frey.player_id, frey.id)] > 0:
                my_ships = [ship for ship in me.ships if shouldAttackOppoShip(ship, frey)]
                my_ships = sorted(my_ships, key=lambda ship: getDistance(ship.position, frey.position))
                for ship in my_ships:
                    if shouldAttackOppoShip(ship, frey):
                        if doAttackOppoShip(ship, frey):
                            frey_slots[(frey.player_id, frey.id)] -= 1
                            if frey_slots[(frey.player_id, frey.id)] == 0:
                                break

    def attackRestrictedShips():
        global frey_slots
        # attack HALITE_POSITIVE_RESTRICTED_MOVE
        freys = [
            oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
            if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_POSITIVE_RESTRICTED_MOVE
        ]
        for frey in freys:
            if frey_slots[(frey.player_id, frey.id)] > 0:
                my_ships = [ship for ship in me.ships if shouldAttackOppoShip(ship, frey)]
                my_ships = sorted(my_ships, key=lambda ship: getDistance(ship.position, frey.position))
                for ship in my_ships:
                    if shouldAttackOppoShip(ship, frey):
                        if doAttackOppoShip(ship, frey):
                            frey_slots[(frey.player_id, frey.id)] -= 1
                            if frey_slots[(frey.player_id, frey.id)] == 0:
                                break

    def attackPositiveShipsInVicinity():
        if not board_info.needMoreAttackShip():
            return

        global frey_slots
        #attack HALITE_POSITIVE_IN_VICINITY
        freys = [
            oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
            if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_POSITIVE_IN_VICINITY
        ]

        my_ships = [ship for ship in me.ships if ship.halite == 0 and shipSuitableAttackOppoShip(ship)]

        attack_line = MyHeap(key=lambda x: x[0][0][0])
        for ship in my_ships:
            weaker_freys = [frey for frey in freys if frey.halite > ship.halite]
            if len(weaker_freys):
                targets = [(getDistance(ship.position, frey.position), frey) for frey in weaker_freys]
                targets = sorted(targets, key=lambda x: (x[0], -x[1].halite))

                attack_line.push((targets, ship))

        while len(attack_line) and board_info.needMoreAttackShip():
            targets, ship = attack_line.pop()
            if ship.id in ship_actions:
                continue

            target = targets[0]
            _, frey = target

            if (frey_slots[(frey.player_id, frey.id)] > 0 and
                 doAttackOppoShip(ship, frey)
                ):
                frey_slots[(frey.player_id, frey.id)] -= 1
            else:
                if len(targets) > 1:
                    attack_line.push((targets[1:], ship))

    def threatenEmptyShipsInVicinity():
        if not board_info.needMoreAttackShip():
            return

        #attack HALITE_0_IN_VICINITY
        my_ships = [ship for ship in me.ships if ship.halite == 0 and shipSuitableAttackOppoShip(ship)]
        threaten_line = MyHeap(key=lambda x: x[0][0][0])
        
        # add empty-freys to queue
        empty_freys = [
            oppo_ship for oppo in board.opponents for oppo_ship in oppo.ships 
            if frey_states[oppo_ship.player_id, oppo_ship.id] == FreyState.HALITE_0_IN_VICINITY
        ]
        if len(empty_freys):
            for ship in my_ships:
                targets = [(getDistance(ship.position, frey.position), frey) for frey in empty_freys]
                targets = sorted(targets, key=lambda x: x[0])

                threaten_line.push((targets, ship))        

        global frey_slots

        while len(threaten_line) and board_info.needMoreAttackShip():
            targets, ship = threaten_line.pop()
            if ship.id in ship_actions:
                continue

            target = targets[0]
            _, frey = target

            if (frey_slots[(frey.player_id, frey.id)] > 0 and
                 doThreatenOppoShip(ship, frey)
                ):
                frey_slots[(frey.player_id, frey.id)] -= 1
            else:
                if len(targets) > 1:
                    threaten_line.push((targets[1:], ship))

    def run_shipAttackOppoShip():
        attackNearYardShips()
        attackRestrictedShips()
        attackPositiveShipsInVicinity()
        # threatenEmptyShipsInVicinity()

    def setShipAttack(ship):
        if shipSuitableAttackOppoShip(ship):
            for frey in board_info.getNearestOppoShips(ship.position, ship.halite+1):
                if len(movesToAttack(ship, frey.position, frey.halite)) > 0:
                    doAttackOppoShip(ship, frey)
                    return True
        return False

    guarded_ships = set()
    def run_shipGuarding():
        def getNewPos(ship):
            direc = ship_actions.get(ship.id, None)
            new_pos = getLocationByMove(ship.cell, direc)
            return new_pos

        def shipNextStepStuck(ship):
            new_pos = getNewPos(ship)

            if len([next_cell for direc, next_cell in getCellDirections(board_info.cell[new_pos.x][new_pos.y], include_stay=True) 
                        if board_info.isSafeMoveTo(next_cell.position, new_pos, ship.halite, safe_level=2)]) == 0:
                return True

            return False

        def goGuard(guard, target_pos):
            if guard.id in ship_actions:
                return True

            if not board_info.needMoreAttackShip():
                return True

            moves = getCellDirections(guard.cell, include_stay=(guard.cell.halite==0))
            moves = [(direc, next_cell) for direc, next_cell in moves if board_info.isSafeMoveTo(next_cell.position, guard.position, guard.halite)]
            moves = sorted(moves, key=lambda move: getDistance(move[1].position, target_pos))

            if len(moves) == 0:
                return False

            direc, next_cell = moves[0]
            curr_dist = getDistance(guard.position, target_pos)
            new_dist = getDistance(next_cell.position, target_pos)
            if new_dist <= curr_dist:
                moveShip(guard, direc, next_cell.position, next_cell.position)
                ship_states[guard.id] = ShipState(ShipState.GUARD, next_cell.position, None)
                printyellow('{}: tell {} to go {} to guard {}'.format(board.step, guard.position, direc, target_pos))

                return True

            return False
        # -----------------
        if not board_info.needMoreAttackShip():
            return

        guarding_line = MyHeap(key=lambda x: x[0][0][:2])
        
        guards = [ship for ship in me.ships if ship.halite == 0 and shipSuitableAttackOppoShip(ship)]
        rich_ships = board_info.getRichShipsNeedGuards()
        rich_ships = [ship for ship in rich_ships if shipNextStepStuck(ship)]

        for guard in guards:            
            targets = []
            for ship in rich_ships:
                new_pos = getNewPos(ship)
                distance = getDistance(guard.position, new_pos)
                if distance <= 2:
                    targets.append((distance, -ship.halite, new_pos, ship))
            targets = sorted(targets, key=lambda x: x[:2])
            if len(targets):
                guarding_line.push((targets, guard))

        while len(guarding_line):
            targets, guard = guarding_line.pop()
            distance, _, new_pos, rich_ship = targets[0]

            if guard.id in ship_actions:
                continue

            if rich_ship.id not in guarded_ships and goGuard(guard, new_pos):
                guarded_ships.add(rich_ship.id)
                board_info.addAttackShip(guard)
            else:
                if len(targets) > 1:
                    guarding_line.push((targets[1:], guard))

    # mine and store
    set_targets = set()
    def shipMineAndStore(ship_min_halite):

        def shouldStore(ship):
            # no shipyard to store
            if board_info.getNumShipyards() == 0:
                return False

            # time is almost out
            if ship.halite > 50 and board_info.getDistToShipyard(ship.position) + 3 >= (399 - board.step):
                return True

            # contains quite a few halites and near enemy
            if ship.halite > THRESHOLD_STORE[0] and not board_info.isPositionSafe(ship.position, ship.halite, safe_level=2):
                return True

            # contain many halites and enemy is in sight
            if ship.halite > THRESHOLD_STORE[1] and not board_info.isPositionSafe(ship.position, ship.halite, safe_level=3):
                return True

            # extremely heavy
            if ship.halite > THRESHOLD_STORE[2] :
                return True

            # on the way back
            if (ship_states[ship.id].state == ShipState.STORE and
                ship.halite > THRESHOLD_STORE[0]-20): # THRESHOLD_STORE might fluctuate
                return True

            return False

        def isHookingWellNow(ship):
            # check if the ship is hooking
            if ship_states[ship.id].state != ShipState.MINE or ship_states[ship.id].target != ship.position:
                return False
            # check if its position is safe
            # if not board_info.isPositionSafe(ship.position, ship.halite, safe_level=1):
            if not board_info.isSafeMoveTo(ship.position, ship.position, ship.halite, safe_level=1):
                return False

            # if ship.halite > 600 and not board_info.isPositionSafe(ship.position, ship.halite, safe_level=2):
            if ship.halite > 600 and not board_info.isSafeMoveTo(ship.position, ship.position, ship.halite, safe_level=2):
                return False
            # check if the hooking is good
            if board.step < 50:
                if board_info.getHaliteAtPos(ship.position) < max(50, board_info.getMinHaliteMine()):
                    return False
            else:
                if board_info.getHaliteAtPos(ship.position) < max(80, board_info.getMinHaliteMine()):
                    return False

            return True

        def setMineOrStoreTargets(ships):
            hooking_line = MyHeap(key=lambda x: x[0][0][0]) # only for uniform
            store_line = MyHeap(key=lambda x: x[0][0][0])
            targeted_line = MyHeap(key=lambda x: x[0][0][0])
            waiting_line = MyHeap(key=lambda x: x[0][0][0])

            ships = set(ships)

            def pushStoreLine(ship):
                nearest_shipyard_pos = board_info.getNearestShipyardPos(ship.position)
                ship_states[ship.id] = ShipState(ShipState.STORE, nearest_shipyard_pos, None)
                targets = [(
                    board_info.getDistToShipyard(ship.position), 
                    nearest_shipyard_pos
                )]
                store_line.push((targets, ship))
                ships.remove(ship)

            def pushHookingLIne(ship):
                # ship_states doesn't change
                ship_states[ship.id] = ShipState(ShipState.MINE, ship.position, None)
                targets = [(0, ship.position)]
                hooking_line.push((targets, ship))
                set_targets.add(ship.position)
                ships.remove(ship)

            def pushTargetedLine(ship):
                target = ship_states[ship.id].target
                targets = [(getDistance(ship.position, target), target)]
                targeted_line.push((targets, ship))
                set_targets.add(target)
                ships.remove(ship)

            def pushWaitingLine(ship):
                ship_states[ship.id] = ShipState(ShipState.MINE, None, None)
                targets = board_info.getSuggestedMinePos(
                    ship.position, 
                    ship.halite,
                    return_length=board_info.getNumMyShips()*2,
                    must_include_yard=False
                )
                
                waiting_line.push((targets, ship))
                ships.remove(ship)
            # -----------------
            examining_ships = list(ships)
            for ship in examining_ships:
                if shouldStore(ship): # these ships go to shipyard
                    pushStoreLine(ship)

            examining_ships = list(ships)
            for ship in examining_ships:
                if (ship_states[ship.id].state == ShipState.MINE and 
                    ship_states[ship.id].target is not None and
                    ship.position == ship_states[ship.id].target and
                    ship.cell.halite <= board_info.getMinHaliteMine()
                   ):
                    if ship.halite > 0:
                        pushStoreLine(ship)
                    else:
                        pushWaitingLine(ship)

            examining_ships = list(ships)
            for ship in examining_ships:
                if isHookingWellNow(ship): # these ships stay to hook more halites
                    pushHookingLIne(ship)

            examining_ships = list(ships)
            for ship in examining_ships: # these ships keep going to their target
                if (ship_states[ship.id].state == ShipState.MINE and 
                    ship_states[ship.id].target is not None and
                    ship.position != ship_states[ship.id].target and
                    board_info.isPositionSafe(ship_states[ship.id].target, ship.halite, safe_level=1) and
                    ship_states[ship.id].target not in set_targets 
                   ):
                    pushTargetedLine(ship)

            examining_ships = list(ships)
            for ship in examining_ships: # the remaining ships go find more halites / or might stay hooking
                pushWaitingLine(ship)

            return hooking_line, store_line, targeted_line, waiting_line
        
        def arrangeMoveShip(ship, dest):
            def findBestDirection(ship, dest, max_dist=15):
                source = ship.position

                # --- find really safe moves ---
                # find good swaps
                safe_swaps = board_info.safeSwapsForShip(ship, brave=0)
                for direc, next_ship in safe_swaps:
                    if shouldSwap(ship, next_ship):
                        return 'swap', (direc, next_ship)

                # find real pathway
                direc, dist = board_info.getRealSafePathAndDistance(ship.cell, dest, ship.halite, max_dist=max_dist)
                if dist <= max_dist:
                    return 'normal', direc

                # find approx pathway
                direcs = getDirection(source, dest, board_info.size)
                for direc in direcs:
                    next_step = getLocationByMove(ship.cell, direc)
                    if board_info.isSafeMoveTo(next_step, ship.position, ship.halite):
                        return 'normal', direc

                # --- find safe moves that ignore equal oppo ships ---
                # find good swaps
                safe_swaps = board_info.safeSwapsForShip(ship, brave=1)
                for direc, next_ship in safe_swaps:
                    # if shouldSwap(ship, next_ship):
                    return 'swap', (direc, next_ship)

                # find real pathway
                direc, dist = board_info.getRealSafePathAndDistance(ship.cell, dest, ship.halite-1, max_dist=max_dist)
                if dist <= max_dist:
                    return 'normal', direc

                # find approx pathway
                direcs = getDirection(source, dest)
                for direc in direcs:
                    next_step = getLocationByMove(ship.cell, direc)
                    if board_info.isSafeMoveTo(next_step, ship.position, ship.halite-1):
                        return 'normal', direc

                # come to here means no moves are safe.
                # stay. i.e. doing nothing OR move to cell with the least oppo ships, near center, least halite, no ally, no oppo yard
                # print('{}: no way at {}'.format(board.step, ship.position))
                
                moves = []
                for direc, next_cell in getCellDirections(ship.cell) + [(None, ship.cell)]:
                    if ((board_info.myShipIn(next_cell.position) - (direc is None)) == 0 and
                        board_info.getDistToOpponentShipyard(next_cell.position) > 0 
                       ):
                        predators = 0
                        for _, near_cell in getCellDirections(next_cell) + [(None, next_cell)]:
                            if (near_cell.ship is not None and 
                                near_cell.ship.player_id != me.id and 
                                near_cell.ship.halite < ship.halite
                               ):
                                predators += 1

                        if next_cell.ship is not None:
                            moves.append((
                                predators, 
                                board_info.getDistToShipyard(next_cell.position), 
                                next_cell.halite + next_cell.ship.halite + (direc is None)*1e6, 
                                direc))
                        else:
                            moves.append((
                                predators, 
                                board_info.getDistToShipyard(next_cell.position), 
                                next_cell.halite, 
                                direc))

                moves = sorted(moves, key=lambda x: x[:3])
                if len(moves) > 0:
                    return 'normal', moves[0][3]

                # moves = []
                # for direc, next_cell in getCellDirections(ship.cell) + [(None, ship.cell)]:
                #     if ((board_info.myShipIn(next_cell.position) - (direc is None)) == 0 and
                #         board_info.getDistToOpponentShipyard(next_cell.position) > 0 
                #        ):
                #         if next_cell.ship is not None:
                #             moves.append((next_cell.halite + next_cell.ship.halite + (direc is None)*1e6, direc))
                #         else:
                #             moves.append((next_cell.halite, direc))
                
                # moves = sorted(moves, key=lambda x: x[0])
                # if len(moves) > 0:
                #     return 'normal', moves[0][1]

                # no way out, just stay and endure
                # print('{}: absolutely no way at {}'.format(board.step, ship.position))
                return 'normal', None

            # --------------------------
            find_best_direction = findBestDirection(ship, dest)
            move_type = find_best_direction[0]

            if move_type == 'swap':
                direction, next_ship = find_best_direction[1]
                swapShips(ship, next_ship)
                # print('{}: swap ships at {} and {}'.format(board.step, ship.position, next_ship.position))
            else: # move_type == 'normal'
                direction = find_best_direction[1]
                next_step = getLocationByMove(ship.cell, direction)
                moveShip(ship, direction, next_step, dest)

        def executeMineOrStore(role_settings):

            def pushToStoreLine(ship):
                nearest_shipyard_pos = board_info.getNearestShipyardPos(ship.position)
                ship_states[ship.id] = ShipState(ShipState.STORE, nearest_shipyard_pos, None)
                targets = [(
                    board_info.getDistToShipyard(ship.position), 
                    nearest_shipyard_pos
                )]
                store_line.push((targets, ship))

            def pushToTargetedLine(ship, dest_pos):
                # print('{}: pushToTargetedLine {}'.format(board.step, ship.position))
                dist = getDistance(ship.position, dest_pos)
                ship_states[ship.id].target = dest_pos
                targeted_line.push( ([(dist, dest_pos)], ship) )
                set_targets.add(dest_pos)

            def pushToTargetedLineAttack(ship):
                # print('{}: pushToTargetedLineAttack {}'.format(board.step, ship.position))
                frey_pos = board_info.getNearestOppoShipPos(ship.position, min_halite=ship.halite+1)
                if board_info.getAxisDistToCenter(frey_pos) > board_info.getVicinity():
                    board_info.addToVicinity(1)
                ship_states[ship.id] = ShipState(ShipState.ATTACK_SHIP, frey_pos, 2)
                targets = [(
                    getDistance(ship.position, frey_pos), 
                    frey_pos
                )]
                targeted_line.push((targets, ship))

            # ---------------------------------
            hooking_line, store_line, targeted_line, waiting_line = role_settings

            # set target for waiting_line
            while len(waiting_line) > 0:
                targets, ship = waiting_line.pop()
                if ship.id in ship_actions:
                    continue

                target = targets[0]
                _, dest_pos = target

                if (dest_pos in set_targets or
                    board_info.getHaliteAtPos(dest_pos) <= board_info.getMinHaliteMine()
                   ): # mining target is occupied or dest is a poor mine
                    del targets[0]
                    if len(targets) > 0:
                        waiting_line.push((targets, ship))
                    else:
                        if ship.halite > 0:
                            pushToStoreLine(ship)
                        else:
                            pushToTargetedLineAttack(ship)
                else:
                    pushToTargetedLine(ship, dest_pos)
                    # print('{}: id ({}), dest ({})'.format(board.step, ship.id, dest_pos))

            # execute all determined-ships
            while len(hooking_line) > 0:
                targets, ship = hooking_line.pop()
                dist, dest_pos = targets[0]
                if ship.id not in ship_actions:
                    arrangeMoveShip(ship, dest_pos)
            while len(store_line) > 0:
                targets, ship = store_line.pop()
                dist, dest_pos = targets[0]
                if ship.id not in ship_actions:
                    arrangeMoveShip(ship, dest_pos)
            while len(targeted_line) > 0:
                targets, ship = targeted_line.pop()
                dist, dest_pos = targets[0]
                if ship.id not in ship_actions:
                    arrangeMoveShip(ship, dest_pos)

        def run():
            # the other ships either mine or store
            remaining_ships = [ship for ship in me.ships if ship.id not in ship_actions and ship.halite >= ship_min_halite]
            role_settings = setMineOrStoreTargets(remaining_ships)
            executeMineOrStore(role_settings)

        run()

    # ---------------------------------

    def moveShip(ship, direc, next_step, dest, postpone_ship_assure=False, override=False):
            # if board.step == 187:# and ship.position == Point(13, 6):
            #     print('step {}: try to move {} ({}) from {} >> {} with dest {}'.format(board.step, ship.id, ship.halite, ship.position, direc, dest))
            #     print('ship state: {}'.format(ship_states[ship.id].state))
            #     print('caller: ', inspect.getouterframes(inspect.currentframe(), 2)[1][3])

            if not override and ship.id in ship_actions:
                print('{}: block attempt to re-assign action for {}, {}, {}, {}'.format(
                    board.step, ship.id, ship.position, next_step, dest))
                print('caller: ', inspect.getouterframes(inspect.currentframe(), 2)[1][3])
                return

            ship_actions[ship.id] = direc
            board_info.shipMove(ship, next_step)
            ship_states[ship.id].target = dest
            #
            if not postpone_ship_assure:
                rescueStrandedShips()

    # ---------------------------------
    formListOfFreys()
    shipConvertToShipyard()
    protectYardEndgame()
    shipCollideEndGame()
    shipProtectShipyard()
    shipAttackShipyard()
    rescueStrandedShips()
    run_SwapShips()
    shipMineAndStore(ship_min_halite=1)
    run_shipGuarding() # only call run_shipGuarding() after have done moving positive ships
    run_shipAttackOppoShip()
    shipMineAndStore(ship_min_halite=0)
    
def lastCheck(board, me, size):
    #
    for shipyard in me.shipyards:
        shipyard.next_action = shipyard_actions.get(shipyard.id, None)
    for ship in me.ships:
        ship.next_action = ship_actions.get(ship.id, None)

    #
    for shipyard in me.shipyards:
        if (shipyard.next_action == ShipyardAction.SPAWN and
            shipyard.cell.ship is not None and
            shipyard.cell.ship.next_action is None
            ):
            shipyard.next_action = None
        if (shipyard.next_action == ShipyardAction.SPAWN and
            board_info.myShipIn(shipyard.position) > 1
           ):
            shipyard.next_action = None
            # printred('{}: Almost has a crash at shipyard {}'.format(board.step, shipyard.position))

    # 
    # for ship in me.ships:
    #     if ship.id not in ship_actions:
    #         printred('{}: ship {} does not has action'.format(board.step, ship.position))

    #
    if board.step % 20 == 0:
        print('{}: perc attack ship: {:.2%} ({})'.format(
            board.step, 
            board_info.num_attack_ship / max(1, board_info.getNumMyShips()), 
            board_info.num_attack_ship)
        )
        print('num guards: {}'.format(len([1 for ship in me.ships if ship_states[ship.id].state == ShipState.GUARD])))

    #
    # if board.step % 5 == 0:
        # print('{}: Ship: {}'.format(board.step, len(me.ships)))
        # store_ships = [ship for ship in me.ships if ship_states[ship.id].state == ShipState.STORE]
        # print('{:3} ship store: {}'.format(len(store_ships), [ship.halite for ship in store_ships]))
        # mine_ships = [ship for ship in me.ships if ship_states[ship.id].state == ShipState.MINE]
        # print('{:3} ship mine: {}'.format(len(mine_ships), [ship.halite for ship in mine_ships]))
        # print('board halite:', board_info.getBoardHalite())

    #
    # for ship in me.ships:
        # print(ship.position, ship_states[ship.id].state, ship.next_action)

def printred(text):
    print("\33[31m{}\033[0m".format(text))
def printyellow(text):
    print("\33[33m{}\x1b[0m".format(text))
def printblue(text):
    print("\33[34m{}\x1b[0m".format(text))
def printgreen(text):
    print("\33[32m{}\x1b[0m".format(text))
def printviolet(text):
    print("\33[35m{}\x1b[0m".format(text))

def agent(obs, config):
    start_time = time.time()

    size = config.size
    board = Board(obs, config)
    me = board.current_player

    global board_info
    board_info = BoardInfo(obs, config)

    updateStates(board, me)
    
    handleShipYards(board, me, size)
    handleShips(board, me, size)

    lastCheck(board, me, size)

    if board.step < 2 or board.step % 40 == 0 or board.step > 397:
        print('Bot 108: finish step {} in {:.2f} s'.format(board.step, time.time() - start_time))
    if board.step == 300 or board.step == 370:
        printgreen('{}: predict halite: {}'.format(board.step, board_info.player_approx_worth))
        print ('my player id: {}'.format(me.id))
        print('oppo id: {}'.format([oppo.id for oppo in board.opponents]))
        print('my worth rank {}'.format(board_info.getWorthRank()))
    if board.step > 0 and board.step % 40 == 0:
        print('--------------------------------------------------')

    return me.next_actions

# ============================
