import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
import logging

logger = logging.getLogger('Can\'t Stop')
logger.setLevel(logging.INFO)


class CantStopMultiplayer(gym.Env):
    """
    Can't Stop is a two phased multiplayer game.
    Each turn per player comprises of:
    Phase 0: player chooses to stop or continue.
    Phase 1: player chooses a dice pair.


    Actions on each turn is:
    - Choose to stop or continue (if phase 0)
    - Choose which die pair (if phase 1)

    Observation:
    - Current phase
    - Dice pairs
    - Current board status (starting w current player)

    Rewards:
    - 100 for winning game
    - 10 per complete column
    - x^2 / 10 where x is steps advanced at end of turn
    """
    def __init__(self, n_players, verbose=False):
        self.n_players = n_players

        self.limits = {}
        for i in range(2, 13):
            self.limits[i] = 12 - 2 * abs(7 - i)
        self.reset()
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Discrete(10+4*11)
        self.rewards = {
            '0_stopping': lambda x: x**2/10,  # fn of len(self.current_choices)
            '0_complete_col': 10,  # complete 1 column
            '0_complete_game': 100,  # 3 complete columns
            '0_cont_feas': 0,  # choose to continue, still have feas opts
            '0_cont_no_feas': 0,  # no more feasible options
            '1_choose_roll': 0,  # for each feasible die chosen
            'wrong_phase': -1,  # choose phase 1 when phase 0, v.v.
        }
        logger.info("Environment for {} players initiated with rewards {}"
                    .format(self.n_players, self.rewards))
        logger.debug("DEBUG MODE")

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        # start randomly
        self.current_player = np.random.randint(self.n_players)
        logger.debug("Starting player: {}".format(self.current_player))
        self.board = {}
        # columns always of length 4 so trained agents can be used against
        # any number of players - each item is position of a player
        for i in range(2, 13):
            self.board[i] = [0, 0, 0, 0]
        self.blocked_states = []
        self.n_complete_columns = [0, 0, 0, 0]
        self.current_choices = []
        self.phase = 1  # always start on choosing which roll
        self.roll_dice()
        self.viewer = None
        return self.state

    @property
    def unique_current(self):
        """
        Number of unique columns in current choices
        """
        return len(list(set(self.current_choices)))

    @property
    def state(self):
        """
        observation:
        0: phase
        1 - 6: rolls
        7, 8, 9: current choices. 0 if nothing
        10 onwards: current board (4*11)
        """

        valid = list(set(self.current_choices))
        valid.sort()
        while len(valid) != 3:
            valid.append(0)
        res = [self.phase] + list(np.array(self.roll).reshape(-1)) + valid
        for i in range(2, 13):
            # giving this state to current player
            # always put his position first so any agent can be any player
            res.append(self.board[i][self.current_player])
            for p in [i for i in range(4) if i != self.current_player]:
                res.append(self.board[i][p])
        return [self.current_player, np.array(res)]

    def step(self, action):
        """
        Accepts an action and returns (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent.
                             0: stop
                             1: continue
                             2-7: choice of die. If both dice in pair are poss,
                             both will be chosen.

        Returns:
            observation (object): board, current_choices
            reward (float) : amount of reward returned after previous action
            done (bool): whether the game is complete
            info (dict): auxiliary diagnostic information
        """
        assert self.action_space.contains(action)
        done = False
        reward = 0
        ###########
        # phase 0 #
        ###########
        # choose to stop or continue
        if self.phase == 0:
            # player chose to stop
            if action == 0:
                logger.debug("Player is stopping! Advanced {} steps"
                             .format(len(self.current_choices)))
                reward = self.rewards['0_stopping'](len(self.current_choices))

                # update board
                for c in self.current_choices:
                    self.board[c][self.current_player] += 1
                # cap at max and check if complete
                for b in self.board:
                    self.board[b] = np.minimum(self.board[b], self.limits[b])
                    if (self.board[b][self.current_player] == self.limits[b]
                            and b not in self.blocked_states):
                        # add one complete column
                        # in multiplayer, dict of each player's
                        logger.debug("Player {} completes column {}"
                                     .format(self.current_player, b))
                        self.n_complete_columns[self.current_player] += 1
                        reward += self.rewards['0_complete_col']
                        # block off this number
                        self.blocked_states.append(b)
                # if player won
                if self.n_complete_columns[self.current_player] >= 3:
                    # >= in case complete 3rd and 4th at same time
                    reward += self.rewards['0_complete_game']
                    done = True
                self.current_choices = []
                self.phase = 1
                self.roll_dice()  # update dice values
                self.current_player = \
                    (self.current_player + 1) % self.n_players
                return self.state, reward, done, {}

            # player chose to continue
            if action == 1:
                logger.debug("Player continues!")
                self.phase = 1
                self.roll_dice()  # update dice values
                # if no feasible choice, reset board and all advances lost
                feasible = (sum(self.roll) != 0)
                reward = self.rewards['0_cont_feas']
                if self.unique_current == 3:
                    for d in self.roll:
                        if d not in self.current_choices:
                            feasible = False
                if not feasible:
                    logger.debug("No feasible choice - all advances lost!")
                    self.current_choices = []
                    reward = self.rewards['0_cont_no_feas']
                    self.roll_dice()
                    self.current_player = \
                        (self.current_player + 1) % self.n_players
                return self.state, reward, done, {}

            # player didn't choose 0 or 1 - punishment
            # end game, so don't get stuck here
            return self.state, self.rewards['wrong_phase'], True, {}

        ###########
        # phase 1 #
        ###########
        # player chooses roll
        if action == 0 or action == 1:
            # player didn't choose 0 or 1 - slight punishment
            # end game, so don't get stuck here
            return self.state, self.rewards['wrong_phase'], True, {}

        chosen_rolls = self.choose_roll(action)
        if not chosen_rolls:
            # player didn't choose feasible actions - slight punishment
            # end game, so don't get stuck here
            return self.state, self.rewards['wrong_phase'], True, {}
        logger.debug("Player chose {}".format(chosen_rolls))

        for c in chosen_rolls:
            # mini reward for each possible choice
            reward += self.rewards['1_choose_roll']
            self.current_choices.append(c)
            # reset phase
            self.phase = 0
        # if nothing in chosen_rolls, wrong choice, so don't change anything
        return self.state, reward, done, {}

    def choose_roll(self, action):
        # 2, 3: roll 1
        # 4, 5: roll 2
        # 6, 7: roll 3
        r = self.roll[action - 2]

        if not self.check_feasible(r) or r == 0:
            # r is not feasible
            return []

        # r is feasible, check if other die is feasible
        if action % 2 == 0:
            r2 = self.roll[action - 1]
        else:
            r2 = self.roll[action - 3]

        if r2 == 0:
            return [r]
        if r2 == r:
            return [r, r2]
        # r2 is not in current_choices and already 3 choices
        if not self.check_feasible(r2):
            return [r]
        # r2 already chosen or only zero / one chosen so far: feasible
        if r2 in self.current_choices or self.unique_current < 2:
            return [r, r2]
        # r2 not in current choices and two chosen so far
        if r in self.current_choices:
            # feasible only if r was already in current_choices
            return [r, r2]
        # r wasn't already in current choices
        return [r]

    def roll_dice(self):
        roll = np.random.randint(1, 7, 4)
        self.roll = [roll[0] + roll[1], roll[2] + roll[3],
                     roll[0] + roll[2], roll[1] + roll[3],
                     roll[0] + roll[3], roll[1] + roll[2]]
        self.roll = self.make_feasible(self.roll)

    def check_feasible(self, r):
        # in choices, or at least one more choice
        return (r in self.current_choices or self.unique_current < 3) \
            and (r not in self.blocked_states)

    def make_feasible(self, roll):
        res = []
        for r in roll:
            if self.check_feasible(r):
                res.append(r)
            else:
                res.append(0)
        return res

    def render(self, mode='human'):
        if self.viewer is None:
            # Initialize plots
            self.viewer = PlotsViewer(self.limits, self.n_players)

        # Updates plots
        return self.viewer.render(self.board, self.current_choices,
                                  self.phase, self.roll, self.current_player)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class PlotsViewer():
    def __init__(self, limits, n_players):
        self.n_players = n_players
        self.fig = plt.figure()
        self.board = self.fig.add_subplot(111)
        plt.ion()

        self.x = np.arange(2, 13, 1)
        self.lim = [limits[x] for x in self.x]

        self.fig.show()
        self.fig.canvas.draw()
        self.colours = ['r', 'black', 'y', 'm']
        # Pause graph at start for video
        plt.pause(3)

    def render(self, board, current_choices, phase, roll, current_player):
        self.board.clear()
        # Top limits
        self.board.plot(self.x, self.lim, marker='o', linestyle='None',
                        label='Goal', color='green')
        complete = []
        complete_per_player = {}

        # Players fixed
        # Always plot current player last
        for player in [i for i in range(self.n_players)
                       if i != current_player] + [current_player]:
            complete_per_player[player] = 0
            for x, lim in zip(self.x, self.lim):
                if board[x][player] == lim:
                    self.board.vlines(x, 0, lim,
                                      color=self.colours[player], alpha=0.5)
                    complete.append(x)
                    complete_per_player[player] += 1
            y1 = [board[x][player] for x in self.x]
            self.board.plot(self.x, y1, marker='o', linestyle='None',
                            label='Player {}'.format(player),
                            color=self.colours[player])

        # Vertical lines
        for x, lim in zip(self.x, self.lim):
            if x not in complete:
                self.board.vlines(x, 0, lim, color='green', alpha=0.5)

        # Current player projected
        y2 = [min(board[x][current_player] + current_choices.count(x), lim)
              for x, lim in zip(self.x, self.lim)]
        self.board.plot(self.x, y2, marker='x', linestyle='None',
                        label='Projected (Player {})'.format(current_player),
                        color=self.colours[current_player])

        handles, labels = self.board.get_legend_handles_labels()

        # Sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: t[0]))
        self.board.legend(handles, labels, loc=1)
        self.board.set_ylim([0, 13])
        self.board.set_yticks(np.arange(0, 14, 1))
        self.board.set_xticks(np.arange(2, 13, 1))
        self.board.grid()
        if phase:
            self.board.set_title("Player {}: choose which roll"
                                 .format(current_player))
            choice_string = "1: {}, {}\n2: {}, {}\n3: {}, {}"\
                .format(roll[0], roll[1], roll[2], roll[3], roll[4], roll[5])
            self.board.text(2, 12, choice_string,
                            verticalalignment='top',
                            bbox=dict(facecolor='white'))
        else:
            self.board.set_title("Player {}: choose to stop or continue"
                                 .format(current_player))
        self.fig.canvas.draw()

        end_game = False
        for player in complete_per_player:
            if complete_per_player[player] == 3:
                end_game = True
        if end_game:
            # Pause graph at end for video
            plt.pause(2)
        else:
            plt.pause(0.4)

    def close(self):
        self.board.clear()
        plt.close()
