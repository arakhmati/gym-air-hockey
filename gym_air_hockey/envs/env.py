import numpy as np
from PIL import Image  # pillow
import sys

import pygame

class Env(object):

    def __init__(self,
                 game, fps=60, frame_skip=1, num_steps=1,
                 reward_values={}, force_fps=True, display_screen=False,
                 add_noop_action=True, state_preprocessor=None, rng=24):

        self.game = game
        self.fps = fps
        self.frame_skip = frame_skip
        self.NOOP = None
        self.num_steps = num_steps
        self.force_fps = force_fps
        self.display_screen = display_screen
        self.add_noop_action = add_noop_action

        self.last_action = []
        self.action = []
        self.previous_score = 0
        self.frame_count = 0

        # update the scores of games with values we pick
        if reward_values:
            self.game.adjustRewards(reward_values)
        
        self.init()

        self.state_preprocessor = state_preprocessor
        self.state_dim = None

        if self.state_preprocessor is not None:
            self.state_dim = self.game.getGameState()

            if self.state_dim is None:
                raise ValueError(
                    "Asked to return non-visual state on game that does not support it!")
            else:
                self.state_dim = self.state_preprocessor(self.state_dim).shape

    def _tick(self):
        """
        Calculates the elapsed time between frames or ticks.
        """
        if self.force_fps:
            return 1000.0 / self.fps
        else:
            return self.game.tick(self.fps)

    def init(self):
        """
        Initializes the game. This depends on the game and could include
        doing things such as setting up the display, clock etc.

        This method should be explicitly called.
        """
        self.game._setup()
        self.game.init() #this is the games setup/init

    def getActionSet(self):
        """
        Gets the actions the game supports. Optionally inserts the NOOP
        action if PLE has add_noop_action set to True.

        Returns
        --------

        list of pygame.constants
            The agent can simply select the index of the action
            to perform.

        """
        actions = self.game.actions

        if (sys.version_info > (3, 0)): #python ver. 3
            if isinstance(actions, dict) or isinstance(actions, dict_values):
                actions = actions.values()
        else:
            if isinstance(actions, dict):
                actions = actions.values()

        actions = list(actions) #.values()
        #print (actions)
        #assert isinstance(actions, list), "actions is not a list"

        if self.add_noop_action:
            actions.append(self.NOOP)

        return actions

    def getFrameNumber(self):
        """
        Gets the current number of frames the agent has seen
        since PLE was initialized.

        Returns
        --------

        int

        """

        return self.frame_count

    def game_over(self):
        """
        Returns True if the game has reached a terminal state and
        False otherwise.

        This state is game dependent.

        Returns
        -------

        bool

        """

        return self.game.game_over()

    def score(self):
        """
        Gets the score the agent currently has in game.

        Returns
        -------

        int

        """

        return self.game.getScore()

    def lives(self):
        """
        Gets the number of lives the agent has left. Not all games have
        the concept of lives.

        Returns
        -------

        int

        """

        return self.game.lives

    def reset_game(self):
        """
        Performs a reset of the games to a clean initial state.
        """
        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.game.reset()

    def getScreenRGB(self):
        """
        Gets the current game screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).


        """

        return self.game.getScreenRGB()

    def getScreenGrayscale(self):
        """
        Gets the current game screen in Grayscale format. Converts from RGB using relative lumiance.

        Returns
        --------
        numpy uint8 array
                Returns a numpy array with the shape (width, height).


        """
        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * \
            frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = np.round(frame).astype(np.uint8)

        return frame

    def saveScreen(self, filename):
        """
        Saves the current screen to png file.

        Parameters
        ----------

        filename : string
            The path with filename to where we want the image saved.

        """
        frame = Image.fromarray(self.getScreenRGB())
        frame.save(filename)

    def getScreenDims(self):
        """
        Gets the games screen dimensions.

        Returns
        -------

        tuple of int
            Returns a tuple of the following format (screen_width, screen_height).
        """
        return self.game.getScreenDims()

    def getGameStateDims(self):
        """
        Gets the games non-visual state dimensions.

        Returns
        -------

        tuple of int or None
            Returns a tuple of the state vectors shape or None if the game does not support it.
        """
        return self.state_dim

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        This can include items such as player position, velocity, ball location and velocity etc.

        Returns
        -------

        dict or None
            It returns a dict of game information. This greatly depends on the game in question and must be referenced against each game.
            If no state is available or supported None will be returned back.

        """
        state = self.game.getGameState()
        if state is not None and self.state_preprocessor is not None:
            return self.state_preprocessor(state)
        else:
            raise ValueError(
                "Was asked to return state vector for game that does not support it!")

    def act(self, action):
        """
        Perform an action on the game. We lockstep frames with actions. If act is not called the game will not run.

        Parameters
        ----------

        action : int
            The index of the action we wish to perform. The index usually corresponds to the index item returned by getActionSet().

        Returns
        -------

        int
            Returns the reward that the agent has accumlated while performing the action.

        """
        return sum(self._oneStepAct(action) for i in range(self.frame_skip))

    def _draw_frame(self):
        """
        Decides if the screen will be drawn too
        """

        self.game._draw_frame(self.display_screen)

    def _oneStepAct(self, action):
        """
        Performs an action on the game. Checks if the game is over or if the provided action is valid based on the allowed action set.
        """
        if self.game_over():
            return 0.0

        if action not in self.getActionSet():
            action = self.NOOP

        self._setAction(action)
        for i in range(self.num_steps):
            time_elapsed = self._tick()
            self.game.step(time_elapsed)
            self._draw_frame()

        self.frame_count += self.num_steps

        return self._getReward()

    def _setAction(self, action):
        """
            Instructs the game to perform an action if its not a NOOP
        """

        if action is not None:
            self.game._setAction(action, self.last_action)

        self.last_action = action

    def _getReward(self):
        """
        Returns the reward the agent has gained as the difference between the last action and the current one.
        """
        reward = self.game.getScore() - self.previous_score
        self.previous_score = self.game.getScore()

        return reward
