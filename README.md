[Halite 4](https://www.kaggle.com/c/halite/) was a competitive simulation challenge run by Kaggle in 2020. It is a 4 player resource management game where teams create agents to compete against other teams' agents.

This is the solution from team "KhaVo Dan Gilles Robga Tung". In a field of 1143 teams, we placed 8th and scored a Gold Medal. You can read our [solution description](https://www.kaggle.com/c/halite/discussion/183312). No ML bot had finished in the top ten of the three forerunner Halite competitions. In Halite 4, ML bots finished 5th and 8th.

Our solution includes code for:
- A notebook to create a numpy dataset from episode.json's
- A notebook to train a model based on the numpy dataset
- Pretrained pytorch weights for the ML agent's model, if you don't want to train
- A Machine Language driven imitation agent
- A manual algorithm/heuristic bot

The ML solution uses semantic segmentation of game boards to predict best next actions for a fleet of competing ships, based on imitating thousands of previous competitor games.

If you want to train a model, first download [11GB of games](https://www.kaggle.com/robga/halitegames/).  Processing the games with the dataset notebook will create a 120GB dataset suitable for training. Training can take anywhere from 5-20 hours on a strong consumer GPU.

Our team also included a manual algorithm/heuristic bot. In our stable of agents, this added diversity to prevent lower ranked agents reaching our ML bot. This manual bot would have scored position 14 on its own at time of writing.

Our team was
- Kha Vo https://www.kaggle.com/khahuras https://github.com/voanhkha
- Dan Grunberg https://www.kaggle.com/solverworld
- Gilles Vandewiele https://www.kaggle.com/group16
- Rob Gardiner https://www.kaggle.com/robga https://github.com/digitalspecialists
- Tung M Phung https://www.kaggle.com/tungmphung https://github.com/Mothaiba

See you in Halite 5?
