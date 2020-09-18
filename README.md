README.md

[Halite 4](https://www.kaggle.com/c/halite/) was a competitive simulation challenge run by Kaggle in 2020. It is a 4 player resource management game where teams create agents to compete against other teams' agents.

This notebook is part of the solution from team "KhaVo Dan Gilles Robga Tung". In a field of 1143 teams, we placed a provisional 8th at the end of the public phase. We hope to remain in the top 10 once the private 7 day phase completes. You can read our [solution description](https://www.kaggle.com/c/halite/discussion/183312). No ML bot had finished in the top ten of the three forerunner Halite competitions.

Our solution includes code for:
- A notebook to create a numpy dataset from episode.json's
- A notebook to train a model based on the numpy dataset
- Pretrained pytorch weights for the ML agent's model, if you don't want to train.
- A machine Llanguage driven imitation agent.

The ML solution ia to use semantic segmentation of game boards to predict best next actions for a fleet of competing ships, based on imitating thousands of previous competitor games.

If you want to train a model, first download [11GB of games](https://www.kaggle.com/robga/halitegames/).  Processing the games with the dataset notebook will create a 120GB dataset suitable for training. Training can take anywhere from 5-20 hours on a strong consumer GPU.

See you in Halite 5?
