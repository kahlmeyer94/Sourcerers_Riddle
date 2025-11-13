# Sorcerers_Riddle
A small pygame that illustrates the difficulties of performing symbolic regression.

## Installation
```
conda create --name sr_game --file requirements.txt
conda activate sr_game
python game.py
```

## Adding Levels
Just change the levels specified in `game.py` main function:

```
lm.add_level(Level(2, lambda x: x**2, [-1, 0, 1], 1e-2, ['x', '1', '+', '-', '*']))
```
The parameters are:
- Tree depth (for skeleton on the left)
- Ground truth function R->R
- x values of enemies
- proximity threshold for hitbox
- library symbols (subset of "+","-","*","/","sin","log","exp","x","1","2","sqrt")
