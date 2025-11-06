# Material Count Feature

## Overview

The GUI now displays material advantage in the info panel, similar to chess.com and other online chess platforms.

## Piece Values

Standard chess piece values are used:
- **Pawn**: 1 point
- **Knight**: 3 points
- **Bishop**: 3 points
- **Rook**: 5 points
- **Queen**: 9 points
- **King**: 0 points (not counted in material)

## Display Features

### Material Advantage Display

The info panel now shows:

1. **"Material:"** label
2. The advantage status in one of three formats:
   - **"White: +X"** - White is ahead by X points
   - **"Black: +X"** - Black is ahead by X points
   - **"Material: Equal"** - Both sides have equal material

### Color Coding

- **Green text**: You are ahead in material
- **Red text**: You are behind in material
- **Gray text**: Material is equal

## Example Scenarios

### Starting Position
```
Material: Equal
```
Both sides start with 39 points each (8 pawns + 2 knights + 2 bishops + 2 rooks + 1 queen)

### After White Captures a Pawn
```
Material:
White: +1
```
(Displayed in green if you're white, red if you're black)

### After Black Wins a Knight
```
Material:
Black: +3
```
(Displayed in green if you're black, red if you're white)

### Complex Position
If white has captured a rook and pawn (6 points) but black has captured a queen (9 points):
```
Material:
Black: +3
```

## How It Works

The `get_material_count()` method:
1. Iterates through all squares on the board
2. Counts up the point value of pieces for each side
3. Calculates the difference
4. Displays the advantage with appropriate color coding

This helps you:
- Track tactical exchanges
- Know when you're ahead or behind
- Make informed decisions about trading pieces
- Understand your position at a glance
