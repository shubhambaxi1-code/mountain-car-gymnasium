# Mountain Car Q-Learning

A small reinforcement learning experiment using [Gymnasium](https://gymnasium.farama.org/)'s `MountainCar-v0` environment. The agent learns a policy with tabular Q-learning after converting the environment's continuous position and velocity observations into a discrete `50 x 50` state grid.

## Features

- Epsilon-greedy exploration with exponential decay
- Tabular Q-learning using the Bellman update rule
- Optional loading of an existing Q-table
- Optional visual playback of the learned policy
- Optional saving to `mountain_car_q_table.npy`

## Requirements

- Python 3.9 or newer
- NumPy
- Gymnasium
- Pygame for graphical playback

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy gymnasium[classic-control]
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

Start the training script with:

```bash
python mountain_car.py
```

The script asks whether to:

1. Load `mountain_car_q_table.npy` if it exists.
2. Watch the trained agent play in a graphical window.
3. Save the current Q-table to `mountain_car_q_table.npy`.

Training is configured for 10,000,000 episodes and may take a while. Press `Ctrl+C` to stop training early; the current in-memory Q-table can still be saved when prompted.

## Project Files

| File | Description |
| --- | --- |
| `mountain_car.py` | Environment setup, Q-learning loop, playback, and persistence |
| `mountain_car_q_table.npy` | Saved NumPy Q-table, when generated |
| `README.md` | Project documentation |

## Algorithm

Each state is mapped to a grid cell containing the best known value for each available action. For every transition, the selected action value is updated using:

```text
Q(state, action) = Q(state, action) + alpha * (
	reward + gamma * max(Q(next_state, *)) - Q(state, action)
)
```

The current hyperparameters are:

| Parameter | Value |
| --- | ---: |
| Learning rate (`alpha`) | `0.1` |
| Discount rate (`gamma`) | `0.95` |
| Initial epsilon | `1.0` |
| Minimum epsilon | `0.01` |
| Epsilon decay rate | `0.001` |
| State grid | `50 x 50` |

## Notes

- The saved `.npy` file is ignored by Git because it is generated training data.
- Playback uses `render_mode="human"` and requires a display environment.
- The default environment gives a reward of `-1` per step, so better policies generally finish episodes in fewer steps.
