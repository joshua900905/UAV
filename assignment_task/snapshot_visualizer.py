"""
Lightweight snapshot visualizer for the three-phase perimeter simulation.
- Uses the existing planner and simulator code in this folder (imports from compare_simplified)
- Ensures UAV 0 and 1 follow the perimeter (three-phase allocation), does NOT plan inner paths initially
- Runs the simulation step-by-step and saves a PNG snapshot each time any outer UAV completes its outer path

Usage (from assignment_task folder):
    python snapshot_visualizer.py --grid-size 12 --num-uavs 8 --seed 42 --max-time 300 --donut --hamiltonian

Outputs:
    snapshot_01.png, snapshot_02.png, ... (one snapshot each time an outer UAV finishes)

Note: this script is intentionally lightweight and reuses Simulator and planner implementations from
`compare_simplified.py`. If you changed class names, adjust imports accordingly.
"""

import os
import sys
import argparse
import math
import matplotlib.pyplot as plt

# Import simulator & planner from compare_simplified.py
# This file expects the following symbols to be available in compare_simplified.py:
# - Environment, Simulator, BoustrophedonPlanner (or Planner used there)
# If you renamed classes, update the imports below.
try:
    from compare_simplified import Environment, Simulator, BoustrophedonPlanner
except Exception as e:
    print("Failed to import required classes from compare_simplified.py:\n", e)
    print("Make sure compare_simplified.py defines Environment, Simulator and BoustrophedonPlanner.")
    raise


def plot_snapshot(env, sim, snap_idx, out_dir):
    """Plot current positions, path progress and discovered targets and save PNG."""
    # Increase figure size to accommodate more UAVs (up to 12)
    fig, ax = plt.subplots(figsize=(10, 8))
    grid = env.grid_size

    # draw grid - extend limits slightly to show edge cells completely
    ax.set_xlim(-0.5, grid + 0.5)
    ax.set_ylim(-0.5, grid + 0.5)
    ax.set_xticks(range(grid + 1))
    ax.set_yticks(range(grid + 1))
    ax.grid(True, color='#cccccc', linewidth=0.5)
    ax.set_aspect('equal')

    # draw reserved area if present
    planner = sim.planner
    if hasattr(planner, 'reserved_area') and planner.reserved_area:
        xs = [c[0] for c in planner.reserved_area]
        ys = [c[1] for c in planner.reserved_area]
        ax.scatter([x+0.5 for x in xs], [y+0.5 for y in ys], s=120, c='lightblue', alpha=0.25, edgecolor='none')

    # draw targets: discovered vs undiscovered
    dx_disc = [t.x+0.5 for t in env.targets if t.discovered]
    dy_disc = [t.y+0.5 for t in env.targets if t.discovered]
    dx_und = [t.x+0.5 for t in env.targets if not t.discovered]
    dy_und = [t.y+0.5 for t in env.targets if not t.discovered]
    ax.scatter(dx_und, dy_und, marker='x', c='black', label='undiscovered', s=100)
    ax.scatter(dx_disc, dy_disc, marker='o', facecolors='none', edgecolors='green', s=100, linewidths=2, label='discovered')

    # Use a color map for up to 12 UAVs
    import matplotlib.cm as cm
    colors = cm.tab20(range(env.num_uavs)) if env.num_uavs <= 20 else cm.rainbow(range(env.num_uavs))
    
    # draw UAV paths and positions
    for uav in env.uavs:
        color = colors[uav.id] if uav.id < len(colors) else 'blue'
        
        # full path (original assigned path saved in outer_path if available)
        if hasattr(uav, 'outer_path') and uav.outer_path:
            px = [p[0]+0.5 for p in uav.outer_path]
            py = [p[1]+0.5 for p in uav.outer_path]
            ax.plot(px, py, linestyle='--', linewidth=1.0, alpha=0.4, color=color)

        # current path remaining
        if uav.path:
            px2 = [p[0]+0.5 for p in uav.path]
            py2 = [p[1]+0.5 for p in uav.path]
            ax.plot(px2, py2, linestyle='-', linewidth=2.0, alpha=0.7, color=color)

        # current position (position is already at cell center, no offset needed)
        # Add label only for position marker to reduce legend clutter
        ax.plot(uav.position[0], uav.position[1], marker='o', markersize=10,
                color=color, markeredgecolor='black', markeredgewidth=1.5, 
                label=f'UAV {uav.id}')

    ax.set_title(f'Snapshot {snap_idx:02d} - t={env.current_time:.1f}s', fontsize=14, fontweight='bold')
    # Adjust legend to accommodate up to 12+ UAVs with smaller font
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)
    plt.tight_layout()

    fname = os.path.join(out_dir, f'snapshot_{snap_idx:02d}.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'  Saved {fname}')


def run_snapshot_run(args):
    # create environment and planner
    env = Environment(grid_size=args.grid_size, num_uavs=args.num_uavs, seed=args.seed)

    planner = BoustrophedonPlanner(speed=1.0, use_donut_strategy=args.donut)

    # plan assignments (this will set outer paths for UAV 0 and 1 per compare_simplified behaviour)
    all_cells = {(x,y) for x in range(env.grid_size) for y in range(env.grid_size)}
    assignments = planner.plan(all_cells, env.num_uavs, env.gcs_pos)

    # apply paths to env.uavs
    for uid, path in assignments.items():
        if uid < len(env.uavs):
            env.uavs[uid].path = path
            env.uavs[uid].outer_path = path.copy() if path else []

    sim = Simulator(env, planner, 'snapshot_run', collect_metrics=False)

    # run until either max_time or we've generated snapshots equal to number of outer UAVs
    out_dir = os.path.join(os.getcwd(), 'snapshots')
    os.makedirs(out_dir, exist_ok=True)

    max_steps = int(args.max_time / 0.1)
    snapshots_taken = 0
    outer_completed_record = set()

    for step in range(max_steps):
        sim.simulate_step(0.1)

        # check ALL outer UAV completions (any UAV that finishes outer search)
        # We'll capture snapshot when ANY UAV completes its outer search
        for uav in env.uavs:
            if uav.search_complete and uav.id not in outer_completed_record:
                snapshots_taken += 1
                outer_completed_record.add(uav.id)
                print(f'UAV {uav.id} completed outer search at t={env.current_time:.1f}s -> snapshot {snapshots_taken}')
                plot_snapshot(env, sim, snapshots_taken, out_dir)

        # stop if all UAVs done or mission complete
        if env.all_missions_complete():
            break

    print(f'Finished simulation. Saved {snapshots_taken} snapshots to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size', type=int, default=12)
    parser.add_argument('--num-uavs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-time', type=float, default=300.0)
    parser.add_argument('--donut', action='store_true', default=True, help='Use donut strategy (default: True)')
    parser.add_argument('--no-donut', dest='donut', action='store_false', help='Disable donut strategy')
    args = parser.parse_args()

    run_snapshot_run(args)
