# ground_plane_scene.py

import argparse
from isaaclab.app import AppLauncher

# Parse CLI args (optional)
parser = argparse.ArgumentParser(description="Simple Isaac Lab scene with ground plane.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab modules ---
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg



# --- Define scene config with ground + light ---
class GroundPlaneSceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(10.0, 10.0),
            color=(0.4, 0.4, 0.4),
            visible=True
        )
    )
    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0)
        )
    )

# --- Main function ---
def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view: (eye_position, look_at_position)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])

    # Build the scene
    scene_cfg = GroundPlaneSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)

    # Reset the simulation (spawns everything)
    sim.reset()

    # Step loop
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


# --- Run ---
if __name__ == "__main__":
    main()
