"""ManiSkill PickCube plus srmp.read_sim, focused on the ground plane.

The ground actor sits at the origin. Its collision plane has a rotated local
pose near z = -0.92. This script prints that pose, imports the SAPIEN scene,
and separates the table from the ground:

- full import: the reset configuration collides
- table removed, ground kept: the reset configuration does not collide
- ground only: distance from the robot to the plane at the reset pose

On srmp 0.2.2 the ground-only clearance is about 0.92 m. A false collision
against the ground alone does not show up; the reset collision is the table.

    pip install "mani_skill==3.0.1" "srmp==0.2.2" "numpy<2"
    # Python 3.11. Vulkan is required for the CPU renderer.
    python scripts/repro_srmp_read_sim_ground.py
"""

import numpy as np
import sapien.physx as physx

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import srmp


def main() -> None:
    env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        sim_backend="physx_cpu",
        render_backend="cpu",
        num_envs=1,
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped
    scene = unwrapped.scene.sub_scenes[0]

    for actor in scene.get_all_actors():
        if "ground" not in actor.get_name():
            continue
        pose = actor.get_pose()
        print(f"ground actor {actor.get_name()}")
        print(f"  actor pose p={np.round(pose.p, 4)} q={np.round(pose.q, 4)}")
        component = actor.find_component_by_type(physx.PhysxRigidBaseComponent)
        for shape in component.get_collision_shapes():
            local = shape.get_local_pose()
            print(f"  {type(shape).__name__} local p={np.round(local.p, 4)} q={np.round(local.q, 4)}")

    planner = srmp.PlannerInterface()
    robot = planner.add_robot("panda")
    base = srmp.Pose()
    base.p = np.asarray(unwrapped.agent.robot.pose.p[0].detach().cpu().numpy(), dtype=np.float64)
    base.q = np.array([1.0, 0.0, 0.0, 0.0])
    planner.set_base_pose(robot, base)
    qpos = np.asarray(unwrapped.agent.robot.qpos[0, :7].detach().cpu().numpy(), dtype=np.float64)
    planner.set_qpos(robot, qpos)
    planner.set_gripper_qpos(robot, np.array([0.04, 0.04]))
    print(f"robot base p={np.round(base.p, 4)}")
    print(f"reset collide before import: {planner.is_state_colliding(robot)}")

    imported = planner.read_sim(scene, "sapien", articulations=["panda"])
    print(f"read_sim: {imported}")
    print(f"reset collide after import: {planner.is_state_colliding(robot)}")

    for actor_name, objects in imported.items():
        if "table" in actor_name:
            for name in objects:
                planner.remove_object(name)
    print(f"reset collide after removing the table: {planner.is_state_colliding(robot)}")

    for actor_name, objects in imported.items():
        if "ground" not in actor_name:
            for name in objects:
                if planner.has_object(name):
                    planner.remove_object(name)
    print(f"objects left: {planner.get_object_names()}")
    planner.set_qpos(robot, qpos)
    print(f"ground-only reset collide: {planner.is_state_colliding(robot)}")
    print(f"ground-only distance at reset: {planner.distance_to_robot_collision():.4f} m")
    env.close()


if __name__ == "__main__":
    main()
