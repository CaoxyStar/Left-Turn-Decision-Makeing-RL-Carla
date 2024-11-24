import carla
import random
from agents.navigation.behavior_agent import BehaviorAgent

def get_autocar_path(map, lane, direction):
    start_location = carla.Location()
    end_location = carla.Location()
    if lane == 'left':
        start_location.x = 31
        start_location.y = random.uniform(105, 115)
        start_location.z = 0
        if direction == 'forward':
            end_location.x = 30
            end_location.y = 40
            end_location.z = 0
        elif direction == 'turn':
            end_location.x = -20
            end_location.y = 88
            end_location.z = 0
        else:
            print("Please choose forward or turn.")
            return
    elif lane == 'right':
        start_location.x = 35
        start_location.y = random.uniform(105, 115)
        start_location.z = 0
        if direction == 'forward':
            end_location.x = 35
            end_location.y = 40
            end_location.z = 0
        elif direction == 'turn':
            end_location.x = 90
            end_location.y = 82
            end_location.z = 0
        else:
            print("Please choose forward or turn.")
            return
    else:
        print("Please choose left or right.")
        return
    start_waypoint = map.get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    end_waypoint = map.get_waypoint(end_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    return start_waypoint, end_waypoint

def main():
    client = carla.Client('localhost', 2000)
    # client.load_world('Town05')

    world = client.get_world()
    spectator = world.get_spectator()
    map = world.get_map()
    vehicle_bps = world.get_blueprint_library()

    origin_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    spectator_transform = carla.Transform()
    spectator_transform.location.x = 25
    spectator_transform.location.y = 90
    spectator_transform.location.z = 40
    spectator_transform.rotation.pitch = -90
    spectator_transform.rotation.yaw = 90
    spectator_transform.rotation.roll = 0
    spectator.set_transform(spectator_transform)

    ego_vehicle_bp = vehicle_bps.find('vehicle.mercedes.coupe')
    location = carla.Location()
    location.x = 28
    location.y = random.uniform(64, 74)
    location.z = 0
    ego_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, random.choice(map.get_spawn_points()))
    ego_vehicle.set_transform(ego_waypoint.transform)

    auto_vehicle_bp = vehicle_bps.find('vehicle.audi.a2')
    start_waypoint, end_waypoint = get_autocar_path(map, 'left')
    auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    auto_vehicle.set_transform(start_waypoint.transform)

    world.tick()

    agent = BehaviorAgent(auto_vehicle, ignore_traffic_light=True, behavior='normal')
    agent.set_destination(agent._vehicle.get_location(), end_waypoint.transform.location, clean=True)

    while True:
        agent.update_information(auto_vehicle)

        world.tick()
            
        if len(agent._local_planner.waypoints_queue)<1:
            print('======== Success, Arrivied at Target Point!')
            break

        speed_limit = auto_vehicle.get_speed_limit()
        agent.get_local_planner().set_speed(speed_limit)

        control = agent.run_step(debug=True)
        auto_vehicle.apply_control(control)

    auto_vehicle.destroy()
    ego_vehicle.destroy()
    world.tick()
    world.apply_settings(origin_settings)


if __name__ == '__main__':
    main()