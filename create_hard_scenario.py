import carla
import random
from agents.navigation.behavior_agent import BehaviorAgent


def main():
    client = carla.Client('localhost', 2000)
    client.load_world('Town05')

    world = client.get_world()
    spectator = world.get_spectator()
    map = world.get_map()
    vehicle_bps = world.get_blueprint_library().filter('*vehicle*')

    origin_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    spectator_transform = carla.Transform()
    spectator_transform.location.x = 25
    spectator_transform.location.y = 90
    spectator_transform.location.z = 50
    spectator_transform.rotation.pitch = -90
    spectator_transform.rotation.yaw = 90
    spectator_transform.rotation.roll = 0
    spectator.set_transform(spectator_transform)

    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.freeze(True)

    ego_vehicle_bp = vehicle_bps.find('vehicle.mercedes.coupe')
    location = carla.Location()
    location.x = 28
    location.y = random.uniform(64, 74)
    location.z = 0
    ego_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, random.choice(map.get_spawn_points()))
    ego_vehicle.set_transform(ego_waypoint.transform)

    forward_inner_num = random.randint(2, 2)
    forward_outer_num = random.randint(2, 2)
    right_inner_num = random.randint(1, 3)
    right_outer_num = random.randint(2, 2)
    left_inner_num = random.randint(2, 2)
    left_outer_num = random.randint(2, 2)
    

    auto_vehicle_bp = vehicle_bps.find('vehicle.audi.a2')
    location = carla.Location()
    location.z = 0

    # location.x = 30
    # location.y = 105
    # for i in range(forward_inner_num):
    #     auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #     auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    #     auto_vehicle.set_transform(auto_waypoint.transform)
    #     location.y = location.y + 7

    # location.x = 35
    # location.y = 105
    # for i in range(forward_outer_num):
    #     auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #     auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    #     auto_vehicle.set_transform(auto_waypoint.transform)
    #     location.y = location.y + 7

    # print("Spawned {} cars on inner lane and {} cars on outer lane at forward direction.".format(forward_inner_num, forward_outer_num))

    location.x = 14
    location.y = 92

    for i in range(right_inner_num):
        auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
        auto_vehicle.set_transform(auto_waypoint.transform)
        location.x = location.x - 7
    
    # location.x = 14
    # location.y = 95

    # for i in range(right_outer_num):
    #     auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #     auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    #     auto_vehicle.set_transform(auto_waypoint.transform)
    #     location.x = location.x - 7
    
    # print("Spawned {} cars on inner lane and {} cars on outer lane at right direction.".format(right_inner_num, right_outer_num))

    # location.x = 45
    # location.y = 88

    # for i in range(left_inner_num):
    #     auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #     auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    #     auto_vehicle.set_transform(auto_waypoint.transform)
    #     location.x = location.x + 7
    
    # location.x = 45
    # location.y = 85

    # for i in range(left_outer_num):
    #     auto_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    #     auto_vehicle = world.spawn_actor(auto_vehicle_bp, random.choice(map.get_spawn_points()))
    #     auto_vehicle.set_transform(auto_waypoint.transform)
    #     location.x = location.x + 7
    
    # print("Spawned {} cars on inner lane and {} cars on outer lane at left direction.".format(left_inner_num, left_outer_num))

    world.tick()

    for vehicle in world.get_actors().filter('vehicle.audi.a2'):
        vehicle.set_autopilot(False)

    ego_dest = carla.Location(x=85, y=81, z=0)
    dest_waypoint = map.get_waypoint(ego_dest, project_to_road=True, lane_type=carla.LaneType.Driving)
    agent = BehaviorAgent(ego_vehicle, ignore_traffic_light=True, behavior='normal')
    agent.set_destination(agent._vehicle.get_location(), dest_waypoint.transform.location, clean=True)

    while True:
        agent.update_information(ego_vehicle)
        print(agent._vehicle.get_location())

        world.tick()
            
        if len(agent._local_planner.waypoints_queue)<1:
            print('======== Success, Arrivied at Target Point!')
            break

        speed_limit = ego_vehicle.get_speed_limit()
        agent.get_local_planner().set_speed(speed_limit)

        control = agent.run_step(debug=True)
        ego_vehicle.apply_control(control)

    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.destroy()

    world.tick()
    world.apply_settings(origin_settings)

if __name__ == '__main__':
    main()