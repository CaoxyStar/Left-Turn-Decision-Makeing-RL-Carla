import random
import queue
import numpy as np
import math

import carla
import pygame

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class CarlaEnv():
    def __init__(self):
        # get objects
        self.client = carla.Client('localhost', 2000)
        self.client.load_world('Town05')
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.map = self.world.get_map()
        self.vehicle_bps = self.world.get_blueprint_library().filter('*vehicle*')
        
        # set synchronous mode
        self.origin_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        # set traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # set spectator view
        spectator_transform = carla.Transform()
        spectator_transform.location.x = 25
        spectator_transform.location.y = 90
        spectator_transform.location.z = 50
        spectator_transform.rotation.pitch = -90
        spectator_transform.rotation.yaw = 90
        spectator_transform.rotation.roll = 0
        self.spectator.set_transform(spectator_transform)

        # close all traffic lights
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)

        # initialize data queue
        self.sensor_queue = queue.Queue(10)

        # update world
        self.world.tick()

        # initialize pygame for rendering
        pygame.init()
        pygame.display.set_caption('Ego Vehicle View')
        self.screen = pygame.display.set_mode((1280, 960))
    

    def reset(self):
        # clear vehicles and sensors in the world        
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for sensor in self.world.get_actors().filter('sensor.camera.rgb'):
            sensor.destroy()
        
        # spawn ego vehicle
        ego_vehicle_bp = self.vehicle_bps.find('vehicle.mercedes.coupe')
        ego_location = carla.Location()
        ego_location.x = 28
        ego_location.y = 74     #random.uniform(64, 74)
        ego_location.z = 0
        ego_waypoint = self.map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, random.choice(self.map.get_spawn_points()))
        self.ego_vehicle.set_transform(ego_waypoint.transform)

        # spawn camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        self.camera = self.world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.0, z=1.8)), attach_to=self.ego_vehicle)
        self.sensor_queue.queue.clear()
        self.camera.listen(lambda data: self.sensor_callback(data))

        # # spawn auto vehicles at right inner lane
        # right_inner_num = random.randint(1, 3)
        # auto_vehicle_bp = self.vehicle_bps.find('vehicle.audi.a2')
        # location = carla.Location()
        # location.z = 0
        # location.x = 14
        # location.y = 92

        # for i in range(right_inner_num):
        #     auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        #     auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
        #     auto_vehicle.set_transform(auto_waypoint.transform)
        #     auto_vehicle.set_autopilot(True, 8000)
        #     location.x = location.x - 7
        
        # update world
        self.world.tick()

        # set record flag
        self.terminated = False
        self.truncated = False
        self.iteration = 0

        # set path
        dao = GlobalRoutePlannerDAO(self.map, 1.0)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        start_waypoint = ego_waypoint
        end_waypoint = self.map.get_waypoint(carla.Location(x=75, y=87, z=0), project_to_road=True, lane_type=carla.LaneType.Driving)
        self.route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        self.last_index = 0

        # # print global route
        # print(len(self.route))
        # for i in range(len(self.route)):
        #     print(self.route[i][0].transform.location)

        # collect observations
        next_point = self.route[self.last_index + 1][0]
        obs = {
            'ego_state': np.array((ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, ego_waypoint.transform.rotation.yaw, 0, 0)),
            'next_point': np.array((next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw))
        }
        
        return obs


    def step(self, action):
        # apply action
        throttle, steer, brake = action
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()
        self.iteration += 1

        # sensor data process
        data = self.sensor_queue.get(block=True)
        raw_image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        raw_image = np.reshape(raw_image, (480, 640, 4))
        raw_image = raw_image[:, :, :3]
        raw_image = raw_image[:, :, ::-1]

        # ego vehicle view rendering
        self.visualize(raw_image)
            
        # get ego state
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_transform = self.ego_vehicle.get_transform()

        index, dist = self.find_nearest_waypoint(ego_transform.location)
        next_point = self.route[index + 1][0]

        # collect observations
        obs = {
            'ego_state': np.array((ego_transform.location.x, ego_transform.location.y, ego_transform.rotation.yaw, ego_velocity.x, ego_velocity.y)),
            'next_point': np.array((next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw))
        }

        # get diff and task status
        position_diff = self.route[self.last_index + 1][0].transform.location.distance(ego_transform.location)
        yaw_inference = (self.route[self.last_index + 1][0].transform.rotation.yaw + self.route[self.last_index][0].transform.rotation.yaw) / 2
        yaw_diff = abs(yaw_inference - ego_transform.rotation.yaw)

        self.last_index = index
        
        if index > len(self.route) - 5:
            self.terminated = True
        elif (position_diff > 5.0) or (yaw_diff > 45):
            self.truncated = True
        elif self.iteration > 500:
            self.truncated = True

        # compute reward
        reward_point = index * 50
        reward_vel = -abs(5 - math.sqrt(ego_velocity.x**2 + ego_velocity.y**2))
        reward_pos = -position_diff * 100
        reward_ang = -yaw_diff * 5
        # reward += (action[0] - action[2]) * 10
        # reward -= action[1] * 10
        
        reward_flag = 0
        if self.terminated:
            reward_flag += 1000
        elif self.truncated:
            reward_flag -= 1000
        
        reward = [reward_flag, reward_point, reward_vel, reward_pos, reward_ang]

        return obs, reward, self.terminated or self.truncated

    def sample_action(self):
        throttle = random.uniform(0.0, 1.0)
        steer = random.uniform(-1.0, 1.0)
        brake = random.uniform(0.0, 1.0)
        return throttle, steer, brake
    
    def close(self):
        # clear vehicles and sensors in the world
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for sensor in self.world.get_actors().filter('sensor.camera.rgb'):
            sensor.destroy()

        # restore origin settings
        self.world.apply_settings(self.origin_settings)
    
    def sensor_callback(self, sensor_data):
        self.sensor_queue.put(sensor_data)

    def visualize(self, raw_image):
        image_surface = pygame.surfarray.make_surface(raw_image.transpose(1, 0, 2))
        scaled_image = pygame.transform.smoothscale(image_surface, (1280, 960))
        self.screen.blit(scaled_image, (0, 0))
        pygame.display.flip()
    
    def find_nearest_waypoint(self, location):
        min_dist = 1000
        index = self.last_index
        for i in range(self.last_index, min(self.last_index + 5, len(self.route))):
            dist = self.route[i][0].transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                index = i
        return index, min_dist
    

# env = CarlaEnv()
# env.reset()
# for i in range(200):
#     if i < 100:
#         obs, reward, down = env.step((0.6, 0, 0))
#     else:
#         obs, reward, down = env.step((0.6, 0, 0.8))
#     # print(obs)
#     # print(reward)
#     # print(down)

# env.close()