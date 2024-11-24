import random
import queue
import math

import numpy as np
import cv2
import carla
import pygame

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class CarlaEnv():
    def __init__(self):
        # get interactive objects in carla
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
        self.settings.fixed_delta_seconds = 0.1
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

        # initialize data queue for senseor data
        self.sensor_queue = queue.Queue(10)

        # update world
        self.world.tick()

        # initialize pygame for rendering
        pygame.init()
        pygame.display.set_caption('Ego Vehicle View')
        self.screen = pygame.display.set_mode((640, 640))
    

    def reset(self, scene='normal'):
        # clear vehicles and sensors actor in the world        
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for sensor in self.world.get_actors().filter('sensor.camera.semantic_segmentation'):
            sensor.destroy()
        for sensor in self.world.get_actors().filter('sensor.other.collision'):
            sensor.destroy()
        
        # spawn ego vehicle at a random location
        ego_vehicle_bp = self.vehicle_bps.find('vehicle.mercedes.coupe')
        ego_location = carla.Location()
        ego_location.x = 28
        ego_location.y = random.uniform(64, 74)
        ego_location.z = 0
        ego_waypoint = self.map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, random.choice(self.map.get_spawn_points()))
        self.ego_vehicle.set_transform(ego_waypoint.transform)

        # spawn camera and attach to ego vehicle
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '64')
        camera_bp.set_attribute('image_size_y', '64')
        camera_bp.set_attribute('fov', '90')
        self.camera = self.world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=13.0, z=15), carla.Rotation(pitch=-90)), attach_to=self.ego_vehicle)
        self.sensor_queue.queue.clear()
        self.camera.listen(lambda data: self.sensor_callback(data))

        # spawn collision sensors and attach to ego vehicle
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self.collision_callback(event))

        # spawn auto-pilot vehicles in the world
        if scene == 'normal':
            self.spawn_normal_vehicle()
        elif scene == 'difficult':
            self.spawn_difficult_vehicle()
        elif scene == 'path_following':
            pass
        else:
            print('No such scene, please input normal, difficult or path_following.')
            exit()
        
        # update world
        self.world.tick()

        # set record flag
        self.terminated = False
        self.truncated = False
        self.iteration = 0

        # get global map and route
        dao = GlobalRoutePlannerDAO(self.map, 1.0)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        start_waypoint = ego_waypoint
        end_waypoint = self.map.get_waypoint(carla.Location(x=75, y=87, z=0), project_to_road=True, lane_type=carla.LaneType.Driving)
        self.route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        self.last_index = 0

        # process sensor data to get observation at bev view
        data = self.sensor_queue.get(block=True)
        raw_image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        raw_image = np.reshape(raw_image, (64, 64, 4))
        raw_image = raw_image[:, :, :3]
        raw_image = raw_image[:, :, ::-1]

        target_color = np.array([0, 0, 142])  
        threshold = 10
        lower_bound = target_color - threshold  
        upper_bound = target_color + threshold  
        mask = cv2.inRange(raw_image, lower_bound.reshape(1, 1, 3), upper_bound.reshape(1, 1, 3))

        # obs visualization
        self.visualize(mask)

        # collect observations
        next_point = self.route[self.last_index + 1][0]
        obs = {
            'ego_state': np.array((ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, ego_waypoint.transform.rotation.yaw, 0, 0)),
            'next_point': np.array((next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw)),
            'image': mask
        }
        
        return obs


    def step_pf(self, action):
        # apply action and update world
        throttle, steer, brake = action
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0))
        self.world.tick()
        self.iteration += 1

        # sensor data process
        data = self.sensor_queue.get(block=True)
        raw_image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        raw_image = np.reshape(raw_image, (64, 64, 4))
        raw_image = raw_image[:, :, :3]
        raw_image = raw_image[:, :, ::-1]

        target_color = np.array([0, 0, 142])  
        threshold = 10
        lower_bound = target_color - threshold  
        upper_bound = target_color + threshold  
        mask = cv2.inRange(raw_image, lower_bound.reshape(1, 1, 3), upper_bound.reshape(1, 1, 3)) 

        # obs visualization
        self.visualize(mask)
            
        # get the current state of ego vehicle
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_transform = self.ego_vehicle.get_transform()
        index, dist = self.find_nearest_waypoint(ego_transform.location)
        next_point = self.route[index + 1][0]

        # collect observations
        obs = {
            'ego_state': np.array((ego_transform.location.x, ego_transform.location.y, ego_transform.rotation.yaw, ego_velocity.x, ego_velocity.y)),
            'next_point': np.array((next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw)),
            'image': mask
        }

        # calculate difference
        position_diff = self.route[self.last_index + 1][0].transform.location.distance(ego_transform.location)
        yaw_inference = (self.route[self.last_index + 1][0].transform.rotation.yaw + self.route[self.last_index][0].transform.rotation.yaw) / 2
        yaw_diff = abs(yaw_inference - ego_transform.rotation.yaw)
        self.last_index = index
        
        # check task status
        if index > len(self.route) - 5:
            self.terminated = True
        elif (position_diff > 3.0) or (yaw_diff > 45):
            self.truncated = True
        elif self.iteration > 250:
            self.truncated = True

        # compute rewards
        reward_point = index * 10
        reward_vel = -abs(5 - math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)) * 10
        reward_pos = -position_diff * 100
        reward_ang = -yaw_diff * 5
        
        reward_flag = 0
        if self.terminated:
            reward_flag += 1000
        elif self.truncated:
            reward_flag -= 1000
        
        reward = [reward_flag, reward_point, reward_vel, reward_pos, reward_ang]

        return obs, reward, self.terminated or self.truncated


    def step(self, action):
        # apply action
        throttle, steer, brake = action
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()
        self.iteration += 1

        # sensor data process
        data = self.sensor_queue.get(block=True)
        raw_image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        raw_image = np.reshape(raw_image, (64, 64, 4))
        raw_image = raw_image[:, :, :3]
        raw_image = raw_image[:, :, ::-1]

        target_color = np.array([0, 0, 142])  
        threshold = 10
        lower_bound = target_color - threshold  
        upper_bound = target_color + threshold  
        mask = cv2.inRange(raw_image, lower_bound.reshape(1, 1, 3), upper_bound.reshape(1, 1, 3))   

        # obs visualization
        self.visualize(mask)
            
        # get ego state
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_transform = self.ego_vehicle.get_transform()
        index, dist = self.find_nearest_waypoint(ego_transform.location)
        next_point = self.route[index + 1][0]

        # collect observations
        obs = {
            'ego_state': np.array((ego_transform.location.x, ego_transform.location.y, ego_transform.rotation.yaw, ego_velocity.x, ego_velocity.y)),
            'next_point': np.array((next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw)),
            'image': mask
        }

        self.last_index = index
        
        # check task status
        if index > len(self.route) - 5:
            self.terminated = True
        elif self.iteration > 200:
            self.truncated = True

        # compute rewards
        reward = -brake
        if self.terminated:
            reward = 100
        elif self.truncated:
            reward = -100

        return obs, reward, self.terminated, self.truncated
    

    def close(self):
        # clear vehicles and sensors in the world
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for sensor in self.world.get_actors().filter('sensor.camera.semantic_segmentation'):
            sensor.destroy()
        for sensor in self.world.get_actors().filter('sensor.other.collision'):
            sensor.destroy()

        # restore origin settings
        self.world.apply_settings(self.origin_settings)
    

    def sensor_callback(self, sensor_data):
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)
        self.sensor_queue.put(sensor_data)


    def collision_callback(self, event):
        self.truncated = True
        print('Collision!')


    def visualize(self, raw_image):
        image_surface = pygame.surfarray.make_surface(raw_image.transpose(1, 0))
        scaled_image = pygame.transform.scale(image_surface, (640, 640))
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


    def spawn_normal_vehicle(self):
        right_inner_num = random.randint(1, 3)
        right_outer_num = random.randint(1, 3)
        auto_vehicle_bp = self.vehicle_bps.find('vehicle.audi.a2')
        location = carla.Location()
        location.z = 0

        location.x = 14
        location.y = 92

        for i in range(right_inner_num):
            auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
            auto_vehicle.set_transform(auto_waypoint.transform)
            auto_vehicle.set_autopilot(True, 8000)
            location.x = location.x - 7

        location.x = 14
        location.y = 95

        for i in range(right_outer_num):
            auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
            auto_vehicle.set_transform(auto_waypoint.transform)
            auto_vehicle.set_autopilot(True, 8000)
            location.x = location.x - 7
    
    def spawn_difficult_vehicle(self):
        forward_outer_num = random.randint(1, 3)
        right_outer_num = random.randint(1, 3)
        left_outer_num = random.randint(1, 3)
        
        auto_vehicle_bp = self.vehicle_bps.find('vehicle.audi.a2')
        location = carla.Location()
        location.z = 0

        # spawn vehicles in the forward direction
        location.x = 35
        location.y = 105
        for i in range(forward_outer_num):
            auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
            auto_vehicle.set_transform(auto_waypoint.transform)
            auto_vehicle.set_autopilot(True, 8000)
            location.y = location.y + 7

        # spawn vehicles in the right direction
        location.x = 14
        location.y = 95

        for i in range(right_outer_num):
            auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
            auto_vehicle.set_transform(auto_waypoint.transform)
            auto_vehicle.set_autopilot(True, 8000)
            location.x = location.x - 7

        # spawn vehicles in the left direction
        location.x = 45
        location.y = 85

        for i in range(left_outer_num):
            auto_waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            auto_vehicle = self.world.spawn_actor(auto_vehicle_bp, random.choice(self.map.get_spawn_points()))
            auto_vehicle.set_transform(auto_waypoint.transform)
            auto_vehicle.set_autopilot(True, 8000)
            location.x = location.x + 7
