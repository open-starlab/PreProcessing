import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
import pdb
import matplotlib.path as mpath

from football_markov.constant import FIELD_LENGTH, FIELD_WIDTH, STOP_THRESHOLD
from football_markov.data.dataclass_LDS import Ball, Player, Position, OnBall, OffBall, AbsoluteState

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from scipy.spatial import Voronoi
from scipy.stats import norm
from shapely.geometry import Polygon
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# claculate space each player has

def generate_heat_map():
    """
    Generate heatmap data for a soccer field.
    """
    # x axis data (exponential weight centered at length/2)
    x = np.linspace(-FIELD_LENGTH/2, FIELD_LENGTH/2, int(FIELD_LENGTH))
    x_weight = 0.1 / (0.1 + np.exp(-0.1 * x))   
    
    # y axis data (gaussian weight)
    y = np.linspace(-FIELD_WIDTH/2, FIELD_WIDTH/2, int(FIELD_WIDTH))
    y_center = 0
    y_weight = 50 * norm.pdf(y, loc=y_center, scale=FIELD_WIDTH / 4)  # loc: mean, scale: standard deviation
    
    # generate heatmap data
    heatmap_data = np.outer(y_weight, x_weight)
    
    return heatmap_data


def judge_offside(points, ball_loc, team_list, team_name):
    # attack_direction = team_info[2]
    team_d = [i for i, team in enumerate(team_list) if team != team_name]
    team_d_loc_x = [points[team][0] for team in team_d]
    second_max = np.partition(team_d_loc_x, -2)[-2]
    offside_line = ball_loc[0] if ball_loc[0] > second_max else second_max; offside_line = FIELD_LENGTH / 2 if offside_line < FIELD_LENGTH / 2 else offside_line

    team_o = [i for i, team in enumerate(team_list) if team == team_name]
    team_o_loc_x = [points[team][0] for team in team_o]
    offside_loc = [loc for i, loc in enumerate(team_o_loc_x) if loc > offside_line]
    offside_idx = [i for i, loc in enumerate(points) if loc[0] in offside_loc]
    offside_f = [0] * len(points)
    if len(offside_idx) > 0:
        for idx in offside_idx:
            offside_f[idx] = 1

    valid_indices = [i for i, flag in enumerate(offside_f) if flag == 0]
    filtered_points = [points[i] for i in valid_indices]

    return filtered_points, offside_f


def velocity_points(points, velocities):
    points = np.array(points)
    velocities = np.array(velocities)
    new_points = points + velocities * 0.1
    return new_points.tolist()


def calculate_voronoi(players: List[Player], ball: Ball, team_info: str, key: str):
    # プレイヤーの位置情報を取得し、スケール変換
    points = [[player.position.x+52.5, player.position.y+34] for player in players]

    velocities = [[player.velocity.x, player.velocity.y] for player in players]
    
    # プレイヤーのチーム情報を取得
    team = [player.team_name for player in players]    

    # ボールの位置
    ball_loc = [ball.position.x+52.5, ball.position.y+34]

    player_name = [player.player_name for player in players]

    if key == "velocity":
        points = velocity_points(points, velocities)

    # judge offside
    filtered_points, offside_f = judge_offside(points, ball_loc, team, team_info)
    
    # ボロノイ図を計算
    vor = Voronoi(filtered_points)
    
    return vor, team, player_name, offside_f


def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        
        if p1 not in all_ridges or p2 not in all_ridges:
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    # サッカーコートの境界内にクリップ
    clipped_regions = []
    field_boundary = Polygon([(0, 0), (FIELD_LENGTH, 0), (FIELD_LENGTH, FIELD_WIDTH), (0, FIELD_WIDTH)])
    for region in new_regions:
        polygon = Polygon([new_vertices[i] for i in region])
        clipped_polygon = polygon.intersection(field_boundary)
        if isinstance(clipped_polygon, Polygon) and not clipped_polygon.is_empty:
            clipped_region = []
            for coord in clipped_polygon.exterior.coords[:-1]:
                if list(coord) not in new_vertices:
                    new_vertices.append(list(coord))
                clipped_region.append(new_vertices.index(list(coord)))
            clipped_regions.append(clipped_region)

    return clipped_regions, np.array(new_vertices)


def weighted_area(polygon, weight_image, team, team_name):
    # 多角形の頂点を取得
    if isinstance(polygon, Polygon):
        vertices = np.array(polygon.exterior.coords, dtype=np.int32)
    else:
        raise TypeError("polygon must be a shapely.geometry.Polygon object")
    
    # 多角形のマスクを作成
    mask = np.zeros(weight_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], 1)

    # team が team_name と異なる場合、weight_image を x 軸方向に反転
    if team != team_name:
        weight_image = np.flip(weight_image, axis=1)

    # 重み画像とマスクを掛け合わせて重み付き領域を取得
    weighted_region = weight_image * mask
    
    # 重み付き領域の合計を計算
    area = np.sum(weighted_region)
    
    return area


# calculate shot block prob
def truncated_normal_pdf(x, mean, std, a, b):
    """
    Calculates the probability density function (PDF) of a truncated normal distribution.

    Args:
        x (tensor): The input tensor.
        mean (float): The mean of the underlying normal distribution.
        std (float): The standard deviation of the underlying normal distribution.
        a (float): The lower bound of the truncation range.
        b (float): The upper bound of the truncation range.

    Returns:
        tensor: The PDF values of the truncated normal distribution for the given input tensor.
    """
    # Convert x, mean, std, a, b into tensors
    x = torch.tensor(x)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    a = torch.tensor(a)
    b = torch.tensor(b)

    z_a = (a - mean) / std
    z_b = (b - mean) / std
    z_x = (x - mean) / std

    if torch.isnan(z_x).any():
        pdb.set_trace() 
    cdf_diff = Normal(0, 1).cdf(z_b) - Normal(0, 1).cdf(z_a)
    truncated_pdf = torch.exp(Normal(0, 1).log_prob(z_x)) / (std * cdf_diff)

    return truncated_pdf


def f_combined(inputs, x):
    result = 0
    prev_term = 1

    for input_set in inputs:
        angle_p, scaler1, mean, std, a, b = input_set
        term = truncated_normal_pdf((x - angle_p) / scaler1, mean, std, a, b)
        result += prev_term * term
        prev_term *= (1 - term)

    return result


def integrate_product_trapezoidal_rule(f, a, b, num_points=1000):
    """
    Numerically integrates the product of a function of multiple variables using the trapezoidal rule.

    Args:
        f (callable): The function to integrate. It should accept a single argument (an array-like object) that represents the variables.
        a (array-like): Lower bounds of the integration interval for each variable.
        b (array-like): Upper bounds of the integration interval for each variable.
        num_points (int): Number of points to sample for the trapezoidal rule.

    Returns:
        float: Approximate integral of the product of the function using the trapezoidal rule.
    """

    num_vars = len(a)
    grid = [np.linspace(a[i], b[i], num_points) for i in range(num_vars)]
    points = np.meshgrid(*grid, indexing='ij')
    values = f(*points)
    integral = np.trapz(values, x=grid[0])
    return integral


def point_position(point, line_point1, line_point2):
    vector1 = line_point2 - line_point1
    vector2 = point - line_point1

    cross_product = np.cross(vector1, vector2)

    if cross_product > 0:
        return "Above"
    elif cross_product == 0:
        return "On the line"
    else:
        return "Below"


def distance_between_two_points(pointA, pointB):
    return np.linalg.norm(pointA - pointB)


def angle_between_three_points(pointA, pointB, pointC):
    #ref https://muthu.co/using-the-law-of-cosines-and-vector-dot-product-formula-to-find-the-angle-between-three-points/
    BA = pointA - pointB
    BC = pointC - pointB

    try:
        cosine_angle = round(np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC)),8)
        angle = np.arccos(cosine_angle)
    except:
        print("exc")
        raise Exception('invalid cosine')

    return np.degrees(angle)


def is_point_within_area(point, bound_points):
    x1, y1 = bound_points[0]
    x2, y2 = bound_points[1]
    x3, y3 = bound_points[2]
    x, y = point
    # Define the triangle path
    triangle_path_data = [
        (x1, y1),
        (x2, y2),
        (x3, y3)
    ]

    # print(f"triangle_path_data: {triangle_path_data}")
    # print(f"point: {point}")
    # print(f"bound_points: {bound_points}")

    triangle_path = mpath.Path(triangle_path_data)

    return triangle_path.contains_point((x, y))


def calculate_shot_block_prob(train, scaler_1=1, scaler_2=1, scaler_3=1, mean=0, sigma=0.4, a=-10, b=10):
    pose_left_x = 52.5
    pose_left_y = -36
    pose_right_x = -52.5
    pose_right_y = 36

    #get the location of the shooter
    shoter_x=train["location_x"].loc[0]
    shoter_y=train["location_y"].loc[0]
    #get the location of the other players
    other_player={}
    for other_player_num in range(22):
        if not train["player"+str(other_player_num)+"_location_x"].isnull().any():
            # print(index,other_player,train["player"+str(other_player)+"_location_x"][index])
            other_player[other_player_num]={"x":train["player"+str(other_player_num)+"_location_x"].loc[0],"y":train["player"+str(other_player_num)+"_location_y"].loc[0],
                                        "teammate":train["player"+str(other_player_num)+"_teammate"].loc[0],"position_name":train["player"+str(other_player_num)+"_position_name"].loc[0]}
            
    # print(f"other_player: {other_player[0]}")

    #filter non teammate, Goalkeeper, and player not in between the shooter and the goal
    filtered_player = {k: v for k, v in other_player.items() if v.get('position_name') != 'Goalkeeper'}
    filtered_player = {k: v for k, v in filtered_player.items() if v.get('teammate') == False}
    keys_to_remove = []
    for k, v in filtered_player.items():
        filter=is_point_within_area((filtered_player[k]["x"], filtered_player[k]["y"]),[(pose_left_x, 30), (pose_right_x, 50), (shoter_x, shoter_y)])
        if not filter:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del filtered_player[k]

    #272 row with player in between the shooter and the goal, total:2060 row

    # print("index",index,"len(filtered_player)",len(filtered_player),filtered_player)

    #model the shot block probability
    if len(filtered_player)>0:

        #calculate the angle from the shooter to the goal left post and shooter to the player
        pointA = np.array([pose_left_x, pose_left_y])
        pointB = np.array([shoter_x,shoter_y])
        pointC = np.array([pose_right_x, pose_right_y])
        total_angle=angle_between_three_points(pointA, pointB, pointC)
        # print("total_angle :",total_angle)

        #calculate the angle between the shooter to the goal left post and shooter to the player
        for k, v in filtered_player.items():
            pointC = np.array([v['x'],v['y']])
            defender_angle=angle_between_three_points(pointA, pointB, pointC)
            defender_distance=distance_between_two_points(pointB, pointC)
            point_position_result=point_position(pointC, pointB, pointA)
            if point_position_result=="Below":
                defender_angle=-defender_angle #if the player is below the line of the shooter and the left pose in the normal coordination, the angle is negative
            v['defender_angle'] = defender_angle
            v['defender_distance'] = defender_distance
        
        #sort the player by distance to the shooter
        sorted_players = sorted(filtered_player.items(), key=lambda item: item[1]['defender_distance'])
        filtered_player = {k: v for k, v in sorted_players}

        #calculate the probability of the shot block
        list(filtered_player.keys())
        function_list=[]
        for k, v in filtered_player.items(): 
            angle_p=v['defender_angle']
            std=sigma+v['defender_distance']/scaler_2
            function_list.append([angle_p,scaler_1,mean,std,a,b])

        integral = integrate_product_trapezoidal_rule(lambda *args: f_combined(function_list, *args), [0], [total_angle], num_points=1000)
        prediction=(integral/total_angle)*scaler_3 
    else:
        prediction=0
    return prediction


def calc_weighted_area(players, ball, team_name_attack, key):

    heatmap_data = generate_heat_map()
    # Assuming event_df is predefined and available
    index = 0

    vor, team_list, player_names, offside_f = calculate_voronoi(players, ball, team_name_attack, key)

    # display Voronoi diagram with finite regions
    regions, vertices = voronoi_finite_polygons_2d(vor)
    weighted_areas: Dict[str, float] = {}
    for team, f, player_name in zip(team_list, offside_f, player_names):
        if f == 0:
            if index < len(regions) and all(v < len(vertices) for v in regions[index]):
                polygon = Polygon(vertices[regions[index]])
                w_area = weighted_area(polygon, heatmap_data, team, team_name_attack)
                polygon = vertices[regions[index]]
                weighted_areas[player_name] = w_area
                index += 1
            else:
                weighted_areas[player_name] = 0.0
        else:
            weighted_areas[player_name] = 0.0
    
    return weighted_areas


# On-ball state variable
def calc_dist_each_opponents(ball: Ball, attackers: List[Player], defenders: List[Player]) -> List[float]:
    attacker_players = sorted(attackers, key=lambda x: x.position.distance_to(ball.position))
    defender_players = sorted(defenders, key=lambda x: x.position.distance_to(ball.position))
    try:
        attacker_to_ball = attacker_players[0].position.distance_to(ball.position)
        defender_to_ball = defender_players[0].position.distance_to(ball.position)
    except:
        print(attacker_players)
        import pdb; pdb.set_trace()
    return [attacker_to_ball, defender_to_ball], [attacker_players[0], defender_players[0]]


def calc_dist_each_goals(players: List[Player], goal_position: Position) -> List[float]:
    attack_goal_position = goal_position
    defense_goal_position = Position(x=-goal_position.x, y=goal_position.y)

    dist_to_attack_goal = players[0].position.distance_to(attack_goal_position)
    dist_to_defense_goal = players[0].position.distance_to(defense_goal_position)
    return  [dist_to_attack_goal, dist_to_defense_goal], [attack_goal_position, defense_goal_position]


def calc_angle_each_goals(players: List[Player], goal_positions: Position) -> List[float]:
    attack_goal_angle = np.arctan2(goal_positions[0].y - players[0].position.y, goal_positions[0].x - players[0].position.x)
    defense_goal_angle = np.arctan2(goal_positions[1].y - players[0].position.y, goal_positions[1].x - players[0].position.x)
    return [attack_goal_angle, defense_goal_angle]
    

def calc_dribble_score(players: List[Player], ball: Ball, team_name_attack: str, player_onball: Player, key: str) -> List[float]:
    # calculate weighted area
    weighted_areas = calc_weighted_area(players, ball, team_name_attack, key)

    if player_onball is not None:
        # extract player_onball's weighted area from weighted_areas
        player_onball_weighted_area = weighted_areas[player_onball.player_name]

        player_onball_point = player_onball.position

        point_up = Position(x=player_onball_point.x+1, y=player_onball_point.y)
        point_up_right = Position(x=player_onball_point.x+(1/np.sqrt(2)), y=player_onball_point.y+(1/np.sqrt(2)))
        point_right = Position(x=player_onball_point.x, y=player_onball_point.y+1)
        point_down_right = Position(x=player_onball_point.x-(1/np.sqrt(2)), y=player_onball_point.y+(1/np.sqrt(2)))
        point_down = Position(x=player_onball_point.x-1, y=player_onball_point.y)
        point_down_left = Position(x=player_onball_point.x-(1/np.sqrt(2)), y=player_onball_point.y-(1/np.sqrt(2)))
        point_left = Position(x=player_onball_point.x, y=player_onball_point.y-1)
        point_up_left = Position(x=player_onball_point.x+(1/np.sqrt(2)), y=player_onball_point.y-(1/np.sqrt(2)))

        direction_points = [point_up, point_up_right, point_right, point_down_right, point_down, point_down_left, point_left, point_up_left]

        # players = copy.deepcopy(players)
        player_onball_weighted_area_list = []

        # calculate weighted area for each direction
        for point in direction_points:
            # insert point to players of player_onball_index 
            for player in players:
                if player.player_name == player_onball.player_name:
                    player.position = point
                    break
            weighted_areas_ = calc_weighted_area(players, ball, team_name_attack, key)
            player_onball_weighted_area_ = weighted_areas_[player_onball.player_name]
            player_onball_weighted_area_list.append(player_onball_weighted_area_ - player_onball_weighted_area)
    
        return player_onball_weighted_area_list, weighted_areas
    else:
        player_onball_weighted_area_list = [np.NaN]*8
        return player_onball_weighted_area_list, weighted_areas


def calc_shot_score(player_onball: Player, onball_team: str, players: List[Player]) -> float:
    onball_id = player_onball.player_id

    df_shot = pd.DataFrame()
    i = 0
    for player in players:
        if player.team_name == onball_team and player.player_id == onball_id:
            df_shot.loc[0, 'location_x'] = player.position.x
            df_shot.loc[0, 'location_y'] = player.position.y
        
        df_shot.loc[0, f'player{i}_location_x'] = player.position.x
        df_shot.loc[0, f'player{i}_location_y'] = player.position.y
        df_shot.loc[0, f'player{i}_teammate'] = True if player.team_name == onball_team else False
        df_shot.loc[0, f'player{i}_position_name'] = player.player_role

        i += 1

    shot_block_prob = calculate_shot_block_prob(df_shot)

    return shot_block_prob


def calc_long_ball_score(attack_players: List[float]) -> List[float]:
    """
    Calculate the long ball score.
    Split the field into 3 zones: center, right, left
    If the highest player is in the center, then center score = 1
    If the highest player is in the right, then right score = 1
    If the highest player is in the left, then left score = 1
    """
    long_ball_score = [0]*3

    # sort the attack players x position and get the top 3 players
    target_players = sorted(attack_players, key=lambda x: x.position.x, reverse=True)[:3]
    highest_player = sorted(target_players, key=lambda x: x.height if hasattr(x, 'height') else 0, reverse=True)[0]

    highest_player_y = highest_player.position.y
    if highest_player_y >= 12:
        long_ball_score[0] = 1
    elif highest_player_y <= -12:
        long_ball_score[2] = 1
    elif -12 < highest_player_y < 12:
        long_ball_score[1] = 1

    return long_ball_score


def calc_transition(players: Player, onball_team: str) -> List[float]:
    transition = []

    for player in players:
        if player.team_name == onball_team:
            transition.append(1.0)
        else:
            transition.append(0.0)
    
    return transition


# Off-ball state variable
def calc_fast_space(players: List[Player], ball: Ball, team_name_attack: str) -> List[float]:
    heatmap_data = generate_heat_map()
    # Assuming event_df is predefined and available
    index = 0

    vor, team_list, _, offside_f = calculate_voronoi(players, ball, team_name_attack)

    # display Voronoi diagram with finite regions
    regions, vertices = voronoi_finite_polygons_2d(vor)
    weighted_areas: List[float] = []
    for team, f in zip(team_list, offside_f):
        if f == 0:
            polygon = Polygon(vertices[regions[index]])
            w_area = weighted_area(polygon, heatmap_data, team, team_name_attack)
            polygon = vertices[regions[index]]
            # weighted_areas[player_name] = w_area
            weighted_areas.append(w_area)
            index += 1
        else:
            # weighted_areas[player_name] = 0.0
            weighted_areas.append(0.0)
    
    return weighted_areas


def calc_dist_ball(ball: Ball, attack_players: List[Player]) -> List[float]:
    # calculate distance to ball for each player
    return [player.position.distance_to(ball.position) for player in attack_players]


def calc_time_to_player(attack_players: List[Player], defense_players: List[Player]) -> List[float]:
    times_to_player = []
    for player in attack_players:
        if player is None:
            # add inf to onball player 
            times_to_player.append(float('inf'))
        else:
            # find the nearest defense player to player
            defense_players_sorted = sorted(defense_players, key=lambda x: x.position.distance_to(player.position))
            nearest_defense = defense_players_sorted[0]

            # calculate time to reach nearest defense player to player
            dist = nearest_defense.position.distance_to(player.position)
            velocity = np.sqrt(nearest_defense.velocity.x**2 + nearest_defense.velocity.y**2)
            if velocity == 0:
                times_to_player.append(float('inf'))  # or handle appropriately
            else:
                times_to_player.append(dist / velocity)
    
    return times_to_player


def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment."""
    px, py = point.x, point.y
    x1, y1 = line_start.x, line_start.y
    x2, y2 = line_end.x, line_end.y

    # Line segment length squared
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if line_len_sq == 0:
        # line_start and line_end are the same point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Projection factor
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

    # Projection point on the line segment
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    # Distance from the point to the projection point
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def calc_time_to_passline(ball: Ball, attack_players: List[Player], defense_players: List[Player]) -> List[float]:
    # calculate time to passline for each player
    times_to_passline = []
    for player in attack_players:
        if player is None:
            # add inf to onball player 
            times_to_passline.append(float('inf'))
        else:
            # get position player and ball
            player_position = Position(x=player.position.x, y=player.position.y)
            ball_position = Position(x=ball.position.x, y=ball.position.y)            

            # player と ball の位置を取得
            player_position = Position(x=player.position.x, y=player.position.y)
            ball_position = Position(x=ball.position.x, y=ball.position.y)

            # defense_players を線分に基づいてソート
            defense_players_sorted = sorted(defense_players, key=lambda x: point_to_line_distance(x.position, ball_position, player_position))

            # 線分に対する最も近い defense player を取得
            nearest_defense = defense_players_sorted[0]

            # 最も近い defense player までの距離と速度を計算
            dist = point_to_line_distance(nearest_defense.position, ball_position, player_position)
            velocity = np.sqrt(nearest_defense.velocity.x**2 + nearest_defense.velocity.y**2)
            if velocity == 0:
                times_to_passline.append(float('inf'))  # or handle appropriately
            else:
                times_to_passline.append(dist / velocity)

    return times_to_passline


def calc_space_score(players: List[Player], ball: Ball, team_name_attack: str, key: str) -> List[float]:
    return calc_fast_space(players, ball, team_name_attack)


def calc_variation_space(
        players: List[Player],
        ball: Ball,
        team_name_attack: str,
        weighted_area: List[float],
        key: str
    ) -> List[List[float]]:
    # loop for each players
    variation_space = []

    attack_players = [player for player in players if player.team_name == team_name_attack]

    for player in attack_players:

        player_weighted_area = weighted_area[player.player_name]
        
        point_up = Position(x=player.position.x+1, y=player.position.y)
        point_up_right = Position(x=player.position.x+(1/np.sqrt(2)), y=player.position.y+(1/np.sqrt(2)))
        point_right = Position(x=player.position.x, y=player.position.y+1)
        point_down_right = Position(x=player.position.x-(1/np.sqrt(2)), y=player.position.y+(1/np.sqrt(2)))
        point_down = Position(x=player.position.x-1, y=player.position.y)
        point_down_left = Position(x=player.position.x-(1/np.sqrt(2)), y=player.position.y-(1/np.sqrt(2)))
        point_left = Position(x=player.position.x, y=player.position.y-1)
        point_up_left = Position(x=player.position.x+(1/np.sqrt(2)), y=player.position.y-(1/np.sqrt(2)))

        direction_points = [
            point_up,
            point_up_right,
            point_right,
            point_down_right,
            point_down,
            point_down_left,
            point_left,
            point_up_left
        ]

        # players = copy.deepcopy(players)
        player_weighted_area_list = []

        # calculate weighted area for each direction
        for point in direction_points:
            # insert point to players of player_onball_index 
            for player_ in players:
                if player_.player_name == player.player_name:
                    player_.position = point
                    break
            weighted_areas_ = calc_weighted_area(players, ball, team_name_attack, key)
            player_weighted_area_ = weighted_areas_[player.player_name]
            player_weighted_area_list.append(player_weighted_area_ - player_weighted_area)
        
        variation_space.append(player_weighted_area_list)
    return variation_space


# Absolute state variable
def calc_dist_offside_line(attack_players: List[Player], defense_players: List[Player], ball: Ball) -> List[float]:
    attack_players_sorted = sorted(attack_players, key=lambda x: x.position.x)
    defense_players_sorted = sorted(defense_players, key=lambda x: x.position.x)

    attack_offside_line = defense_players_sorted[1].position.x
    # find max value from attack_offside_line, ball.position.x, FIELD_LENGTH / 2
    attack_offside_line = max(attack_offside_line, ball.position.x, FIELD_LENGTH / 2)
    
    defense_offside_line = attack_players_sorted[-2].position.x
    # find min value from defense_offside_line, FIELD_LENGTH / 2
    defense_offside_line = min(defense_offside_line, FIELD_LENGTH / 2)

    dist_attack_offside_line = abs(ball.position.x - attack_offside_line)
    dist_defense_offside_line = abs(defense_offside_line - ball.position.x)

    return [dist_attack_offside_line, dist_defense_offside_line]


# onball function
def calc_onball(
        players: List[Player],
        attack_players: List[Player],
        defense_players: List[Player],
        player_onball: Player,
        ball: Ball,
        goal_position: Position,
        team_name_attack: str,
        onball_team: str
    ) -> OnBall:

    # someone is on the ball
    dist_ball_opponent, nearest_players = calc_dist_each_opponents(ball, attack_players, defense_players)
    dist_goal, goal_positions = calc_dist_each_goals(nearest_players, goal_position)
    angle_goal = calc_angle_each_goals(nearest_players, goal_positions)
    ball_speed = np.sqrt(ball.position.x**2 + ball.position.y**2)
    dribble_score, weighted_area = calc_dribble_score(players, ball, team_name_attack, player_onball, "position")
    dribble_score_vel, weighted_area_vel = calc_dribble_score(players, ball, team_name_attack, player_onball, "velocity")
    shot_score = np.NaN
    long_ball_score = [np.NaN]*3
    transition = [np.NaN]*len(players)
    if player_onball is not None:
        if player_onball.player_role == "GK":
            long_ball_score = calc_long_ball_score(attack_players)
        if player_onball.position.x >= 75.0:
            shot_score = calc_shot_score(player_onball, goal_position, defense_players)
    else:
        transition = calc_transition(players, onball_team)
        
    return OnBall(
        dist_ball_opponent=dist_ball_opponent,
        dribble_score=dribble_score,
        dribble_score_vel=dribble_score_vel,
        dist_goal=dist_goal,
        angle_goal=angle_goal,
        ball_speed=ball_speed,
        transition = transition,
        shot_score=shot_score,
        long_ball_score=long_ball_score,
    ), weighted_area, weighted_area_vel


# offball function
def calc_offball(
        players: List[Player],
        attack_players: List[Player],
        defense_players: List[Player],
        player_onball: Player,
        ball: Ball,
        weighted_areas: Dict[str, float],
        weighted_areas_vel: Dict[str, float],
        team_name_attack: str
    ) -> OffBall:

    # convert weighted area: dict to weighted area: List[float]
    # weighted_areas_list: List[float] = list(weighted_areas.values())
    if player_onball is not None:
        attack_players = [player for player in attack_players if player.player_name != player_onball.player_name]
    fast_space = [weighted_areas[player.player_name] for player in attack_players]
    fast_space_vel = [weighted_areas_vel[player.player_name] for player in attack_players]
    variation_space = calc_variation_space(players, ball, team_name_attack, weighted_areas, "position")
    variation_space_vel = calc_variation_space(players, ball, team_name_attack, weighted_areas_vel, "velocity")
    dist_ball = calc_dist_ball(ball, attack_players)
    defense_space = [weighted_areas[player.player_name] for player in defense_players]
    defense_space_vel = [weighted_areas_vel[player.player_name] for player in defense_players]
    defense_dist_ball = calc_dist_ball(ball, defense_players)
    if player_onball is not None:
        # remove player_onball from attack_players
        time_to_player = calc_time_to_player(attack_players, defense_players)
        time_to_passline = calc_time_to_passline(ball, attack_players, defense_players)
    else:
        # fast_space = [np.NaN]*len(attack_players)
        # fast_space_vel = [np.NaN]*len(attack_players)
        # variation_space = [[np.NaN]*8]*len(attack_players)
        # variation_space_vel = [[np.NaN]*8]*len(attack_players)
        time_to_player = [np.NaN]*len(attack_players)
        time_to_passline = [np.NaN]*len(attack_players)
        # defense_space = [np.NaN]*len(defense_players)
        # defense_space_vel = [np.NaN]*len(defense_players)

    
    # This is a placeholder. You should implement the actual calculations here.
    return OffBall(
        fast_space=fast_space,
        fast_space_vel=fast_space_vel,
        dist_ball=dist_ball,
        time_to_player=time_to_player,
        time_to_passline=time_to_passline,
        variation_space=variation_space,
        variation_space_vel=variation_space_vel,
        defense_space=defense_space,
        defense_space_vel=defense_space_vel,
        defense_dist_ball=defense_dist_ball
    )


# absolute state function
def calc_absolute_state(players: Player, ball: Ball, attack_players: List[Player], defense_players: List[Player], formation: int) -> AbsoluteState:
    # This is a placeholder. You should implement the actual calculations here

    # calculate distance to offside line
    dist_offside_line = calc_dist_offside_line(attack_players, defense_players, ball)
    attacker_action = [player.action for player in attack_players]
    defender_action = [player.action for player in defense_players]

    return AbsoluteState(
        dist_offside_line=dist_offside_line,
        formation=str(formation) if formation is not None else "",  # Example formation
        attack_action=attacker_action,
        defense_action=defender_action
    )

