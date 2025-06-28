import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import heapq

class Node:
    """A* 알고리즘을 위한 노드 클래스"""
    def __init__(self, x, z, cost=0, heuristic=0, parent=None):
        self.x = x  # 그리드 X 좌표
        self.z = z  # 그리드 Z 좌표
        self.cost = cost  # 시작점으로부터의 실제 비용 (g_cost)
        self.heuristic = heuristic  # 목적지까지의 추정 비용 (h_cost)
        self.parent = parent  # 부모 노드
        self.f_cost = self.cost + self.heuristic # 총 비용 (f_cost)

    def __lt__(self, other):
        """우선순위 큐를 위한 비교 연산자 (f_cost 기준)"""
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        """노드 동일성 비교 (좌표 기준)"""
        return isinstance(other, Node) and self.x == other.x and self.z == other.z

    def __hash__(self):
        """해시 함수 (좌표 기준, 딕셔너리 키로 사용하기 위함)"""
        return hash((self.x, self.z))

class AStarPathfinder:
    def __init__(self, map_width, map_height, grid_size, grid):
        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size
        self.grid_width = int(map_width / grid_size)
        self.grid_height = int(map_height / grid_size)

        # grid: 0 = free, 1 = obstacle
        self.grid = grid
        self.obstacles = [] # 원본 장애물 데이터 저장 (시각화에 사용)

        # 인접한 그리드 셀 (8방향 이동) 및 이동 비용
        self.neighbors = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # 상하좌우
            (1, 1), (1, -1), (-1, 1), (-1, -1) # 대각선
        ]
        self.move_costs = {
            (0, 1): 1, (0, -1): 1, (1, 0): 1, (-1, 0): 1,
            (1, 1): math.sqrt(2), (1, -1): math.sqrt(2), (-1, 1): math.sqrt(2), (-1, -1): math.sqrt(2)
        }

    def _world_to_grid_coord(self, val, axis):
        """월드 좌표를 그리드 좌표로 변환"""
        # 월드 좌표가 음수일 경우를 고려하여 그리드 인덱스를 안전하게 계산
        grid_coord = int(val / self.grid_size)
        if axis == 'x':
            return max(0, min(grid_coord, self.grid_width - 1))
        elif axis == 'z':
            return max(0, min(grid_coord, self.grid_height - 1))
        return -1 # 에러 발생 시

    def _grid_to_world_center(self, grid_x, grid_z):
        """그리드 좌표를 월드 중심 좌표로 변환"""
        world_x = (grid_x + 0.5) * self.grid_size
        world_z = (grid_z + 0.5) * self.grid_size
        return world_x, world_z


    def update_obstacles(self, obstacles, start_pos=None, end_pos=None):
            """ 
                그리드 맵에 장애물 정보를 업데이트합니다.   
                update_obstacles의 기본정보 : x, z, radius
            """
            self.grid.fill(0) # 기존 그리드 초기화 (모두 자유 공간으로)
            self.obstacles = obstacles # 원본 장애물 데이터 저장

            for obs in obstacles:
                x, z, radius = obs['x'], obs['z'], obs['radius']
                
                # 장애물 반경을 고려하여 영향을 받는 그리드 셀 범위를 계산
                # 실제 그리드 셀의 중심과 장애물 중심 간의 거리를 사용
                min_gx = self._world_to_grid_coord(x - radius, 'x')
                max_gx = self._world_to_grid_coord(x + radius, 'x')
                min_gz = self._world_to_grid_coord(z - radius, 'z')
                max_gz = self._world_to_grid_coord(z + radius, 'z')

                # 그리드 경계를 벗어나지 않도록 클램프
                # _world_to_grid_coord 자체에서 클램핑되지만, 명시적으로 다시 적용하여 안전성 확보
                min_gx = max(0, min_gx)
                max_gx = min(self.grid_width - 1, max_gx)
                min_gz = max(0, min_gz)
                max_gz = min(self.grid_height - 1, max_gz) 

                for gx in range(min_gx, max_gx + 1):
                    for gz in range(min_gz, max_gz + 1):
                        cell_center_x, cell_center_z = self._grid_to_world_center(gx, gz)
                        distance_to_obstacle_center = math.sqrt((cell_center_x - x)**2 + (cell_center_z - z)**2)
                        
                        # 셀 중심이 장애물 반지름 내에 있거나, 셀이 장애물과 충분히 가까이 있다면 장애물로 표시
                        # 여기서는 셀의 모서리 중 하나라도 장애물 원 안에 들어오면 장애물로 간주하도록 보수적으로 처리
                        # 또는, 간단하게 dist <= radius + grid_size/2.0 와 같이 마진을 줄 수 있습니다.
                        if distance_to_obstacle_center <= radius + 2.5: # 그리드 셀 크기의 절반만큼 마진 추가 ##################################################################### 여기 수정해야됨
                            self.grid[gx][gz] = 1 # 장애물로 표시
                            
            # 예외 처리: 시작점과 도착점은 통과 가능하게 강제 개방
            if start_pos:
                sx = self._world_to_grid_coord(start_pos[0], 'x')
                sz = self._world_to_grid_coord(start_pos[1], 'z')
                self.grid[sx][sz] = 0

            if end_pos:
                ex = self._world_to_grid_coord(end_pos[0], 'x')
                ez = self._world_to_grid_coord(end_pos[1], 'z')
                self.grid[ex][ez] = 0
            

    def _is_valid(self, x, z):
        """그리드 좌표가 유효한 범위 내에 있고 장애물이 아닌지 확인"""
        return 0 <= x < self.grid_width and 0 <= z < self.grid_height and self.grid[x][z] != 1

    def _calculate_heuristic(self, node_x, node_z, target_x, target_z):
        """휴리스틱 비용 계산 (유클리드 거리)"""
        return math.sqrt((node_x - target_x)**2 + (node_z - target_z)**2)



    def find_path(self, start_world, end_world):
        """A* 알고리즘을 사용하여 최적 경로를 찾습니다."""
        start_gx, start_gz = self._world_to_grid_coord(start_world[0], 'x'), self._world_to_grid_coord(start_world[1], 'z')
        end_gx, end_gz = self._world_to_grid_coord(end_world[0], 'x'), self._world_to_grid_coord(end_world[1], 'z')

        if not self._is_valid(start_gx, start_gz) or not self._is_valid(end_gx, end_gz):
            print(f"Warning: Start ({start_world}) or End ({end_world}) position is invalid or on an obstacle.")
            return []

        # Open Set (우선순위 큐)
        open_set = [] 
        # 초기 노드 추가: g_cost = 0, h_cost 계산
        heapq.heappush(open_set, Node(start_gx, start_gz, 0, self._calculate_heuristic(start_gx, start_gz, end_gx, end_gz)))
        
        # g_costs: 시작점으로부터의 실제 비용 저장 (좌표 -> 비용)
        g_costs = {(start_gx, start_gz): 0}
        
        # came_from: 경로 재구성을 위한 부모 노드 추적 (좌표 -> Node 객체)
        came_from = {(start_gx, start_gz): None}

        while open_set:
            current_node = heapq.heappop(open_set)

            # 목적지에 도달했으면 경로 재구성
            if (current_node.x, current_node.z) == (end_gx, end_gz):
                path = []
                temp = current_node
                while temp:
                    world_x, world_z = self._grid_to_world_center(temp.x, temp.z)
                    path.append({'x': world_x, 'z': world_z})
                    temp = temp.parent
                return path[::-1] # 역순으로 저장되었으므로 뒤집어서 반환

            # 인접 노드 탐색
            for dx, dz in self.neighbors:
                neighbor_x, neighbor_z = current_node.x + dx, current_node.z + dz

                # 유효성 검사 (맵 범위 내, 장애물 아님)
                if not self._is_valid(neighbor_x, neighbor_z):
                    continue

                new_g_cost = current_node.cost + self.move_costs[(dx, dz)]

                # 더 짧은 경로를 찾았거나, 처음 방문하는 경우
                if (neighbor_x, neighbor_z) not in g_costs or new_g_cost < g_costs[(neighbor_x, neighbor_z)]:
                    g_costs[(neighbor_x, neighbor_z)] = new_g_cost
                    heuristic = self._calculate_heuristic(neighbor_x, neighbor_z, end_gx, end_gz)
                    new_node = Node(neighbor_x, neighbor_z, new_g_cost, heuristic, current_node)
                    heapq.heappush(open_set, new_node)
                    came_from[(neighbor_x, neighbor_z)] = current_node # 부모 노드 기록

        return [] # 경로를 찾지 못한 경우

   

    
    matplotlib.use("TkAgg")

    def visualize_astar_grid(self, obstacles, path=None, start_pos=None, end_pos=None, grid_size=5, map_width=300, map_height=300):
        """
        A* 결과를 시각화합니다.

        Parameters:
            grid: 2D numpy 배열 (0=빈 공간, 1=장애물)
            obstacles: [{'x': x, 'z': z, 'radius': r}, ...]
            path: [{'x': x, 'z': z}, ...]
            start_pos: (x, z)
            end_pos: (x, z)
        """
        # ✅ 완전 새 figure + 새 axes
        fig, ax = plt.subplots(figsize=(10, 10))  

        # 1. 그리드 배경
        ax.imshow(self.grid.T, cmap='Greys_r', origin='lower',
                extent=[0, map_width, 0, map_height])

        # 2. 장애물 (원형)
        for obs in obstacles:
            circle = plt.Circle((obs['x'], obs['z']), obs['radius'], color='red', alpha=0.5)
            ax.add_patch(circle)

        # 3. 시작점
        if start_pos:
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
            ax.text(start_pos[0], start_pos[1] + 1, 'Start', ha='center', fontsize=8)

        # 4. 도착점
        if end_pos:
            ax.plot(end_pos[0], end_pos[1], 'bo', markersize=10, label='End')
            ax.text(end_pos[0], end_pos[1] + 1, 'End', ha='center', fontsize=8)

        # 5. 경로
        if path:
            path_x = [p['x'] for p in path]
            path_z = [p['z'] for p in path]
            ax.plot(path_x, path_z, 'y-', linewidth=2, label='Path')

        # 6. 기타 시각적 설정
        ax.set_xticks(np.arange(0, map_width + 1, grid_size))
        ax.set_yticks(np.arange(0, map_height + 1, grid_size))
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title("A* Pathfinding Grid")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.legend(loc='upper right')


        # 7. 시각화 표시 및 클리어 (중복 방지)
        plt.show()

