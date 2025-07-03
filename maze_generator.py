import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import random
from collections import deque
from enum import Enum

class MazeType(Enum):
    RECURSIVE_BACKTRACK = "recursive_backtrack"
    BINARY_TREE = "binary_tree"
    ELLER = "eller"
    KRUSKAL = "kruskal"

class MazeGenerator:
    def __init__(self, width=21, height=21, seed=None):
        """
        Initialize maze generator
        Args:
            width, height: Must be odd numbers for proper maze structure
            seed: Random seed for reproducibility
        """
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        # 0 = wall, 1 = path
        self.maze = np.zeros((self.height, self.width), dtype=np.uint8)
        
    def recursive_backtrack(self):
        """
        Recursive backtracking algorithm - creates mazes with long winding paths
        Good for RL training (complex navigation)
        """
        # Start from (1,1)
        start_x, start_y = 1, 1
        self.maze[start_y, start_x] = 1
        
        stack = [(start_x, start_y)]
        
        while stack:
            current_x, current_y = stack[-1]
            
            # Get unvisited neighbors (2 cells away)
            neighbors = []
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]  # up, right, down, left
            
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if (0 < nx < self.width-1 and 0 < ny < self.height-1 and 
                    self.maze[ny, nx] == 0):
                    neighbors.append((nx, ny, dx//2, dy//2))
            
            if neighbors:
                # Choose random neighbor
                nx, ny, wall_x, wall_y = random.choice(neighbors)
                
                # Remove wall between current and chosen cell
                self.maze[current_y + wall_y, current_x + wall_x] = 1
                self.maze[ny, nx] = 1
                
                stack.append((nx, ny))
            else:
                stack.pop()
                
    def binary_tree(self):
        """
        Binary tree algorithm - creates mazes with bias toward one corner
        Good for testing algorithm robustness
        """
        # Create paths at all odd positions
        for y in range(1, self.height, 2):
            for x in range(1, self.width, 2):
                self.maze[y, x] = 1
                
                # Randomly choose to go north or east (if possible)
                directions = []
                if y > 1:  # Can go north
                    directions.append((0, -1))
                if x < self.width - 2:  # Can go east
                    directions.append((1, 0))
                
                if directions:
                    dx, dy = random.choice(directions)
                    self.maze[y + dy, x + dx] = 1
                    
    def kruskal(self):
        """
        Kruskal's algorithm using Union-Find
        Creates mazes with more uniform distribution of short paths
        Good for CNN pattern recognition
        """
        # Initialize cells
        cells = []
        parent = {}
        
        for y in range(1, self.height, 2):
            for x in range(1, self.width, 2):
                self.maze[y, x] = 1
                cells.append((x, y))
                parent[(x, y)] = (x, y)
        
        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]
        
        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                parent[root2] = root1
                return True
            return False
        
        # Get all possible walls between cells
        walls = []
        for x, y in cells:
            if x < self.width - 2:  # Right wall
                walls.append(((x, y), (x + 2, y), (x + 1, y)))
            if y < self.height - 2:  # Bottom wall
                walls.append(((x, y), (x, y + 2), (x, y + 1)))
        
        random.shuffle(walls)
        
        # Process walls
        for cell1, cell2, wall in walls:
            if union(cell1, cell2):
                self.maze[wall[1], wall[0]] = 1
                
    def add_entrance_exit(self):
        """Add entrance and exit to maze with reachability validation"""
        entrance, exit_point = self.find_valid_entrance_exit()
        
        # Ensure the points are actually accessible
        self.maze[entrance[1], entrance[0]] = 1
        self.maze[exit_point[1], exit_point[0]] = 1
        
        # Store entrance and exit for later use
        self.entrance = entrance
        self.exit_point = exit_point
        
        return entrance, exit_point
        
    def find_valid_entrance_exit(self):
        """Find valid entrance and exit points that are reachable"""
        # Find all path cells on borders
        entrance_candidates = []
        exit_candidates = []
        
        # Top and bottom borders
        for x in range(self.width):
            if self.maze[1, x] == 1:  # Near top border
                entrance_candidates.append((x, 0))
            if self.maze[self.height-2, x] == 1:  # Near bottom border
                exit_candidates.append((x, self.height-1))
        
        # Left and right borders
        for y in range(self.height):
            if self.maze[y, 1] == 1:  # Near left border
                entrance_candidates.append((0, y))
            if self.maze[y, self.width-2] == 1:  # Near right border
                exit_candidates.append((self.width-1, y))
        
        # Try to find reachable entrance-exit pairs
        for entrance in entrance_candidates:
            for exit_point in exit_candidates:
                if self.is_reachable(entrance, exit_point):
                    return entrance, exit_point
        
        # If no valid pair found, create one by forcing paths
        # Default entrance and exit
        entrance = (1, 0)
        exit_point = (self.width-2, self.height-1)
        
        # Ensure entrance and exit are paths
        self.maze[0, 1] = 1  # entrance
        self.maze[1, 1] = 1  # connect to interior
        self.maze[self.height-1, self.width-2] = 1  # exit
        self.maze[self.height-2, self.width-2] = 1  # connect to interior
        
        return entrance, exit_point
    
    def is_reachable(self, start, goal):
        """
        Check if goal is reachable from start using BFS
        Returns True if path exists, False otherwise
        """
        if start == goal:
            return True
            
        # Convert border points to internal accessible points
        start_internal = self.get_internal_point(start)
        goal_internal = self.get_internal_point(goal)
        
        if start_internal is None or goal_internal is None:
            return False
        
        visited = set()
        queue = deque([start_internal])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal_internal:
                return True
            
            x, y = current
            # Check all 4 directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.maze[ny, nx] == 1 and (nx, ny) not in visited):
                    queue.append((nx, ny))
        
        return False
    
    def get_internal_point(self, border_point):
        """Convert border point to nearest internal path point"""
        x, y = border_point
        
        # Check adjacent internal cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and
                self.maze[ny, nx] == 1):
                return (nx, ny)
        
        return None
    
    def find_shortest_path(self, start, goal):
        """
        Find shortest path from start to goal using BFS
        Returns path as list of coordinates, or None if no path exists
        """
        if not self.is_reachable(start, goal):
            return None
        
        start_internal = self.get_internal_point(start)
        goal_internal = self.get_internal_point(goal)
        
        visited = set()
        queue = deque([(start_internal, [start_internal])])
        
        while queue:
            current, path = queue.popleft()
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal_internal:
                return [start] + path + [goal]
            
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.maze[ny, nx] == 1 and (nx, ny) not in visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return None
    
    def get_complexity_metrics(self, entrance=None, exit_point=None):
        """Calculate maze complexity metrics for dataset analysis"""
        # Path density
        path_cells = np.sum(self.maze == 1)
        total_cells = self.width * self.height
        path_density = path_cells / total_cells
        
        # Dead ends count
        dead_ends = 0
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.maze[y, x] == 1:
                    neighbors = [
                        self.maze[y-1, x], self.maze[y+1, x],
                        self.maze[y, x-1], self.maze[y, x+1]
                    ]
                    if sum(neighbors) == 1:  # Only one neighbor is path
                        dead_ends += 1
        
        # Longest path (approximate using BFS)
        def bfs_longest_path():
            start = (1, 1)
            visited = set()
            queue = deque([(start, 0)])
            max_distance = 0
            
            while queue:
                (x, y), dist = queue.popleft()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                max_distance = max(max_distance, dist)
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and
                        self.maze[ny, nx] == 1 and (nx, ny) not in visited):
                        queue.append(((nx, ny), dist + 1))
            
            return max_distance
        
        longest_path = bfs_longest_path()
        
        # Calculate shortest path length if entrance and exit are provided
        shortest_path_length = 0
        if entrance and exit_point:
            shortest_path = self.find_shortest_path(entrance, exit_point)
            shortest_path_length = len(shortest_path) - 1 if shortest_path else 0
        
        return {
            'path_density': path_density,
            'dead_ends': dead_ends,
            'longest_path': longest_path,
            'shortest_path_length': shortest_path_length,
            'branching_factor': path_density * 4,  # Approximate branching
            'solvable': shortest_path_length > 0
        }

class MazeDatasetGenerator:
    def __init__(self, output_dir="maze_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/arrays", exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        
    def generate_dataset(self, 
                        total_samples=1000,
                        size_range=(15, 31),  # (min, max) - odd numbers only
                        algorithms=[MazeType.RECURSIVE_BACKTRACK, MazeType.KRUSKAL],
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15):
        """
        Generate complete maze dataset
        """
        dataset_info = {
            'total_samples': total_samples,
            'size_range': size_range,
            'algorithms': [alg.value for alg in algorithms],
            'splits': {
                'train': int(total_samples * train_ratio),
                'val': int(total_samples * val_ratio),
                'test': int(total_samples * test_ratio)
            }
        }
        
        mazes_data = []
        
        print(f"Generating {total_samples} mazes...")
        
        for i in range(total_samples):
            # Random parameters
            size = random.choice(range(size_range[0], size_range[1] + 1, 2))  # Odd numbers only
            algorithm = random.choice(algorithms)
            
            # Keep generating until we get a solvable maze
            max_attempts = 10  # Prevent infinite loops
            attempt = 0
            solvable = False
            
            while not solvable and attempt < max_attempts:
                attempt += 1
                
                # Generate maze with unique seed for each attempt
                seed = i * 1000 + attempt
                generator = MazeGenerator(size, size, seed=seed)
                
                if algorithm == MazeType.RECURSIVE_BACKTRACK:
                    generator.recursive_backtrack()
                elif algorithm == MazeType.KRUSKAL:
                    generator.kruskal()
                elif algorithm == MazeType.BINARY_TREE:
                    generator.binary_tree()
                
                entrance, exit_point = generator.add_entrance_exit()
                
                # Check if maze is solvable
                if generator.is_reachable(entrance, exit_point):
                    solvable = True
                    if attempt > 1:
                        print(f"Maze {i}: Found solvable maze on attempt {attempt}")
                else:
                    if attempt == 1:
                        print(f"Maze {i}: Attempt {attempt} failed, regenerating...")
                    else:
                        print(f"Maze {i}: Attempt {attempt} failed, trying again...")
            
            if not solvable:
                print(f"ERROR: Could not generate solvable maze {i} after {max_attempts} attempts!")
                print(f"This might indicate an issue with the algorithm or parameters.")
                continue  # Skip this maze and continue with next
            
            # Get metrics for the solvable maze
            metrics = generator.get_complexity_metrics(generator.entrance, generator.exit_point)
            
            # Verify once more that it's actually solvable
            if not metrics['solvable']:
                print(f"ERROR: Maze {i} metrics indicate unsolvable despite passing reachability test!")
                continue
            
            # Save data
            maze_info = {
                'id': i,
                'size': size,
                'algorithm': algorithm.value,
                'entrance': generator.entrance,
                'exit': generator.exit_point,
                'generation_attempts': attempt,
                'metrics': metrics
            }
            
            # Save as image (for CNN)
            img = Image.fromarray((generator.maze * 255).astype(np.uint8), mode='L')
            img.save(f"{self.output_dir}/images/maze_{i:06d}.png")
            
            # Save as numpy array (for general use)
            np.save(f"{self.output_dir}/arrays/maze_{i:06d}.npy", generator.maze)
            
            # Save metadata
            with open(f"{self.output_dir}/metadata/maze_{i:06d}.json", 'w') as f:
                json.dump(maze_info, f)
            
            mazes_data.append(maze_info)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{total_samples} mazes")
        
        # Final dataset validation
        successful_mazes = len(mazes_data)
        if successful_mazes < total_samples:
            print(f"\nWarning: Only generated {successful_mazes}/{total_samples} mazes successfully")
            # Update dataset info with actual count
            dataset_info['total_samples'] = successful_mazes
            dataset_info['splits'] = {
                'train': int(successful_mazes * train_ratio),
                'val': int(successful_mazes * val_ratio),
                'test': int(successful_mazes * test_ratio)
            }
        
        # Create train/val/test splits with actual maze count
        indices = list(range(successful_mazes))
        random.shuffle(indices)
        
        train_end = dataset_info['splits']['train']
        val_end = train_end + dataset_info['splits']['val']
        
        splits = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:]
        }
        
        # Calculate generation statistics
        total_attempts = sum(maze['generation_attempts'] for maze in mazes_data)
        avg_attempts = total_attempts / successful_mazes if successful_mazes > 0 else 0
        difficult_mazes = sum(1 for maze in mazes_data if maze['generation_attempts'] > 1)
        
        print(f"\nGeneration Statistics:")
        print(f"Total attempts needed: {total_attempts}")
        print(f"Average attempts per maze: {avg_attempts:.2f}")
        print(f"Mazes requiring multiple attempts: {difficult_mazes}/{successful_mazes}")
        
        # Save dataset info and splits
        dataset_info['mazes'] = mazes_data
        dataset_info['splits_indices'] = splits
        dataset_info['generation_stats'] = {
            'total_attempts': total_attempts,
            'avg_attempts_per_maze': avg_attempts,
            'difficult_mazes': difficult_mazes
        }
        
        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generated successfully!")
        print(f"Successful mazes: {successful_mazes}")
        print(f"Train: {len(splits['train'])} samples")
        print(f"Val: {len(splits['val'])} samples")
        print(f"Test: {len(splits['test'])} samples")
        
        return dataset_info
    
    def visualize_sample(self, maze_id, save_path=None, show_solution=True):
        """Visualize a maze sample with optional solution path"""
        maze = np.load(f"{self.output_dir}/arrays/maze_{maze_id:06d}.npy")
        
        with open(f"{self.output_dir}/metadata/maze_{maze_id:06d}.json", 'r') as f:
            info = json.load(f)
        
        plt.figure(figsize=(10, 8))
        
        # Create visualization
        display_maze = maze.copy().astype(float)
        
        if show_solution and 'entrance' in info and 'exit' in info:
            # Load maze generator to find path
            temp_gen = MazeGenerator(info['size'], info['size'])
            temp_gen.maze = maze
            temp_gen.entrance = tuple(info['entrance'])
            temp_gen.exit_point = tuple(info['exit'])
            
            # Find and display shortest path
            path = temp_gen.find_shortest_path(temp_gen.entrance, temp_gen.exit_point)
            if path:
                for x, y in path[1:-1]:  # Exclude entrance and exit from path coloring
                    if 0 <= x < maze.shape[1] and 0 <= y < maze.shape[0]:
                        display_maze[y, x] = 0.5  # Gray for path
                
                # Mark entrance and exit
                ex, ey = temp_gen.entrance
                if 0 <= ex < maze.shape[1] and 0 <= ey < maze.shape[0]:
                    display_maze[ey, ex] = 0.3  # Darker for entrance
                    
                gx, gy = temp_gen.exit_point
                if 0 <= gx < maze.shape[1] and 0 <= gy < maze.shape[0]:
                    display_maze[gy, gx] = 0.7  # Lighter for exit
        
        plt.imshow(display_maze, cmap='RdYlBu_r')
        plt.title(f"Maze {maze_id} - {info['algorithm']} - Size: {info['size']}x{info['size']}\n" +
                 f"Solvable: {info['metrics']['solvable']} - " +
                 f"Shortest Path: {info['metrics']['shortest_path_length']} steps")
        
        # Add legend if showing solution
        if show_solution:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='darkblue', label='Wall'),
                Patch(facecolor='yellow', label='Path'),
                Patch(facecolor='orange', label='Solution'),
                Patch(facecolor='darkred', label='Entrance'),
                Patch(facecolor='lightblue', label='Exit')
            ]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def validate_dataset(self):
        """Validate that all mazes in dataset are solvable"""
        with open(f"{self.output_dir}/dataset_info.json", 'r') as f:
            dataset_info = json.load(f)
        
        solvable_count = 0
        total_count = len(dataset_info['mazes'])
        
        print("Validating dataset...")
        for maze_info in dataset_info['mazes']:
            if maze_info['metrics']['solvable']:
                solvable_count += 1
            else:
                print(f"ERROR: Maze {maze_info['id']} is marked as not solvable!")
                print(f"This should not happen with the new generation method!")
        
        print(f"Dataset validation complete:")
        print(f"Solvable mazes: {solvable_count}/{total_count} ({solvable_count/total_count*100:.1f}%)")
        
        if 'generation_stats' in dataset_info:
            stats = dataset_info['generation_stats']
            print(f"Generation efficiency: {stats['avg_attempts_per_maze']:.2f} attempts per maze")
            print(f"Difficult mazes: {stats['difficult_mazes']} required multiple attempts")
        
        return solvable_count == total_count

# Example usage
if __name__ == "__main__":
    # Generate dataset
    generator = MazeDatasetGenerator("maze_benchmark_dataset")
    
    dataset_info = generator.generate_dataset(
        total_samples=2000,
        size_range=(50, 200),
        algorithms=[MazeType.RECURSIVE_BACKTRACK, MazeType.KRUSKAL],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Visualize some samples with solutions
    print("\nVisualizing sample mazes with solutions...")
    for i in [0, 1, 2]:
        generator.visualize_sample(i, show_solution=True)
    
    # Validate entire dataset
    print("\nValidating dataset...")
    all_solvable = generator.validate_dataset()
    if all_solvable:
        print("✅ All mazes are solvable!")
    else:
        print("⚠️ Some mazes are not solvable - check warnings above")
    
    # Print final dataset statistics
    successful_mazes = len(dataset_info['mazes'])
    if successful_mazes > 0:
        avg_path_length = np.mean([maze['metrics']['shortest_path_length'] 
                                   for maze in dataset_info['mazes']])
        avg_attempts = dataset_info['generation_stats']['avg_attempts_per_maze']
        
        print(f"\nFinal Dataset Statistics:")
        print(f"Successfully generated mazes: {successful_mazes}")
        print(f"All mazes are guaranteed solvable: ✅")
        print(f"Average solution length: {avg_path_length:.1f} steps")
        print(f"Generation efficiency: {avg_attempts:.2f} attempts per maze")
        
        # Show algorithm distribution
        algorithm_counts = {}
        for maze in dataset_info['mazes']:
            alg = maze['algorithm']
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        
        print(f"\nAlgorithm distribution:")
        for alg, count in algorithm_counts.items():
            print(f"  {alg}: {count} mazes ({count/successful_mazes*100:.1f}%)")
    else:
        print("❌ No mazes were successfully generated!")