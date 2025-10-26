import numpy as np

class TileCoder:
    def __init__(self, num_tiles = 10, num_tilings = 8, state_bounds = None):
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.state_bounds = state_bounds
        
        self.tile_widths = []
        for low, high in state_bounds:
            self.tile_widths.append((high - low) / num_tiles)
            
        self.offset = []
        
        for i in range(num_tilings):
            offsets_for_dim = []
            for tile_width in self.tile_widths:
                offsets_for_dim.append((i * tile_width) / num_tilings)
            self.offset.append(offsets_for_dim)
        
        self.tiles_per_tiling = num_tiles ** len(state_bounds)
        self.total_num_tiles = self.tiles_per_tiling * num_tilings
    
    def get_tiles(self, state):
        active_tiles = []
        
        # For each tiling, compute which tile the state falls into
        for tiling_idx in range(self.num_tilings):
            # Find the tile indices for each dimension
            tile_indices = []
            for dim, (state_val, tile_width, offset) in enumerate(zip(state, self.tile_widths, self.offset[tiling_idx])):
                # Apply offset for this tiling
                adjusted_val = state_val - self.state_bounds[dim][0] - offset
                # Compute tile index (ensure it's within bounds)
                tile_idx = int(adjusted_val / tile_width)
                tile_idx = max(0, min(tile_idx, self.num_tiles - 1))
                tile_indices.append(tile_idx)
            
            # Convert multi-dimensional tile indices to a single index
            # Using row-major order
            flat_index = 0
            multiplier = 1
            for i in range(len(tile_indices) - 1, -1, -1):
                flat_index += tile_indices[i] * multiplier
                multiplier *= self.num_tiles
            
            # Add offset for this tiling to ensure unique indices across tilings
            active_tile = tiling_idx * self.tiles_per_tiling + flat_index
            active_tiles.append(active_tile)
        
        return active_tiles