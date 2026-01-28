# %%
import math
import os
import numpy as np
def create_star_mask(height, width, center_x=None, center_y=None, outer_radius=None, inner_radius=None, num_points=5):
        """
        Create a binary mask with a star shape filled with ones.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            center_x (float): X coordinate of star center (default: width/2)
            center_y (float): Y coordinate of star center (default: height/2)
            outer_radius (float): Outer radius of star points (default: min(height,width)/3)
            inner_radius (float): Inner radius between points (default: outer_radius/2.5)
            num_points (int): Number of star points (default: 5)
        
        Returns:
            torch.Tensor: Binary mask with star shape
        """
        if center_x is None:
            center_x = width / 2
        if center_y is None:
            center_y = height / 2
        if outer_radius is None:
            outer_radius = min(height, width) / 3
        if inner_radius is None:
            inner_radius = outer_radius / 2.5
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(height, dtype=np.float16),
                            np.arange(width, dtype=np.float16),
                            indexing='ij')
        
        # Translate coordinates to center
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Convert to polar coordinates
        r = np.sqrt(x_centered**2 + y_centered**2)
        theta = np.arctan2(y_centered, x_centered)
        
        # Normalize angle to [0, 2π]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        # Calculate the radius threshold for star shape
        angle_per_section = 2 * math.pi / num_points
        
        # For each point, calculate which section of the star we're in
        section = np.floor(theta / angle_per_section)
        angle_in_section = theta - section * angle_per_section
        
        # Calculate the radius boundary for the star at this angle
        # Use linear interpolation between inner and outer radius
        angle_ratio = np.abs(angle_in_section - angle_per_section / 2) / (angle_per_section / 2)
        radius_threshold = inner_radius + (outer_radius - inner_radius) * (1 - angle_ratio)
        
        # Create binary mask: 1 where r <= radius_threshold, 0 elsewhere
        mask = (r <= radius_threshold).astype(np.float16)
        
        return mask
def create_synthetic_image(img_size=64):
    """
    Creates a synthetic image with random geometric shapes (circle, square, star).
    
    Args:
        img_size (int): Size of the square image (img_size x img_size)
    Returns:
        input_img (np.ndarray): Synthetic input image with one channel
        target_img (np.ndarray): Target segmentation mask Containg the star
    """
# Generate a background
    input_img = np.random.rand(1,img_size,img_size)* 0.2  # Start with a dark background

    # Create coordinate grids
    x = np.arange(img_size).reshape(img_size, 1).repeat(img_size, axis=1).astype(np.float32)
    y = np.arange(img_size).reshape(1, img_size).repeat(img_size, axis=0).astype(np.float32)
    
    # Set random positions of centers for shapes
    centers_x = np.random.randint(0, img_size, (3,))
    centers_y = np.random.randint(0, img_size, (3,))
    # Create a circle
    center_x_circle = centers_x[0]
    center_y_circle = centers_y[0]
    # Define radius between 10% to 20% of image size
    radius = np.random.randint(img_size // 10,img_size // 5)
    circle_mask = ((x - center_x_circle)**2 + (y - center_y_circle)**2 <= radius**2).astype(np.float32)

    # Create a square
    # Square side size: maximum 30% of image size
    max_side_size = int(img_size * 0.3)
    side_size = np.random.randint(max_side_size // 2, max_side_size + 1)
    
    # Random center for square
    center_x_square = centers_x[1]
    center_y_square = centers_y[1]
    
    # Create square mask
    half_side = side_size // 2
    square_mask = ((np.abs(x - center_x_square) <= half_side) & 
                    (np.abs(y - center_y_square) <= half_side)).astype(np.float32)  
    
    # Create a star mask
    max_radius_size = int(img_size * 0.2)  # Slightly smaller to avoid edge issues
    radius_size = np.random.randint(max_radius_size // 2, max_radius_size + 1)
    center_x_star = centers_x[2]
    center_y_star = centers_y[2]
    
    star_mask =create_star_mask(img_size, img_size, 
                                        center_x=center_x_star, center_y=center_y_star, 
                                        outer_radius=radius_size, inner_radius=radius_size//2)

    # Add all shapes to the input image

    # Combine them with random weights
    alpha , beta, gamma = np.random.dirichlet(np.ones(3))
    input_img[0] += circle_mask * alpha
    input_img[0] += square_mask * beta
    input_img[0] += star_mask * gamma
    input_img = np.clip(input_img, 0, 1)
    # Create the taget channel segmentation mask
    # Here we only keep the star mask as target
    target_img = star_mask
    return input_img, target_img

def simulate_dataset(num_samples, img_size, root_dir):
     os.makedirs(root_dir, exist_ok=True)
     for i in range(num_samples):
        input_img, target_mask = create_synthetic_image(img_size)
        np.savez_compressed(f"{root_dir}/sample_{i:d}.npy", input=input_img, target=target_mask)

        
if __name__ == "__main__":
    # Example usage
    simulate_dataset(num_samples=1000, img_size=128, root_dir="./data/all")
# %%
