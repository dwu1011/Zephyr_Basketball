from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 20):
        self.buffer = deque(maxlen=buffer_size)
        self.centroid = None

    # def update(self, detections: sv.Detections) -> sv.Detections:
    #     """
    #     Updates the buffer with new detections and returns the detection closest to the
    #     centroid of recent positions.

    #     Args:
    #         detections (sv.Detections): The current frame's ball detections.

    #     Returns:
    #         sv.Detections: The detection closest to the centroid of recent positions.
    #         If there are no detections, returns the input detections.
    #     """
    #     xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
    #     centroid = np.mean(self.buffer)
    #     if np.linalg.norm(xy-centroid) <= 25:
    #         self.buffer.append(xy)

    #     if len(detections) == 0:
    #         return detections

    #     centroid = np.mean(np.concatenate(self.buffer), axis=0)
    #     distances = np.linalg.norm(xy - centroid, axis=1)
    #     index = np.argmin(distances)
    #     return detections[[index]]
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the most likely ball position.
        Uses velocity-based prediction when detections are missing or unreliable.
        
        Args:
            detections (sv.Detections): The current frame's ball detections.
            
        Returns:
            sv.Detections: The most likely ball position, either from detection or prediction.
        """
        # Track how many consecutive predictions we've made without real detections
        if not hasattr(self, 'consecutive_predictions'):
            self.consecutive_predictions = 0
        
        # If buffer is empty, just use the detection (if available)
        if len(self.buffer) == 0 and len(detections) > 0:
            xy = detections.get_anchors_coordinates(sv.Position.CENTER)
            if len(xy) > 0:
                self.buffer.append(xy[0])  # Store just the coordinates
                self.consecutive_predictions = 0
                return detections[[0]]  # Return first detection
            return detections
        
        # Get coordinates from detections
        if len(detections) > 0:
            xy = detections.get_anchors_coordinates(sv.Position.CENTER)
            
            # If we've been predicting for too long, prioritize actual detections
            if self.consecutive_predictions > 10:  # Adjust threshold as needed
                # Just use the most confident detection
                most_conf_idx = np.argmax(detections.confidence)
                self.buffer.append(xy[most_conf_idx])
                self.consecutive_predictions = 0
                return detections[[most_conf_idx]]
                
            # Calculate predicted position based on recent movement
            if len(self.buffer) >= 2:
                # Get recent positions
                recent_positions = np.array(list(self.buffer)[-3:])
                
                # Calculate velocity vector from last positions
                if len(recent_positions) >= 2:
                    velocity = recent_positions[-1] - recent_positions[0]
                    velocity = velocity / len(recent_positions)
                    
                    # Predict next position
                    predicted_position = recent_positions[-1] + velocity
                    
                    # If we have detections, compare with prediction
                    if len(xy) > 0:
                        distances = np.linalg.norm(xy - predicted_position, axis=1)
                        closest_idx = np.argmin(distances)
                        closest_distance = distances[closest_idx]
                        
                        # If closest detection is too far from prediction but we've been predicting
                        # for a while, trust the detection more
                        trust_threshold = 50  # Base threshold
                        if self.consecutive_predictions > 5:
                            # Gradually increase acceptable distance as predictions continue
                            trust_threshold = trust_threshold * (1 + 0.2 * self.consecutive_predictions)
                        
                        # If closest detection is too far from prediction, use prediction
                        if closest_distance > trust_threshold:
                            # Create a synthetic detection at the predicted position
                            xyxy = np.array([[
                                predicted_position[0] - 10, 
                                predicted_position[1] - 10,
                                predicted_position[0] + 10, 
                                predicted_position[1] + 10
                            ]])
                            confidence = np.array([0.5])
                            class_id = np.array([0])  # Ball class ID
                            synthetic_detection = sv.Detections(
                                xyxy=xyxy,
                                confidence=confidence,
                                class_id=class_id
                            )
                            self.buffer.append(predicted_position)
                            self.consecutive_predictions += 1
                            return synthetic_detection
                        else:
                            # Detection is close to prediction, use the detection
                            selected_detection = detections[[closest_idx]]
                            self.buffer.append(xy[closest_idx])
                            self.consecutive_predictions = 0
                            return selected_detection
            
            # No velocity calculation possible yet, use centroid-based approach
            if len(self.buffer) > 0:
                buffer_array = np.array(list(self.buffer))
                centroid = np.mean(buffer_array, axis=0)
                distances = np.linalg.norm(xy - centroid, axis=1)
                index = np.argmin(distances)
                self.buffer.append(xy[index])
                self.consecutive_predictions = 0
                return detections[[index]]
            else:
                # First detection, just add to buffer and return
                self.buffer.append(xy[0])
                self.consecutive_predictions = 0
                return detections[[0]]
        
        # No detections - predict based on velocity if possible
        elif len(self.buffer) >= 2:
            recent_positions = np.array(list(self.buffer)[-3:])
            if len(recent_positions) >= 2:
                velocity = recent_positions[-1] - recent_positions[0]
                velocity = velocity / len(recent_positions)
                predicted_position = recent_positions[-1] + velocity
                
                # Create synthetic detection
                xyxy = np.array([[
                    predicted_position[0] - 10, 
                    predicted_position[1] - 10,
                    predicted_position[0] + 10, 
                    predicted_position[1] + 10
                ]])
                confidence = np.array([0.5])
                class_id = np.array([0])  # Ball class ID
                synthetic_detection = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id
                )
                self.buffer.append(predicted_position)
                self.consecutive_predictions += 1
                return synthetic_detection
        
        # No detection and no prediction possible
        return detections
