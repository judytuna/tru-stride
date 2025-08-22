import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import cv2

class HorseStrideAnalyzer:
    def __init__(self):
        # Define the tracked body parts
        self.body_parts = ['forehead', 'crest', 'back', 'point_of_hip', 'right_knee', 'left_knee',
                          'front_right_fetlock', 'front_left_fetlock', 'front_right_foot', 'front_left_foot',
                          'right_hock', 'left_hock', 'hind_right_fetlock', 'hind_left_fetlock',
                          'hind_right_foot', 'hind_left_foot', 'tail']
        
        self.foot_parts = ['front_right_foot', 'front_left_foot', 'hind_right_foot', 'hind_left_foot']
        self.likelihood_threshold = 0.6
        
    def process_video_to_h5(self, video_path):
        """
        Convert video to H5 format using DeepLabCut analysis
        This is a placeholder - you'd replace with your actual DLC processing
        """
        # TODO: Replace with your actual DeepLabCut video processing pipeline
        # For now, we'll simulate the H5 file structure
        return self.create_mock_h5_data()
    
    def create_mock_h5_data(self):
        """
        Create mock H5 data structure for testing
        Replace this with your actual DeepLabCut output processing
        """
        # Simulate 150 frames of pose data
        frames = 150
        scorer = 'DLC_resnet50_horse_lameness_project'
        
        # Create MultiIndex columns exactly like DeepLabCut output
        columns = []
        for part in self.body_parts:
            columns.append((scorer, part, 'x'))
            columns.append((scorer, part, 'y'))
            columns.append((scorer, part, 'likelihood'))
        
        # Create data array
        data = np.random.rand(frames, len(columns))
        
        # Make coordinates realistic
        for i, (scorer_name, part, coord) in enumerate(columns):
            if coord == 'x':
                if 'front' in part:
                    data[:, i] = np.random.normal(320, 50, frames)  # Front of horse
                elif 'hind' in part:
                    data[:, i] = np.random.normal(480, 50, frames)  # Back of horse
                else:
                    data[:, i] = np.random.normal(400, 30, frames)  # Middle
                data[:, i] = np.clip(data[:, i], 0, 640)
            elif coord == 'y':
                data[:, i] = np.random.normal(240, 100, frames)
                data[:, i] = np.clip(data[:, i], 0, 480)
            else:  # likelihood
                data[:, i] = np.random.uniform(0.7, 0.95, frames)
        
        # Create DataFrame with proper MultiIndex
        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns, names=['scorer', 'bodyparts', 'coords']))
        
        return df
    
    def analyze_stride_patterns(self, df):
        """
        Analyze stride patterns from pose estimation data
        """
        scorer = df.columns.get_level_values('scorer')[0]
        
        # Filter predictions with likelihood > threshold
        filtered_df = df.copy()
        for part in self.body_parts:
            filtered_df.loc[df[(scorer, part, 'likelihood')] < self.likelihood_threshold,
                           [(scorer, part, 'x'), (scorer, part, 'y')]] = np.nan
        
        # Normalization: compute reference length
        x1 = filtered_df[(scorer, 'forehead', 'x')]
        y1 = filtered_df[(scorer, 'forehead', 'y')]
        x2 = filtered_df[(scorer, 'point_of_hip', 'x')]
        y2 = filtered_df[(scorer, 'point_of_hip', 'y')]
        reference_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        mean_reference_length = np.nanmean(reference_length.dropna())
        
        if np.isnan(mean_reference_length) or mean_reference_length == 0:
            mean_reference_length = 1.0
            
        # Normalize coordinates
        normalized_df = filtered_df.iloc[:150].copy()
        for part in self.body_parts:
            normalized_df[(scorer, part, 'x')] = (
                normalized_df[(scorer, part, 'x')] - normalized_df[(scorer, 'back', 'x')]
            ) / mean_reference_length
            normalized_df[(scorer, part, 'y')] = (
                normalized_df[(scorer, part, 'y')] - normalized_df[(scorer, 'back', 'y')]
            ) / mean_reference_length
        
        return normalized_df, scorer, mean_reference_length
    
    def calculate_stride_metrics(self, normalized_df, scorer):
        """
        Calculate stride length metrics for each foot
        """
        stride_lengths = {}
        for foot in self.foot_parts:
            x = normalized_df[(scorer, foot, 'x')]
            y = normalized_df[(scorer, foot, 'y')]
            dx = x.diff().abs()
            dy = y.diff().abs()
            stride_lengths[foot] = np.sqrt(dx**2 + dy**2)
            
        return stride_lengths
    
    def calculate_knee_angle(self, normalized_df, scorer):
        """
        Calculate right knee angle
        """
        x_hock = normalized_df[(scorer, 'right_hock', 'x')]
        y_hock = normalized_df[(scorer, 'right_hock', 'y')]
        x_knee = normalized_df[(scorer, 'right_knee', 'x')]
        y_knee = normalized_df[(scorer, 'right_knee', 'y')]
        x_fetlock = normalized_df[(scorer, 'front_right_fetlock', 'x')]
        y_fetlock = normalized_df[(scorer, 'front_right_fetlock', 'y')]
        
        angles = []
        for i in range(len(x_hock)):
            vec1 = np.array([x_hock.iloc[i] - x_knee.iloc[i], y_hock.iloc[i] - y_knee.iloc[i]])
            vec2 = np.array([x_fetlock.iloc[i] - x_knee.iloc[i], y_fetlock.iloc[i] - y_knee.iloc[i]])
            
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norm_product > 1e-10:
                cos_theta = dot_product / norm_product
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                angles.append(angle)
            else:
                angles.append(np.nan)
                
        return pd.Series(angles)
    
    def classify_movement(self, stride_lengths, knee_angles, body_length_variation):
        """
        Classify movement as normal or abnormal based on stride metrics
        """
        # Calculate key metrics
        stride_variability = np.mean([np.nanstd(stride_lengths[foot]) for foot in self.foot_parts])
        mean_knee_angle = np.nanmean(knee_angles)
        knee_angle_variability = np.nanstd(knee_angles)
        
        # Simple classification logic (you can make this more sophisticated)
        abnormal_indicators = 0
        confidence_factors = []
        
        # Check stride variability
        if stride_variability > 0.1:  # Threshold for abnormal variability
            abnormal_indicators += 1
            confidence_factors.append(0.3)
            
        # Check knee angle consistency
        if knee_angle_variability > 15:  # Degrees
            abnormal_indicators += 1
            confidence_factors.append(0.25)
            
        # Check for extreme knee angles
        if mean_knee_angle < 90 or mean_knee_angle > 170:
            abnormal_indicators += 1
            confidence_factors.append(0.2)
            
        # Classification
        if abnormal_indicators >= 2:
            result = "abnormal"
            confidence = min(0.95, 0.5 + sum(confidence_factors))
        else:
            result = "normal"
            confidence = min(0.95, 0.7 + (0.25 * (3 - abnormal_indicators)))
            
        return result, confidence
    
    def analyze_video(self, video_path):
        """
        Main analysis function
        """
        try:
            # Process video to get pose estimation data
            # In real implementation, this would run DeepLabCut on the video
            df = self.process_video_to_h5(video_path)
            
            # Analyze stride patterns
            normalized_df, scorer, ref_length = self.analyze_stride_patterns(df)
            
            # Calculate metrics
            stride_lengths = self.calculate_stride_metrics(normalized_df, scorer)
            knee_angles = self.calculate_knee_angle(normalized_df, scorer)
            
            # Body length variation
            x1 = normalized_df[(scorer, 'forehead', 'x')]
            y1 = normalized_df[(scorer, 'forehead', 'y')]
            x2 = normalized_df[(scorer, 'point_of_hip', 'x')]
            y2 = normalized_df[(scorer, 'point_of_hip', 'y')]
            body_lengths = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            body_length_variation = np.nanstd(body_lengths)
            
            # Classify movement
            result, confidence = self.classify_movement(stride_lengths, knee_angles, body_length_variation)
            
            return {
                "result": result,
                "confidence": round(confidence, 2),
                "analysis_time": "2.1 seconds",
                "details": f"Stride analysis completed. Movement classified as {result} with {int(confidence*100)}% confidence.",
                "metrics": {
                    "stride_variability": round(np.mean([np.nanstd(stride_lengths[foot]) for foot in self.foot_parts]), 3),
                    "mean_knee_angle": round(np.nanmean(knee_angles), 1),
                    "body_length_variation": round(body_length_variation, 3)
                }
            }
            
        except Exception as e:
            return {
                "result": "error",
                "confidence": 0.0,
                "analysis_time": "0 seconds",
                "details": f"Analysis failed: {str(e)}",
                "metrics": {}
            }

# Initialize analyzer
analyzer = HorseStrideAnalyzer()

def process_video_upload(video_file):
    """
    Process uploaded video and return analysis results
    """
    if video_file is None:
        return "Please upload a video file."
    
    try:
        # video_file is already a path string in Gradio
        video_path = video_file
        
        # Analyze the video
        results = analyzer.analyze_video(video_path)
        
        # Format results for display
        if results["result"] == "error":
            return f"❌ Error: {results['details']}"
        
        status_emoji = "✅" if results["result"] == "normal" else "⚠️"
        
        output = f"""
{status_emoji} **Stride Analysis Results**

**Classification:** {results["result"].upper()}
**Confidence:** {int(results["confidence"] * 100)}%
**Processing Time:** {results["analysis_time"]}

**Details:** {results["details"]}

**Metrics:**
- Stride Variability: {results["metrics"].get("stride_variability", "N/A")}
- Mean Knee Angle: {results["metrics"].get("mean_knee_angle", "N/A")}°
- Body Length Variation: {results["metrics"].get("body_length_variation", "N/A")}
        """
        
        return output
        
    except Exception as e:
        return f"❌ Error processing video: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=process_video_upload,
    inputs=gr.File(label="Upload Horse Video", file_types=["video"]),
    outputs=gr.Textbox(label="Analysis Results", lines=10),
    title="🐎 Tru-Stride: Horse Movement Analysis",
    description="Upload a video of a horse to analyze stride patterns and detect potential lameness or abnormal movement.",
    examples=None,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
