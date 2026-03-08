import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
from datetime import datetime, timedelta
import os
import html


# Path to ONNX model (relative to repo root for HF Spaces)
ONNX_PATH = "model/final_model.onnx"

# Load ONNX runtime session
ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

print("✅ ONNX model loaded")

# Class labels (VERY IMPORTANT: order must match training)
CLASS_NAMES = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

# Same preprocessing as training
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Maintenance recommendations database
MAINTENANCE_RECOMMENDATIONS = {
    'Bird-drop': {
        'severity': 'Medium',
        'severity_color': '🟡',
        'urgency': 'Schedule within 1-2 weeks',
        'impact': '5-15% efficiency loss',
        'actions': [
            'Clean affected panels with soft brush and water',
            'Install bird deterrents (spikes, netting, or reflective tape)',
            'Inspect for corrosion under droppings',
            'Apply protective coating if acid damage detected'
        ],
        'frequency': 'Inspect monthly in areas with high bird activity',
        'degradation_rate': 0.8
    },
    'Clean': {
        'severity': 'Low',
        'severity_color': '🟢',
        'urgency': 'Routine maintenance only',
        'impact': 'Optimal performance (0-2% below peak)',
        'actions': [
            'Continue regular monitoring schedule',
            'Quarterly visual inspections recommended',
            'Annual professional inspection',
            'Maintain vegetation clearance around panels'
        ],
        'frequency': 'Quarterly inspections',
        'degradation_rate': 0.5
    },
    'Dusty': {
        'severity': 'Medium',
        'severity_color': '🟡',
        'urgency': 'Schedule within 2-4 weeks',
        'impact': '10-25% efficiency loss depending on dust thickness',
        'actions': [
            'Clean panels with deionized water and soft microfiber cloth',
            'Consider automated cleaning system for frequent dust',
            'Apply anti-soiling nano-coating',
            'Schedule cleaning before monsoon/rain season'
        ],
        'frequency': 'Clean every 2-6 months (varies by location)',
        'degradation_rate': 1.2
    },
    'Electrical-damage': {
        'severity': 'High',
        'severity_color': '🔴',
        'urgency': 'URGENT - Address within 24-48 hours',
        'impact': '30-100% efficiency loss, fire/safety risk',
        'actions': [
            '⚠️ IMMEDIATELY disconnect affected panel circuit',
            'Call certified solar technician for inspection',
            'Check for loose connections, burnt wiring, or junction box damage',
            'Perform thermographic scan of entire array',
            'Replace damaged components (bypass diodes, connectors)',
            'Test electrical continuity and insulation resistance'
        ],
        'frequency': 'Emergency response, then quarterly electrical audits',
        'degradation_rate': 5.0
    },
    'Physical-Damage': {
        'severity': 'High',
        'severity_color': '🔴',
        'urgency': 'URGENT - Address within 1 week',
        'impact': '25-100% efficiency loss, water ingress risk',
        'actions': [
            'Assess crack severity (micro-cracks vs. major breaks)',
            'Seal minor cracks with UV-resistant clear sealant',
            'Replace severely damaged panels',
            'Check for moisture ingress in junction box',
            'Inspect mounting hardware and structural integrity',
            'Document damage for warranty/insurance claims'
        ],
        'frequency': 'Immediate repair, then bi-annual structural inspections',
        'degradation_rate': 3.5
    },
    'Snow-Covered': {
        'severity': 'Medium',
        'severity_color': '🟡',
        'urgency': 'Monitor and clear when safe',
        'impact': '80-100% temporary efficiency loss (recovers after melting)',
        'actions': [
            'Allow natural melting when possible (panels generate some heat)',
            'Use soft snow rake with non-abrasive head if necessary',
            '⚠️ NEVER use hot water (thermal shock can crack panels)',
            'Adjust panel tilt angle to 45°+ in snowy regions',
            'Install heating cables for persistent snow areas',
            'Clear bottom panels first to enable snow sliding'
        ],
        'frequency': 'As needed during winter months',
        'degradation_rate': 0.0
    }
}

def predict_degradation(defect_class, current_efficiency=100, time_horizon_months=12):
    """
    Predict solar panel efficiency degradation over time
    """
    maintenance = MAINTENANCE_RECOMMENDATIONS[defect_class]
    degradation_rate = maintenance['degradation_rate']
    
    timeline = []
    efficiency = current_efficiency
    
    if defect_class in ['Electrical-damage', 'Physical-Damage']:
        for week in range(0, min(time_horizon_months * 4, 52), 2):
            date = datetime.now() + timedelta(weeks=week)
            timeline.append({
                'date': date.strftime('%b %d, %Y'),
                'efficiency': max(0, efficiency),
                'status': '🔴 Critical' if efficiency < 50 else '🟡 Degraded'
            })
            efficiency -= degradation_rate
    else:
        for month in range(0, time_horizon_months + 1, 2):
            date = datetime.now() + timedelta(days=month * 30)
            timeline.append({
                'date': date.strftime('%b %d, %Y'),
                'efficiency': max(0, efficiency),
                'status': '🟢 Good' if efficiency > 85 else '🟡 Fair' if efficiency > 70 else '🔴 Poor'
            })
            efficiency -= degradation_rate
    
    return timeline

def format_maintenance_report(defect_class, confidence):
    """
    Generate comprehensive maintenance report
    """
    maint = MAINTENANCE_RECOMMENDATIONS[defect_class]
    
    actions_formatted = '\n'.join([f'**{i+1}.** {action}' for i, action in enumerate(maint['actions'])])
    
    report = f"""
<div style="background: linear-gradient(135deg, #525252 0%, #404040 100%); padding: 20px; border-radius: 10px; color: #fafafa; margin-bottom: 20px; border: 1px solid #404040;">
    <h2 style="margin: 0; font-size: 24px; color: #fafafa;">🔧 Maintenance Report</h2>
</div>

<div style="background: #171717; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #262626;">
    <h3 style="color: #fafafa; margin-top: 0;">📋 Detection Summary</h3>
    <p style="font-size: 16px; margin: 10px 0; color: #d4d4d4;"><strong>Condition Detected:</strong> <span style="color: #a3a3a3; font-size: 18px;">{defect_class}</span></p>
    <p style="font-size: 16px; margin: 10px 0; color: #d4d4d4;"><strong>Confidence Level:</strong> <span style="color: #a3a3a3; font-size: 18px;">{confidence*100:.1f}%</span></p>
    <p style="font-size: 16px; margin: 10px 0; color: #d4d4d4;"><strong>Severity:</strong> {maint['severity_color']} <span style="font-weight: bold;">{maint['severity']}</span></p>
</div>

<div style="background: #422006; padding: 15px; border-left: 4px solid #f59e0b; border-radius: 5px; margin-bottom: 15px;">
    <p style="margin: 0; font-size: 16px; color: #fef3c7;"><strong>⏰ Urgency:</strong> {maint['urgency']}</p>
</div>

<div style="background: #171717; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #262626;">
    <h3 style="color: #fafafa; margin-top: 0;">📊 Performance Impact</h3>
    <p style="font-size: 15px; line-height: 1.6; color: #d4d4d4;">{maint['impact']}</p>
</div>

<div style="background: #022c22; padding: 20px; border-left: 4px solid #10b981; border-radius: 5px; margin-bottom: 15px;">
    <h3 style="color: #d1fae5; margin-top: 0;">✅ Recommended Actions</h3>
    <div style="font-size: 15px; line-height: 1.8; color: #d1fae5;">
        {actions_formatted}
    </div>
</div>

<div style="background: #171717; padding: 15px; border-radius: 10px; border: 1px solid #262626;">
    <p style="margin: 5px 0; font-size: 15px; color: #d4d4d4;"><strong>📅 Maintenance Frequency:</strong> {maint['frequency']}</p>
    <p style="margin: 5px 0; font-size: 15px; color: #d4d4d4;"><strong>⚠️ Degradation Rate:</strong> {maint['degradation_rate']}% efficiency loss per {'week' if defect_class in ['Electrical-damage'] else 'month'} if untreated</p>
</div>
"""
    return report

def format_degradation_prediction(timeline):
    """
    Format degradation prediction timeline
    """
    table_rows = '\n'.join([
        f"<tr><td style='padding: 12px; border-bottom: 1px solid #262626; color: #d4d4d4;'>{entry['date']}</td>"
        f"<td style='padding: 12px; border-bottom: 1px solid #262626; font-weight: bold; color: #fafafa;'>{entry['efficiency']:.1f}%</td>"
        f"<td style='padding: 12px; border-bottom: 1px solid #262626; color: #d4d4d4;'>{entry['status']}</td></tr>"
        for entry in timeline
    ])
    
    report = f"""
<div style="background: linear-gradient(135deg, #525252 0%, #404040 100%); padding: 20px; border-radius: 10px; color: #fafafa; margin-bottom: 20px; border: 1px solid #404040;">
    <h2 style="margin: 0; font-size: 24px;">📉 Degradation Forecast</h2>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">Projected efficiency without maintenance intervention</p>
</div>

<div style="background: #171717; padding: 20px; border-radius: 10px; border: 1px solid #262626; margin-bottom: 20px;">
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background: #0a0a0a;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #404040; color: #fafafa;">Date</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #404040; color: #fafafa;">Efficiency</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #404040; color: #fafafa;">Status</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</div>

<div style="background: #022c22; padding: 20px; border-left: 4px solid #10b981; border-radius: 5px; margin-bottom: 15px;">
    <h3 style="color: #d1fae5; margin-top: 0;">💡 Prevention Strategies</h3>
    <ul style="color: #d1fae5; line-height: 1.8; font-size: 15px;">
        <li><strong>Regular Monitoring:</strong> Track daily energy output to detect issues early</li>
        <li><strong>Scheduled Maintenance:</strong> Follow recommended cleaning and inspection schedules</li>
        <li><strong>Professional Audits:</strong> Annual thermographic scans detect hidden problems</li>
        <li><strong>Protective Measures:</strong> Install bird deterrents, anti-soiling coatings, and proper drainage</li>
        <li><strong>Documentation:</strong> Keep maintenance records for warranty compliance</li>
    </ul>
</div>

<div style="background: #422006; padding: 20px; border-radius: 10px; border-left: 4px solid #f59e0b;">
    <h3 style="color: #fef3c7; margin-top: 0;">⚡ Performance Optimization Tips</h3>
    <ul style="color: #fef3c7; line-height: 1.8; font-size: 15px;">
        <li>Clean panels during early morning or late evening (avoid thermal shock)</li>
        <li>Trim nearby vegetation to prevent shading and debris accumulation</li>
        <li>Inspect wiring and connections for corrosion every 6 months</li>
        <li>Keep inverter and electrical components clean and ventilated</li>
        <li>Consider microinverters for better partial-shading performance</li>
    </ul>
</div>
"""
    return report

def get_gradcam_heatmap(pil_image, pred_idx):
    img = preprocess(pil_image).unsqueeze(0).numpy().astype(np.float32)
    outputs = ort_session.run(None, {"input_image": img})[0]
    
    img_array = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    heatmap = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def create_heatmap_overlay(pil_image, heatmap):
    img_array = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    return Image.fromarray(overlay.astype(np.uint8))

def predict_image(pil_image):
    img = preprocess(pil_image).unsqueeze(0).numpy().astype(np.float32)
    outputs = ort_session.run(None, {"input_image": img})[0]
    exp_scores = np.exp(outputs)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    probs = probs[0]
    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, prob_dict, pred_idx

def extract_frame_from_video(video_path):
    """
    Extract the middle frame from a video file as a PIL Image.
    Used as a fallback when the user uploads a video instead of an image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read a frame from the uploaded video.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

print("✅ Inference pipeline ready")

def gradio_predict(image, video, current_efficiency, time_horizon):
    # Video fallback: if no image provided but a video is, extract its middle frame
    if image is None and video is not None:
        try:
            image = extract_frame_from_video(video)
        except Exception as e:
            error_msg = f"<p style='color:#f87171;'>⚠️ Could not process video: {html.escape(str(e))}</p>"
            return None, None, None, None, error_msg, ""

    if image is None:
        return None, None, None, None, "<p style='color:#f87171;'>⚠️ Please upload an image or video to analyze.</p>", ""
    
    pred_class, confidence, prob_dict, pred_idx = predict_image(image)
    heatmap = get_gradcam_heatmap(image, pred_idx)
    heatmap_overlay = create_heatmap_overlay(image, heatmap)
    
    maintenance_report = format_maintenance_report(pred_class, confidence)
    timeline = predict_degradation(pred_class, current_efficiency, int(time_horizon))
    degradation_report = format_degradation_prediction(timeline)

    return pred_class, f"{confidence * 100:.2f}%", prob_dict, heatmap_overlay, maintenance_report, degradation_report

# Custom CSS for dark neutral theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #525252 0%, #404040 100%);
    padding: 40px;
    border-radius: 15px;
    color: #fafafa;
    margin-bottom: 30px;
    border: 1px solid #404040;
}

.upload-section {
    background: #171717;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #262626;
}

.dark {
    background-color: #0a0a0a !important;
}

/* Override Gradio's default backgrounds */
.gr-box, .gr-form, .gr-panel {
    background-color: #171717 !important;
    border-color: #262626 !important;
}

.gr-input, .gr-text-input {
    background-color: #0a0a0a !important;
    border-color: #404040 !important;
    color: #fafafa !important;
}

.gr-button {
    background: linear-gradient(135deg, #525252 0%, #404040 100%) !important;
    color: #fafafa !important;
    border: 1px solid #404040 !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, #737373 0%, #525252 100%) !important;
}

label {
    color: #d4d4d4 !important;
}

.gr-prose {
    color: #d4d4d4 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="neutral", secondary_hue="neutral")) as iface:
    gr.HTML("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 42px; font-weight: 700; color: #f5f5f5;">☀️ Solar Panel AI Diagnostics</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px; opacity: 0.95; color: #a3a3a3;">Intelligent defect detection, maintenance planning & performance forecasting</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="upload-section">')
            input_image = gr.Image(
                type="pil", 
                label="📸 Upload Solar Panel Image",
                height=300
            )
            input_video = gr.Video(
                label="🎬 Or Upload a Video (fallback — middle frame will be used)",
                height=300
            )
            gr.HTML('</div>')
            
            with gr.Row():
                current_eff = gr.Slider(
                    minimum=50, 
                    maximum=100, 
                    value=95, 
                    step=1, 
                    label="⚡ Current System Efficiency (%)",
                    info="Set your panel's current performance level"
                )
            
            with gr.Row():
                time_horiz = gr.Slider(
                    minimum=3, 
                    maximum=24, 
                    value=12, 
                    step=3,
                    label="📅 Forecast Period (months)",
                    info="Choose prediction time horizon"
                )
            
            predict_btn = gr.Button(
                "🔍 Analyze Solar Panel", 
                variant="primary", 
                size="lg",
                scale=1
            )
        
        with gr.Column(scale=1):
            with gr.Group():
                pred_class = gr.Textbox(
                    label="🎯 Detected Condition",
                    interactive=False,
                    container=True
                )
                confidence = gr.Textbox(
                    label="📊 Confidence Score",
                    interactive=False,
                    container=True
                )
            
            prob_dist = gr.Label(
                label="📈 Classification Probabilities",
                num_top_classes=6
            )
            
            heatmap_img = gr.Image(
                type="pil", 
                label="🔥 AI Attention Heatmap",
                height=300
            )
    
    with gr.Row():
        with gr.Column():
            maintenance_output = gr.HTML(label="Maintenance Report")
    
    with gr.Row():
        with gr.Column():
            degradation_output = gr.HTML(label="Degradation Forecast")
    
    predict_btn.click(
        fn=gradio_predict,
        inputs=[input_image, input_video, current_eff, time_horiz],
        outputs=[pred_class, confidence, prob_dist, heatmap_img, 
                maintenance_output, degradation_output]
    )
    
    gr.HTML("""
        <div style="background: #171717; padding: 30px; border-radius: 12px; margin-top: 30px; border: 1px solid #262626;">
            <h3 style="color: #fafafa; margin-top: 0;">📖 How to Use This System</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="background: #0a0a0a; padding: 20px; border-radius: 8px; border: 1px solid #262626;">
                    <h4 style="color: #a3a3a3; margin-top: 0;">1️⃣ Upload Image or Video</h4>
                    <p style="color: #737373; font-size: 14px; line-height: 1.6;">Upload a photo of your solar panel (thermal, infrared, or RGB). You can also upload a short video — the middle frame will be extracted and analyzed automatically.</p>
                </div>
                <div style="background: #0a0a0a; padding: 20px; border-radius: 8px; border: 1px solid #262626;">
                    <h4 style="color: #a3a3a3; margin-top: 0;">2️⃣ Set Parameters</h4>
                    <p style="color: #737373; font-size: 14px; line-height: 1.6;">Adjust current efficiency and forecast timeframe</p>
                </div>
                <div style="background: #0a0a0a; padding: 20px; border-radius: 8px; border: 1px solid #262626;">
                    <h4 style="color: #a3a3a3; margin-top: 0;">3️⃣ Analyze</h4>
                    <p style="color: #737373; font-size: 14px; line-height: 1.6;">Click analyze to get comprehensive diagnostics</p>
                </div>
                <div style="background: #0a0a0a; padding: 20px; border-radius: 8px; border: 1px solid #262626;">
                    <h4 style="color: #a3a3a3; margin-top: 0;">4️⃣ Review & Act</h4>
                    <p style="color: #737373; font-size: 14px; line-height: 1.6;">Check maintenance actions and follow recommendations</p>
                </div>
            </div>
            
            <div style="margin-top: 25px; padding: 20px; background: #0a0a0a; border-radius: 8px; border: 1px solid #262626;">
                <h4 style="color: #fafafa; margin-top: 0;">⚡ Detection Capabilities</h4>
                <p style="color: #a3a3a3; font-size: 14px; line-height: 1.8;">
                    This AI system detects <strong style="color: #d4d4d4;">6 types of solar panel conditions</strong>: Bird droppings, Clean panels, 
                    Dust accumulation, Electrical damage, Physical damage, and Snow coverage. The attention heatmap 
                    visualizes which areas influenced the AI's decision-making process.
                </p>
            </div>
        </div>
    """)

if __name__ == "__main__":
    iface.launch()
