from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import sys
sys.path.append("..")
from src.compliance import check_compliance
import base64
import cv2
import tempfile
import os
from collections import defaultdict

app = FastAPI()
model = YOLO("models/best.pt")


# ============================================
# Endpoint للصور
# ============================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    
    # NMS أقوى لتقليل الكشوفات المكررة
    results = model(image, conf=0.25, iou=0.4)
    
    annotated = results[0].plot()
    img = Image.fromarray(annotated)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    compliance = check_compliance(results, model)
    
    total = len(compliance)
    compliant = sum(1 for c in compliance if c['status'] == 'COMPLIANT')
    non_compliant = total - compliant
    
    return {
        'annotated_image': img_base64,
        'results': compliance,
        'summary': {
            'total_persons': total,
            'compliant': compliant,
            'non_compliant': non_compliant
        }
    }


# ============================================
# Endpoint للفيديو (مع Tracking)
# ============================================
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # 1. احفظي الفيديو مؤقتاً
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
        temp_in.write(await file.read())
        input_path = temp_in.name
    
    # 2. افتحيه
    cap = cv2.VideoCapture(input_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. إعدادات
    SAMPLING_RATE = 5
    
    # تتبع كل شخص
    person_data = defaultdict(lambda: {
        'compliant_frames': 0,
        'non_compliant_frames': 0,
        'violations': set(),
        'confidences': [],
        'best_frame': None,
        'best_conf': 0
    })
    
    total_processed = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % SAMPLING_RATE == 0:
            # NMS أقوى + Tracking
            results = model.track(frame, persist=True, conf=0.25, iou=0.4, verbose=False)
            compliance = check_compliance(results, model)
            
            annotated = results[0].plot()
            h, w = annotated.shape[:2]
            
            for c in compliance:
                track_id = c.get('track_id')
                
                if track_id is None:
                    track_id = f"frame_{frame_count}_person_{c['person_id']}"
                
                conf = c['conf']
                person_data[track_id]['confidences'].append(conf)
                
                # نحفظ أحسن frame
                if conf > person_data[track_id]['best_conf']:
                    x1, y1, x2, y2 = [int(v) for v in c['box']]
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        person_crop = annotated[y1:y2, x1:x2]
                        
                        if person_crop.size > 0:
                            try:
                                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                                img_pil = Image.fromarray(person_rgb)
                                buffer = io.BytesIO()
                                img_pil.save(buffer, format='JPEG', quality=85)
                                frame_base64 = base64.b64encode(buffer.getvalue()).decode()
                                
                                person_data[track_id]['best_frame'] = frame_base64
                                person_data[track_id]['best_conf'] = conf
                            except Exception:
                                pass
                
                if c['status'] == 'COMPLIANT':
                    person_data[track_id]['compliant_frames'] += 1
                else:
                    person_data[track_id]['non_compliant_frames'] += 1
                    for v in c.get('violations', []):
                        person_data[track_id]['violations'].add(v)
            
            total_processed += 1
    
    cap.release()
    
    # 4. احسبي القرار النهائي لكل شخص
    total_unique_persons = len(person_data)
    total_compliant = 0
    total_non_compliant = 0
    persons_summary = []
    
    for track_id, data in person_data.items():
        total = data['compliant_frames'] + data['non_compliant_frames']
        
        if data['compliant_frames'] > data['non_compliant_frames']:
            status = 'COMPLIANT'
            violations = []
            total_compliant += 1
        else:
            status = 'NON-COMPLIANT'
            violations = list(data['violations'])
            total_non_compliant += 1
        
        avg_conf = sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        
        persons_summary.append({
            'track_id': str(track_id),
            'status': status,
            'violations': violations,
            'appearances': total,
            'compliant_frames': data['compliant_frames'],
            'non_compliant_frames': data['non_compliant_frames'],
            'avg_confidence': round(avg_conf, 3),
            'image': data['best_frame']
        })
    
    persons_summary.sort(key=lambda x: x['track_id'])
    
    # 5. امسحي الملف المؤقت
    os.unlink(input_path)
    
    # 6. ارجّعي
    return {
        'persons': persons_summary,
        'summary': {
            'total_frames': total_frames_in_video,
            'processed_frames': total_processed,
            'unique_persons': total_unique_persons,
            'compliant': total_compliant,
            'non_compliant': total_non_compliant,
            'fps': fps
        }
    }