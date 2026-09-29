def is_inside(ppe_box, person_box):
    """تحديد إذا كان مركز الـ PPE داخل الـ Person box"""
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    x1, y1, x2, y2 = person_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def check_compliance(results, model):
    """
    تحليل نتائج YOLO وتحديد حالة كل شخص.
    
    Args:
        results: نتائج YOLO
        model: الموديل (للأسماء)
    
    Returns:
        list: قايمة بالشخاص وحالتهم
    """
    persons = []
    hardhats = []
    safety_vests = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        
        # الحصول على track_id
        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])
        
        item = {'box': xyxy, 'conf': conf, 'track_id': track_id}
        
        if class_name == 'Person':
            persons.append(item)
        elif class_name == 'Hardhat':
            hardhats.append(item)
        elif class_name == 'Safety Vest':
            safety_vests.append(item)
    
    results_list = []
    for i, person in enumerate(persons):
        person_box = person['box']
        
        # البحث عن Hardhat
        has_hardhat = False
        for h in hardhats:
            if is_inside(h['box'], person_box):
                has_hardhat = True
                break
        
        # البحث عن Safety Vest
        has_vest = False
        for v in safety_vests:
            if is_inside(v['box'], person_box):
                has_vest = True
                break
        
        # تحديد المخالفات
        violations = []
        if not has_hardhat:
            violations.append('Missing Helmet')
        if not has_vest:
            violations.append('Missing Vest')
        
        results_list.append({
            'person_id': i + 1,
            'track_id': person.get('track_id'),
            'box': person_box,
            'conf': person['conf'],
            'has_hardhat': has_hardhat,
            'has_vest': has_vest,
            'violations': violations,
            'status': 'NON-COMPLIANT' if violations else 'COMPLIANT'
        })
    
    return results_list