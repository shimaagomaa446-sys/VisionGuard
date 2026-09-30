def is_inside(ppe_box, person_box):
    """
    Check if the center of a PPE box is inside a person box.
    Uses Point-in-Box instead of IoU (better for small PPE objects).
    """
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    x1, y1, x2, y2 = person_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def check_compliance(results, model):
    persons = []
    hardhats = []
    no_hardhats = []
    safety_vests = []
    no_safety_vests = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        
        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])
        
        item = {'box': xyxy, 'conf': conf, 'track_id': track_id}
        
        if class_name == 'Person':
            persons.append(item)
        elif class_name == 'Hardhat':
            hardhats.append(item)
        elif class_name == 'NO-Hardhat':
            no_hardhats.append(item)
        elif class_name == 'Safety Vest':
            safety_vests.append(item)
        elif class_name == 'NO-Safety Vest':
            no_safety_vests.append(item)
    
    results_list = []
    for i, person in enumerate(persons):
        person_box = person['box']
        
        has_hardhat = any(is_inside(h['box'], person_box) for h in hardhats)
        has_no_hardhat = any(is_inside(nh['box'], person_box) for nh in no_hardhats)
        has_vest = any(is_inside(v['box'], person_box) for v in safety_vests)
        has_no_vest = any(is_inside(nv['box'], person_box) for nv in no_safety_vests)
        
        violations = []
        
        if has_no_hardhat:
            violations.append('Missing Helmet')
        elif not has_hardhat:
            violations.append('Missing Helmet')
        
        if has_no_vest:
            violations.append('Missing Vest')
        elif not has_vest:
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