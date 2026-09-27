# Error Analysis — VisionGuard

**Project:** AI Safety Monitoring System for Construction Sites
**Model:** YOLOv8s (Experiment 3)
**Dataset:** Construction Site Safety Image Dataset
**Test Set:** 82 images, 476 instances

---

## Executive Summary

We analyzed 10 failure cases from the test set to understand the model's limitations. The cases were selected to cover diverse error types and challenging scenarios.

### Error Type Distribution

| Error Type | Count |
|------------|------:|
| False Positives (FP) | 5 |
| False Negatives (FN) | 7 |
| Misclassifications | 2 |
| **Cases with multiple errors** | 5 |

### Root Cause Distribution

| Cause | Count |
|-------|------:|
| Crowded scenes | 3 |
| Small objects | 3 |
| Unusual context | 3 |
| Challenging lighting/background | 2 |
| Out-of-domain images | 1 |

---

## Summary Table

| # | Image | Error Type | Primary Cause |
|---|-------|------------|---------------|
| 1 | truck_scene.jpg | FP + FN | Small person on truck (unusual context) |
| 2 | loader_product.jpg | FP | Product image (white background) |
| 3 | tunnel_worker.jpg | FN (Hardhat) | Small object + tunnel lighting |
| 4 | crowd_5_workers.jpg | FP + FN + MC | Crowd (5 workers) |
| 5 | industrial_metal.jpg | FN (Hardhat) | Industrial metal scene |
| 6 | crowd_15_workers.jpg | FP + FN + MC | Extreme crowding (15 workers) |
| 7 | crowd_10_workers.jpg | FN (Person) | NMS issue in dense crowd |
| 8 | building_edge.jpg | FP + FN | Tiny objects at image edge |
| 9 | bookstore.jpg | FP (×2) | Out-of-domain image |
| 10 | mountain_scene.jpg | FP + FN | Backlighting + mountain background |

---

## Case 1: Person on Truck (FP + FN)

**Image:** `test_001_truck.jpg`

**Ground Truth:**
- Person: 1 (small, on truck)

**Prediction:**
- Person: 2 (both at wrong locations)

**Error Type:**
- False Positive: 2 persons detected instead of 1
- False Negative: actual person location missed

**Cause:**
- Object is very small (~5% of image)
- Unusual context: person on top of a truck
- High visual contrast (orange truck)
- Low confidence predictions (0.59 and 0.27)

**Impact:**
- System reports 2 people instead of 1
- Wrong locations → could confuse safety officer

**Screenshot:** `case_01_truck.png`

**Analysis:**
The model failed to detect the small person on the truck. Instead, it produced two low-confidence false positives in different locations (windshield area). This suggests the model struggles with small objects in unusual contexts.

---

## Case 2: False Positive on Construction Machinery (FP)

**Image:** `test_002_loader.jpg`

**Ground Truth:**
- No Person (empty)

**Prediction:**
- Person: 1 (confidence 0.32, on loader cabin)

**Error Type:**
- False Positive

**Cause:**
- Unusual context: product photo with white background
- Window/glass area of loader resembles a person
- Low confidence (0.32) but above threshold (0.25)

**Impact:**
- System reports a person where none exists
- False alarm → reduces user trust

**Screenshot:** `case_02_loader.png`

**Analysis:**
The model mistakenly detected a "Person" in the cabin area of the blue wheel loader. The dark window region with yellow frame resembles a construction worker wearing a hardhat. The 0.32 confidence indicates uncertainty, but our threshold (0.25) allows this false positive through.

---

## Case 3: Missed Hardhat in Tunnel Scene (FN)

**Image:** `test_003_tunnel.jpg`

**Ground Truth:**
- Person: 1
- Hardhat: 1
- Safety Vest: 1

**Prediction:**
- Person: ✅
- Safety Vest: ✅
- Hardhat: ❌ NOT DETECTED

**Error Type:**
- False Negative (Hardhat)

**Cause:**
1. Very small object (Hardhat is ~1.7% × 3.3% of image)
2. Difficult lighting: tunnel with uneven illumination
3. Low contrast: yellow hardhat on yellow CAT machinery
4. Occlusion: helmet partially obscured by head position

**Impact:**
- **Critical:** In Compliance Logic, this triggers a FALSE ALARM
- Person detected without Hardhat → "Missing Helmet" warning
- But the worker IS wearing a hardhat

**Screenshot:** `case_03_tunnel.png`

**Analysis:**
The model failed to detect a small hardhat in a challenging environment. The yellow hardhat blends with the yellow construction machinery, and the tunnel lighting creates shadows that obscure the object. Despite correctly detecting Person and Safety Vest, the missed Hardhat creates a cascading error in the compliance pipeline.

---

## Case 4: Multiple Errors in Crowd Scene (FP + FN + MC)

**Image:** `test_004_crowd5.jpg`

**Ground Truth:**
- Person: 5
- Hardhat: 5
- NO-Hardhat: 1
- Safety Vest: 1
- NO-Safety Vest: 5

**Prediction:**
- Person: 4 (❌ 1 missed)
- Hardhat: 2 (❌ 3 missed)
- NO-Hardhat: 0 (❌ 1 missed)
- Safety Vest: 1 (✅)
- NO-Safety Vest: 6 (❌ 1 extra)

**Error Type:**
- Multiple False Negatives
- Multiple False Positives
- Potential Misclassification

**Cause:**
- Crowded scene (5 workers in close proximity)
- Overlapping bounding boxes
- Small hardhats in low-light conditions
- Yellow hardhats blend with each other
- Model confusion between Hardhat and NO-Hardhat classes

**Impact:**
- In Compliance Logic:
  - Missing Hardhat detections → false "Missing Helmet" alarms
  - Extra NO-Safety Vest → false "Missing Vest" alarms
- Multiple errors compound in a single frame

**Screenshot:** `case_04_crowd5.png`

**Analysis:**
This case combines multiple error types. The model's performance degrades significantly in crowded scenes where workers overlap. The 3 missed Hardhats and 1 missed Person suggest the model struggles with overlapping objects of similar appearance.

---

## Case 5: Missed Hardhat in Industrial Scene (FN)

**Image:** `test_005_industrial.jpg`

**Ground Truth:**
- Person: 1
- Hardhat: 1
- NO-Safety Vest: 1

**Prediction:**
- Person: 1 (0.66) ✅
- NO-Safety Vest: 1 (0.55) ✅
- Hardhat: 0 ❌

**Error Type:**
- False Negative (Hardhat)

**Cause:**
1. Small object: Hardhat is only 4% × 3.4% of image
2. Unusual context: industrial scene with large metal structures
3. Difficult lighting: strong shadows + reflections on metal
4. Partial occlusion: Hardhat might be partially hidden

**Impact:**
- Compliance Logic reports "Missing Helmet" (false alarm)
- Combined with NO-Safety Vest detection:
  - System says: "Missing Helmet + Missing Vest"
  - Reality: "Missing Vest" only

**Screenshot:** `case_05_industrial.png`

**Analysis:**
The model correctly detected Person and NO-Safety Vest but missed the small Hardhat. This is likely due to the small object size in a complex scene with unusual background (industrial metal structures).

---

## Case 6: Severe Errors in Very Crowded Scene (FP + FN + MC)

**Image:** `test_006_crowd15.jpg`

**Ground Truth:**
- Person: 15
- Hardhat: 12
- NO-Safety Vest: 8
- Safety Vest: 1
- Total: 36

**Prediction:**
- Person: ~6 (❌ 9 missed)
- Hardhat: ~9 (❌ 3 missed)
- NO-Safety Vest: ~3 (❌ 5 missed)
- Safety Vest: 0 or 1 (❌ uncertain)

**Error Type:**
- Multiple False Negatives
- Some False Positives
- Some Misclassifications

**Cause:**
1. Extreme crowding: 15 workers in close proximity
2. Overlapping bounding boxes heavily
3. Small object size: Each person occupies <5% of image
4. Motion blur: image from video (MOV-12)
5. Model capacity: YOLOv8s struggles with this many objects

**Impact:**
- In Compliance Logic:
  - Many workers not detected → no checks
  - Some detected wrongly → false alarms
  - Very unreliable output for this frame

**Screenshot:** `case_06_crowd15.png`

**Analysis:**
This case demonstrates the model's limitations with dense crowds (15+ objects), small objects at 640 resolution, and video frames with motion blur. The error rate is severe with more than half of the persons missed.

---

## Case 7: Person Detection Failure in Crowd (FN)

**Image:** `test_007_crowd10.jpg`

**Ground Truth:**
- Person: 10
- Hardhat: 9
- NO-Safety Vest: 11
- Safety Vest: 0
- Total: 30

**Prediction:**
- Person: 3 (❌ 7 missed)
- Hardhat: 7 (❌ 2 missed)
- NO-Safety Vest: 9 (❌ 2 missed)
- Safety Vest: 0 (✅ correct)

**Error Type:**
- Multiple False Negatives (especially Person)
- PPE detected correctly but not associated with persons

**Cause:**
1. Extreme crowding: 10 workers standing very close
2. Overlapping boxes: NMS (Non-Maximum Suppression) likely removed valid Person detections
3. Partial occlusion: workers blocking each other
4. Similar clothing: workers in similar outfits

**Impact:**
- **CRITICAL:** In Compliance Logic:
  - Only 3 persons detected → 7 workers NOT CHECKED
  - PPE detected without Person → orphan detections
  - System would miss violations for 7 workers

**Screenshot:** `case_07_crowd10.png`

**Analysis:**
Interesting case where PPE (Hardhat, NO-Safety Vest) are detected correctly but Person detections are severely underrepresented. This suggests YOLO's NMS is too aggressive for crowded scenes, filtering out valid Person detections that overlap.

---

## Case 8: Tiny Objects at Image Edge (FP + FN)

**Image:** `test_008_building.jpg`

**Ground Truth:**
- Person: 2 (very small, at left edge)
- Hardhat: 2 (extremely small, <1.5% of image)
- NO-Safety Vest: 2

**Prediction:**
- Person: 2 (one at wrong location - right side)
- Hardhat: 2 (low confidence 0.44, 0.49)
- NO-Safety Vest: 2 (low confidence 0.38, 0.60)

**Error Type:**
- False Positive: Person detected on the right (no person there)
- Small object challenges

**Cause:**
1. Extremely small objects: Hardhats <1.5% of image
2. Edge of frame: Persons at image borders
3. Background confusion: Cars on the right detected as Person
4. Low confidence: All predictions <0.7

**Impact:**
- In Compliance Logic:
  - Person detected on right (car) → false alarm
  - Small persons on left may be missed in some frames

**Screenshot:** `case_08_building.png`

**Analysis:**
The model struggles with very small objects (<2% of image). Interestingly, it produces a False Positive on a car that looks similar to a person at this scale. The actual persons at the left edge are correctly detected but with low confidence.

---

## Case 9: False Positives on Out-of-Domain Image (FP ×2)

**Image:** `test_009_bookstore.jpg`

**Ground Truth:**
- Empty (no objects)
- This is an out-of-domain image (person in a bookstore)

**Prediction:**
- Person 0.27 (on chest area)
- NO-Safety Vest 0.38 (on head area)

**Error Type:**
- False Positive (2 detections)
- Both predictions are low confidence but above threshold (0.25)

**Cause:**
1. Out-of-domain image: Bookstore scene, not construction site
2. Wrong clothing: Star Trek uniform in gray resembles safety gear
3. Low confidence: Model is uncertain (0.27 and 0.38)
4. Threshold: With threshold=0.25, these pass through

**Impact:**
- In Compliance Logic:
  - Person detected → triggers checks
  - NO-Safety Vest detected → false "Missing Vest" alarm
- User sees false alarm for a non-construction scene

**Screenshot:** `case_09_bookstore.png`

**Analysis:**
This case highlights two key issues: (1) dataset quality — out-of-domain images in training/test data, and (2) threshold choice — 0.25 may be too low for some scenarios. This image was previously identified as an empty label file during the Data Audit phase.

---

## Case 10: Multiple Errors in Mountain Construction Scene (FP + FN)

**Image:** `test_010_mountain.jpg`

**Ground Truth:**
- Person: 3
- Hardhat: 3
- NO-Safety Vest: 3

**Prediction:**
- Person: 5 (❌ 2 extra False Positives)
- Hardhat: 2 (❌ 1 missed)
- NO-Safety Vest: 3 (✅)

**Error Type:**
- False Positive: 2 extra Person detections
- False Negative: 1 Hardhat missed

**Cause:**
1. Complex background: Large mountain cliff behind workers
2. Backlighting: Strong sunlight from behind
3. Scale variation: Workers at different distances
4. Potential FPs: Model may be confusing shadows/rock edges with workers

**Impact:**
- In Compliance Logic:
  - Over-counting persons → incorrect worker count
  - Missing Hardhat → false "Missing Helmet" alarm
  - System reports wrong numbers

**Screenshot:** `case_10_mountain.png`

**Analysis:**
This case demonstrates multiple challenges: backlighting makes workers appear as silhouettes, the mountain background creates visual noise, and the worker in the background is partially occluded. The model produces 2 extra Person detections while missing 1 hardhat.

---

## Key Findings

### 1. Most Common Error: False Negatives
- 7 out of 10 cases involve False Negatives
- **Hardhat** is the most missed class
- Small objects (<3% of image) are systematically missed

### 2. Crowded Scenes are Challenging
- 3 cases involve crowds of 5+ workers
- NMS (Non-Maximum Suppression) removes valid detections
- Person count is often underreported in crowds

### 3. Out-of-Domain Images Cause FPs
- Book images, product photos cause false positives
- Dataset contains non-construction images
- Domain classifier would help filter these

### 4. Threshold Trade-off
- Threshold=0.25 allows low-confidence FPs
- Threshold=0.4 would filter most FPs but miss real detections
- Current choice balances both concerns

### 5. Class Imbalance Affects Performance
- NO-Hardhat is the weakest class
- Person is overrepresented in training
- Hardhat detection suffers from imbalance

---

## Recommendations

### Short-term (for this project):
1. **Document limitations** in the final report
2. **Consider raising threshold** to 0.4 for production
3. **Implement consistency checks** in Compliance Logic:
   - If Person has Hardhat nearby, don't flag missing
   - Require minimum confidence for violation alarms
4. **Add domain validation** before detection

### Long-term (future work):
1. **Train at higher resolution** (imgsz=800 or 1024)
2. **Use larger model** (YOLOv8m or YOLOv8l)
3. **Data augmentation** for crowded scenes and backlighting
4. **Test-time augmentation** (TTA)
5. **Hard negative mining** for FP reduction
6. **More PPE data** for NO-Hardhat class

---

## Impact on Compliance Logic

Based on the error analysis, Compliance Logic must handle:

1. **Missing detections (FN):** Don't assume "missing = violation"
2. **Low confidence:** Use confidence thresholds before alarming
3. **Orphan PPE:** Handle PPE without associated Person
4. **Crowded scenes:** May need manual review flag
5. **Out-of-domain:** Reject non-construction images early

---

## Conclusion

The error analysis reveals that while the model achieves strong overall performance (mAP50 = 0.811), it has significant limitations in:
- **Crowded scenes** (systematic NMS issues)
- **Small objects** (hardhats at distance)
- **Unusual contexts** (industrial, non-construction)
- **Challenging lighting** (tunnels, backlighting)

These limitations must be communicated to users and considered in the compliance logic design.

---

**End of Error Analysis**