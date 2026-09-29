# Compliance Logic — VisionGuard

## Overview

The Compliance Logic module converts raw YOLO detections into 
product-level compliance decisions.

## Methodology

### Input
- YOLO detections (Person, Hardhat, Safety Vest)

### Logic
1. Group detections by class
2. For each Person:
   - Check if a Hardhat is inside the Person box
   - Check if a Safety Vest is inside the Person box
3. Determine violations

### Association Method: Point-in-Box

**Why not IoU?**
IoU fails for small PPE objects. A Person box (300×500) and 
Hardhat box (40×30) have IoU < 0.05 even when correctly placed.

**Point-in-Box:**
Check if the center of the PPE box is inside the Person box.

## Rules

| Condition | Decision |
|-----------|----------|
| Person + Hardhat inside | ✅ Helmet OK |
| Person + No Hardhat | ❌ Missing Helmet |
| Person + Safety Vest inside | ✅ Vest OK |
| Person + No Safety Vest | ❌ Missing Vest |

## Output Format

```json
{
  "person_id": 1,
  "box": [x1, y1, x2, y2],
  "conf": 0.86,
  "has_hardhat": true,
  "has_vest": false,
  "violations": ["Missing Vest"],
  "status": "NON-COMPLIANT"
}