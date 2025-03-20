# Geographic Domain Adaptation (GDA)
**Official implementation for "Learning Region-specific Features and Matching Distributions Across Regions in Geographic Domain Adaptation"**

*Authors: Takashi Horihata, Soh Yoshida, Mitsuji Muneyasu*  
*Kansai University, Osaka, Japan*

> **Note**: This repository accompanies the paper currently under review.

---

## Overview

This repository provides the official implementation of our Geographic Domain Adaptation (GDA) method introduced in the paper:

**"Learning Region-specific Features and Matching Distributions Across Regions in Geographic Domain Adaptation"**

Our method explicitly addresses geographic domain adaptation by modeling and adapting to regional variations such as architectural styles, backgrounds, and context-specific object appearances. The method significantly improves model accuracy across geographic domains, such as USA ↔ Asia transfers.

## Highlights

- **Region-specific feature learning**: Captures and leverages region-specific visual characteristics.
- **Cross-region distribution matching**: Aligns domain distributions selectively, preserving discriminative information.
- **Superclass-level adaptation**: Utilizes semantic class hierarchies to enhance adaptation performance.

## Results

The proposed method significantly outperforms state-of-the-art domain adaptation methods on GeoNet:

| Dataset   | Task           | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-----------|----------------|--------------------|--------------------|
| GeoImNet  | USA → Asia     | **50.23**          | **75.39**          |
| GeoImNet  | Asia → USA     | **58.76**          | **81.54**          |
| GeoPlaces | USA → Asia     | **44.06**          | **74.92**          |
| GeoPlaces | Asia → USA     | **41.94**          | **70.46**          |

## Superclasses
For detailed superclass definitions, see [superclass list](https://github.com/meruemon/GDA/blob/main/CLASSES.md)
