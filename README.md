# PDing Video Recommendation System 

This is the Hybrid Recommendation System for PDing platform

## Overview

This project aims to  recommend related videos for PDing' users using Collaborative Filtering and Content-Based approaches.

## Recommendation Approaches

### 1. Collaborative Filtering
- Uses historical user-video interactions (currently using **rating score**)
- Matrix-based approach with similarity metrics like **cosine similarity** or **alternating least squares (ALS)**

### 2. Content-Based Filtering
- Uses video metadata such as:
  - title
  - description
  - trees_consumed
  - video_duration
  - purchase_tier
  - pd_category
- Computes video-video similarity using **TF-IDF**

### 3. Hybrid Model
- Combines collaborative top-N results with content-similar expansion
- Handles cold-start users and items

## Features
- **Collaborative Filtering** using user-video interactions
- **Content-Based Filtering** using video attributes 
- Hybrid logic for improved cold-start handling
- Caching layer **(Redis + Memory cache)** for I/O bound task optimization
- Configurable pipeline and scoring
- REST API via FastAPI for real-time recommendations
- Unit tests and reusable components

## Installation
### Prerequisite: 
* Python 3.10.12
* Docker
* docker-compose
* git
* virtualenv
* CUDA (change the package faiss-gpu to faiss cpu if CUDA bot compatible)


1. Create production environment
```
# Create env
virtual env
source venv/bin/activate

# Install libraries
pip install -r requirements.txt
```

2. Prepare data for model training
```
bash data_preparing.sh
```

3. Model training
```
bash model_training.sh
```

4. Project serving and API hosting
```
docker-compose up -d --build 
```