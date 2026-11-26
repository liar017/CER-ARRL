# Rating and Emoji Dynamic Completion Experimental Platform

## Overview

This experimental platform provides datasets, baseline algorithms, model implementations, and experimental tools for research on rating and emoji completion.

## 1. Datasets

The `datasets/` directory contains five datasets in `.mat` format:

- BeerAdvocate
- HotelRec
- OpenRiceEmoji
- OpenRiceText
- Yelp

## 2. Baseline Algorithms

The `baselines/` directory contains four baseline implementations:

- AMCF
- BPER-J
- PITF
- SULM

The `datapreprocess.py` script converts the datasets into the input formats required by the baseline algorithms.

## 3. Model

The `model/` directory contains the MATLAB implementation of CARAR (Correlation-aware Review Aspect Recommender System), obtained from the [original repository](https://github.com/Joinn99/CARAR).

Before running CARAR, copy the required dataset files from `datasets/` into the `model/CARAR/Data/` directory.

## 4. Experimental Tools

The platform provides experimental utilities for:

- Data preprocessing and format conversion
- Data loading and splitting
- Model training and prediction
- Performance evaluation

These utilities are integrated into the corresponding algorithm implementations.

For detailed instructions, please refer to the README files in the corresponding directories.
