#!/bin/bash
dataset_dict = {
    "case01_bias_m20": [10, 250],
    "case01_bias_p20": [10, 250],
    "case01_bias_o0": [10, 250],
    "case02_bias_m20": [10, 250],
    "case02_bias_p20": [10, 250],
    "case02_bias_o0": [10, 250],
    "case03_bias_m20": [10, 250],
    "case03_bias_p20": [10, 250],
    "case03_bias_o0": [10, 250],
    "case04_bias_m20": [10, 250],
    "case04_bias_p20": [10, 250],
    "case04_bias_o0": [10, 250],
    "case05_bias_m20": [10, 250],
    "case05_bias_p20": [10, 250],
    "case05_bias_o0": [10, 250],
    "case06_bias_m20": [10, 500],
    "case06_bias_p20": [10, 500],
    "case06_bias_o0": [10, 500],
    "case07_bias_m20": [10, 500],
    "case07_bias_p20": [10, 500],
    "case07_bias_o0": [10, 500],
    "challenge_ball": [10, 180],
    "challenge_hand": [10, 180],
    "challenge_paper": [10, 180],
    "challenge_train": [10, 180],
    "challenge_umbrella": [10, 180],
    "demo": [10, 500]
}



test_key = [
    "case01_bias_m20",
    "case01_bias_p20",
    "case01_bias_o0",
    "case02_bias_m20",
    "case02_bias_p20",
    "case02_bias_o0",
    "case03_bias_m20",
    "case03_bias_p20",
    "case03_bias_o0",
    "case04_bias_m20",
    "case04_bias_p20",
    "case04_bias_o0",
    "case05_bias_m20",
    "case05_bias_p20",
    "case05_bias_o0",
    "case06_bias_m20",
    "case06_bias_p20",
    "case06_bias_o0",
    "case07_bias_m20",
    "case07_bias_p20",
    "case07_bias_o0",
    "challenge_ball",
    "challenge_hand",
    "challenge_paper",
    "challenge_train",
    "challenge_umbrella",
    "demo"
]


# dataset_dict = {
#     "scene01_m20":[10,250],
#     "scene01_p20":[10,250],
#     "scene01_o0":[10,250],
#     "scene02_m20":[10,250],
#     "scene02_p20":[10,250],
#     "scene02_o0":[10,250],
#     "scene03_m20":[10,250],
#     "scene03_p20":[10,250],
#     "scene03_o0":[10,250],
#     "scene04_m20":[10,250],
#     "scene04_p20":[10,250],
#     "scene04_o0":[10,250],
#     "scene05_m20":[10,250],
#     "scene05_p20":[10,250],
#     "scene05_o0":[10,250],
#     "2024-05-09-23-40_ball_m20":[100,500],
#     "2024-05-09-23-27_ball_p20":[100,500],
#     "2024-05-09-23-20_ball_o0":[100,500],
#     "2024-05-09-23-45_paper_m20":[100,500],
#     "2024-05-09-23-50_paper_p20":[100,500],
#     "2024-05-09-23-57_paper_o0":[100,500],
#     "2024-04-10-21-38":[10,180],
#     "2024-04-10-21-43":[10,180],
#     "2024-04-10-21-47":[10,180],
#     "2024-04-10-21-51":[10,180],
#     "2024-04-10-21-55":[10,180],
#     "haze_20lux":[2,502],
# }

# test_key = [
#     "haze_20lux",
#     "2024-05-09-23-40_ball_m20", 
#     "2024-05-09-23-27_ball_p20",
#     "2024-05-09-23-20_ball_o0",
#     "2024-05-09-23-45_paper_m20",
#     "2024-05-09-23-50_paper_p20", 
#     "2024-05-09-23-57_paper_o0", 
#     "scene01_m20",
#     "scene01_p20",  
#     "scene01_o0", 
#     "scene02_m20", 
#     "scene02_p20", 
#     "scene02_o0",
#     "scene03_m20",
#     "scene03_p20",
#     "scene03_o0", 
#     "scene04_m20",  
#     "scene04_p20", 
#     "scene04_o0", 
#     "scene05_m20",  
#     "scene05_p20", 
#     "scene05_o0",  
#     "2024-04-10-21-38", 
#     "2024-04-10-21-43",
#     "2024-04-10-21-47",
#     "2024-04-10-21-51",
#     "2024-04-10-21-55"
# ]