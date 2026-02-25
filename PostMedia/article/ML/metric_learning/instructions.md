## Goal: 
Make a plotly animation showing the training of a metric learning model and export that animation as an interactive HTML page and a GIF animation for a slide. 

## Animation details:
1. There should be 4 key frames showing epoch 0, epoch 5, epoch 25 and epoch 50.
2. During each epoch
    - 3 images (`./anchor.png`, `./Negative.png` and `./positive.png`) are passed through an encoder (depicted via a trapezium rotated 90 degrees) and projected onto a coordinate space. 
    - The distances are measured for triplet loss. 
3. Each epoch should learn a better mapping

## ENV: 
Use the `web` conda env