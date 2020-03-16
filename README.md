# Sci-Inq
Attempting to improve the current OpenPilot image detection algorithm.

Dataset:
https://bdd-data.berkeley.edu
Download 'images' and 'labels' under 'bdd100k' into Data folder
Structure should be Data -> [
    train.json,
    val.json,
    train   -> [All images in bdd100k train folder],
    val     -> [All images in bdd100k val folder]
    ]
    (May have to rename files / folders)

Set path in Data.py to you path to Sci-Inq directory