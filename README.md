# High-Resolution Road Vehicle Collision Prediction for the City of Montreal

This repository contains the source code developed for a study of road vehicle collisions in the city of Montreal.
Three datasets provided by the city of Montreal and the Government of Canada were used: a dataset containing road vehicle collisions, a dataset describing the Canadian road network, and a dataset containing historical weather information.
These datasets have been fused to generate examples corresponding to an hour period and a road segment delimited by intersections.
A binary classification has been performed with positive examples, corresponding to the occurrence of a collision, and negative examples, corresponding to the non-occurrence of a collision.
Four models have been built and compared, a first basic model using only the count of accident during previous years on the road segment, a model built using random forest with under-sampling of the majority class, a model using balanced random forest and a model using XGBoost. 
The best performances were obtained by the balanced random forest model.
It identifies as positives the 13% most dangerous examples which correspond to 85% of vehicle collisions.

For more information read [the corresponding scientific paper](https://arxiv.org/abs/1905.08770 "Paper on ArXiv").

## Folder Structure
- mains: contains the scripts for the generation of the dataset, the hyperparameter tuning, the training and the evaluation of the models
- notebooks: Jupyter notebooks used during development for interactive exploration of the data and experimentations
- results: results of the four models

## License
[MIT](https://choosealicense.com/licenses/mit/)
