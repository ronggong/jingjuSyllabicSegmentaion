schulter_jan_cv: first jan implementation
    essentia log-mel, complicate sample weighting
    jan model, ELU, L2 weight decay, ADAM

schulter_jan_madmom:
    Madmom log-mel, complicate sample weighting
    jan model, ELU, L2 weight decay, ADAM

schulter_jan_madmom_simpleSampleWeighting:
    Madmom log-mel, simple sample weighting
    jan_original, CNN, dense relu, SGD

schluter_jan_madmom_nodropout_simpleSampleWeighting:
    Madmom log-mel, simple sample weighting
    jan_original, CNN, dense sigmoid, SGD, no dropout

schluter_jan_madmom_simpleSampleWeighting_100epochs_adam
    Madmom log-mel, simple sample weighting
    jan_original, CNN, dense sigmoid, ADAM, dropout, 100 epochs, StandardScaler
    parameters: 289,273
('threshold', 0.5700000000000001)
('jan_madmom_simpleSampleWeighting_100epochs_adam', 'recal, precision, F1', 82.7043572200216, 91.15697559039492, 86.7251958840744)

schluter_jan_madmom_simpleSampleWeighting_early_stopping_adam
    Madmom log-mel, simple sample weighting
    jan_original, CNN, dense sigmoid, ADAM, dropout, early stopping, StandardScaler
    parameters: 289,273
('threshold', 0.47000000000000003)
('jan_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 83.26251350378105, 90.36267000156323, 86.66741631995201)

schluter_jan_madmom_simpleSampleWeighting_100epochs_3channels_adam
    Madmom 3 channels log-mel, simple sample weighting
    jan_original, CNN, dense sigmoid, ADAM, dropout, 100 epochs, StandardScaler
    parameters: 289,693
('threshold', 0.54)
('jan_madmom_simpleSampleWeighting_100epochs_3channels_adam', 'recal, precision, F1', 84.22398271516023, 90.63747335787639, 87.31311245916939)

schluter_jan_madmom_simpleSampleWeighting_early_stopping_3channels_adam
('threshold', 0.53)
('jan_madmom_simpleSampleWeighting_early_stopping_3channels_adam', 'recal, precision, F1', 83.02124594886568, 91.74293672900914, 87.16446124763705)

schluter_jordi_temporal_schluter_madmom_simpleSampleWeighting_early_stopping_adam
('threshold', 0.39)
('jordi_temporal_schluter_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 81.13071660064819, 90.34767614388258, 85.49149069383573)

params: 254,750
('threshold', 0.51)
('jordi_temporal_schluter_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 83.42095786820309, 91.93951660912013, 87.4733324522816)
ttest p-value: 0.18957215163218821

schluter_jordi_timbral_schluter_madmom_simpleSampleWeighting_early_stopping_adam
('threshold', 0.52)
('jordi_timbral_schluter_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 79.8487576521426, 90.19687601692158, 84.70794972685947)

params: 262,535
('threshold', 0.58)
('jordi_timbral_schluter_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 81.0190853438963, 90.80965450435906, 85.63544323069311)

fast implementation 0.58
('jordi_timbral_schluter_madmom_simpleSampleWeighting_early_stopping_adam', 'recal, precision, F1', 80.33129276197334, 89.5364238410596, 84.68444528803265)
