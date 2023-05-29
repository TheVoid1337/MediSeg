# MediSeg
Dieses Projekt unterteilt sich in die folgenden Ordner

### data_loader: 
Dient zum Laden des Trainingsdatensatzes. Dieser ist darauf ausgerichtet den Liver Tumor Datensatz [1],[2] der Liver Tumor Segmentation Challenge (LiTS) nach Bilic et al. [3] in das numpy Binärformat umzuwandeln. 
### preprocessing: 
Hier wird der Datensatz in Trainings- und Testdatensatz aufgeteilt. Anschließend werden diese normalisiert. Die Maskendaten werden anstatt der Normalisierung einem One-Hot Encoding unterzogen.
### models:
Hier befindet sich der Programmcode der U-Net Architekturen: <br>
U-Net nach Ronneberger et al. [4], <br>
Attention U-Net nach Oktay et al. [5], <br>
Bi-ConvLSTM U-Net nach Azad et al. [6], <br>
### metrics:
Dice-Koeffizent für binäre und mehrklassen Segmentierung, welche für das Training der Modelle und deren Evaluation genutzt wurde.
### training:
Trainingsmethode als template für das Training. 
### results: 
Ergebnisse der Trainingsverläufe, sowie der generierten Modellvohersagen in quantitativer (Box-Plots) als auch qualitativer Form (Bilder). 
### nets:
Ordner zum Speichern der U-Net Modelle in Form der H5-Dateien der Keras-API.
### evaluation:
Programmcode zur erzeugung der Ergebnisse, welche sich in den Unterordnern von results befinden.




## Quellen 
[1] P. Bilic, P. F. Christ, E. Vorontsov et al., Liver Tumor Segmentation, 2017. Adresse:
https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation?select=volume_pt1, Teil 1 des Datensatzes. Download am 06.04.2023. <br>
[2] P. Bilic, P. F. Christ, E. Vorontsov et al., Liver Tumor Segmentation - Part 2, 2017.
Adresse: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2, Download am 06.04.2023. <br>
[3] P. Bilic, P. F. Christ, E. Vorontsov et al., The Liver Tumor Segmentation Benchmark
(LiTS), 2019. [arXiv: 1901.04056 [cs.CV]](https://arxiv.org/abs/1901.04056). <br>
[4] O. Ronneberger, P. Fischer und T. Brox, U-Net: Convolutional Networks for Bio-
medical Image Segmentation, 2015. arXiv: [1505.04597 [cs.CV]](https://arxiv.org/abs/1505.04597). <br>
[5] O. Oktay, J. Schlemper, L. L. Folgoc et al., Attention U-Net: Learning Where to
Look for the Pancreas, 2018. arXiv: [1804.03999 [cs.CV]](https://arxiv.org/abs/1804.03999). <br>
[6] R. Azad, M. Asadi-Aghbolaghi, M. Fathy und S. Escalera, „Bi-Directional ConvLSTM
U-Net with Densley Connected Convolutions,“ in 2019 IEEE/CVF International
Conference on Computer Vision Workshop (ICCVW), 2019, S. 406–415. [DOI: 10
.1109/ICCVW.2019.00052](https://ieeexplore.ieee.org/document/9022282). <br>




