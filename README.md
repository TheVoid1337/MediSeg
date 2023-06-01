# MediSeg
Heterogene Datensätze sind im alltäglichen Gebrauch der Medizin gängig, bringen jedoch herausforderungen mit sich, welche sich in schwer zu Segmentierenden bereichen äußert.
Um dem entgegenzuwirken, setzt die Forschung auf die Erweiterung von Long Short Term Memories (LSTMs) [1] und Attention [2]. Jedoch ist der konkrete Vergleich beider Ansätze auf die Leistungsfähigkeit auf das U-Net [3] bisher nur indirekt untersucht worden [4]-[6]. 
Dieses Projekt dient zur Messung des Leistungseinflusses von (LSTMs) und Attention auf die Leistungsfähigkeit der U-Net Architektur. Hierfür wird die Leistungsfähigkeit der folgenden vier Ansätze mit dem LiTS Datensatz [7,8] der Liver Tumor Segmentation Challenge nach Bilic et al. [4] trainiert und anschließend evaluiert:
Architekturmodelle: <br>

#### U-Net: nach Ronneberger et al. [3], <br> 
#### Attention U-Net: nach Oktay et al. [5], <br>
#### Bi-ConvLSTM U-Net: nach Azad et al. [6], <br>
#### Attention LSTM U-Net: als Kombination der oberen Ansätze.


## Projektaufbau
Das Projekt ist in neun Ordner unterteilt, wovon einer den LiTS-Datensatz beinhaltet. Die Ordnerstruktur ist im Folgenden aufgelistet:

### data_loader: 
Enthält einen speziellen loader zum Laden des LiTS-Datensatzes. Dieser lädt die Bilddaten und speichert diese im numpy binärformat. 

### preprocessing:
Sobald der Datensatz geladen, gefiltert und gespeichert wurde, wird der Datensatz mit der Methode prepare für das Training präpariert. Dieser befindet sich im preprocessing Ordner.

### models:
In diesem Ordner befinden sich die Modelle nach den obengenannten Forschungsansätzen, welche nach der Vorverarbeitung erzeugt werden.

### metrics:
Hier befindet sich die Binär- und Mehrklassendefinition des Dice-Koeffizienten, welche für den Trainingsverlauf und die evaluation genutzt wird.

### training:
Trainingsmethode als template für das Training der U-Net Modelle.

### results: 
Ergebnisse der Trainingsverläufe, sowie der generierten Modellvorhersagen in quantitativer (Box-Plots) als auch qualitativer Form (Bilder).
Diese unterteilen sich in vier Unterordner: <br>
#### graphics: Box-Plots der Leber und Tumor Segmente mit den Metriken des Dice- und des Jaccard-Koeffizienten. <br>
#### plots: Zusammengefasste Modellvorhersagen aller Modelle in Bildform. <br>
#### test_data: Übereinstimmungsergebnisse als CSV-Datei für Leber und Tumor Segment der einzelnen Modelle. <br>
#### training_plots: Grafische Darstellung der Trainingsverläufe (Dice, IoU, Kreuzentropieverlust) aller Modelle. <br>
### nets:
Ordner zum Speichern der U-Net Modelle in Form der H5-Dateien der Keras-API.

### evaluation:
Programmcode zur erzeugung der Ergebnisse, welche sich in den Unterordnern von results befinden.

### LiverTumorDataset (Nur Lokal) 
Dieser Ordner enthalte den LiTS-Datensatz Teil 1 und 2 inklusive der numpy Dateien. Aufgrund der Größe des Datensatzes 
ist das Hochladen jedoch nicht möglich. Der Datensatz ist erhältlich unter [7,8].

### Hinweis:
Der Programmcode der Architekturen ist inspiriert von den Autoren der oben genannten Paper. 
Referenz auf die originalen Umsetzungen der Modell architekturen finden sich hier: <br>
#### U-Net: [https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). <br>
#### Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions: [https://github.com/rezazad68/BCDU-Net](https://github.com/rezazad68/BCDU-Net). <br>
#### Attention U-Net: [https://github.com/ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks). <br>
#### Weitere Information zum Programmcode befindet sich in den entsprechenden Dateien des Pythoncodes.

## Quellen 

[1] S. Hochreiter und J. Schmidhuber, „Long Short Term Memory Technical Report
FKI-207-95,“ Technische Universität München, IDSIA, 1995 <br>

[2] D. Bahdanau, K. Cho und Y. Bengio, Neural Machine Translation by Jointly Lear-
ning to Align and Translate, 2016. arXiv: [1409.0473 [cs.CL]](https://arxiv.org/abs/1409.0473). <br>

[3] O. Ronneberger, P. Fischer und T. Brox, U-Net: Convolutional Networks for Bio-
medical Image Segmentation, 2015. arXiv: [1505.04597 [cs.CV]](https://arxiv.org/abs/1505.04597). <br>

[4] P. Bilic, P. F. Christ, E. Vorontsov et al., The Liver Tumor Segmentation Benchmark
(LiTS), 2019. [arXiv: 1901.04056 [cs.CV]](https://arxiv.org/abs/1901.04056). <br>

[5] O. Oktay, J. Schlemper, L. L. Folgoc et al., Attention U-Net: Learning Where to
Look for the Pancreas, 2018. arXiv: [1804.03999 [cs.CV]](https://arxiv.org/abs/1804.03999). <br>

[6] R. Azad, M. Asadi-Aghbolaghi, M. Fathy und S. Escalera, „Bi-Directional ConvLSTM
U-Net with Densley Connected Convolutions,“ in 2019 IEEE/CVF International
Conference on Computer Vision Workshop (ICCVW), 2019, S. 406–415. [DOI: 10
.1109/ICCVW.2019.00052](https://ieeexplore.ieee.org/document/9022282). <br>

[7] P. Bilic, P. F. Christ, E. Vorontsov et al., Liver Tumor Segmentation, 2017. Adresse:
https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation?select=volume_pt1, Teil 1 des Datensatzes. Download am 06.04.2023. <br>

[8] P. Bilic, P. F. Christ, E. Vorontsov et al., Liver Tumor Segmentation - Part 2, 2017.
Adresse: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2, Download am 06.04.2023. <br>







