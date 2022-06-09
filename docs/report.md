# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objhectives, approach, and results.

# 1. Introduction

This section should cover the following items:

* Motivation  <br />
We observe a surge in the count of smart devices that uses voice control as its interface to users over the past decade. And since voice control is used in numerous senstive applications such as Banking, Healthcare and Smart Homes, Automatic Speaker Verification systems are used as a form of biometric identification of the speaker. Over the past few years, there has been many attempts to spoof the ASV system by various attacks like Impersonation attacks, Replay Speech Synthesis, Voice Conversion. Hence a countermeasure system is needed to identify such attacks. The Countermeasure System will complement the ASV system in its identification process. The goal of the project is to develop a Countermeasure (CM) system to complement the ASV system to verify the authenticity (original/fake) of a given audio sample.

* Objective  <br />
Given an Automatic Speaker Verification (ASV) system whose role is verify if the input speech sample belongs to the target user or not, this system is however vulenerable to attacks such as the following nature:
Physical Access Attacks: Replay attacks where in the attacker captures the voice of the target user using a recording device and plays the recording to the ASV system)
Voice Conversion/Speech Synthesis attacks : Attacks where in an attacker utilizes TTS (text to speech) modules to generate speech to mimic the target user or use voice conversion techniques to convert the attacker’s voice to the target user’s voice).
Our aim to develop an countermeasure system which helps us tackle the Speech Synthesis/Voice Conversion attacks commonly referred to as Logical Access attacks. We will explore various feature extraction techniques such as MFCC, CQCC, Mel Spectrum and couple it with both DNN and Non Neural Network architectures to understand the performance of the resulting models.


* State of the Art & Its Limitations: 
Architecture Details: - Fusion Light CNN (LCNN) which operates upon LFCC features (feature extraction technique to covert the raw acoustic waveform into its fourier domain) employing angular-margin-based softmax loss, batch normalization and a normal Kaiming initialization.

* Limitations: 
For the physical access attacks, it has been found that the system does not handle real world conditions charecterized by nuisance variations such as Additive noise.
Does not model channel variability.


* Novelty & Rationale:
Our intention is to develop an architecture combining CNN model with sequencing models. Since we are dealing with temporal data, we believe sequence models such as RNNs, LSTMs and GRUs can be utilized to capture the temporal information present in the data and when coupled with CNNs which are known to extract both spatial and temporal information, the combination of both could yield us better results

* Potential Impact: 
On a general front, Counter Measure systems can be employed in tandem to ASV systems, thereby increasing the reliability of the system. On a technical front, the above architecture can handle the Logical Access and Physical Access attacks.

* Challenges: What are the challenges and risks?

* Requirements for Success: What skills and resources are necessary to perform the project?
Based on the work done thus far, we understand that various filtering and pre-processing techniques such as MQCC, Mel Spectrum and CQCCs are required as these domain transformation techniques are widely used to convert the data from its acoustic waveform to its fourier domain. Hence speech processing/filtering domain knowledge is needed.

The second leg of the architecture involves the use of Deep Learning models such as CNNs, RNNs along with traditional statistical/machine learning models such as the GMMs. Hence Statistical, Machine Learning and Deep Learning domain knowledge will be reuqired.

* Metrics of Success: What are metrics by which you would check for success?
As the problem is defined by the ASV spoof competion organizers (https://www.asvspoof.org/), the metrics defined are as follows:

t-DCF [1] : the tandem detection cost function is the new primary metric in the ASVSpoof 2019 challenge. It was proposed as a reliable scoring metric to evaluate the combined performance of ASV and CMs.
EER : the Equal Error Rate is used as a secondary metric. EER is determined by the point at which the miss (false negative) rate and false alarm (false positive) rate are equal to each other.



# 2. Related Work

# 3. Technical Approach

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
[1] Kanervisto, Anssi & Hautamäki, Ville & Kinnunen, Tomi & Yamagishi, Junichi. (2022). Optimizing Tandem Speaker Verification and Anti-Spoofing Systems.

[2] https://datashare.ed.ac.uk/handle/10283/3336

[3] https://datashare.ed.ac.uk/handle/10283/3055

[4] Nautsch, Andreas et al. “ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech.” IEEE Transactions on Biometrics, Behavior, and Identity Science 3 (2021): 252-265.

[5] X. Cheng, M. Xu and T. F. Zheng, “Replay detection using CQT-based modified group delay feature and ResNeWt network in ASVspoof 2019,” 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2019, pp. 540-545, doi: 10.1109/APSIPAASC47483.2019.9023158.

[6] Chettri, Bhusan, et al. “A study on convolutional neural network based end-to-end replay anti-spoofing.” arXiv preprint arXiv:1805.09164 (2018).

[7] Alzantot, Moustafa, Ziqi Wang, and Mani B. Srivastava. “Deep residual neural networks for audio spoofing detection.” arXiv preprint arXiv:1907.00501 (2019).

[8] Wang, Xin, et al. “ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.” Computer Speech & Language 64 (2020): 101114.

[9] R. K. Das, J. Yang and H. Li, “Long Range Acoustic and Deep Features Perspective on ASVspoof 2019,” 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2019, pp. 1018-1025, doi: 10.1109/ASRU46091.2019.9003845.

[10] O. Abdel-Hamid, A. Mohamed, H. Jiang, L. Deng, G. Penn and D. Yu, “Convolutional Neural Networks for Speech Recognition,” in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 10, pp. 1533-1545, Oct. 2014, doi: 10.1109/TASLP.2014.2339736.

[11] Zhang, Xiang, Junbo Zhao, and Yann LeCun. “Character-level convolutional networks for text classification.” Advances in neural information processing systems 28 (2015).

[12] McFee, Brian & Raffel, Colin & Liang, Dawen & Ellis, Daniel & Mcvicar, Matt & Battenberg, Eric & Nieto, Oriol. (2015). librosa: Audio and Music Signal Analysis in Python. 18-24. 10.25080/Majora-7b98e3ed-003.

[13] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems,

Software available from tensorflow.org.
[14] Ioffe, Sergey, and Christian Szegedy. “Batch normalization: Accelerating deep network training by reducing internal covariate shift.” International conference on machine learning. PMLR, 2015.

[15] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
[16] Lavrentyeva, S. Novoselov, A. Tseren, M. Volkova, A. Gorlanov, and A. Kozlov, “STC antispoofing systems for the ASVspoof2019 challenge,” in Proc. Interspeech, 2019, pp. 1033–1037. 
[17] Chen, A. Kumar, P. Nagarsheth, G. Sivaraman, and E. Khoury, “Generalization of Audio Deepfake Detection,” in Proc. Odyssey, 2020, pp. 132–137. 
[18] Yamagishi, Junichi; Todisco, Massimiliano; Sahidullah, Md; Delgado, Héctor; Wang, Xin; Evans, Nicolas; Kinnunen, Tomi; Lee, Kong Aik; Vestman, Ville; Nautsch, Andreas. (2019). ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database, [sound]. University of Edinburgh. The Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2555
[19] Y. Zhang, F. Jiang and Z. Duan, "One-Class Learning Towards Synthetic Voice Spoofing Detection," in IEEE Signal Processing Letters, vol. 28, pp. 937-941, 2021, doi: 10.1109/LSP.2021.3076358.

