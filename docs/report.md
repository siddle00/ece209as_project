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
  -We observe a surge in the count of smart devices that uses voice control as its interface to users over the past decade. And since voice control is used in numerous senstive applications such as Banking, Healthcare and Smart Homes, Automatic Speaker Verification systems are used as a form of biometric identification of the speaker. Over the past few years, there has been many attempts to spoof the ASV system by various attacks like Impersonation attacks, Replay Speech Synthesis, Voice Conversion. Hence a countermeasure system is needed to identify such attacks. The Countermeasure System will complement the ASV system in its identification process. The goal of the project is to develop a Countermeasure (CM) system to complement the ASV system to verify the authenticity (original/fake) of a given audio sample.

* Objective  <br />
  -The Automatic Speaker Verification (ASV) system ideally aims to verify the identity and authenticity of a target user given an audio sample. However, these ASV systems are vulnerable to spoofing attacks of the following kind:
    - Impersonation attacks 
    - Replay 
    - Speech Synthesis 
    - Voice Conversion 
 
  -Physical Access Attacks: Replay attacks where in the attacker captures the voice of the target user using a recording device and plays the recording to the ASV system)  <br />

  -Voice Conversion/Speech Synthesis attacks : Attacks where in an attacker utilizes TTS (text to speech) modules to generate speech to mimic the target user or use voice conversion techniques to convert the attacker’s voice to the target user’s voice). <br />

  -Our aim to develop an countermeasure system which helps us tackle the Speech Synthesis/Voice Conversion attacks commonly referred to as Logical Access attacks. We will explore various feature extraction techniques such as MFCC, CQCC, Mel Spectrum and couple it with both DNN and Non Neural Network architectures to understand the performance of the resulting models.
<br />

* State of the Art & Its Limitations:  <br />
Architecture Details: - Fusion Light CNN (LCNN) which operates upon LFCC features (feature extraction technique to covert the raw acoustic waveform into its fourier domain) employing angular-margin-based softmax loss, batch normalization and a normal Kaiming initialization. <br />

* Limitations:          <br />
  - For the physical access attacks, it has been found that the system does not handle real world conditions charecterized by nuisance variations such as Additive noise.           <br />
  - Does not model channel variability. <br />
  - The system cannot defend against unknown spoofing attacks a.k.a generalization ability 
  - How to match single system performances to fusion model (state of the art) which are computationally expensive ?


* Novelty & Rationale:   <br />
  - How can the system defend against unknown spoofing attacks a.k.a generalization ability ?
        - Intuition: One class classification approach with modified loss function to shrink the embedding space of the target class.
  - How to match single system performances to fusion model which are computationally expensive ?
        - Intuition: Using Generative models or Auto encoder models as an alternate to Deep fused models <br /> 

* Potential Impact:     <br />
  - On a general front, Counter Measure systems can be employed in tandem to ASV systems, thereby increasing the reliability of the system. On a technical front, the above architecture can handle the Logical Access and Physical Access attacks.
  - Our proposed model relies on a combination of feature extraction and a traditional ML classifer and hence the model is lightweight. Hence it could be potentially emplpoyed in every day wireless devices without any demand for higher computational power or edge computation.       <br />

* Challenges: What are the challenges and risks?   
  -  Developing a model that can match the performance of fusion models. Fusion models are usally a combination on multiple sub systems and hence they are computationally expensive. Hence the light weight model must match the performance of the fusion models would be a challenge <br />

* Requirements for Success: What skills and resources are necessary to perform the project?     <br />
  -Based on the work done thus far, we understand that various filtering and pre-processing techniques such as MQCC, Mel Spectrum and CQCCs are required as these domain transformation techniques are widely used to convert the data from its acoustic waveform to its fourier domain. Hence speech processing/filtering domain knowledge is needed.
  
  - The second leg of the architecture involves the use of Deep Learning models such as CNNs, RNNs along with traditional statistical/machine learning models such as the GMMs. Hence Statistical, Machine Learning and Deep Learning domain knowledge will be reuqired. 
  - Apart from that, models will be trained using Google Collab Pro GPUs and that should be the computational resources required.  <br />

* Metrics of Success: <br />
What are metrics by which you would check for success?
  -As the problem is defined by the ASV spoof competion organizers (https://www.asvspoof.org/), the metrics defined are as follows: 
    - t-DCF [1] : the tandem detection cost function is the new primary metric in the ASVSpoof 2019 challenge. It was proposed as a reliable scoring metric to evaluate the combined performance of ASV and CMs.      <br />
    -EER : the Equal Error Rate is used as a secondary metric. EER is determined by the point at which the miss (false negative) rate and false alarm (false positive) rate are equal to each other.                 <br />



# 2. Related Work

![This is an image](https://github.com/siddle00/ece209as_project/blob/main/Images/LR.png)


# 3. Technical Approach


- **Feature Extraction** 
  - MFCC (Mel Frequency Cepstral Coefficients) - Available @ Librosa python.
  - CQCC (Constant Q Cepstral Coefficients) - Implemented in python based on the block diagram below:

![This is CQCC ](https://github.com/siddle00/ece209as_project/blob/main/Images/CQCC.png)

- **Classifier**
  - GMM: 3 GMMs of 144, 256 & 512 mixture components modules with expectation-maximization (EM) algorithm with random initialisation were trained. 
        - Score for a given test occurrence is computed as the log-likelihood ratio as following :
      ![This is CQCC ](https://github.com/siddle00/ece209as_project/blob/main/Images/GMM.png)
         - where X: Test utterance feature vectors, L: Likelihood function, Θn: GMMs for bonafide speech, Θs: GMM for spoofed speech.
  - SVM: 2 SVMs with mean-variance normalisation performed on the extracted features applied on a linear/RBF kernel and the default parameters of the Scikit-Learn library.


- **Dataset and Protocols**
  - Publicly available ASVspoof 2019 LA [3] - Based on the VLTK corpus, a multi-speaker (46 male, 61 female) speech database. 
        - Training set: 25380 with 2580 bonafide, 22800 spoofed utterances 
        - Development set: 24987 with 2548 bonafide, 22296 spoofed utterances.
        - Testing set: 71934 with 7355 bonafide, 63882 spoofed utterances.

  - Spoofed data is generated by using 17 TTS and VC algorithms.
        - 6 known spoofing systems with 2 VC and 4 TTS.
        - 11 unknown spoofing systems with unknown division.


- **Platform**
  - Models were trained on Google Collab Pro on K80 and T4 GPUs with 32 GB RAM.
  - Few of the pre-processing blocks were run on local machine.


# 4. Evaluation and Results

**-Evaluation Metric**

  - Equal Error Rate (EER): Decision threshold where the false acceptance and the false rejection rates are equal.
  - Tandem Detection Cost Function (t-DCF): Takes into account both the ASV system error and CM system error into consideration.
      ![This is tdcf ](https://github.com/siddle00/ece209as_project/blob/main/Images/tdcf.png)
      
      - Casv miss - Cost of ASV system rejecting a target trial.
      - Casv fa - Cost of ASV system accepting a non-target trial.
      - Ccm miss - Cost of CM system rejecting a bonafide trial. 
      - Ccm fa - Cost of ASV system accepting a spoof trial.
      - π - Priori probabilities, P• - Error rates


**-Results**

- **Development Set Results**

  ![This is dev ](https://github.com/siddle00/ece209as_project/blob/main/Images/dev.png)
  
  
- **Evaluation Set Results** 

  ![This is eval ](https://github.com/siddle00/ece209as_project/blob/main/Images/eval.png)

# 5. Discussion and Conclusions

  -**What worked?**: Adopting the one class learning approach helped in generalising the model for unknown spoof attacks.
  
  -**What did not work?**: Although single systems did give comparable results to state of the art fusion models, better performance was expected. Probably a feature fusion could have aided in better results.
  
  -**What could have been done differently?**: Using deep models to extract features rather than using MFCC, CQCC. 
  
  -**Future directions**: Exploring performances on individual spoof attacks and propose maybe an ensemble architecture to handle different spoofing attacks.

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

