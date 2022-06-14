# Project Proposal

## 1. Motivation & Objective


### **Motivation**

We observe a surge in the count of smart devices that uses voice control as its interface to users over the past decade. And since voice control is used in numerous senstive applications such as Banking, Healthcare and Smart Homes, Automatic Speaker Verification systems are used as a form of biometric identification of the speaker. Over the past few years, there has been many attempts to spoof the ASV system by various attacks (discussed below) and hence a countermeasure system is needed to identify such attacks. The Countermeasure System will complement the ASV system in its identification process.

### **Objective/Problem Statement** 

Given an Automatic Speaker Verification (ASV) system whose role is verify if the input speech sample belongs to the target user or not, this system is however vulenerable to attacks such as the following nature: 

- Physical Access Attacks: Replay attacks where in the attacker captures the voice of the target user using a recording device and plays the recording to the ASV system)
- Voice Conversion/Speech Synthesis attacks : Attacks where in an attacker utilizes TTS (text to speech) modules to generate speech to mimic the target user or use voice conversion techniques to convert the attacker's voice to the target user's voice). 

**Our aim to develop an countermeasure system which helps us prevent either of the above mentioned type of attacks.**


## 2. State of the Art & Its Limitations

- **Architecture Details:** - Fusion Light CNN (LCNN) which operates upon LFCC features (feature extraction technique to covert the raw acoustic waveform into its fourier domain) employing angular-margin-based softmax loss, batch normalization and a normal Kaiming initialization. 

- **Limitations:**  
   1. For the physical access attacks, it has been found that the system does not handle real world conditions charecterized by nuisance variations such as Additive noise.
    2. Does not model channel variability. 
   

## 3. Novelty & Rationale

Our intention is to develop an architecture combining CNN model with sequencing models. Since we are dealing with temporal data, we beleive sequence models such as RNNs, LSTMs and GRUs can be utilized to capture the temporal information present in the data and when coupled with CNNs which are known to extract both spatial and temporal information, the combination of both could yield us better results. 

## 4. Potential Impact

On a general front, Counter Measure systems can be employed in tandem to ASV systems, thereby increasing the reliability of the system. On a technical front, the above architecture can handle the Logical Access and Physical Access attacks. 

## 5. Challenges

Developing a model that can match the performance of fusion models. Fusion models are usally a combination on multiple sub systems and hence they are computationally expensive. Hence the light weight model must match the performance of the fusion models would be a challenge.

## 6. Requirements for Success


Based on the work done thus far, we understand that various filtering and pre-processing techniques such as MQCC, Mel Spectrum and CQCCs are required as these domain transformation techniques are widely used to convert the data from its acoustic waveform to its fourier domain. Hence speech processing/filtering domain knowledge is needed. 

The second leg of the architecture involves the use of Deep Learning models such as CNNs, RNNs along with traditional statistical/machine learning models such as the GMMs. Hence Statistical, Machine Learning and Deep Learning domain knowledge will be reuqired. 


## 7. Metrics of Success


As the problem is defined by the ASV spoof competion organizers (https://www.asvspoof.org/), the metrics defined are as follows: 

  1. t-DCF [[1]](#1) : the tandem detection cost function is the new primary metric in the ASVSpoof 2019 challenge. It was proposed as a reliable scoring metric to evaluate the combined performance of ASV and CMs.
  2. EER : the Equal Error Rate is used as a secondary metric. EER is determined by the point at which the miss (false negative) rate and false alarm (false positive) rate are equal to each other.


## 8. Execution Plan

We would compare the performance of different models based on the evaluation metrics specified below, 
  - Equal Error Rate (EER): Decision threshold where the false acceptance and the false rejection rates are equal.
  - Tandem Detection Cost Function (t-DCF): Takes into account both the ASV system error and CM system error into consideration.
      ![This is tdcf ](https://github.com/siddle00/ece209as_project/blob/main/Images/tdcf.png)
      
      - Casv miss - Cost of ASV system rejecting a target trial.
      - Casv fa - Cost of ASV system accepting a non-target trial.
      - Ccm miss - Cost of CM system rejecting a bonafide trial. 
      - Ccm fa - Cost of ASV system accepting a spoof trial.
      - π - Priori probabilities, P• - Error rates

## 9. Related Work

### 9.a. Papers

List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

|  Paper/Article | Summary | 
| ------------- | ------------- |
| ***Core Papers i.e papers that are directly relevant to our work*** | | 
| ASVspoof 2019: spoofing countermeasures for the detection of synthesized, converted and replayed speechhttps [[4]](#4) | Summarising the architectures of 2019 asvspoof for LA and PA. Fusion algorithm is shown to have a better performance for logical Access systems , where the inputs are text to speech. Fusion is not as effective for Physical access. However, a slight modification of the Same called oracle fusion is shown to perform better for PA. |
| Replay detection using CQT-based modified group delay feature and ResNeWt network in ASVspoof 2019 [[5]](#5)| Describes the implementation of CQT based MGT preprocesssing technique to improve the performance for physical access based spoofing attacks.|
| A Study On Convolutional Neural Network Based End-To-End Replay Anti-Spoofing [[6]](#6) | Explores and compares four end to end deep CNN architecture for replay attacks detection (PA audio spoofing). Also explores the diversity in the data set as the results are quite different for evaluation dataset and development dataset.|
| Deep Residual Neural Networks for Audio Spoofing Detection [[7]](#7)|  Model is inspired by the success of residual convolutional networks in many classification tasks. We build three variants of a residual convolutional neural network that accept different feature representations (MFCC, Log-magnitude STFT, and CQCC) of input. We compare the performance achieved by our model variants and the competition baseline models. In the logical access scenario, the fusion of our models has zero t-DCF cost and zero equal error rate (EER), as evaluated on the development set. On the evaluation set, our model fusion improves the t-DCF and EER by 25% compared to the baseline algorithms. Against physical access replay attacks, our model fusion improves the baseline algorithms t-DCF and EER scores by 71% and 75% on the evaluation set, respectively.Employed multiple feature extraction techqniues and applied a CNN + Residual architecuture| 
| ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech [[8]](#8) | This paper gives a brief overview for the creation of asvspoof database both for Logical access and Physical access scenarios with various use cases.|
| Long Range Acoustic and Deep Features Perspective on ASVspoof 2019 [[9]](#9)| The work considered novel countermeasures based on long range acoustic features, that are unique in many ways as they are derived using octave power spectrum and subbands, as opposed to the commonly used linear power spectrum. During the post-challenge study, the work further investigate the use of deep features that enhances the discriminative ability between genuine and spoofed speech. In this paper,they summarize the findings from the perspective of long range acoustic and deep features for spoof detection and make a comprehensive analysis on the nature of different kinds of spoofing attacks and system development.|
| ***Helper Papers - Enables us to understand the different concepts needed for the core paper. For example usage of CNN architectures for speech processing*** ||
|Convolutional Neural Networks for Speech Recognition [[#10]](#10)| This paper shows that error rate reduction can be obtained by using convolutional neural networks (CNNs). It first presents a concise description of the basic CNN and explain how it can be used for speech recognition. And then further propose a limited-weight-sharing scheme that can better model speech features. The special structure such as local connectivity, weight sharing, and pooling in CNNs exhibits some degree of invariance to small shifts of speech features along the frequency axis, which is important to deal with speaker and environment variations. Experimental results show that CNNs reduce the error rate by 6%-10% compared with DNNs on the TIMIT phone recognition and the voice search large vocabulary speech recognition tasks.|
|Character-level Convolutional Networks for Text Classification [[#11]](#11)| This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.|
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[#14]](#14) | Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.|



### 9.b. Datasets

1. ASV Spoof 2019 dataset [[2]](#2) 
2. ASV Spoof 2017 dataset V2 [[3]](#3)

### 9.c. Software

List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

1. librosa: Audio and Music Signal Analysis in Python [[12]](#12) 
2. TensorFlow: A System for Large-Scale Machine Learning [[#13]](#13)
3. PyTorch: An Imperative Style, High-Performance Deep Learning Library [[#15]](#15)


## 10. References

List references correspondign to citations in your text above. For papers please include full citation and URL. For datasets and software include name and URL.

<a id="1">[1]</a> 
Kanervisto, Anssi & Hautamäki, Ville & Kinnunen, Tomi & Yamagishi, Junichi. (2022). Optimizing Tandem Speaker Verification and Anti-Spoofing Systems. 

<a id="2">[2]</a> 
https://datashare.ed.ac.uk/handle/10283/3336

<a id="3">[3]</a>
https://datashare.ed.ac.uk/handle/10283/3055

<a id="4">[4]</a>
Nautsch, Andreas et al. “ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech.” IEEE Transactions on Biometrics, Behavior, and Identity Science 3 (2021): 252-265.

<a id="5">[5]</a>
X. Cheng, M. Xu and T. F. Zheng, "Replay detection using CQT-based modified group delay feature and ResNeWt network in ASVspoof 2019," 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2019, pp. 540-545, doi: 10.1109/APSIPAASC47483.2019.9023158.

<a id="6">[6]</a>
Chettri, Bhusan, et al. "A study on convolutional neural network based end-to-end replay anti-spoofing." arXiv preprint arXiv:1805.09164 (2018).

<a id="7">[7]</a>
Alzantot, Moustafa, Ziqi Wang, and Mani B. Srivastava. "Deep residual neural networks for audio spoofing detection." arXiv preprint arXiv:1907.00501 (2019).

<a id="8">[8]</a>
Wang, Xin, et al. "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech." Computer Speech & Language 64 (2020): 101114.

<a id="9">[9]</a>
R. K. Das, J. Yang and H. Li, "Long Range Acoustic and Deep Features Perspective on ASVspoof 2019," 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2019, pp. 1018-1025, doi: 10.1109/ASRU46091.2019.9003845.


<a id="10">[10]</a>
O. Abdel-Hamid, A. Mohamed, H. Jiang, L. Deng, G. Penn and D. Yu, "Convolutional Neural Networks for Speech Recognition," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 10, pp. 1533-1545, Oct. 2014, doi: 10.1109/TASLP.2014.2339736.


<a id="11">[11]</a>
Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." Advances in neural information processing systems 28 (2015).

<a id="12">[12]</a>
McFee, Brian & Raffel, Colin & Liang, Dawen & Ellis, Daniel & Mcvicar, Matt & Battenberg, Eric & Nieto, Oriol. (2015). librosa: Audio and Music Signal Analysis in Python. 18-24. 10.25080/Majora-7b98e3ed-003. 

<a id="13">[13]</a>
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.

<a id="14">[14]</a>
Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International conference on machine learning. PMLR, 2015.

<a id="15">[15]</a>
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
