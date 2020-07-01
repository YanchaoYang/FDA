# FDA: Fourier Domain Adaptation for Semantic Segmentation.

This is the Pytorch implementation of our [FDA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) paper published in CVPR 2020.

Domain adaptation via *style transfer* made easy using Fourier Transform. FDA needs **no deep networks** for style transfer, and involves **no adversarial training**. Below is the diagram of the proposed Fourier Domain Adaptation method:

Step 1: Apply FFT to source and target images.

Step 2: Replace the low frequency part of the source amplitude with that from the target.

Step 3: Apply inverse FFT to the modified source spectrum.

![Image of FDA](https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png)

# Usage

1. FDA Demo
   
   > python3 FDA_demo.py
   
   An example of FDA for domain adaptation. (source: GTA5, target: CityScapes, with beta 0.01)
   
   ![Image of Source](https://github.com/YanchaoYang/FDA/blob/master/demo_images/example.png)


2. Sim2Real Adaptation Using FDA (single beta)

   > python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                      --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0

   *Important*: use the original images for FDA, then do mean subtraction, normalization, etc. Otherwise, will be numerical artifacts.

   DeepLab initialization can be downloaded through this [link.](https://drive.google.com/file/d/1dk_4JJZBj4OZ1mkfJ-iLLWPIulQqvHQd/view?usp=sharing)

   LB: beta in the paper, controls the size of the low frequency window to be replaced.

   entW: weight on the entropy term.
   
   ita: coefficient for the robust norm on entropy.
   
   switch2entropy: entropy minimization kicks in after this many steps.


3. Evaluation of the Segmentation Networks Adapted with Multi-band Transfer (multiple betas)

   > python3 evaluation_multi.py --model='DeepLab' --save='../results' 
                                 --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                 --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                 --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"

   Pretrained models on the GTA5 -> CityScapes task using DeepLab backbone can be downloaded [here.](https://drive.google.com/file/d/1HueawBlg6RFaKNt2wAX__1vmmupKqHmS/view?usp=sharing)
   
   The above command should output:
       ===> mIoU19: 50.45
       ===> mIoU16: 54.23
       ===> mIoU13: 59.78
       

4. Get Pseudo Labels for Self-supervised Training

   > python3 getSudoLabel_multi.py --model='DeepLab' --data-list-target='./dataset/cityscapes_list/train.txt' --set='train' 
                                   --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                   --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                   --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"


5. Self-supervised Training with Pseudo Labels

   > python3 SStrain.py --model='DeepLab' --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                        --label-folder='cs_pseudo_label' --LB=0.01 --entW=0.005 --ita=2.0

6. Other Models

   VGG initializations can be downloaded through this [link.](https://drive.google.com/file/d/1pgHtwBKUcbAyItnU4hgMb96UfY1PGiCv/view?usp=sharing)
   
   Pretrained models on the Synthia -> CityScapes task using DeepLab backbone [link.](https://drive.google.com/file/d/1FRI_KIWnubyknChhTOAVl6ZsPxzvEXce/view?usp=sharing)
   
   Pretrained models on the GTA5 -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/15Az8DFaLw1kTgt82KX9rI6S85n7iesdc/view?usp=sharing)
   
   Pretrained models on the Synthia -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/1SC7sxKtic_7ClFmAZDlrBqRaL0pvKYZ8/view?usp=sharing)
   
   
**Acknowledgment**

Code adapted from [BDL.](https://github.com/liyunsheng13/BDL)
