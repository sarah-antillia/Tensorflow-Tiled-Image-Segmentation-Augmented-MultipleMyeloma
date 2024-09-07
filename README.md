<h2>Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma (Updated: 2024/09/08)</h2>

This is the first experiment of Tiled Image Segmentation for <a href="https://bcnb.grand-challenge.org/Dataset/">
(MultipleMyeloma Cell Segmentation)
</a>
 based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/105Ppwc5X92_qJhreS1NWUx1-DaCuQd6I/view?usp=sharing">
Tiled-MultipleMyeloma-ImageMask-Dataset.zip</a>, which was derived by us from <a href="https://bcnb.grand-challenge.org/Dataset/">MultipleMyeloma Dataset</a>
<br>
<br>
<!--
Please see also <br> 
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-MultipleMyeloma">
Tiled-ImageMask-Dataset-MultipleMyeloma</a><br><br>
-->
<b>Experiment Strategies:</b><br>
<b>1.Tiled-ImageMask-Dataset:</b><br>
 We trained and validated a TensorFlow UNet model using the Tiled-MultipleMyeloma-ImageMask-Dataset, 
which was tiledly-splitted to 512x512
 and reduced to 512x512 image and mask dataset.<br>

<b>2. Tiled-ImaeSegmentation Method:</b><br>
 We applied the Tiled-Image Segmentation inference method to predict the cell regions for the mini_test images 
with a resolution of 2K pixels. <br>


<br>  

<hr>
<b>Actual Tiled Image Segmentation for Images of 2K pixels</b><br>
As shown below, the tiled inferred masks look similar to the ground truth masks. <br>
<!--
<b>TO DO to improve segmentation accuracy:</b><br>
Change model or training parameters for our Segmentation model.<br>
Use more sophiscated Segmentation model.<br>
-->
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/1991.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1991.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1991.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/1993.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1993.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1993.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/1998.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1998.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1998.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this MultipleMyeloma Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
We used the following dataset in kaggle web-site:<br>
<b>SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images</b><br>
<a href="https://www.kaggle.com/datasets/sbilab/segpc2021dataset">
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</a>
<br><br>

<b>Citation:</b><br>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: <br>
 Segmentation of Multiple Myeloma Plasma Cells <br>
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.<br>
BibTex<br>
@data{segpc2021,<br>
doi = {10.21227/7np1-2q42},<br>
url = {https://dx.doi.org/10.21227/7np1-2q42},<br>
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },<br>
publisher = {IEEE Dataport},<br>
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},<br>
year = {2021} }<br>
IMPORTANT:<br>
If you use this dataset, please cite below publications-<br>
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy,<br> 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," <br>
 Medical Image Analysis, vol. 65, Oct 2020. DOI: <br>
 (2020 IF: 11.148)<br>
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, <br>
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"<br>
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), <br>
 Barcelona, Spain, 2020, pp. 1389-1393.<br>
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal,<br> 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," <br>
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908<br>
<br>
<b>License: </b>CC BY-NC-SA 4.0
<br>

<h3>
<a id="2">
2 Tiled-MultipleMyeloma ImageMask Dataset
</a>
</h3>
 If you would like to train this Tiled-MultipleMyeloma Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/105Ppwc5X92_qJhreS1NWUx1-DaCuQd6I/view?usp=sharing">
Tiled-MultipleMyeloma-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-MultipleMyeloma
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Tiled-MultipleMyeloma DataDatset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/Tiled-MultipleMyeloma-ImageMask-Dataset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set for our segmentation model, 
but we used an online augmentation tool <a href="./src/ImageMaskAugmentor.py">ImageMaskAugmentor.py</a> 
to improve generalization performance.
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Tiled-MultipleMyeloma TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Tiled-MultipleMyeloma and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<pre>
; train_eval_infer.config
; 2024/09/05 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = False
normalization  = False
num_classes    = 1
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.00007
clipvalue      = 0.5
dilation       = (1,1)
;loss           = "bce_iou_loss"
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
show_summary   = False

[dataset]
datasetclass  = "ImageMaskDataset"
;Please specify a resize interpolation algorithm in case of ImageMaskDatast.
resize_interpolation = "cv2.INTER_CUBIC"


[train]
epochs        = 100
batch_size    = 2
steps_per_epoch  = 320
validation_steps = 80
patience      = 10

;metrics       = ["iou_coef", "val_iou_coef"]
metrics       = ["dice_coef", "val_dice_coef"]

model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Tiled-MultipleMyeloma/train/images/"
mask_datapath  = "../../../dataset/Tiled-MultipleMyeloma/train/masks/"

epoch_change_infer     = True
epoch_change_infer_dir = "./epoch_change_infer"
epoch_change_tiledinfer     = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 1

create_backup  = False

learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
save_weights_only  = True

[eval]
image_datapath = "../../../dataset/Tiled-MultipleMyeloma/valid/images/"
mask_datapath  = "../../../dataset/Tiled-MultipleMyeloma/valid/masks/"

[test] 
image_datapath = "../../../dataset/Tiled-MultipleMyeloma/test/images/"
mask_datapath  = "../../../dataset/Tiled-MultipleMyeloma/test/masks/"

[infer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output"

[tiledinfer] 
overlapping   = 128
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"

[segmentation]
colorize      = True
black         = "black"
white         = "green"
blursize      = None

[image]
color_converter = None
;color_converter = "cv2.COLOR_BGR2HSV_FULL"
gamma           = 0
sharpening      = 0

[mask]
blur      = False
blur_size = (3,3)
binarize  = False
;threshold = 128
threshold = 80

[generator]
debug        = False
augmentation = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [60, 120, 180, 240, 300,]
shrinks  = [0.8,]
shears   = [0.1]

deformation = True
distortion  = Truw
sharpening  = False
brightening = False
barrdistortion = True
pincdistortion = True

[deformation]
alpah     = 1300
sigmoids  = [8.0,]

[distortion]
gaussian_filter_rsigma= 40
gaussian_filter_sigma = 0.5
distortions           = [0.02, ]

[barrdistortion]
radius = 0.3
amount = 0.3
centers =  [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

[pincdistortion]
radius = 0.3
amount = -0.3
centers =  [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

[sharpening]
k        = 1.0

[brightening]
alpha  = 1.2
beta   = 10  
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer and epoch_change_tiledinfer callbacks.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_change_tiledinfer  = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images         = 1
</pre>

By using these callbacks, on every epoch_change, the inference procedures can be called
 for an image in <b>mini_test</b> folder. These will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_tiled_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/epoch_change_tiledinfer.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, the training process was stopped at epoch 52 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_console_output_at_epoch_52.png" width="720" height="auto"><br>
<br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Tiled-MultipleMyeloma.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/evaluate_console_output_at_epoch_52.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) and dice_coef to this Tiled-MultipleMyeloma/test were poor results as shown below.
<br>
<pre>
loss,0.348
dice_coef,0.6234
</pre>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-MultipleMyeloma.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<h3>
6 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-MultipleMyeloma.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer_aug.config
</pre>

<hr>
<b>Tiled inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_output_tiledinfer.png" width="1024" height="auto"><br>
<br>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/1994.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1994.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1994.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/2005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/2005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/2005.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/2025.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/2025.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/2025.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/2042.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/2042.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/2042.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/2045.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/2045.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/2045.jpg" width="320" height="auto"></td>
</tr>

</table>

<br>
<br>
<!--
  -->
<b>Comparison of Non-tiled inferred mask and Tiled-Inferred mask</b><br>
As shown below, both non-tiled and tiled inferred masks are far from the ground truth masks, 
but tiled inferencer can generate better results than non-tiled inferencer.

<br>
<table>
<tr>
<th>Mask (ground_truth)</th>

<th>Non-tiled-inferred-mask</th>
<th>Tiled-inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1992.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output/1992.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1992.jpg" width="320" height="auto"></td>

</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/1994.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output/1994.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/1994.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/2045.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output/2045.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/2045.jpg" width="320" height="auto"></td>
</tr>
</table>
<br>

<h3>
References
</h3>
<b>1. SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
<pre>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</pre>
Citation:<br>
<pre>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells 
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.
BibTex
@data{segpc2021,
doi = {10.21227/7np1-2q42},
url = {https://dx.doi.org/10.21227/7np1-2q42},
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },
publisher = {IEEE Dataport},
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},
year = {2021} }
IMPORTANT:
If you use this dataset, please cite below publications-
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy, 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," 
 Medical Image Analysis, vol. 65, Oct 2020. DOI: 
 (2020 IF: 11.148)
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, 
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
 Barcelona, Spain, 2020, pp. 1389-1393.
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal, 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," 
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908
License
CC BY-NC-SA 4.0
</pre>

<b>2. Deep Learning Based Approach For MultipleMyeloma Detection</b><br>
Vyshnav M T, Sowmya V, Gopalakrishnan E A, Sajith Variyar V V, Vijay Krishna Menon, Soman K P<br>
<a href="https://www.researchgate.net/publication/346238471_Deep_Learning_Based_Approach_for_Multiple_Myeloma_Detection">
https://www.researchgate.net/publication/346238471_Deep_Learning_Based_Approach_for_Multiple_Myeloma_Detection
</a>
<br>
<br>
<b>3. Image-Segmentation-Multiple-Myeloma</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/atlan-antillia/Image-Segmentation-Multiple-Myeloma">
https://github.com/atlan-antillia/Image-Segmentation-Multiple-Myeloma
</a>
<br>
<br>
<b>4. Tiled-ImageMask-Dataset-MultipleMyeloma
</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-MultipleMyeloma">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-MultipleMyeloma</a>
