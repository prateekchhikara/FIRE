# FIRE: Food Image to REcipe generation

<p align="center">
<!-- Link to tutorials badge using shields.io -->
<!-- Follow on twitter badge using shields.io -->
  <a href="https://openaccess.thecvf.com/content/WACV2024/html/Chhikara_FIRE_Food_Image_to_REcipe_Generation_WACV_2024_paper.html">
    <img src="https://img.shields.io/badge/Paper-📝-blue">
  </a> 
  <a href="https://www.youtube.com/watch?v=Y6Fuoptl8Rs&t=460s&ab_channel=ComputerVisionFoundationVideos">
    <img src="https://img.shields.io/badge/Youtube-💻-red">
  </a> 
  <a href="https://arxiv.org/abs/2308.14391">
    <img src="https://img.shields.io/badge/ArXiv-📝-yellow">
  </a>
</p>

### Files Description:

- **download.py**: 
  - As the dataset contains over 500,000 images, downloading them sequentially was time-consuming. Therefore, we download the images using 40 processes to speed up the process.

- **data_clean.ipynb**: 
  - While downloading the images from the web, there are cases when the images were corrupt or not able to download. This notebook iterates through all the recipe folders and deletes the folders which are empty. Moreover, it deletes the instances of such recipes from the json files as well. This file performs the deletion of empty folders and cleans the json files which will be used during training and validation.

- **requirements.txt**: 
  - File containing all the libraries and dependencies which should be installed.

- **ingredients/train.py**: 
  - To train the ingredients model, please run:
    ```
    python3 train.py --use_vision_transformer --model_name=ingr_vit --ingrs_only
    ```

- **title.ipynb**: 
  - This is a notebook. Run this in sequence to fine-tune the BLIP model on the Recipe1M dataset.

- **cooking_instructions.ipynb**: 
  - This is a notebook. Run this in sequence to fine-tune the T5 model on the Recipe1M dataset.

- **code_generation.ipynb**: 
  - Notebook for the recipe to code generation based on few-shot prompting. (Requires OpenAI API keys.)

- **customization.ipynb**: 
  - Notebook for the recipe customization based on few-shot prompting. (Requires OpenAI API keys.)

- **metric.ipynb**: 
  - To evaluate the performance of title, ingredients, and cooking instructions output, use this notebook.


## Citation 

```bib
@inproceedings{chhikara2024fire,
  title={Fire: Food image to recipe generation},
  author={Chhikara, Prateek and Chaurasia, Dhiraj and Jiang, Yifan and Masur, Omkar and Ilievski, Filip},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={8184--8194},
  year={2024}
}
```