# FIRE: Food Image to REcipe generation

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
