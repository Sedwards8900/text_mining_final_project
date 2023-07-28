# CSC 8980 - Special Topics in CS: Application-Oriented Text and Web Mining

## PROJECT NAME: Orpheus Text Miner

We have developed this Text Miner to simplify and enhance our further research focused on disaster preparedness, revolving storms and disasters within the 50 U.S. States. The Text Miner is able to extract data and also help us sort the data according to categories found through the implementation of the k-means clustering algorithm.

Below are instructions you must follow to successfully run any of the files and code within this project. You may also copy directly from the github repository at https://github.com/Sedwards8900/text_mining_final_project.

# 1. Instructions on how to run the program:

In order to successfully run the files part_one.ipynb and part_two.ipynb, you must first install all required modules by implementing the following command from terminal:

pip install -r requirements.txt

It is recommended that you search for a Jupyter Notebook extension for the used IDE in addition to installing Jupyter Notebooks within the IDE. If working with VS Code, please refer to https://code.visualstudio.com/docs/datascience/jupyter-notebooks for more information.

After all modules have been installed, open the file part_one.ipynb in the folder and run each cell in a descending order (top-bottom), this program works if you have created a Google API account and have set up your keys and environment variables. For more information on Google's API, please go to https://apilayer.com/marketplace/google_search-api. 

If you fork this repository from github, all you will need to do is use the part_one.ipynb file to extract the df from the already created JSON files obtained via the Google Scraper API, then save the df as a Pickle file for further use in part_two.ipynb.

You can then move on to perform the k-clustering analysis using the k-means algorithm at the end of part_two.ipynb file after having followed the instructions from top to bottom to filter out and clean the text within the obtained weblinks for purposes of this final project.


# 2. Installed required dependencies

The following are the modules necessary to run the program:
- beautifulsoup4==4.12.2
- demoji
- dotenv
- fasttext==0.9.2
- gensim==4.3.1
- numpy==1.24.3
- pandas==2.0.1
- regex==2023.6.3
- requests==2.30.0
- scikit-learn==1.2.2
- scipy==1.10.1
- spacy==3.5.3
- spacy-legacy==3.0.12
- spacy-loggers==1.0.4
- tqdm==4.65.0

# 3. File organization

## A. This project contains the following data files:
- urls.txt, urls 2.txt, urls_all_sme.txt: Text files containing different versions of the links scraped throught the google Scraper API.

- part_one.ipynb: Jupyter Notebook containing code to scrape web content and text capable of allowing an user to get the number of desired links from the google search engine for a particular query.

- part_two.ipynb: Jupyter Notebook containing code to clean, tokenize, POS tagging, and vectorization of words and documents within the set of texts from the files obtained with the help part_one.ipynb.

- orpheus_scraper.py and orpheus_cluster.py: Python files containing same code found in jupyter notebooks but for simplified use through implentation via console commands. You may run the files by using the following command:

    python orpheus_scraper.py

    or
    
    python orpheus_cluster.py