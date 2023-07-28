# %%
import pandas as pd
import requests
import os
import dotenv   
from bs4 import BeautifulSoup
# turn off warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Get an api key from : https://apilayer.com/marketplace/google_search-api 
# 
# ### Then set it to a variable named 'google' in your .env file 

# %%
# import .env file
dotenv.load_dotenv()

# %%
# list of all 50 US states
united_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
                    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
                    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
                    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
                    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
                    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
                    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

# %%
# creating the folders
os.makedirs('articles', exist_ok=True)
os.makedirs('more_requests', exist_ok=True)
os.makedirs('requests', exist_ok=True)

# %% [markdown]
# ### Below is how we query the api and get the results split into the first 10 results and the next 90 results

# %%
# for state in united_states:
#   q = f'{state}%20storm%20article'
#   url = f"https://api.apilayer.com/google_search?q={q}"

#   payload = {}
#   headers= {
#   "apikey": os.getenv('google'),
#   }

#   response = requests.request("GET", url, headers=headers, data = payload)

#   status_code = response.status_code
#   result = response.text
# requests folder holds the json files for the first page of results for each state
#   with open(f'requests/{state}.json', 'w') as f:
#         f.write(response.text)
#   print(f'{state} status code: {status_code}')

# %%
# for state in united_states:
#   for i in range(10):

#       q = f'{state}%20storm%20article'
#       url = f"https://api.apilayer.com/google_search?q={q}&start={11+i}"

#       payload = {}
#       headers= {
#       "apikey": os.getenv('google'),
#       }

#       response = requests.request("GET", url, headers=headers, data = payload)

#       status_code = response.status_code
#       result = response.text
#       name = f'{state}_{i}'
# more_requests folder holds the json files for pages 2-10 of results for each state
#       with open(f'more_requests/{name}.json', 'w') as f:
#             f.write(response.text)
#       print(f'{state}_{i} status code: {status_code}')

# %% [markdown]
# ### below gets the urls found in the requests and stores them in a list organized by State

# %%
import json
# urls
urls = {}

# step through the json files and get the urls for the articles
for file in os.listdir('./requests/'):
    with open(f'./requests/{file}') as f:
        urls[file[:-5]] = []
        data = json.load(f)
        for item in data['organic']:
            url = item['link']
            urls[file[:-5]].append(url)
urls

            

# %%
# add more reqeusts to the urls dictionary
for file in os.listdir('./more_requests/'):
    with open(f'./more_requests/{file}') as f:
        # urls[file[:-5]] = []
        data = json.load(f)
        for item in data['organic']:
            url = item['link']
            urls[file[:-7]].append(url)

# %%
# print the total number of urls
total = 0
for key in urls.keys():
    total += len(urls[key])
print(total)

# %% [markdown]
# ### below the urls are iterated through and the webpages are downloaded and stored in a folder named 'articles'

# %%
# download the articles from the urls in parallel
articles = os.listdir('./articles')

def download_article(state, i, url):
    print(f'{state}_{i}')
    if f'{state}_{i}.html' in articles:
        return
    try:
        # stop requests from hanging
        r = requests.get(url, timeout=5)
        
        
        with open(f'./articles/{state}_{i}.html', 'w') as f:
            f.write(r.text)
    except:
        print(f'error with {state}_{i}')

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
print(num_cores)

downloads = Parallel(n_jobs=num_cores)(delayed(download_article)(state, i, url) for state in urls.keys() for i, url in enumerate(urls[state]))



# %% [markdown]
# ### below the articles are read and the text is extracted and stored in a dataframe 

# %%
# organize the articles into a dataframe
articles = os.listdir('./articles')

# create a dataframe
columns = ['title', 'author', 'publication', 'body_text' , 'url', 'state']
df = pd.DataFrame(columns=columns)

# step through the articles and add them to the dataframe

for article in articles:
    with open(f'./articles/{article}') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        title = soup.find('title')
        if title:
            title = title.text
        author = soup.find('meta', {'name': 'author'})
        if author:
            author = author['content']
        publication = soup.find('meta', {'name': 'publication'})
        if publication:
            publication = publication['content']
        body_text = soup.find('body')
        if body_text:
            body_text = body_text.text
        url = soup.find('meta', {'property': 'og:url'})
        if url:
            url = url['content']
        state = article.split('_')[0]
        df = df.append({'title': title, 'author': author, 'publication': publication, 'body_text': body_text, 'url': url, 'state': state}, ignore_index=True)

df

# %%
# Pickle file for use on part 2
df = pd.to_pickle('articles.pkl')
# df

# %% [markdown]
# # Phase 1 complete

# %%



