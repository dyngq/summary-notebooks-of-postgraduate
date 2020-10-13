
import requests
from bs4 import BeautifulSoup
url = "..."
headers = {'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
}

payload = {'vToken': '','rdid': '...', 'rdPasswd': '...','returnUrl':'', 'password':''}
response = requests.post(url, data=payload, headers = headers)
# response = requests.post(url)
soup = BeautifulSoup(response.text,'html')
print(soup.prettify())