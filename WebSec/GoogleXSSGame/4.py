import urllib.request
response = urllib.request.urlopen('http://www.xssgame.com/f/__58a1wgqGgI/confirm?next=javascript:alert("a");')
html = response.read()
print(html)