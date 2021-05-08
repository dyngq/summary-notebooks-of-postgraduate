import requests
from bs4 import BeautifulSoup
import json
import re

import pandas as pd
import csv
import time
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
}
# 00000489422529DB90F9AE48E0F9D8E0
# while(1):
#     url = f'http://159.75.70.9:8081/pull?u=0000083547D054D91E8D106EFA199372'
#     reponse = requests.get(url,headers=headers)
#     a = json.loads(reponse.text)
#     print(a['c'],a['a'][0],a['t'])
#     aa = a['a'][0]
#     aa = aa*(aa+1)
#     tt = a['t']
#     url_ = f'http://159.75.70.9:8081/push?t={tt}&a={aa}'
#     reponse_ = requests.get(url_,headers=headers)
#     a_ = json.loads(reponse_.text)
#     print(a_)
#     time.sleep(0)


while(1):
    url = f'http://159.75.70.9:8081/pull?u=00000489422529DB90F9AE48E0F9D8E0'
    reponse = requests.get(url,headers=headers)
    a = json.loads(reponse.text)
    print(a['c'],a['a'][0],a['t'])
    a_0 = a['a'][0]
    a_1 = a['a'][1]
    a_2 = a['a'][2]
    n = 199999
    for i in range(0,199999):
        # print(i)
        temp = 0
        if (a_0*i % a_2 == a_1) and i < n:
            n = i
            break
    tt = a['t']
    url_ = f'http://159.75.70.9:8081/push?t={tt}&a={n}'
    reponse_ = requests.get(url_,headers=headers)
    a_ = json.loads(reponse_.text)
    print(a_)
    time.sleep(0)



# window.A274075A = async
# function({
#     a
# }) {
#     return new Promise(_ = >setTimeout(__ = >_(a[0]), 2000))
# }


# window.A3C2EA99 = async
# function({
#     a
# }) {
#     return new Promise(_ = >setTimeout(__ = >_(a[0] * a[0] + a[0]), 2000))
# }


# eval(atob(
#     var _0xe936 = ['A5473788']; (function(_0x48e85c, _0xe936d8) {
#         var _0x23fc5a = function(_0x2858d9) {
#             while (--_0x2858d9) {
#                 _0x48e85c['push'](_0x48e85c['shift']());
#             }
#         };
#         _0x23fc5a(++_0xe936d8);
#     } (_0xe936, 0x196));
#     var _0x23fc = function(_0x48e85c, _0xe936d8) {
#         _0x48e85c = _0x48e85c - 0x0;
#         var _0x23fc5a = _0xe936[_0x48e85c];
#         return _0x23fc5a;
#     };
#     window[_0x23fc('0x0')] = function(_0x335437) {
#         var _0x1aac02 = 0x30d3f;
#         for (var _0x3bed6a = 0x30d3f; _0x3bed6a > 0x0; _0x3bed6a--) {
#             var _0x375340 = 0x0;
#             for (var _0x1ddb77 = 0x0; _0x1ddb77 < _0x3bed6a; _0x1ddb77++) {
#                 _0x375340 += _0x335437['a'][0x0];
#             }
#             _0x375340 % _0x335437['a'][0x2] == _0x335437['a'][0x1] && _0x3bed6a < _0x1aac02 && (_0x1aac02 = _0x3bed6a);
#         }
#         return _0x1aac02;
#     };
# ))

# eval(atob(
#     var _0xe936 = ['A5473788']; (function(_0x48e85c, _0xe936d8) {
#         var _0x23fc5a = function(_0x2858d9) {
#             while (--_0x2858d9) {
#                 _0x48e85c['push'](_0x48e85c['shift']());
#             }
#         };
#         _0x23fc5a(++_0xe936d8);
#     } (_0xe936, 406));
#     var _0x23fc = function(_0x48e85c, _0xe936d8) {
#         _0x48e85c = _0x48e85c - 0x0;
#         var _0x23fc5a = _0xe936[_0x48e85c];
#         return _0x23fc5a;
#     };
#     window[_0x23fc('0x0')] = function(x) {
#         var n = 199999;
#         for (var i = 199999; i > 0; i--) {
#             var temp = 0;
#             for (var j = 0; j < i; j++) {
#                 temp += x['a'][0];
#             }
#             temp % x['a'][2] == x['a'][1] && i < n && (n = i);
#         }
#         return n;
#     };
# ))