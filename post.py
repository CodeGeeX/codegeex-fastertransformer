import json
import time
import requests

url = 'http://0.0.0.0:5000/code'

headers = {
    "Content-Type": "application/json; charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
}

sentence = ["# language: python\n# write a merge sort function\ndef",
        "# language: python\n# write a merge sort function\ndef",
        "# language: python\n# write a merge sort function\ndef",
]

results=[]
for i in range(1):
    data = json.dumps({'ability': 'seo_article_creation', 
                       'context': sentence, 
                       'temperature':0.2 ,
                       'top_k': 0,
                       'top_p': 0.9,
                       'max_seq_len': 256,
                       'len_penalty': 1.0,
                       'repetition_penalty': 1.0,
                       'presence_penalty': 1.0,
                       'frequency_penalty': 1.0,
                       'end_tokens': [],
                    })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    rdict=json.loads(r.text)
    #result={"sentence":sentence,"result":rdict['generated']}
    #results.append(result)
