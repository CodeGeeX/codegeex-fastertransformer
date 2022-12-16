import json
import time
import requests
#url = 'http://192.168.249.19:9162/fasttrans'
url = 'http://0.0.0.0:5000/code'
#url = 'http://172.17.0.3:5000/fasttrans'
# headers = {
#     "Content-Type": "application/json; charset=UTF-8",
#     "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
# }

# url = 'https://algo.wudaoai.com/test/glmseries'
headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
    }
# context = '工业互联网（Industrial Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。'
#context = '凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。'
# context = '问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]'


# file = open("data/seo_key_title_part2.txt", "r", encoding='utf-8')

# outfile = open("data/seo_key_title_pages_long_part.txt","w", encoding="utf-8")


# for line in file:
#     key = line.strip().split("\t")[0]
#     title = line.strip().split("\t")[1]

    #print(sentence)

# data = json.dumps({'ability': 'content_creation', 
#                    'context': context, 
# #                    'temperature':1.2 ,
#                      'top_k': 1,
# #                    'top_p': 0,
# #                    'generated_length': 20,
# #                    'presence_penalty': 1,
# #                    'frequency_penalty': 1,
#                    'end_tokens': ['<n>']
#                   })

# sentence = "随意吃一点水过吧"

# pre_prompt = "将错误文本："
# post_prompt =  "纠正为："

    # pre_prompt = '依据关键词"'
    # middle = '"，以<<'
    # post_prompt = '>>为标题创作一篇长文。正文：'
sentence=""
with open("prompt.txt") as f:
    # lines=f.readlines()
    # for line in lines:
    #     sentence+=(line)
    lines = f.readlines()
    sentence = "".join(lines)

results=[]
for i in range(1):
    #sentence = "创作一篇关于云南除虫的文章："
    #sentence = "大模型"
    #sentence = "剧本杀我的同学才不可能是妖怪"
    # content = "酒店信息：大众酒店是一个非常干净的酒店，服务质量很好，欢迎入住 顾客评论：总体很好 酒店回复："

    #sentence = '依据关键词"出国留学 条件 要求"，以<<出国留学的条件和要求>>为标题创作一篇短文。正文：'
    #sentence = '有的菜感觉不新鲜了，居然还是米其林一星，空有其名'
    #sentence = "新华社北京11月4日电 国家主席习近平11月4日上午在人民大会堂会见来华正式访问的德国总理朔尔茨。习近平指出，你是中共二十大召开后首位来访的欧洲领导人，这也是你就任以来首次访华。相信访问将增进双方了解和互信，深化各领域务实合作，为下阶段中德关系发展做好谋划。习近平强调，中德关系发展到今天的高水平，离不开中德几代领导人的高瞻远瞩和政治魄力。今年恰逢中德建交50周年。50载历程表明，只要秉持相互尊重、求同存异、交流互鉴、合作共赢原则，两国关系的大方向就不会偏，步子也会走得很稳。当前，国际形势复杂多变，中德作为有影响力的大国，在变局、乱局中更应该携手合作，为世界和平与发展作出更多贡献。"
    #data = json.dumps({'ability': 'content_creation', 
    data = json.dumps({'ability': 'seo_article_creation', 
        'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':1.0 ,
                            'top_k': 1,
                            'top_p': 0.0,
                            'max_seq_len': 256,
                            'len_penalty': 1.0,
                            'repetition_penalty': 1.0,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 1.0,
                            'end_tokens': [],#'<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json()['generated'])
    rdict=json.loads(r.text)
    result={"sentence":sentence,"result":rdict['generated']}
    results.append(result)
    #print(sentence)
with open("test_result1202.txt","w", encoding="utf-8") as outfile:
    outfile.write(results[0]['result']+"\n")
    #json.dump({"RECORD":results},outfile)
    '''
    sentence = '新华社北京11月4日电 国家主席习近平11月4日上午在人民大会堂会见来华正式访问的德国总理朔尔茨。习近平指出，你是中共二十大召开后首位来访的欧洲领导人，这也是你就任以来首次访华。相信访问将增进双方了解和互信，深化各领域务实合作，为下阶段中德关系发展做好谋划。习近平强调，中德关系发展到今天的高水平，离不开中德几代领导人的高瞻远瞩和政治魄力。今年恰逢中德建交50周年。50载历程表明，只要秉持相互尊重、求同存异、交流互鉴、合作共赢原则，两国关系的大方向就不会偏，步子也会走得很稳。当前，国际形势复杂多变，中德作为有影响力的大国，在变局、乱局中更应该携手合作，为世界和平与发展作出更多贡献。中方愿同德方共同努力，构建面向未来的全方位战略伙伴关系，推动中德、中欧关系取得新的发展。'
    data = json.dumps({'ability': 'abstract_generation', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':0.9 ,
                            'top_k': 1,
                            'top_p': 0.9,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 2,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 2,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    sentence = '便宜坊 烤鸭'
    data = json.dumps({'ability': 'advertisement_generation', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':0.9 ,
                            'top_k': 3,
                            'top_p': 0.9,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 2,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 2,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    sentence = '北京 全聚德'
    data = json.dumps({'ability': 'titleDes_generation', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':0.9 ,
                            'top_k': 3,
                            'top_p': 0.9,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 2,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 2,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    sentence = '北京 全聚德'
    data = json.dumps({'ability': 'titleKey_generation', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':0.9 ,
                            'top_k': 3,
                            'top_p': 0.9,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 2,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 2,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    sentence = '北京'
    data = json.dumps({'ability': 'poem_creation', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':1.0 ,
                            'top_k': 10,
                            'top_p': 0.0,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 1.5,
                            'presence_penalty': 1.5,
                            'frequency_penalty': 1.5,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    sentence = '我们都拥有梅好光明的未来，我们都是国家的栋梁'
    data = json.dumps({'ability': 'correction', 
                            'context': sentence, 
                            # 'context': pre_prompt + sentence + post_prompt,
                            'temperature':1.0 ,
                            'top_k': 2,
                            'top_p': 0,
                            'max_seq_len': 512,
                            'len_penalty': 0,
                            'repetition_penalty': 1,
                            'presence_penalty': 1.0,
                            'frequency_penalty': 1,
                            'end_tokens': ['<n>'],
                            })
    time1=time.time()
    r = requests.post(url, data, headers=headers)
    time2=time.time()
    print("time used",time2-time1)
    print(r.json())
    #print(r.json()['data']['outputText'].replace("<n>", "\n"))
    '''
# print(r.json()['data']['outputText'].replace("<n>", "\n"))
    # print('---------------------------------------------------------------------')
    # print("orii:" + sentence + "  correct:"  + r.json()['data']['outputText'])
    # outfile.write(r.json()['data']['outputText'] + "\n")
