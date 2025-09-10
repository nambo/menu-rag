"""
大模型调用工具

1.0 @nambo 2025-07-20
"""
import sys
import logging
import json
import os
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

current_file_path = os.path.abspath(__file__)  
current_dir = os.path.dirname(current_file_path)  
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import config
from mcps.common.cache import removeCache, getCache, setCache, get_md5

class InvokeUtil:
  llm = None
  agent = None
  client = None

  # 初始化方法（构造方法）
  def __init__(self, llm=None, client=None, agent=None):
    # 实例属性
    if llm is None:
      # 使用语言模型 通义千问
      llm = ChatTongyi(model=config.conf['APP_MODEL_VERSION'], model_kwargs={
        'enable_thinking': False
      })

    # 配置 MCP 服务器参数
    print('模型初始化完成')
    if client is None:
      client = MultiServerMCPClient({
        "stock": {
          "command": config.conf['python_path'],
          "args": ["./mcps/server_stock.py"],
          "transport": "stdio",
        }
      })
    self.llm = llm
    self.agent = agent
    self.client = client

  async def ainvoke(self, msgs, call_type='llm', res_type='str'):
    res = None
    i = 0
    cache_key = 'llm_invoke,{0},{1},{2}'.format(str(msgs), call_type, res_type)
    while i < 5:
      try:
        res = getCache(cache_key)
        if res is None:
          if call_type == 'llm':
            ag = self.llm
            res = await ag.ainvoke(msgs)
          elif call_type == 'agent':
            if self.agent is None:
              tools = await self.client.get_tools()
              self.agent = create_react_agent(self.llm, tools)
            ag = self.agent
            res = await ag.ainvoke({
              "messages": msgs
            })

          if call_type == 'agent':
            res = res['messages'][-1]

          try:
            config.count_tokens(res.response_metadata['token_usage'])
          except Exception as e:
            print('【统计token】保存失败', res)


          res = res.content
          setCache(cache_key, res)

        else:
          print("从缓存中获取到结果")

        if res_type == 'json':
          if '```' in res:
            res = res.replace('```json', '```')
            res = res.split('```')[1]
          
          res = json.loads(res)
        return res
      except Exception as e:
        print(e)
        if i >= 3:
            removeCache(cache_key)
            raise ValueError('大模型调用失败：' + str(e))
      
      i = i + 1

    return res

if __name__ == '__main__':
  from langchain_community.chat_models import ChatTongyi
  from prompts import stock_prompts

  llm = ChatTongyi(model=config.conf['APP_MODEL_VERSION'], model_kwargs={
    'enable_thinking': False
  })
  invoke_util = InvokeUtil(llm)
  
  msgs = stock_prompts['tpl_get_comp_info'].invoke({"roles": "股票分析师", "target": "个股", "msg": "生成宁德时代研报"}).to_messages()
  import asyncio
  print(config.total_token_usage)
  asyncio.run(invoke_util.ainvoke(msgs, 'llm'))
  print(config.total_token_usage)
  asyncio.run(invoke_util.ainvoke(msgs, 'agent'))
  print(config.total_token_usage)


