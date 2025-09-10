"""
应用的配置文件

1.0 @nambo 2025-07-20
"""
import os

conf = {
  # 阿里百炼的API KEY
  'DASHSCOPE_API_KEY': 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx',
  # 开启LangSmith（暂不开启）
  # 'LANGSMITH_TRACING': "true",
  # LangSmith的token
  # 'LANGSMITH_API_KEY': 'lsv2_pt_xxxxxxxxxxxxxxxxxxxx_xxxxx',
  # 百炼的模型版本
  'APP_MODEL_VERSION': 'qwen3-235b-a22b',
  # 'APP_MODEL_VERSION': 'qwen3-32b',
  # 雪球的token
  'xq_token': 'xq_a_token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx;u=3782822713',
  # 百度股票的token
  'bd_stock_token': 'BAIDUID=xxxxxxxxxxxxxxxxxxxxxxxxxxx:FG=1',
  # 人民政府网的token配置
  'rmzfw_token': {
    'athenaappkey': 'W6kZwu%2FD5BO4T7pn1zn7FkV8XoWNn3%2FDAuL17RxFh8QM2UM2XazHiTST3ThKlhiJmA8Vt5EfQLq7rrn1z5Lm5zICI7gsn2WSGttCY67ePCWjJSGZotVnXCEuCFEkui%2BXAL3FajWCE2LqWHdmpD%2BmPfHCatk5aP18umuG8nh2cus%3D',
    'athenaappname': '%E5%9B%BD%E7%BD%91%E6%90%9C%E7%B4%A2',
    'Cookie': 'acw_tc=ac11000117529741836235758e00576561739a4292cbea38c2e0e4c30f9a21',
  },
  # python所在目录
  "python_path": '/usr/local/bin/python',
  # Zilliz Cloud的Milvus配置
  'zilliz_milvus_url': 'https://xxxxxxxx.cloud.zilliz.com.cn',
  'zilliz_milvus_token': 'xxxxxxxxxxxxxxxxx',
  'zilliz_milvus_db_name': 'report_agent_db',
  # 多线程配置
  'parallelism_count': 5
}


total_token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
def count_tokens(token_usage):
  global total_token_usage
  if token_usage is not None:
    for key in total_token_usage.keys():
      if key in token_usage:
        total_token_usage[key] += token_usage[key]

  print('【统计token】:', total_token_usage, token_usage)
  return total_token_usage

for key in conf.keys():
  if type(conf[key]) is str and conf[key].strip() != '':
    os.environ[key] = conf[key]