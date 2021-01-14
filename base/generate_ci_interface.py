import argparse
import json
import logging
import traceback
import difflib
import torch

from torch.autograd import Variable
from trainer import Vrae
from config import get_config
import iterator
from utils import prepare_dirs_and_logger
import time

from aiohttp import web  # TODO aiohttp
parser = argparse.ArgumentParser()
parser.add_argument('-lp', '--local_port', required=True,
                    help='local port which runs the service for kb')
args = parser.parse_args()
logger = logging.getLogger('generate_ci')
# 宋词生成预加载
config, unparsed = get_config()
config.is_gpu = True
prepare_dirs_and_logger(config)# ci_generation\base\utils.py 日志输出
step = iterator.Iterator(config) # without1/iterator.py
trainer = Vrae(config, step)
print(config.is_gpu)


routes = web.RouteTableDef()  # TODO aiohttp
def dump(data):
    return json.dumps(data, ensure_ascii=False)
@routes.get('/')  # TODO aiohttp
async def answer(request):
    res = ''
    try:
        if 'key' in request.query:
            key = request.query['key']
            if key.strip():
                print(key)
                res = trainer.Generate_ci(key)
            else:
                res = "输入查询参数不能为空"
        else:
            res = "输入查询参数不能为空"
        return web.json_response(res, dumps=dump)  # TODO aiohttp
    except Exception as e:
        logger.warning(msg=traceback.format_exc())
    return web.json_response({})  # TODO aiohttp


def run_server():
    server = web.Application()  # TODO aiohttp
    server.add_routes(routes)  # TODO aiohttp
    web.run_app(server, port=args.local_port)  # TODO aiohttp


if __name__ == "__main__":
    run_server()